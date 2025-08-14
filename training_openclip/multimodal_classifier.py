import csv
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import tqdm
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    Resize,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from mydata import MyFungiModule
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score
from torchvision import models


class SeparateWeightsSaveCallback(Callback):
    """
    Custom callback to save image encoder and classifier weights separately.
    """

    def __init__(self, save_dir, monitor="val_f1", mode="max"):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the monitored metric
        current_score = trainer.logged_metrics.get(self.monitor, None)

        if current_score is not None:
            # Check if this is the best score so far
            is_best = False
            if self.mode == "max" and current_score > self.best_score:
                is_best = True
                self.best_score = current_score
            elif self.mode == "min" and current_score < self.best_score:
                is_best = True
                self.best_score = current_score

            if is_best:
                print(f"New best {self.monitor}: {current_score:.4f}")
                # Save separate weights
                pl_module.save_weights_separately(self.save_dir)


def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == "train":
        return Compose(
            [
                RandomResizedCrop((width, height), scale=(0.8, 1.0)),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Resize(width, height),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")


class MultimodalClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes=1000,
        lr=1e-4,
        embedding_dim=19,  # Dimension of the embeddings: 12 (text) + 7 (seasonal) = 19
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.embedding_dim = embedding_dim

        # Use EfficientNet-B0 as image encoder
        self.efficientnet_model = models.efficientnet_b0(pretrained=True)
        # Remove the classifier to use only as feature extractor
        self.image_encoder = nn.Sequential(*list(self.efficientnet_model.children())[:-1])
        # Add global average pooling to get fixed size features
        self.image_encoder.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        self.image_encoder.add_module("flatten", nn.Flatten())

        # For EfficientNet-B0, the feature dimension is 1280
        feature_dim = 1280

        # self.loss_fn = nn.CrossEntropyLoss()
        # Class-weighted CrossEntropyLoss for better macro F1 optimization
        # Will be computed dynamically based on class frequencies
        self.loss_fn = None  # Will be set after seeing class distribution
        self.use_focal_loss = False  # Use focal loss for hard example mining
        self.focal_alpha = 1.0
        self.focal_gamma = 2.0

        # Embedding preprocessing layer - single hidden layer
        self.embedding_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        # Calculate combined features dimension after embedding processing
        combined_features_dim = (
            feature_dim + self.embedding_dim
        )  # Image features + embeddings (unchanged size)

        # Normalization layers for balanced multimodal fusion
        self.image_norm = nn.LayerNorm(feature_dim)
        self.embedding_norm = nn.LayerNorm(self.embedding_dim)

        # Classifier with no hidden layers - direct mapping to output classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(combined_features_dim, num_classes),
        )

        # Initialize F1Score metrics for training and validation
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def compute_class_weights(self, metadata_df=None):
        """
        Compute class weights based on inverse class frequency for better macro F1.
        Uses metadata dataframe for faster computation.
        """
        if metadata_df is None:
            # Load metadata if not provided
            metadata_path = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/metadata/metadata.csv"
            metadata_df = pd.read_csv(metadata_path)

        # Filter for training data only (files starting with "fungi_train")
        train_df = metadata_df[
            metadata_df["filename_index"].str.startswith("fungi_train")
        ]

        # Get valid labels (non-null taxonID_index)
        valid_labels = train_df["taxonID_index"].dropna()

        # Count samples per class
        class_counts = torch.zeros(self.hparams.num_classes)
        total_samples = len(valid_labels)

        for label in valid_labels:
            if 0 <= label < self.hparams.num_classes:
                class_counts[int(label)] += 1

        # Compute inverse frequency weights
        class_weights = total_samples / (self.hparams.num_classes * class_counts + 1e-8)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * self.hparams.num_classes

        return class_weights

    def focal_loss(self, logits, targets, class_weights=None):
        """
        Focal Loss implementation for hard example mining and class imbalance.
        """
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=class_weights, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def compute_loss(self, logits, labels):
        """
        Compute loss as average of class-weighted and standard cross-entropy losses.
        """
        if self.loss_fn is None:
            # Use simple CrossEntropyLoss if class weights not computed yet
            return nn.functional.cross_entropy(logits, labels)

        # Compute standard cross-entropy loss (unweighted)
        standard_loss = nn.functional.cross_entropy(logits, labels)

        # Compute class-weighted loss
        if self.use_focal_loss:
            weighted_loss = self.focal_loss(logits, labels, self.class_weights)
        else:
            weighted_loss = self.loss_fn(logits, labels)

        # Return average of both losses
        combined_loss = (standard_loss + weighted_loss) / 2.0
        return combined_loss

    def forward(self, images, embeddings):
        """
        Forward pass through the network.

        Args:
            images: Input images tensor
            embeddings: Input embeddings tensor

        Returns:
            logits: Output logits for classification
        """
        # Ensure embedding is a tensor and on the correct device
        if not isinstance(embeddings, torch.Tensor):
            # If embedding is a list of tensors, stack them
            if (
                isinstance(embeddings, list)
                and len(embeddings) > 0
                and isinstance(embeddings[0], torch.Tensor)
            ):
                embeddings = torch.stack(embeddings).T
            else:
                embeddings = torch.tensor(embeddings)
        embeddings = embeddings.to(self.device, torch.float)

        # Get image features
        image_features = self.image_encoder(images)

        # Process embeddings through two hidden layers
        processed_embeddings = self.embedding_processor(embeddings)

        # Normalize both modalities for balanced fusion
        normalized_image_features = self.image_norm(image_features)
        normalized_embeddings = self.embedding_norm(processed_embeddings)

        # Concatenate normalized features from both modalities
        combined_features = torch.cat(
            [normalized_image_features, normalized_embeddings], dim=1
        )

        # Get classification logits
        logits = self.classifier(combined_features)

        return logits

    def save_weights_separately(self, save_dir):
        """
        Save image encoder and classifier weights to separate files.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save image encoder weights
        image_encoder_path = os.path.join(save_dir, "efficientnet_encoder_weights.pth")
        torch.save(self.image_encoder.state_dict(), image_encoder_path)
        print(f"Image encoder weights saved to: {image_encoder_path}")

        # Save classifier weights
        classifier_path = os.path.join(save_dir, "efficientnet_classifier_weights.pth")
        torch.save(self.classifier.state_dict(), classifier_path)
        print(f"Classifier weights saved to: {classifier_path}")

        return image_encoder_path, classifier_path

    def load_image_encoder_weights(self, weights_path):
        """
        Load image encoder weights from a file.
        """
        if os.path.exists(weights_path):
            self.image_encoder.load_state_dict(torch.load(weights_path))
            print(f"Image encoder weights loaded from: {weights_path}")
        else:
            print(f"Warning: Image encoder weights file not found: {weights_path}")

    def load_classifier_weights(self, weights_path):
        """
        Load classifier weights from a file.
        """
        if os.path.exists(weights_path):
            self.classifier.load_state_dict(torch.load(weights_path))
            print(f"Classifier weights loaded from: {weights_path}")
        else:
            print(f"Warning: Classifier weights file not found: {weights_path}")

    def training_step(self, batch, batch_idx):
        images, labels, embeddings, file_paths = batch

        # Forward pass
        logits = self.forward(images, embeddings)
        loss = self.compute_loss(logits, labels)

        # Calculate training accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        # Calculate F1 score
        f1 = self.train_f1(preds, labels)

        # Log loss, accuracy, and F1 score
        self.log("train_loss", loss, batch_size=len(labels))
        self.log("train_acc", acc, batch_size=len(labels))
        self.log("train_f1", f1, batch_size=len(labels))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, embeddings, file_paths = batch

        # Forward pass
        logits = self.forward(images, embeddings)
        loss = self.compute_loss(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        # Calculate F1 score
        f1 = self.val_f1(preds, labels)

        # Log loss, accuracy, and F1 score
        self.log("val_loss", loss, batch_size=len(labels))
        self.log("val_acc", acc, batch_size=len(labels))
        self.log("val_f1", f1, batch_size=len(labels))
        return loss

    def setup(self, stage=None):
        """
        Setup method called before training to compute class weights.
        """
        if stage == "fit" or stage is None:
            # Get the datamodule from trainer
            if (
                hasattr(self.trainer, "datamodule")
                and self.trainer.datamodule is not None
            ):
                # Use the metadata dataframe from the datamodule for faster computation
                metadata_df = self.trainer.datamodule.df
                print("Computing class weights for macro F1 optimization...")
                self.class_weights = self.compute_class_weights(metadata_df).to(
                    self.device
                )

                # Initialize class-weighted loss function
                self.loss_fn = nn.CrossEntropyLoss()  # weight=self.class_weights

                print(
                    f"Class weights computed: {self.class_weights[:10]}..."
                )  # Show first 10 weights
            else:
                print("Warning: Could not compute class weights, using unweighted loss")
                self.class_weights = None
                self.loss_fn = nn.CrossEntropyLoss()

    def on_train_epoch_end(self):
        # Get the logged metrics for this epoch
        train_acc = self.trainer.logged_metrics.get("train_acc", 0.0)
        train_loss = self.trainer.logged_metrics.get("train_loss", 0.0)
        train_f1 = self.trainer.logged_metrics.get("train_f1", 0.0)
        print(
            f"Epoch {self.current_epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}"
        )

    def on_validation_epoch_end(self):
        # Get the logged metrics for this epoch
        val_acc = self.trainer.logged_metrics.get("val_acc", 0.0)
        val_loss = self.trainer.logged_metrics.get("val_loss", 0.0)
        val_f1 = self.trainer.logged_metrics.get("val_f1", 0.0)
        print(
            f"Epoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
        )

    def configure_optimizers(self):
        # Optimize all parameters (image encoder + classifier)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.9, patience=1, eps=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": (
                    "val_loss"
                ),  # Monitor validation loss for learning rate reduction
                "frequency": 1,
            },
        }


def train_model():
    """
    Train the multimodal classifier.
    """
    # Load data module
    data_module = MyFungiModule()
    data_module.setup()

    # Create model
    model = MultimodalClassifier(
        num_classes=data_module.num_classes,
        lr=1e-3,
        embedding_dim=19,  # Dimension of embeddings: 12 (text) + 7 (seasonal) = 19
    )

    # Directory for saving separate weights
    results_dir = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"

    # ModelCheckpoint callback to save best model based on validation F1 score
    checkpoint_callback = ModelCheckpoint(
        dirpath=results_dir,
        filename="multimodal_best_model",
        monitor="val_f1",  # Monitor validation F1 score
        mode="max",
        save_top_k=1,
        save_weights_only=False,
    )

    # Custom callback to save separate weights
    separate_weights_callback = SeparateWeightsSaveCallback(
        save_dir=results_dir, monitor="val_f1", mode="max"
    )

    # EarlyStopping callback to stop training when validation loss stops improving
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        mode="min",  # Stop when validation loss stops decreasing
        patience=10,  # Wait 10 epochs before stopping
        verbose=True,  # Print message when early stopping is triggered
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        callbacks=[
            checkpoint_callback,
            separate_weights_callback,
            early_stopping_callback,
        ],
    )

    # Compute class weights before training
    print("Computing class weights for macro F1 optimization...")
    class_weights = model.compute_class_weights(data_module.df).to(model.device)
    model.class_weights = class_weights
    model.loss_fn = nn.CrossEntropyLoss()  # weight=class_weights
    print(f"Class weights computed: {class_weights[:10]}...")  # Show first 10 weights

    trainer.fit(model, datamodule=data_module)

    print("Training completed!")
    return model


def evaluate_on_test_set(
    session_name,
    checkpoint_path=None,
    image_encoder_path=None,
    classifier_path=None,
):
    """
    Evaluate the trained model on the test set and save predictions to a CSV file.
    """
    # Ensure results directory exists
    results_dir = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
    os.makedirs(results_dir, exist_ok=True)

    output_csv_path = os.path.join(results_dir, "test_predictions.csv")

    # Load data module and setup test data
    data_module = MyFungiModule()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # Load from full checkpoint
        model = MultimodalClassifier.load_from_checkpoint(
            checkpoint_path, num_classes=data_module.num_classes
        )
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    elif image_encoder_path is not None and classifier_path is not None:
        # Load from separate weight files
        model = MultimodalClassifier(
            num_classes=data_module.num_classes,
            lr=0.001,
            embedding_dim=19,  # Make sure this matches your training embedding dimension
        )
        model.load_image_encoder_weights(image_encoder_path)
        model.load_classifier_weights(classifier_path)
        print("Model loaded from separate weight files")
    else:
        # Use untrained model
        model = MultimodalClassifier(
            num_classes=data_module.num_classes,
            lr=0.001,
            embedding_dim=19,  # Make sure this matches your training embedding dimension
        )
        print("Using untrained model (only pretrained weights)")

    model.to(device)
    model.eval()

    # Collect Predictions
    results = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating on test set"):
            if len(batch) == 4:
                # Handle case with embeddings: images, labels, embeddings, file_paths
                images, labels, embeddings, file_paths = batch
                images = images.to(device)

                # Forward pass
                logits = model.forward(images, embeddings)
                filenames = file_paths
            elif len(batch) == 3:
                # Handle case with embeddings but no file paths: images, labels, embeddings
                images, labels, embeddings = batch
                images = images.to(device)

                # Forward pass
                logits = model.forward(images, embeddings)
                filenames = [f"sample_{i}" for i in range(len(images))]
            else:
                print("Did not load the test embeddings as was hoped...")
                # Handle case without embeddings (fallback)
                images, filenames = batch[0], batch[-1]
                images = images.to(device)
                # For models expecting embeddings, we need to create dummy embeddings
                batch_size = images.shape[0]
                dummy_embeddings = torch.zeros(batch_size, model.embedding_dim).to(
                    device, torch.float
                )
                logits = model.forward(images, dummy_embeddings)

            predictions = logits.argmax(1).cpu().numpy()

            # Store filenames and predictions
            for filename, pred in zip(filenames, predictions):
                results.append([filename, pred])

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions

    print(f"Test predictions saved to {output_csv_path}")
    return output_csv_path


def evaluate_on_validation_set(
    checkpoint_path=None,
    image_encoder_path=None,
    classifier_path=None,
):
    """
    Evaluate the trained model on the validation set and provide detailed class-wise metrics.
    """
    # Load data module and setup validation data
    data_module = MyFungiModule()
    data_module.setup(stage="fit")  # This sets up both train and val data
    val_loader = data_module.val_dataloader()

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        # Load from full checkpoint
        model = MultimodalClassifier.load_from_checkpoint(
            checkpoint_path, num_classes=data_module.num_classes
        )
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    elif image_encoder_path is not None and classifier_path is not None:
        # Load from separate weight files
        model = MultimodalClassifier(
            num_classes=data_module.num_classes,
            lr=0.001,
            embedding_dim=19,  # Make sure this matches your training embedding dimension
        )
        model.load_image_encoder_weights(image_encoder_path)
        model.load_classifier_weights(classifier_path)
        print("Model loaded from separate weight files")
    else:
        print("No trained model found. Please provide valid model weights.")
        return

    model.to(device)
    model.eval()

    # Collect all predictions and true labels
    all_predictions = []
    all_labels = []

    print("Evaluating on validation set...")
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc="Processing validation batches"):
            images, labels, embeddings, file_paths = batch
            images = images.to(device)

            # Forward pass
            logits = model.forward(images, embeddings)
            predictions = logits.argmax(1).cpu().numpy()

            # Store predictions and labels
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for easier processing
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate overall accuracy
    overall_accuracy = (all_predictions == all_labels).mean()
    print(f"\nOverall Validation Accuracy: {overall_accuracy:.4f}")

    # Calculate class-wise metrics
    num_classes = data_module.num_classes
    class_results = []  # Store class results for sorting

    print("\nClass-wise Results:")
    print("=" * 60)
    print(f"{'Class':<8} {'Count':<8} {'Correct':<8} {'Accuracy':<10}")
    print("=" * 60)

    for class_id in range(num_classes):
        # Find instances of this class
        class_mask = all_labels == class_id
        class_count = class_mask.sum()

        if class_count > 0:
            # Calculate accuracy for this class
            class_predictions = all_predictions[class_mask]
            class_true_labels = all_labels[class_mask]
            correct_predictions = (class_predictions == class_true_labels).sum()
            class_accuracy = correct_predictions / class_count

            # Store class results for sorting
            class_results.append(
                {
                    "class_id": class_id,
                    "count": class_count,
                    "correct": correct_predictions,
                    "accuracy": class_accuracy,
                }
            )

    # Sort class results by accuracy (descending order - best first)
    class_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Display sorted results
    class_accuracies = []
    class_counts = []

    for result in class_results:
        class_accuracies.append(result["accuracy"])
        class_counts.append(result["count"])

        print(
            f"{result['class_id']:<8} {result['count']:<8} {result['correct']:<8} {result['accuracy']:<10.4f}"
        )

    # Display classes with no samples
    for class_id in range(num_classes):
        class_mask = all_labels == class_id
        class_count = class_mask.sum()
        if class_count == 0:
            print(f"{class_id:<8} {0:<8} {0:<8} {'N/A':<10}")

    print("=" * 60)

    # Calculate and display summary statistics
    if class_accuracies:
        mean_class_accuracy = np.mean(class_accuracies)
        std_class_accuracy = np.std(class_accuracies)
        min_class_accuracy = np.min(class_accuracies)
        max_class_accuracy = np.max(class_accuracies)

        print("\nSummary Statistics:")
        print(
            f"Mean class accuracy: {mean_class_accuracy:.4f} ± {std_class_accuracy:.4f}"
        )
        print(f"Min class accuracy: {min_class_accuracy:.4f}")
        print(f"Max class accuracy: {max_class_accuracy:.4f}")
        print(f"Total validation samples: {len(all_labels)}")
        print(f"Classes with samples: {len(class_accuracies)}/{num_classes}")

    # Check metadata completeness
    print("\nChecking metadata completeness for validation samples...")

    # Load metadata to check completeness
    metadata_path = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/metadata/metadata.csv"
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)

        # Get validation file paths from data module
        val_dataset = data_module.val_dataset
        validation_files = [os.path.basename(fp) for fp in val_dataset.file_paths]

        # Count samples with complete metadata
        complete_metadata_count = 0
        total_validation_samples = len(validation_files)

        for filename in validation_files:
            try:
                # Find the corresponding row in metadata using full filename
                metadata_row = metadata_df[metadata_df["filename_index"] == filename]

                if not metadata_row.empty:
                    row = metadata_row.iloc[0]
                    # Check if all required metadata fields are present and not null
                    required_fields = [
                        "Longitude",
                        "Latitude",
                        "eventDate",
                        "Habitat",
                        "Substrate",
                    ]
                    has_complete_metadata = all(
                        pd.notna(row[field]) and str(row[field]).strip() != ""
                        for field in required_fields
                    )

                    if has_complete_metadata:
                        complete_metadata_count += 1
            except Exception:
                # Handle any errors in metadata lookup
                continue

        metadata_completeness = (
            complete_metadata_count / total_validation_samples
            if total_validation_samples > 0
            else 0
        )
        print(
            f"Validation samples with complete metadata: {complete_metadata_count}/{total_validation_samples} ({metadata_completeness:.2%})"
        )
    else:
        print("Metadata file not found - cannot check completeness")

    return {
        "overall_accuracy": overall_accuracy,
        "class_accuracies": class_accuracies,
        "class_counts": class_counts,
        "predictions": all_predictions,
        "labels": all_labels,
    }


if __name__ == "__main__":
    run_mode = "test"  # "train" "test" "validate"

    if run_mode == "train":
        train_model()
        print("Done training! Will now evaluate the performance!")

    elif run_mode == "test":
        # Run the best model parameters on the test set
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )
        checkpoint_path = os.path.join(results_dir, "multimodal_best_model.ckpt")

        # Check for separate weight files
        efficientnet_encoder_path = os.path.join(
            results_dir, "efficientnet_encoder_weights.pth"
        )
        efficientnet_classifier_path = os.path.join(
            results_dir, "efficientnet_classifier_weights.pth"
        )

        session_name = "MultimodalClassifier_test"

        if os.path.exists(checkpoint_path):
            # Use the full checkpoint if available
            evaluate_on_test_set(session_name, checkpoint_path=checkpoint_path)
        elif os.path.exists(efficientnet_encoder_path) and os.path.exists(
            efficientnet_classifier_path
        ):
            # Use separate weight files if available
            print("Loading model from separate weight files...")
            evaluate_on_test_set(
                session_name,
                image_encoder_path=efficientnet_encoder_path,
                classifier_path=efficientnet_classifier_path,
            )
        else:
            print("No trained model found. Please train the model first.")
            print(f"Looking for: {checkpoint_path}")
            print(
                f"Or separate files: {efficientnet_encoder_path} and {efficientnet_classifier_path}"
            )

    elif run_mode == "validate":
        # Evaluate the best model on the validation set with detailed class-wise metrics
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )
        checkpoint_path = os.path.join(results_dir, "multimodal_best_model.ckpt")

        # Check for separate weight files
        efficientnet_encoder_path = os.path.join(
            results_dir, "efficientnet_encoder_weights.pth"
        )
        efficientnet_classifier_path = os.path.join(
            results_dir, "efficientnet_classifier_weights.pth"
        )

        if os.path.exists(checkpoint_path):
            # Use the full checkpoint if available
            print("Evaluating with full checkpoint...")
            evaluate_on_validation_set(checkpoint_path=checkpoint_path)
        elif os.path.exists(efficientnet_encoder_path) and os.path.exists(
            efficientnet_classifier_path
        ):
            # Use separate weight files if available
            print("Evaluating model from separate weight files...")
            evaluate_on_validation_set(
                image_encoder_path=efficientnet_encoder_path,
                classifier_path=efficientnet_classifier_path,
            )
        else:
            print("No trained model found. Please train the model first.")
            print(f"Looking for: {checkpoint_path}")
            print(
                f"Or separate files: {efficientnet_encoder_path} and {efficientnet_classifier_path}"
            )
