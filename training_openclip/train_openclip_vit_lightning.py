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
from mydata import (
    FungiDataset,
    MyFungiModule,
    create_text_from_habitat_and_substrate,
)  # Example data module

# from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, tokenize
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


class OpenCLIPViTLightning(pl.LightningModule):
    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        num_classes=1000,
        lr=1e-4,
        train_image_encoder=True,
        use_efficientnet=False,  # New flag to use EfficientNet instead of CLIP ViT
        embedding_dim=12,  # Dimension of the embeddings to concatenate with image features
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.train_image_encoder = train_image_encoder
        self.use_efficientnet = use_efficientnet
        self.embedding_dim = embedding_dim

        if use_efficientnet:
            # Use EfficientNet-B0 as image encoder
            self.efficientnet_model = models.efficientnet_b0(pretrained=True)
            # Remove the classifier to use only as feature extractor
            self.image_encoder = nn.Sequential(
                *list(self.efficientnet_model.children())[:-1]
            )
            # Add global average pooling to get fixed size features
            self.image_encoder.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
            self.image_encoder.add_module("flatten", nn.Flatten())

            # For EfficientNet-B0, the feature dimension is 1280
            feature_dim = 1280

            # Create dummy CLIP model for text encoding only
            self.model, _, self.preprocess = create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.text_encoder = self.model.transformer
            self.logit_scale = self.model.logit_scale
        else:
            # Use CLIP ViT as image encoder (original behavior)
            self.model, _, self.preprocess = create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.image_encoder = self.model.visual
            self.text_encoder = self.model.transformer
            self.logit_scale = self.model.logit_scale

            # Get feature dimension from a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                dummy_features = self.image_encoder(dummy_input)
                feature_dim = dummy_features.shape[1]

        self.loss_fn = nn.CrossEntropyLoss()

        # Set image encoder training mode
        if not self.train_image_encoder:
            # Freeze the image encoder parameters
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        else:
            # Unfreeze the image encoder parameters for training
            for param in self.image_encoder.parameters():
                param.requires_grad = True

        # Always freeze the text encoder parameters (we're not using it for classification)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Calculate hidden layer size
        combined_features_dim = (
            feature_dim + self.embedding_dim
        )  # Image features + embeddings
        hidden_dim = combined_features_dim // 2  # Hidden layer: half of combined features

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(combined_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        # self.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(feature_dim, num_classes),
        # )

        # Initialize F1Score metrics for training and validation
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def save_weights_separately(self, save_dir):
        """
        Save image encoder and classifier weights to separate files.
        Filenames will reflect the encoder type (CLIP ViT or EfficientNet).
        """
        os.makedirs(save_dir, exist_ok=True)

        # Choose filename based on encoder type
        if self.use_efficientnet:
            image_encoder_filename = "efficientnet_encoder_weights.pth"
            classifier_filename = "efficientnet_classifier_weights.pth"
        else:
            image_encoder_filename = "image_encoder_weights.pth"
            classifier_filename = "classifier_weights.pth"

        # Save image encoder weights
        image_encoder_path = os.path.join(save_dir, image_encoder_filename)
        torch.save(self.image_encoder.state_dict(), image_encoder_path)
        print(f"Image encoder weights saved to: {image_encoder_path}")

        # Save classifier weights
        classifier_path = os.path.join(save_dir, classifier_filename)
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

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_tokens = tokenize(text)
        text_features = self.text_encoder(text_tokens)
        return image_features, text_features

    def training_step(self, batch, batch_idx):
        images, labels, embedding, file_paths = batch
        # Ensure embedding is a tensor and on the correct device
        if not isinstance(embedding, torch.Tensor):
            # If embedding is a list of tensors, stack them
            if (
                isinstance(embedding, list)
                and len(embedding) > 0
                and isinstance(embedding[0], torch.Tensor)
            ):
                embedding = torch.stack(embedding).T
            else:
                embedding = torch.tensor(embedding)
        embedding = embedding.to(self.device, torch.float)

        # # Flatten embedding if it has more than 2 dimensions
        # if embedding.dim() > 2:
        #     embedding = embedding.flatten(start_dim=1)
        # elif embedding.dim() == 1:
        #     embedding = embedding.unsqueeze(0)

        # For CLIP training without text, we'll use image features for classification
        image_features = self.image_encoder(images)

        # Concatenate image features with embeddings
        combined_features = torch.cat([image_features, embedding], dim=1)

        logits = self.classifier(combined_features)
        loss = self.loss_fn(logits, labels)

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
        images, labels, embedding, file_paths = batch
        # Ensure embedding is a tensor and on the correct device
        if not isinstance(embedding, torch.Tensor):
            # If embedding is a list of tensors, stack them
            if (
                isinstance(embedding, list)
                and len(embedding) > 0
                and isinstance(embedding[0], torch.Tensor)
            ):
                embedding = torch.stack(embedding).T
            else:
                embedding = torch.tensor(embedding)

        embedding = embedding.to(self.device, torch.float)

        # # Flatten embedding if it has more than 2 dimensions
        # if embedding.dim() > 2:
        #     embedding = embedding.flatten(start_dim=1)
        # elif embedding.dim() == 1:
        #     embedding = embedding.unsqueeze(0)

        image_features = self.image_encoder(images)

        # Concatenate image features with embeddings
        combined_features = torch.cat([image_features, embedding], dim=1)

        logits = self.classifier(combined_features)
        loss = self.loss_fn(logits, labels)

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
        # Create list of parameters to optimize
        params_to_optimize = []

        if self.train_image_encoder:
            # Include image encoder parameters if training is enabled
            params_to_optimize.extend(list(self.image_encoder.parameters()))
            print("Training image encoder parameters")

        # Always include classifier parameters
        params_to_optimize.extend(list(self.classifier.parameters()))
        print("Training classifier parameters")

        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
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


def evaluate_clip_network_on_test_set(
    checkpoint_path,
    session_name,
    image_encoder_path=None,
    classifier_path=None,
    use_efficientnet=False,
):
    """
    Evaluate CLIP network on the test set and save predictions to a CSV file.
    Can load from either a full checkpoint or separate weight files.
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
        model = OpenCLIPViTLightning.load_from_checkpoint(
            checkpoint_path, num_classes=data_module.num_classes
        )
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    elif image_encoder_path is not None and classifier_path is not None:
        # Load from separate weight files
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=0.001,
            train_image_encoder=True,
            use_efficientnet=use_efficientnet,
            embedding_dim=12,  # Make sure this matches your training embedding dimension
        )
        model.load_image_encoder_weights(image_encoder_path)
        model.load_classifier_weights(classifier_path)
        print("Model loaded from separate weight files")
    else:
        # Use untrained model
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=0.001,
            use_efficientnet=use_efficientnet,
            embedding_dim=12,  # Make sure this matches your training embedding dimension
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
                images, labels, embedding, file_paths = batch
                images = images.to(device)
                image_features = model.image_encoder(images)

                # Ensure embedding is a tensor and on the correct device
                if not isinstance(embedding, torch.Tensor):
                    # If embedding is a list of tensors, stack them
                    if (
                        isinstance(embedding, list)
                        and len(embedding) > 0
                        and isinstance(embedding[0], torch.Tensor)
                    ):
                        embedding = torch.stack(embedding).T
                    else:
                        embedding = torch.tensor(embedding)
                embedding = embedding.to(device, torch.float)
                combined_features = torch.cat([image_features, embedding], dim=1)
                logits = model.classifier(combined_features)

                # Use the actual file paths from the batch
                filenames = file_paths
            elif len(batch) == 3:
                # Handle case with embeddings but no file paths: images, labels, embeddings
                images, labels, embedding = batch
                images = images.to(device)
                image_features = model.image_encoder(images)

                # Ensure embedding is a tensor and on the correct device
                if not isinstance(embedding, torch.Tensor):
                    # If embedding is a list of tensors, stack them
                    if (
                        isinstance(embedding, list)
                        and len(embedding) > 0
                        and isinstance(embedding[0], torch.Tensor)
                    ):
                        embedding = torch.stack(embedding).T
                    else:
                        embedding = torch.tensor(embedding)
                embedding = embedding.to(device, torch.float)
                combined_features = torch.cat([image_features, embedding], dim=1)
                logits = model.classifier(combined_features)

                # Generate filenames as fallback
                filenames = [f"sample_{i}" for i in range(len(images))]
            else:
                print("Did not load the test embeddings as was hoped...")
                # Handle case without embeddings (fallback)
                images, filenames = batch[0], batch[-1]
                images = images.to(device)
                image_features = model.image_encoder(images)
                # For models expecting embeddings, we need to create dummy embeddings
                # This should match the embedding dimension used in training
                batch_size = images.shape[0]
                dummy_embeddings = torch.zeros(batch_size, model.embedding_dim).to(
                    device, torch.float
                )
                combined_features = torch.cat([image_features, dummy_embeddings], dim=1)
                logits = model.classifier(combined_features)

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
    checkpoint_path,
    image_encoder_path=None,
    classifier_path=None,
    use_efficientnet=False,
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
        model = OpenCLIPViTLightning.load_from_checkpoint(
            checkpoint_path, num_classes=data_module.num_classes
        )
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    elif image_encoder_path is not None and classifier_path is not None:
        # Load from separate weight files
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=0.001,
            train_image_encoder=True,
            use_efficientnet=use_efficientnet,
            embedding_dim=12,  # Make sure this matches your training embedding dimension
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
            images, labels, embedding, file_paths = batch
            images = images.to(device)

            # Handle embeddings properly (same as in training/validation)
            if not isinstance(embedding, torch.Tensor):
                if (
                    isinstance(embedding, list)
                    and len(embedding) > 0
                    and isinstance(embedding[0], torch.Tensor)
                ):
                    embedding = torch.stack(embedding).T
                else:
                    embedding = torch.tensor(embedding)
            embedding = embedding.to(device, torch.float)

            # Get predictions
            image_features = model.image_encoder(images)
            combined_features = torch.cat([image_features, embedding], dim=1)
            logits = model.classifier(combined_features)
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

    return {
        "overall_accuracy": overall_accuracy,
        "class_accuracies": class_accuracies,
        "class_counts": class_counts,
        "predictions": all_predictions,
        "labels": all_labels,
    }


if __name__ == "__main__":
    run_mode = "validate"  # "train" "test" "encode" "validate"

    if run_mode == "train":
        # Example usage with MyFungiModule (adapt as needed)
        data_module = MyFungiModule()  # reduced_datarate=0.2
        data_module.setup()
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=1e-3,
            train_image_encoder=True,  # Set to True to train the image encoder
            use_efficientnet=True,  # Set to True to use EfficientNet-B0 instead of CLIP ViT
            embedding_dim=12,  # Dimension of embeddings from your data
        )

        # Directory for saving separate weights
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )

        # ModelCheckpoint callback to save best model based on validation F1 score
        checkpoint_callback = ModelCheckpoint(
            dirpath=results_dir,
            filename="best_model_full",
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
            patience=10,  # Wait 10 epochs before stopping (same as EfficientNet)
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
        trainer.fit(model, datamodule=data_module)

        print("Done fitting! Will now evaluate the performance!")
    elif run_mode == "test":
        # Run the best model parameters on the test set
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )
        checkpoint_path = os.path.join(results_dir, "best_model_full.ckpt")

        # Check for both CLIP ViT and EfficientNet weight files
        image_encoder_path = os.path.join(results_dir, "image_encoder_weights.pth")
        classifier_path = os.path.join(results_dir, "classifier_weights.pth")
        efficientnet_encoder_path = os.path.join(
            results_dir, "efficientnet_encoder_weights.pth"
        )
        efficientnet_classifier_path = os.path.join(
            results_dir, "efficientnet_classifier_weights.pth"
        )

        session_name = "OpenCLIP_test"

        if os.path.exists(checkpoint_path):
            # Use the full checkpoint if available
            evaluate_clip_network_on_test_set(checkpoint_path, session_name)
        elif os.path.exists(efficientnet_encoder_path) and os.path.exists(
            efficientnet_classifier_path
        ):
            # Use EfficientNet separate weight files if available
            print("Loading EfficientNet model from separate weight files...")
            evaluate_clip_network_on_test_set(
                None,
                session_name,
                image_encoder_path=efficientnet_encoder_path,
                classifier_path=efficientnet_classifier_path,
                use_efficientnet=True,
            )
        elif os.path.exists(image_encoder_path) and os.path.exists(classifier_path):
            # Use CLIP ViT separate weight files if available
            print("Loading CLIP ViT model from separate weight files...")
            evaluate_clip_network_on_test_set(
                None,
                session_name,
                image_encoder_path=image_encoder_path,
                classifier_path=classifier_path,
                use_efficientnet=False,
            )
        else:
            print("No trained model found. Please train the model first.")
            print(f"Looking for: {checkpoint_path}")
            print(f"Or separate files: {image_encoder_path} and {classifier_path}")
    elif run_mode == "validate":
        # Evaluate the best model on the validation set with detailed class-wise metrics
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )
        checkpoint_path = os.path.join(results_dir, "best_model_full.ckpt")

        # Check for both CLIP ViT and EfficientNet weight files
        image_encoder_path = os.path.join(results_dir, "image_encoder_weights.pth")
        classifier_path = os.path.join(results_dir, "classifier_weights.pth")
        efficientnet_encoder_path = os.path.join(
            results_dir, "efficientnet_encoder_weights.pth"
        )
        efficientnet_classifier_path = os.path.join(
            results_dir, "efficientnet_classifier_weights.pth"
        )

        if os.path.exists(checkpoint_path):
            # Use the full checkpoint if available
            print("Evaluating with full checkpoint...")
            evaluate_on_validation_set(checkpoint_path)
        elif os.path.exists(efficientnet_encoder_path) and os.path.exists(
            efficientnet_classifier_path
        ):
            # Use EfficientNet separate weight files if available
            print("Evaluating EfficientNet model from separate weight files...")
            evaluate_on_validation_set(
                None,
                image_encoder_path=efficientnet_encoder_path,
                classifier_path=efficientnet_classifier_path,
                use_efficientnet=True,
            )
        elif os.path.exists(image_encoder_path) and os.path.exists(classifier_path):
            # Use CLIP ViT separate weight files if available
            print("Evaluating CLIP ViT model from separate weight files...")
            evaluate_on_validation_set(
                None,
                image_encoder_path=image_encoder_path,
                classifier_path=classifier_path,
                use_efficientnet=False,
            )
        else:
            print("No trained model found. Please train the model first.")
            print(f"Looking for: {checkpoint_path}")
            print(f"Or separate files: {image_encoder_path} and {classifier_path}")
    elif run_mode == "encode":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file_path = "../data/metadata/metadata.csv"
        image_path = ("../data/FungiImages",)

        df = pd.read_csv(file_path)

        data = FungiDataset(
            df=df,
            path=image_path,
            embedding_path="/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/lookup/text_embeddings_dictionary.json",
            full_data=False,
        )

        text2encode = data.df.apply(
            lambda x: create_text_from_habitat_and_substrate(
                x["Habitat"], x["Substrate"]
            ),
            axis=1,
        )

        # text2encode = "testing that fungi is fun!"

        # Use the original CLIP model directly for text encoding
        clip_model, _, _ = create_model_and_transforms("ViT-B-32", pretrained="openai")
        clip_model.to(device)
        clip_model.eval()

        # Tokenize and encode text
        text_tokens = tokenize(text2encode).to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)

        print(f"Text: '{text2encode}'")
        print(f"Text features shape: {text_features.shape}")
        print(f"Text features: {text_features}")
    elif run_mode == "load_weights_demo":
        # Demonstration of how to load separate weights
        results_dir = (
            "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
        )

        # Check for both CLIP ViT and EfficientNet weight files
        image_encoder_path = os.path.join(results_dir, "image_encoder_weights.pth")
        classifier_path = os.path.join(results_dir, "classifier_weights.pth")
        efficientnet_encoder_path = os.path.join(
            results_dir, "efficientnet_encoder_weights.pth"
        )
        efficientnet_classifier_path = os.path.join(
            results_dir, "efficientnet_classifier_weights.pth"
        )

        # Create a new model
        data_module = MyFungiModule()
        data_module.setup()

        # Try to load EfficientNet weights first, then CLIP ViT weights
        if os.path.exists(efficientnet_encoder_path) and os.path.exists(
            efficientnet_classifier_path
        ):
            print("Loading EfficientNet model...")
            model = OpenCLIPViTLightning(
                model_name="ViT-B-32",
                pretrained="openai",
                num_classes=data_module.num_classes,
                lr=0.001,
                train_image_encoder=True,
                use_efficientnet=True,  # Use EfficientNet
            )
            model.load_image_encoder_weights(efficientnet_encoder_path)
            model.load_classifier_weights(efficientnet_classifier_path)
            print("Successfully loaded EfficientNet separate weights!")
        elif os.path.exists(image_encoder_path) and os.path.exists(classifier_path):
            print("Loading CLIP ViT model...")
            model = OpenCLIPViTLightning(
                model_name="ViT-B-32",
                pretrained="openai",
                num_classes=data_module.num_classes,
                lr=0.001,
                train_image_encoder=True,
                use_efficientnet=False,  # Use CLIP ViT
            )
            model.load_image_encoder_weights(image_encoder_path)
            model.load_classifier_weights(classifier_path)
            print("Successfully loaded CLIP ViT separate weights!")
        else:
            print("No separate weight files found. Please train the model first.")
            print(f"Looking for CLIP ViT: {image_encoder_path} and {classifier_path}")
            print(
                f"Or EfficientNet: {efficientnet_encoder_path} and {efficientnet_classifier_path}"
            )
            model = None

        if model is not None:
            # Save weights again to demonstrate saving functionality
            save_dir = os.path.join(results_dir, "demo_weights")
            model.save_weights_separately(save_dir)
