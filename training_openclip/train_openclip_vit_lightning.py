import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, tokenize
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from albumentations import Compose, Normalize, Resize
from albumentations import (
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
)
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score
import tqdm
import csv

import torch.nn as nn
from mydata import MyFungiModule  # Example data module


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
        self, model_name="ViT-B-32", pretrained="openai", num_classes=1000, lr=1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.image_encoder = self.model.visual
        self.text_encoder = self.model.transformer
        self.logit_scale = self.model.logit_scale
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        # Freeze the image encoder parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Freeze the text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Create the classifier layer immediately
        # We'll get the feature dimension from a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.image_encoder(dummy_input)
            feature_dim = dummy_features.shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes),
        )

        # Initialize F1Score metrics for training and validation
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_tokens = tokenize(text)
        text_features = self.text_encoder(text_tokens)
        return image_features, text_features

    def training_step(self, batch, batch_idx):
        images, labels, file_paths = batch
        # For CLIP training without text, we'll use image features for classification
        image_features = self.image_encoder(images)
        logits = self.classifier(image_features)
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
        images, labels, file_paths = batch
        image_features = self.image_encoder(images)
        logits = self.classifier(image_features)
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
        # Only optimize the classifier parameters since image encoder is frozen
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
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


def evaluate_clip_network_on_test_set(checkpoint_path, session_name):
    """
    Evaluate CLIP network on the test set and save predictions to a CSV file.
    """
    # Ensure results directory exists
    results_dir = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results"
    os.makedirs(results_dir, exist_ok=True)

    output_csv_path = os.path.join(results_dir, "test_predictions.csv")

    # Load data module and setup test data
    data_module = MyFungiModule()
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()

    # Load the trained model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if checkpoint_path is not None:
        model = OpenCLIPViTLightning.load_from_checkpoint(
            checkpoint_path, num_classes=data_module.num_classes
        )
    else:
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=0.001,
        )
    model.to(device)
    model.eval()

    # Collect Predictions
    results = []
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(
            test_loader, desc="Evaluating on test set"
        ):
            images = images.to(device)
            image_features = model.image_encoder(images)
            logits = model.classifier(image_features)
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


if __name__ == "__main__":
    run_mode = "encode"  # "train" "test"

    if run_mode == "train":
        # Example usage with MyFungiModule (adapt as needed)
        data_module = MyFungiModule()  # reduced_datarate=0.2
        data_module.setup()
        model = OpenCLIPViTLightning(
            model_name="ViT-B-32",
            pretrained="openai",
            num_classes=data_module.num_classes,
            lr=0.001,
        )

        # ModelCheckpoint callback to save best model based on validation F1 score
        checkpoint_callback = ModelCheckpoint(
            dirpath="/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results",
            filename="best_model",
            monitor="val_f1",  # Monitor validation F1 score
            mode="max",
            save_top_k=1,
            save_weights_only=False,
        )

        # EarlyStopping callback to stop training when validation loss stops improving
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            mode="min",  # Stop when validation loss stops decreasing
            patience=10,  # Wait 10 epochs before stopping (same as EfficientNet)
            verbose=True,  # Print message when early stopping is triggered
        )

        trainer = pl.Trainer(
            max_epochs=25,
            accelerator="auto",
            callbacks=[checkpoint_callback, early_stopping_callback],
        )
        trainer.fit(model, datamodule=data_module)

        print("Done fitting! Will now evaluate the performance!")
    elif run_mode == "test":
        # Run the best model parameters on the test set
        checkpoint_path = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/results/best_model.ckpt"
        session_name = "OpenCLIP_test"

        if os.path.exists(checkpoint_path):
            evaluate_clip_network_on_test_set(checkpoint_path, session_name)
        else:
            print(
                f"Checkpoint not found at {checkpoint_path}. Please train the model first."
            )
    elif run_mode == "encode":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        text2encode = "testing that fungi is fun!"

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
