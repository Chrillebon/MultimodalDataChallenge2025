from glob import glob
from os.path import join, isfile
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from albumentations import Compose, Normalize, Resize
from albumentations import (
    RandomResizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
)
from albumentations.pytorch import ToTensorV2
import json


def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == "train":
        return Compose(
            [
                RandomResizedCrop((width, height), scale=(0.8, 1.0)),
                # HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomBrightnessContrast(p=0.1),
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


def norm_substrate(x):
    if pd.isna(x) or str(x).strip().lower() in ("", "nan", "none", "null", "unknown"):
        return "Unknown"
    return str(x).strip()


def make_sentence(substrate: str) -> str:
    return f"The substrate of this fungi is {norm_substrate(substrate)}."


class FungiDataset(Dataset):
    def __init__(self, df, path, embedding_path, transform=None, full_data=True):

        if full_data is False:
            df = df.dropna().reset_index(
                drop=True
            )  # NOTE: this assumes that we've bought ALL data for the training set
        self.df = df
        self.transform = transform
        self.path = path
        if embedding_path is not None and isfile(embedding_path):
            with open(embedding_path, "r") as f:
                self.embedding_dict = json.load(f)
            # print(f"Debugging: {type(self.embedding_dict)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df["filename_index"].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df["taxonID_index"].values[idx]  # Get label

        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)

        with Image.open(join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert("RGB")
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        habitat = self.df["Habitat"].values[idx]
        substrate = self.df["Substrate"].values[idx]

        text = make_sentence(substrate=substrate)
        if text not in self.embedding_dict:
            text = make_sentence(substrate=None)
        assert (
            text in self.embedding_dict
        ), f"Error... text not embedded... Found text {text}"

        # Get the original text embedding (12 dimensions)
        text_embedding = self.embedding_dict[text]

        # Normalize text embedding (similar to image normalization)
        # Using computed statistics from the text embedding dataset
        text_means = [
            0.0023,
            0.0156,
            -0.0089,
            0.0145,
            -0.0067,
            0.0034,
            0.0098,
            -0.0045,
            0.0123,
            -0.0078,
            0.0056,
            -0.0012,
        ]
        text_stds = [
            0.0892,
            0.0834,
            0.0798,
            0.0845,
            0.0823,
            0.0856,
            0.0867,
            0.0834,
            0.0789,
            0.0823,
            0.0845,
            0.0878,
        ]

        normalized_text_embedding = [
            (feature - mean) / std
            for feature, mean, std in zip(text_embedding, text_means, text_stds)
        ]

        # Get the 7 seasonal features from metadata
        seasonal_features = [
            self.df["season_k1_sin"].values[idx],
            self.df["season_k1_cos"].values[idx],
            self.df["season_k2_sin"].values[idx],
            self.df["season_k2_cos"].values[idx],
            self.df["season_k3_sin"].values[idx],
            self.df["season_k3_cos"].values[idx],
            self.df["season_abs"].values[idx],
        ]

        # Convert to float and handle any NaN values
        seasonal_features = [
            float(val) if not pd.isna(val) else 0.0 for val in seasonal_features
        ]

        # Normalize seasonal features (similar to image normalization)
        # Using computed statistics from the dataset
        seasonal_means = [-0.4150, -0.2159, 0.2768, 0.1591, -0.0057, 0.3407, 0.5619]
        seasonal_stds = [0.4983, 0.7300, 0.5859, 0.7448, 0.6331, 0.6950, 0.3285]

        normalized_seasonal_features = [
            (feature - mean) / std
            for feature, mean, std in zip(
                seasonal_features, seasonal_means, seasonal_stds
            )
        ]

        # Combine normalized text embedding with normalized seasonal features (12 + 7 = 19 dimensions)
        extended_embedding = normalized_text_embedding + normalized_seasonal_features

        return image, label, extended_embedding, file_path


class MyFungiModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(
        self,
        data_file: str = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/metadata/metadata.csv",
        image_path: str = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/FungiImages",
        embedding_path: str = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/lookup/text_embeddings_dictionary.json",
        reduced_datarate=None,
    ):
        super(MyFungiModule).__init__()
        self.df = pd.read_csv(data_file)
        self.train_df, self.val_df = None, None
        self.image_path = image_path
        self.embedding_path = embedding_path
        # Count unique classes from the last column (taxonID_index)
        self.num_classes = self.df["taxonID_index"].nunique()
        self.reduced_datarate = reduced_datarate

        # Add required PyTorch Lightning attributes
        self.prepare_data_per_node = True
        self.allow_zero_length_dataloader_with_multiple_devices = False
        self._log_hyperparams = True

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None, transform=None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            train_df = self.df[self.df["filename_index"].str.startswith("fungi_train")]
            if self.reduced_datarate is not None:
                train_df = train_df.sample(
                    frac=self.reduced_datarate, random_state=42
                ).reset_index(drop=True)
            self.train_df, self.val_df = train_test_split(
                train_df, test_size=0.2, random_state=42
            )
            print("Training size", len(self.train_df))
            print("Validation size", len(self.val_df))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_df = self.df[
                self.df["filename_index"].str.startswith("fungi_final")
            ]  # fungi_test

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self, **kwargs):
        if self.train_df is None:
            self.setup(stage="fit")
        train_dataset = FungiDataset(
            self.train_df,
            self.image_path,
            embedding_path=self.embedding_path,
            transform=get_transforms(data="train"),
        )
        return DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self, **kwargs):
        if self.val_df is None:
            self.setup(stage="fit")
        valid_dataset = FungiDataset(
            self.val_df,
            self.image_path,
            embedding_path=self.embedding_path,
            transform=get_transforms(data="valid"),
        )
        return DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    def test_dataloader(self, **kwargs):
        if self.test_df is None:
            self.setup(stage="test")
        test_dataset = FungiDataset(
            self.test_df,
            self.image_path,
            embedding_path=self.embedding_path,
            transform=get_transforms(data="valid"),
        )
        return DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


if __name__ == "__main__":

    # # train -> 25863 (~809 batches of size 32)
    # # test  -> 6552
    # # final -> 3600
    # test = MyCustomDataset(tag="final")
    # print(f"testing that it works? {len(test)}")

    test = MyFungiModule(reduced_datarate=0.01)
    test.setup()
    print(len(test.train_dataloader()))
    print(len(test.val_dataloader()))
