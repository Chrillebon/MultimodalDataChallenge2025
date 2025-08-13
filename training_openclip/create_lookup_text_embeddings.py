import pandas as pd
import pickle as pkl

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
import pandas as pd

import torch.nn as nn
from mydata import MyFungiModule, create_text_from_habitat_and_substrate, FungiDataset  # Example data module

data = pd.read_csv("../data/metadata/metadata.csv")

all_habitats = list(data["Habitat"].unique())

all_substrates = list(data["Substrate"].unique())

all_habitats.extend(["Unknown"])
all_substrates.extend(["Unknown"])

def create_text_from_habitat_and_substrate(habitat, substrate):

    if pd.isna(habitat):
        habitat = "Unknown"
    if pd.isna(substrate):
        substrate = "Unknown"

    text = f"The habitat is {habitat}, and the substrate is {substrate}."

    return text

list_of_combinations = []

for habitat in all_habitats:
    for substrate in all_substrates:
        list_of_combinations.append(create_text_from_habitat_and_substrate(habitat, substrate))


if __name__ == "__main__":
    run_mode = "encode"  # "train" "test"
    if run_mode == "encode":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file_path = "../data/metadata/metadata.csv"
        image_path = "../data/FungiImages",

        df = pd.read_csv(file_path)

        data = FungiDataset(df = df, path = image_path, full_data=False)

        text2encode = data.df.apply(lambda x: create_text_from_habitat_and_substrate(x["Habitat"], x["Substrate"]), axis = 1)

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

        look_up_dict = dict(zip(list_of_combinations, text_features))

        import numpy as np
        from sklearn.decomposition import PCA
        import pickle
        import json

        # Suppose your original lookup is: dict[str or tuple, np.ndarray[512]]
        orig_lookup = look_up_dict  # load your dict

        # Stack all embeddings
        keys = list(orig_lookup.keys())
        embeds = np.stack([orig_lookup[k] for k in keys])  # shape [N, 512]

        # Fit PCA
        target_dim = 12   # or 32, 128 — depends on how compact you want
        pca = PCA(n_components=target_dim, random_state=42)
        embeds_reduced = pca.fit_transform(embeds)

        # Create reduced lookup
        reduced_lookup = {k: embeds_reduced[i].tolist() for i, k in enumerate(keys)}

        # Save to json
        with open("../lookup/text_embeddings_dictionary.json", "w") as f:
            json.dump(reduced_lookup, f)