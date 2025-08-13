import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import pandas as pd
import pickle as pkl

def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

with open("../lookup/text_embeddings_dictionary.pkl", "rb") as f:
    lookup = pkl.load(f)


# --- add near top ---
def get_text_clip_embedding(lookup, habitat, substrate, text=None, d_text=512):
    """
    lookup: dict mapping either text OR (habitat, substrate) -> np.ndarray float32[D]
    If 'text' is supplied and present in lookup, that key is preferred.
    Fallback: zeros if not found or missing.
    """
    if text is not None and text in lookup:
        v = lookup[text]
    elif (habitat, substrate) in lookup:
        v = lookup[(habitat, substrate)]
    else:
        v = None
    if v is None:
        return np.zeros(d_text, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    # (optional but helpful) L2-normalize CLIP embeddings
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


def create_text_from_habitat_and_substrate(habitat, substrate):

    if pd.isna(habitat):
        habitat = "Unknown"
    if pd.isna(substrate):
        substrate = "Unknown"

    text = f"The habitat is {habitat}, and the substrate is {substrate}."

    return text


def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])

def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop((width, height), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")

class FungiDataset(Dataset):
    def __init__(self, df, path, transform=None, full_data=True, text_lookup=None, d_text=512):
        if full_data == False:
            df = df.dropna().reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path = path
        self.text_lookup = text_lookup or {}   # dict
        self.d_text = d_text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["filename_index"]
        label = row["taxonID_index"]
        label = -1 if pd.isnull(label) else int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        # Build text and fetch CLIP embedding
        habitat = row.get("Habitat", None)
        substrate = row.get("Substrate", None)
        text = create_text_from_habitat_and_substrate(habitat=habitat, substrate=substrate)
        text_vec = get_text_clip_embedding(self.text_lookup, habitat, substrate, text=text, d_text=self.d_text)
        text_vec = torch.from_numpy(text_vec)   # FloatTensor [D_text]

        return image, text_vec, label, file_path
    
class TextBranch(nn.Module):
    def __init__(self, d_text, out_dim=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(p),
        )
        self.out_dim = out_dim
    def forward(self, x): return self.net(x)

class EffNetTextFusion(nn.Module):
    def __init__(self, n_classes, d_text=512, img_backbone='efficientnet_b0'):
        super().__init__()
        # Image backbone -> embedding
        self.backbone = getattr(models, img_backbone)(pretrained=True)
        feat_dim = self.backbone.classifier[1].in_features  # 1280 for b0
        # turn classifier into embedding head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True)
        )
        # Text branch
        self.text = TextBranch(d_text=d_text, out_dim=128, p=0.2)
        # Projectors for a cheap interaction (Hadamard)
        self.proj_img = nn.Linear(feat_dim, 128)
        # Classifier
        fused_dim = feat_dim + self.text.out_dim + 128  # + interaction
        self.cls = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

    def forward(self, images, text_vec):
        img_emb = self.backbone(images)      # [B, 1280]
        txt_emb = self.text(text_vec)        # [B, 128]
        inter   = self.proj_img(img_emb) * txt_emb   # [B, 128]
        x = torch.cat([img_emb, txt_emb, inter], dim=1)
        return self.cls(x)

def train_fungi_network(data_file, image_path, checkpoint_dir):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Initialize DataLoaders
    #train_dataset = FungiDataset(train_df, image_path, transform=get_transforms(data='train'))
    #valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms(data='valid'))
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    #valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    D_TEXT = next(iter(text_lookup.values())).shape[0] if len(text_lookup) else 512

    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms('train'),
                                text_lookup=lookup, d_text=D_TEXT)
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms('valid'),
                                text_lookup=lookup, d_text=D_TEXT)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = train_df['taxonID_index'].nunique()
    model = EffNetTextFusion(n_classes=n_classes, d_text=D_TEXT).to(device)
    

    # Network Setup
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = models.efficientnet_b0(pretrained=True)
    #model.classifier = nn.Sequential(
    #    nn.Dropout(0.2),
    #    nn.Linear(model.classifier[1].in_features, len(train_df['taxonID_index'].unique()))
    #)
    model.to(device)

    # Define Optimization, Scheduler, and Criterion
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # Training Loop
    for epoch in range(100):  # Maximum epochs
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        for images, text_vec, labels, _ in train_loader:
            images, text_vec, labels = images.to(device), text_vec.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, text_vec)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate train accuracy
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        # Validation Loop
        with torch.no_grad():
            for images, text_vec, labels, _ in valid_loader:
                images, text_vec, labels = images.to(device), text_vec.to(device), labels.to(device)
                outputs = model(images, text_vec)
                val_loss += criterion(outputs, labels).item()
                
                # Calculate validation accuracy
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print summary at the end of the epoch
        print(f"Epoch {epoch + 1} Summary: "
            f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
            f"Epoch Time = {epoch_time:.2f} seconds")
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 183)  # Number of classes
    )
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Path to fungi images
    image_path = '/novo/projects/shared_projects/eye_imaging/data/FungiImages/'
    # Path to metadata file
    data_file = str('/novo/projects/shared_projects/eye_imaging/data/FungiImages/metadata.csv')

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = "EfficientNet"

    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"/novo/projects/shared_projects/eye_imaging/code/FungiChallenge/results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)