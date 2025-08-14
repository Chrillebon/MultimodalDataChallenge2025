import os
import csv
import time
import random
import json
import pickle as pkl
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from torchvision import models
from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
import tqdm

# -----------------------------
# Utils
# -----------------------------
def ensure_folder(folder):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_csv_logger(file_path):
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])

def find_grid_cols(df, prefix="geo"):
    cols = [c for c in df.columns if c.startswith(prefix + "_grid") and c.endswith("km_id")]
    import re
    def scale_from_name(c):
        m = re.search(r"grid(\d+)km_id", c)
        return int(m.group(1)) if m else 0
    cols = sorted(cols, key=scale_from_name)  # ascending: 2,5,10...
    return cols

def build_grid_vocabs(train_df, grid_cols):
    """Return list of dicts (raw_id -> compact_idx) per grid col; 0 is pad."""
    vocabs = []
    for c in grid_cols:
        ids = train_df[c].dropna().astype(np.int64).values
        ids = ids[ids >= 0]  # -1 is missing
        uniq = np.unique(ids)
        mapping = {int(x): i+1 for i, x in enumerate(uniq)}  # 1..V
        vocabs.append(mapping)
    return vocabs

def map_grids_row(row, grid_cols, vocabs, pad_idx=0):
    mapped = []
    for c, vocab in zip(grid_cols, vocabs):
        v = row.get(c, -1)
        if pd.isna(v) or int(v) < 0:
            mapped.append(pad_idx)
        else:
            mapped.append(vocab.get(int(v), pad_idx))  # unseen -> pad
    return np.array(mapped, dtype=np.int64)

def get_transforms(split):
    W, H = 224, 224
    if split == 'train':
        return Compose([
            RandomResizedCrop((W, H), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif split == 'valid':
        return Compose([
            Resize(W, H),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("split must be 'train' or 'valid'")

# -----------------------------
# Load PCA-reduced CLIP text lookup (JSON, 12-D)
# -----------------------------
def load_text_lookup_json(path):
    """
    Supports either:
      - { "lookup": { key: [floats], ... } }
      - { key: [floats], ... }
    Returns: dict[str -> np.ndarray(float32, [D])]
    """
    with open(path, "r") as f:
        obj = json.load(f)
    d = obj.get("lookup", obj)
    out = {}
    for k, v in d.items():
        arr = np.asarray(v, dtype=np.float32)
        out[str(k)] = arr
    return out

def get_text_clip_embedding(lookup, habitat, substrate, text=None, d_text=12):
    """
    lookup keys: either exact 'text' or a composite 'Habitat||Substrate'.
    Returns L2-normalized vector or zeros if missing.
    """
    v = None
    if text is not None and text in lookup:
        v = lookup[text]
    else:
        h = "Unknown" if pd.isna(habitat) else str(habitat)
        s = "Unknown" if pd.isna(substrate) else str(substrate)
        comp = f"{h}||{s}"
        v = lookup.get(comp, None)

    if v is None:
        return np.zeros(d_text, dtype=np.float32)

    v = np.asarray(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v

def create_text_from_habitat_and_substrate(habitat, substrate):
    habitat = "Unknown" if pd.isna(habitat) else habitat
    substrate = "Unknown" if pd.isna(substrate) else substrate
    return f"The habitat is {habitat}, and the substrate is {substrate}."

# -----------------------------
# Column helpers (season + geo numeric)
# -----------------------------
def infer_meta_columns(df, season_prefix="season", geo_prefix="geo"):
    season_cols = [c for c in df.columns if c.startswith(season_prefix + "_")]
    geo_cols = [c for c in df.columns if c.startswith(geo_prefix + "_")]
    geo_cols_numeric = [c for c in geo_cols if not c.endswith("km_id")]  # keep numeric only
    return season_cols, geo_cols_numeric

# -----------------------------
# Dataset
# -----------------------------
class FungiDataset(Dataset):
    def __init__(self, df, img_root, transform=None,
                 text_lookup=None, d_text=12,
                 season_cols=None, geo_cols=None,
                 grid_cols=None, grid_vocabs=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        self.text_lookup = text_lookup or {}
        self.d_text = d_text
        self.season_cols = season_cols or []
        self.geo_cols = geo_cols or []
        self.grid_cols = grid_cols or []
        self.grid_vocabs = grid_vocabs or []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        file_path = row["filename_index"]
        with Image.open(os.path.join(self.img_root, file_path)) as img:
            image = img.convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]

        # label
        label = row["taxonID_index"]
        label = -1 if pd.isnull(label) else int(label)

        # text vec
        habitat = row.get("Habitat", None)
        substrate = row.get("Substrate", None)
        text = create_text_from_habitat_and_substrate(habitat, substrate)
        text_vec = get_text_clip_embedding(self.text_lookup, habitat, substrate, text=text, d_text=self.d_text)
        text_vec = torch.from_numpy(text_vec)  # float32 [d_text]

        # numeric meta
        season_vec = torch.from_numpy(row[self.season_cols].astype(np.float32).values) if self.season_cols else torch.zeros(0, dtype=torch.float32)
        geo_vec    = torch.from_numpy(row[self.geo_cols].astype(np.float32).values)    if self.geo_cols    else torch.zeros(0, dtype=torch.float32)

        # grid indices (compact)
        if self.grid_cols and self.grid_vocabs:
            grid_idx_np = map_grids_row(row, self.grid_cols, self.grid_vocabs, pad_idx=0)
            grid_idx = torch.from_numpy(grid_idx_np)  # int64 [S]
        else:
            grid_idx = torch.zeros(0, dtype=torch.long)

        return image, text_vec, season_vec, geo_vec, grid_idx, label, file_path

# -----------------------------
# Model: image + text + numeric(meta) + grids
# -----------------------------
class TextBranch(nn.Module):
    # Slim for 12-D inputs
    def __init__(self, d_text, out_dim=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, out_dim), nn.ReLU(), nn.Dropout(p),
        )
        self.out_dim = out_dim
    def forward(self, x):
        return self.net(x)

class NumericMetaBranch(nn.Module):
    def __init__(self, in_dim, out_dim=128, p=0.2):
        super().__init__()
        if in_dim == 0:
            self.net = nn.Identity()
            self.out_dim = 0
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(p),
                nn.Linear(256, out_dim), nn.ReLU(), nn.Dropout(p),
            )
            self.out_dim = out_dim
    def forward(self, x):
        return self.net(x)

class GridBranch(nn.Module):
    def __init__(self, vocab_sizes, emb_dims=None, pad_idx=0, p=0.1):
        super().__init__()
        S = len(vocab_sizes)
        if emb_dims is None:
            emb_dims = [16, 24, 32][:S] + [32] * max(0, S - 3)
        assert len(emb_dims) == S
        self.embs = nn.ModuleList([
            nn.Embedding(vsz + 1, d, padding_idx=pad_idx)
            for vsz, d in zip(vocab_sizes, emb_dims)
        ])
        self.out_dim = sum(emb_dims)
        self.dropout = nn.Dropout(p)

    def forward(self, grid_idx):  # LongTensor [B, S]
        if grid_idx.numel() == 0:
            return torch.zeros(grid_idx.size(0), 0, device=grid_idx.device)
        embs = [emb(grid_idx[:, j]) for j, emb in enumerate(self.embs)]
        return self.dropout(torch.cat(embs, dim=1))

class EffNetFusion(nn.Module):
    def __init__(self, n_classes, d_text=12, d_numeric=0,
                 grid_vocab_sizes=None, img_backbone='efficientnet_b0'):
        super().__init__()
        # image backbone → embedding
        self.backbone = getattr(models, img_backbone)(pretrained=True)
        feat_dim = self.backbone.classifier[1].in_features  # 1280 for b0
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

        # branches
        self.text = TextBranch(d_text, out_dim=64, p=0.1)         # 12 -> 64
        self.numeric = NumericMetaBranch(d_numeric, out_dim=128)  # seasonal+geo numeric

        # grid branch (optional)
        self.has_grid = grid_vocab_sizes is not None and len(grid_vocab_sizes) > 0
        if self.has_grid:
            self.grid = GridBranch(grid_vocab_sizes, emb_dims=None, pad_idx=0, p=0.1)
            grid_out = self.grid.out_dim
        else:
            grid_out = 0

        # cheap interaction: text gates image (project image -> 64)
        self.proj_img_to_64 = nn.Linear(feat_dim, 64)

        # fused: img(1280) + text(64) + numeric(128) + inter(64) + grid(sum emb_dims)
        fused_dim = feat_dim + self.text.out_dim + self.numeric.out_dim + 64 + grid_out
        self.cls = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, n_classes)
        )

    def forward(self, images, text_vec, numeric_vec, grid_idx=None):
        img_emb = self.backbone(images)               # [B, 1280]
        txt_emb = self.text(text_vec)                 # [B, 64]
        num_emb = self.numeric(numeric_vec)           # [B, 128] or [B,0]
        inter   = self.proj_img_to_64(img_emb) * txt_emb  # [B,64]

        parts = [img_emb, txt_emb, num_emb, inter]
        if self.has_grid and grid_idx is not None and grid_idx.numel() > 0:
            parts.append(self.grid(grid_idx))
        x = torch.cat(parts, dim=1)
        return self.cls(x)

# -----------------------------
# Training / Eval
# -----------------------------
def train_fungi_network(data_file, image_path, checkpoint_dir,
                        text_lookup_json_path="../lookup/text_embeddings_dictionary.json",
                        season_prefix="season", geo_prefix="geo"):
    ensure_folder(checkpoint_dir)
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # metadata with season_* / geo_* columns
    df = pd.read_csv(data_file)

    SEASON_COLS, GEO_NUM_COLS = infer_meta_columns(df, season_prefix, geo_prefix)
    GRID_COLS = find_grid_cols(df, geo_prefix)

    # split
    train_df = df[df['filename_index'].str.startswith('fungi_train')].copy()
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, shuffle=True, stratify=train_df['taxonID_index']
    )

    # grid vocabs (train only)
    grid_vocabs = build_grid_vocabs(train_df, GRID_COLS)
    grid_vocab_sizes = [len(v) for v in grid_vocabs]
    with open(os.path.join(checkpoint_dir, "grid_vocabs.pkl"), "wb") as f:
        pkl.dump({"grid_cols": GRID_COLS, "vocabs": grid_vocabs}, f)

    # text lookup
    text_lookup = load_text_lookup_json(text_lookup_json_path)
    if len(text_lookup) == 0:
        D_TEXT = 12
    else:
        dims = {np.asarray(v).shape[0] for v in text_lookup.values()}
        assert len(dims) == 1, f"Inconsistent text dims in JSON: {dims}"
        D_TEXT = dims.pop()
        assert D_TEXT == 12, f"Expected 12-D PCA text embeddings, got {D_TEXT}"

    D_NUMERIC = len(SEASON_COLS) + len(GEO_NUM_COLS)

    # datasets
    train_dataset = FungiDataset(train_df, image_path, transform=get_transforms('train'),
                                 text_lookup=text_lookup, d_text=D_TEXT,
                                 season_cols=SEASON_COLS, geo_cols=GEO_NUM_COLS,
                                 grid_cols=GRID_COLS, grid_vocabs=grid_vocabs)
    valid_dataset = FungiDataset(val_df, image_path, transform=get_transforms('valid'),
                                 text_lookup=text_lookup, d_text=D_TEXT,
                                 season_cols=SEASON_COLS, geo_cols=GEO_NUM_COLS,
                                 grid_cols=GRID_COLS, grid_vocabs=grid_vocabs)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # device / classes / model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = int(train_df['taxonID_index'].nunique())
    model = EffNetFusion(n_classes=n_classes, d_text=D_TEXT, d_numeric=D_NUMERIC,
                         grid_vocab_sizes=grid_vocab_sizes).to(device)

    # optim / sched / loss
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    # early stopping
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    # train
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        t0 = time.time()

        for images, text_vec, season_vec, geo_vec, grid_idx, labels, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            images = images.to(device)
            text_vec = text_vec.to(device)
            numeric_vec = torch.cat([season_vec, geo_vec], dim=1).to(device)
            grid_idx = grid_idx.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, text_vec, numeric_vec, grid_idx)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)

        avg_train_loss = train_loss / max(1, len(train_loader))
        train_accuracy = total_correct_train / max(1, total_train_samples)

        # val
        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        with torch.no_grad():
            for images, text_vec, season_vec, geo_vec, grid_idx, labels, _ in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1} [val]"):
                images = images.to(device)
                text_vec = text_vec.to(device)
                numeric_vec = torch.cat([season_vec, geo_vec], dim=1).to(device)
                grid_idx = grid_idx.to(device)
                labels = labels.to(device)

                outputs = model(images, text_vec, numeric_vec, grid_idx)
                val_loss += criterion(outputs, labels).item()
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = val_loss / max(1, len(valid_loader))
        val_accuracy = total_correct_val / max(1, total_val_samples)

        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1:03d} | Train Loss {avg_train_loss:.4f} Acc {train_accuracy:.4f} "
              f"| Val Loss {avg_val_loss:.4f} Acc {val_accuracy:.4f} | {epoch_time:.1f}s")

        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"  ✓ Best accuracy updated to {best_accuracy:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"  ✓ Best loss updated to {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step(avg_val_loss)
        if patience_counter >= patience:
            print(f"Early stopping: no val loss improvement for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name,
                                 text_lookup_json_path="../lookup/text_embeddings_dictionary.json",
                                 season_prefix="season", geo_prefix="geo"):
    ensure_folder(checkpoint_dir)
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    SEASON_COLS, GEO_NUM_COLS = infer_meta_columns(df, season_prefix, geo_prefix)
    test_df = df[df['filename_index'].str.startswith('fungi_test')].copy()

    # text lookup
    text_lookup = load_text_lookup_json(text_lookup_json_path)
    if len(text_lookup) == 0:
        D_TEXT = 12
    else:
        dims = {np.asarray(v).shape[0] for v in text_lookup.values()}
        assert len(dims) == 1, f"Inconsistent text dims in JSON: {dims}"
        D_TEXT = dims.pop()
        assert D_TEXT == 12, f"Expected 12-D PCA text embeddings, got {D_TEXT}"

    D_NUMERIC = len(SEASON_COLS) + len(GEO_NUM_COLS)

    # load grid vocabs (training)
    voc_path = os.path.join(checkpoint_dir, "grid_vocabs.pkl")
    with open(voc_path, "rb") as f:
        gsaved = pkl.load(f)
    GRID_COLS = gsaved["grid_cols"]
    grid_vocabs = gsaved["vocabs"]
    grid_vocab_sizes = [len(v) for v in grid_vocabs]

    # dataset/loader
    test_dataset = FungiDataset(test_df, image_path, transform=get_transforms('valid'), text_lookup=text_lookup, d_text=D_TEXT, season_cols=SEASON_COLS, geo_cols=GEO_NUM_COLS, grid_cols=GRID_COLS, grid_vocabs=grid_vocabs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # model
    n_classes = int(df['taxonID_index'].nunique())  # or fixed 183
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EffNetFusion(n_classes=n_classes, d_text=D_TEXT, d_numeric=D_NUMERIC, grid_vocab_sizes=grid_vocab_sizes).to(device)
    model.load_state_dict(torch.load(best_trained_model, map_location=device))
    model.eval()

    # predict
    results = []
    with torch.no_grad():
        for images, text_vec, season_vec, geo_vec, grid_idx, labels, filenames in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            text_vec = text_vec.to(device)
            numeric_vec = torch.cat([season_vec, geo_vec], dim=1).to(device)
            grid_idx = grid_idx.to(device)
            logits = model(images, text_vec, numeric_vec, grid_idx)
            preds = logits.argmax(1).cpu().numpy()
            results.extend(zip(filenames, preds))

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])
        writer.writerows(results)
    print(f"Results saved to {output_csv_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    seed_torch(777)

    image_path = '../data/FungiImages/'
    # CSV must already contain season_* and geo_* columns
    data_file = '../data/metadata/metadata_with_geo_time.csv'
    session = "EfficientNet_Fusion_Text12D_SeasonGeo_Grid"
    checkpoint_dir = os.path.join(f"../results/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir)
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session)
