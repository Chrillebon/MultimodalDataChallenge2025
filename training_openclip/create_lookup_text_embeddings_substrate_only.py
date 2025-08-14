import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from open_clip import create_model_and_transforms, tokenize
from sklearn.decomposition import PCA

# -----------------------------
# Helpers
# -----------------------------
def norm_substrate(x):
    if pd.isna(x) or str(x).strip().lower() in ("", "nan", "none", "null", "unknown"):
        return "Unknown"
    return str(x).strip()

def make_sentence(substrate: str) -> str:
    return f"The substrate of this fungi is {substrate}."

# -----------------------------
# Main
# -----------------------------
def main(
    metadata_csv="../data/metadata/metadata.csv",
    out_json="../lookup/text_embeddings_dictionary.json",
    model_name="ViT-B-32",
    pretrained="openai",
    target_dim=12,
    batch_size=256,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Load metadata and collect unique substrates
    df = pd.read_csv(metadata_csv)
    if "Substrate" not in df.columns:
        raise ValueError("Expected a 'Substrate' column in the metadata CSV.")
    subs = sorted({norm_substrate(s) for s in df["Substrate"].tolist()} | {"Unknown"})
    texts = [make_sentence(s) for s in subs]
    print(f"Found {len(subs)} unique substrates.")

    # 2) Load CLIP model for text encoding
    clip_model, _, _ = create_model_and_transforms(model_name, pretrained=pretrained)
    clip_model = clip_model.to(device)
    clip_model.eval()

    # 3) Encode texts in batches
    all_feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
            chunk = texts[i : i + batch_size]
            toks = tokenize(chunk).to(device)  # (B, 77)
            feats = clip_model.encode_text(toks)  # (B, 512) for ViT-B/32
            # L2-normalize (optional but typical for CLIP)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            all_feats.append(feats.cpu().numpy())
    feats_512 = np.concatenate(all_feats, axis=0)  # [N, 512]
    assert feats_512.shape[0] == len(texts)

    # 4) PCA to target_dim (e.g., 12)
    pca = PCA(n_components=target_dim, random_state=seed)
    feats_reduced = pca.fit_transform(feats_512)  # [N, target_dim]

    # 5) Build lookup {text -> list[float]}
    lookup = {texts[i]: feats_reduced[i].astype(np.float32).tolist() for i in range(len(texts))}

    # 6) Save JSON in a format your loader accepts
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(lookup, f)
    print(f"Saved {len(lookup)} substrate embeddings to: {out_json}")
    print(f"PCA explained variance (first {target_dim} comps): {pca.explained_variance_ratio_.sum():.3f}")

if __name__ == "__main__":
    main()

# python create_lookup_text_embeddings_substrate_only.py \
#   --metadata_csv ../data/metadata/metadata.csv \
#   --out_json ../lookup/substrate_embeddings_dictionary.json
