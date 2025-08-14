#!/usr/bin/env python3
import csv
import re
import argparse
import pandas as pd
import numpy as np

# -----------------------------
# Parse the report .txt
# -----------------------------
def load_class_report_txt(path):
    """
    Returns a DataFrame with columns: class_id, count, correct, accuracy
    """
    pat = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)", re.M)
    with open(path, "r") as f:
        text = f.read()
    rows = [(int(c), int(n), int(k), float(a)) for (c, n, k, a) in pat.findall(text)]
    df = pd.DataFrame(rows, columns=["class_id", "count", "correct", "accuracy"])
    return df

# -----------------------------
# Load metadata + detect cols
# -----------------------------
def load_metadata(path):
    df = pd.read_csv(path)
    # filename col
    cand_fn = [c for c in df.columns if c.lower() in ("filename", "filename_index", "image", "image_path")]
    filename_col = cand_fn[0] if cand_fn else df.columns[0]
    # class col
    cand_cls = [c for c in df.columns if c.lower() in ("taxonid_index", "class", "label")]
    class_col = cand_cls[0] if cand_cls else df.columns[-1]
    # substrate/date columns if present (not required for listing)
    sub_col = "Substrate" if "Substrate" in df.columns else None
    date_col = "eventDate" if "eventDate" in df.columns else None
    return df, {"filename": filename_col, "class_id": class_col, "substrate": sub_col, "eventDate": date_col}

# -----------------------------
# Build shopping list (budget-aware)
# -----------------------------
def build_buy_all_for_each_class(meta_df, colmap, class_priority_df,
                                 fields=("Substrate", "eventDate"),
                                 only_train_pattern=r"fungi_train\d{6}\.jpg",
                                 max_samples=None):
    """
    For each class (worst accuracy first), add BOTH requested fields
    for EVERY training image of that class, up to max_samples unique images.
    Returns (rows, n_unique_images, capped).
    """
    filename_col = colmap["filename"]
    class_col = colmap["class_id"]

    # filter to training images by filename pattern
    pat = re.compile(only_train_pattern)
    df = meta_df[[filename_col, class_col]].copy()
    df = df[df[filename_col].apply(lambda s: isinstance(s, str) and bool(pat.fullmatch(s)))]

    # ensure class ids are int
    df[class_col] = pd.to_numeric(df[class_col], errors="coerce").astype("Int64")

    # merge accuracy
    pr = class_priority_df[["class_id", "accuracy"]].copy()
    pr = pr.rename(columns={"class_id": class_col})
    df = df.merge(pr, on=class_col, how="left")
    # classes not in report → push to the end (treat as good accuracy)
    df["accuracy"].fillna(1.0, inplace=True)

    # order classes by worst accuracy first
    class_order = (
        df.groupby(class_col)["accuracy"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    rows = []
    seen = set()
    capped = False

    for cid in class_order:
        subdf = df[df[class_col] == cid]
        for fn in subdf[filename_col].unique():
            if fn in seen:
                continue
            # budget cap by unique images
            if (max_samples is not None) and (len(seen) >= max_samples):
                capped = True
                return rows, len(seen), capped

            seen.add(fn)
            for field in fields:
                rows.append([fn, field])

    return rows, len(seen), capped

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Make shopping list buying Substrate & eventDate for all images per class (worst-first), with budget cap.")
    ap.add_argument("--report_txt", required=True, help="Path to validation report .txt")
    ap.add_argument("--metadata_csv", required=True, help="Path to metadata.csv")
    ap.add_argument("--output_csv", default="shoppinglist.csv", help="Output shopping list CSV")
    ap.add_argument("--fields", default="Substrate,eventDate", help="Comma-separated fields to buy (default: Substrate,eventDate)")
    ap.add_argument("--cost_per_sample", type=float, default=4.0, help="Credits per image (bundle of fields)")
    ap.add_argument("--budget_credits", type=int, default=None, help="Total credits available; caps by unique images")
    ap.add_argument("--train_pattern", default=r"fungi_train\d{6}\.jpg", help="Regex for training files")
    args = ap.parse_args()

    fields = tuple([f.strip() for f in args.fields.split(",") if f.strip()])

    # Load inputs
    class_df = load_class_report_txt(args.report_txt)
    meta_df, colmap = load_metadata(args.metadata_csv)

    # Compute cap from budget
    max_samples = None
    if args.budget_credits is not None:
        if args.cost_per_sample <= 0:
            raise ValueError("--cost_per_sample must be > 0 when using --budget_credits")
        max_samples = args.budget_credits // args.cost_per_sample

    # Build list (no per-class limit; budget-aware)
    rows, n_unique, capped = build_buy_all_for_each_class(
        meta_df, colmap, class_df,
        fields=fields,
        only_train_pattern=args.train_pattern,
        max_samples=max_samples
    )

    # Write output
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # Cost summary
    cost_per_sample = args.cost_per_sample
    total_cost = n_unique * cost_per_sample

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Unique images (samples): {n_unique}")
    print(f"Estimated cost per sample: {cost_per_sample:.2f} credits")
    print(f"Total estimated cost: {total_cost:.2f} credits")
    if args.budget_credits is not None:
        print(f"Budget: {args.budget_credits} credits | Max samples allowed: {max_samples}")
        if capped:
            print("Stopped early due to budget cap.")

if __name__ == "__main__":
    main()

# to run:
# python prioritized_shopping_list.py \
#   --report_txt report.txt \
#   --metadata_csv /zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/metadata/metadata.csv \
#   --output_csv shoppinglist.csv \
#   --fields Substrate,eventDate \
#   --cost_per_sample 4 \
#   --budget_credits 72406