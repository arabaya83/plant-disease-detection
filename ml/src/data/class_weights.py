"""Compute balanced class weights from training split labels."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def main():
    """CLI entrypoint for writing `class_index -> weight` JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    y = df["label_idx"].values
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    payload = {int(c): float(w) for c, w in zip(classes, weights)}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved class weights to {args.out_json}")


if __name__ == "__main__":
    main()
