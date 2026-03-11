"""Compute balanced loss weights from the training split labels.

The training scripts use weighted cross-entropy to reduce the effect of class
imbalance. This script converts the training CSV's integer labels into a JSON
mapping that can be loaded directly by the model-training entrypoints.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def main() -> None:
    """Compute class weights and write them to a JSON artifact.

    Side Effects:
        Writes a JSON file mapping class indices to balanced loss weights.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    label_indices = train_df["label_idx"].values
    unique_classes = np.unique(label_indices)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=label_indices,
    )
    payload = {int(class_index): float(weight) for class_index, weight in zip(unique_classes, class_weights)}

    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved class weights to {output_path}")


if __name__ == "__main__":
    main()
