"""Generate reproducible stratified dataset splits from PlantVillage folders.

This script scans the raw class-folder dataset layout and writes train,
validation, and test CSV manifests. Those manifests become the canonical data
inputs for the rest of the ML pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def collect_samples(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Scan dataset folders and assemble sample metadata.

    Args:
        data_dir: Root directory containing one subdirectory per class.

    Returns:
        A tuple of the sample dataframe and the sorted class-name list used to
        assign class indices.
    """
    rows = []
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        for img_path in (data_dir / class_name).glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            rows.append(
                {
                    "image_path": str(img_path.resolve()),
                    "class_name": class_name,
                    "label_idx": class_to_idx[class_name],
                }
            )
    return pd.DataFrame(rows), class_names


def main() -> None:
    """Create the canonical 80/10/10 split artifacts for the project.

    Side Effects:
        Writes ``train.csv``, ``val.csv``, ``test.csv``, and ``classes.txt``
        into the requested output directory.

    Raises:
        ValueError: If no valid images are found in the dataset directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, class_names = collect_samples(data_dir)
    if df.empty:
        raise ValueError("No images found in dataset directory")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label_idx"],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label_idx"],
        random_state=42,
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    (out_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    print(f"Saved train/val/test splits to {out_dir}")


if __name__ == "__main__":
    main()
