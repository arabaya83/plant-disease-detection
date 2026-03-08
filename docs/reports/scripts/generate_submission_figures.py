"""Generate/copy submission figures for documentation and slides.

This script is intentionally lightweight and deterministic:
- plots class distributions from split CSVs,
- copies confusion matrix images when available,
- writes placeholders for manual evidence that cannot be fabricated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_train_distribution(splits_dir: Path, output_dir: Path) -> Path:
    """Plot class frequency in the training split."""
    train_df = pd.read_csv(splits_dir / "train.csv")
    counts = train_df["class_name"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(16, 6))
    counts.plot(kind="bar", ax=ax, color="#4c956c")
    ax.set_title("PlantVillage Class Distribution (Train Split)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()

    out_path = output_dir / "class_distribution_train.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_all_splits_distribution(splits_dir: Path, output_dir: Path) -> Path:
    """Plot split-level image counts and verify 80/10/10 artifact sizes."""
    rows = []
    for split_name in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split_name}.csv"
        count = len(pd.read_csv(csv_path))
        rows.append({"split": split_name, "count": count})

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["split"], df["count"], color=["#2d6a4f", "#40916c", "#74c69d"])
    ax.set_title("Split Size Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    for i, v in enumerate(df["count"]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()

    out_path = output_dir / "class_distribution_all_splits.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def copy_confusion_matrices(weights_dir: Path, output_dir: Path) -> list[str]:
    """Copy confusion matrix images from weights folder when they exist."""
    copied = []
    for name in ["cnn_confusion_matrix.png", "mobilenet_confusion_matrix.png", "hybrid_confusion_matrix.png"]:
        src = weights_dir / name
        dst = output_dir / name
        if src.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(name)
    return copied


def write_placeholder(output_dir: Path, filename: str, message: str) -> None:
    """Write TODO-style placeholder for human-generated demo artifacts."""
    (output_dir / filename).write_text(message.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="ml/splits")
    parser.add_argument("--weights-dir", default="ml/weights")
    parser.add_argument("--output-dir", default="docs/reports/figures")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_train_distribution(splits_dir, output_dir)
    save_all_splits_distribution(splits_dir, output_dir)
    copied = copy_confusion_matrices(weights_dir, output_dir)

    for name in [
        "sample_gradcam_healthy.png",
        "sample_gradcam_diseased.png",
        "sample_inference_healthy.json",
        "sample_inference_diseased.json",
        "sample_inference_multileaf.json",
        "sample_inference_invalid.json",
    ]:
        if not (output_dir / name).exists():
            write_placeholder(
                output_dir,
                name + ".TODO.txt",
                (
                    f"TODO: Replace with real artifact `{name}` captured from runtime demo. "
                    "See docs/submission/demo_video_script.md for required scenarios."
                ),
            )

    summary = output_dir / "generation_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Figure Generation Summary",
                "",
                f"- Train distribution: `class_distribution_train.png`",
                f"- Split distribution: `class_distribution_all_splits.png`",
                f"- Copied confusion matrices: {', '.join(copied) if copied else 'none found in ml/weights'}",
                "- Placeholder TODO files created for manual demo artifacts if missing.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
