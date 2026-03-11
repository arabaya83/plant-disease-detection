"""Confusion matrix visualization helpers for evaluation reports.

These utilities save labeled confusion matrices as PNG images so model quality
can be reviewed visually in reports, slides, and comparison discussions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: str,
) -> None:
    """Render and save a labeled confusion matrix image.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Ordered class labels aligned with the indices.
        out_path: Destination path for the PNG image.

    Side Effects:
        Creates parent directories if needed and writes the image to disk.
    """
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion, cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
