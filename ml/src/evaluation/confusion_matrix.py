"""Confusion matrix visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true, y_pred, class_names, out_path: str):
    """Render and save labeled confusion matrix image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
