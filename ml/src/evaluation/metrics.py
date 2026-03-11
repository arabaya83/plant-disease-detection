"""Metric helpers for model evaluation outputs.

This module keeps the scalar metrics reported in JSON artifacts consistent
across all evaluated architectures.
"""

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | str]:
    """Compute scalar metrics and a detailed text classification report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Dictionary containing scalar summary metrics plus the sklearn text
        classification report.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
