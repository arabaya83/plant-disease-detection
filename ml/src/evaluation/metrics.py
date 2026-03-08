"""Metric helpers for classification evaluation outputs."""

from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred):
    """Compute scalar summary metrics and text classification report."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
