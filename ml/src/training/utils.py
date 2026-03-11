"""Reusable helpers shared by the model-training scripts.

This module centralizes the training loop used by all three classifier
entrypoints so experiment behavior stays consistent. It tracks train/validation
metrics, applies early stopping on validation loss, and writes reusable
checkpoint/history artifacts.
"""

import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Validation-loss early stopping with best-state checkpointing.

    The class records the best validation-loss model state seen so far and
    signals when no meaningful improvement has been observed for a configured
    number of epochs.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """Configure early-stopping behavior.

        Args:
            patience: Number of consecutive non-improving epochs tolerated.
            min_delta: Minimum validation-loss improvement required to reset the
                patience counter.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Update internal state after a validation epoch.

        Args:
            val_loss: Validation loss from the current epoch.
            model: Model whose state should be snapshotted when validation loss
                improves.

        Returns:
            ``True`` when training should stop, otherwise ``False``.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Any,
    optimizer: Any,
    device: str,
    train: bool = True,
) -> dict[str, float]:
    """Run one training or evaluation epoch.

    Args:
        model: Model being trained or evaluated.
        loader: DataLoader for the current split.
        criterion: Loss function.
        optimizer: Optimizer used during training. It is ignored when
            ``train=False``.
        device: Torch device string.
        train: Whether to enable gradient updates.

    Returns:
        Dictionary containing mean loss and macro classification metrics.
    """
    model.train(mode=train)
    losses = []
    all_targets, all_predictions = [], []

    for input_batch, target_batch in tqdm(loader, leave=False):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(input_batch)
        loss = criterion(logits, target_batch)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        predictions = torch.argmax(logits, dim=1)
        all_targets.extend(target_batch.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "precision": precision_score(all_targets, all_predictions, average="macro", zero_division=0),
        "recall": recall_score(all_targets, all_predictions, average="macro", zero_division=0),
        "f1": f1_score(all_targets, all_predictions, average="macro", zero_division=0),
    }
    return metrics


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    max_epochs: int,
    patience: int,
    out_weights: str,
    out_history: str,
 ) -> list[dict[str, Any]]:
    """Train a model with validation monitoring and artifact persistence.

    Args:
        model: Model to train.
        train_loader: Training split loader.
        val_loader: Validation split loader.
        criterion: Loss function.
        optimizer: Optimizer instance.
        device: Torch device string.
        max_epochs: Maximum number of training epochs.
        patience: Early stopping patience based on validation loss.
        out_weights: Destination path for the best checkpoint weights.
        out_history: Destination path for the training-history JSON.

    Returns:
        List of per-epoch metric records written to the history file.

    Side Effects:
        Saves the best model weights and a JSON file containing metric history
        plus total training time.
    """
    stopper = EarlyStopping(patience=patience)
    history = []

    start = perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        should_stop = stopper.step(val_metrics["loss"], model)
        if should_stop:
            break

    elapsed = perf_counter() - start

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    Path(out_weights).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_weights)

    Path(out_history).parent.mkdir(parents=True, exist_ok=True)
    with open(out_history, "w", encoding="utf-8") as f:
        json.dump({"history": history, "train_time_sec": elapsed}, f, indent=2)

    return history
