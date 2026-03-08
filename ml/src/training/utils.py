"""Reusable training utilities: epoch loops, metrics, early stopping."""

import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Validation-loss early stopping with best-state checkpointing."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """Update stopper state and return True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    """Run one train/eval pass and return loss + macro metrics."""
    model.train(mode=train)
    losses = []
    all_y, all_pred = [], []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        all_y.extend(y.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": accuracy_score(all_y, all_pred),
        "precision": precision_score(all_y, all_pred, average="macro", zero_division=0),
        "recall": recall_score(all_y, all_pred, average="macro", zero_division=0),
        "f1": f1_score(all_y, all_pred, average="macro", zero_division=0),
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
):
    """Train with early stopping and persist best weights + history JSON."""
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
