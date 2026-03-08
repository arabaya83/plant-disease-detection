"""CLI training entrypoint for the SimpleCNN baseline."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data.augmentations import build_eval_transforms, build_train_transforms
from ml.src.data.dataset import PlantVillageSplitDataset
from ml.src.models.cnn_baseline import SimpleCNN
from ml.src.training.utils import train_model


def load_class_names(split_dir: Path):
    """Load class name order used to align class-weight vectors."""
    return [line.strip() for line in (split_dir / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    """Configure data/model/loss and train CNN with early stopping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="ml/splits")
    parser.add_argument("--weights-json", default="ml/splits/class_weights.json")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-weights", default="ml/weights/cnn_best.pt")
    parser.add_argument("--out-history", default="ml/weights/cnn_history.json")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    classes = load_class_names(split_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PlantVillageSplitDataset(str(split_dir / "train.csv"), transform=build_train_transforms(args.image_size))
    val_ds = PlantVillageSplitDataset(str(split_dir / "val.csv"), transform=build_eval_transforms(args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with open(args.weights_json, "r", encoding="utf-8") as f:
        w_map = json.load(f)
    weight_tensor = torch.tensor([w_map[str(i)] for i in range(len(classes))], dtype=torch.float32).to(device)

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = Adam(model.parameters(), lr=args.lr)

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        out_weights=args.out_weights,
        out_history=args.out_history,
    )


if __name__ == "__main__":
    main()
