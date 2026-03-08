"""Evaluate trained model on test split and persist metrics artifacts."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data.augmentations import build_eval_transforms
from ml.src.data.dataset import PlantVillageSplitDataset
from ml.src.evaluation.confusion_matrix import save_confusion_matrix
from ml.src.evaluation.metrics import compute_metrics
from ml.src.models.cnn_baseline import SimpleCNN
from ml.src.models.hybrid_model import HybridPlantDiseaseModel
from ml.src.models.mobilenet_baseline import build_mobilenet_v2


def build_model(name: str, num_classes: int):
    """Instantiate evaluation model by architecture name."""
    name = name.lower()
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "mobilenet":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)


def main():
    """CLI entrypoint for test-set evaluation and confusion matrix export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "mobilenet", "hybrid"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split-dir", default="ml/splits")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    class_names = [line.strip() for line in (split_dir / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PlantVillageSplitDataset(str(split_dir / "test.csv"), transform=build_eval_transforms(args.image_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())

    metrics = compute_metrics(y_true, y_pred)
    out_json = Path("ml/weights") / f"{args.model}_test_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_true, y_pred, class_names, f"ml/weights/{args.model}_confusion_matrix.png")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
