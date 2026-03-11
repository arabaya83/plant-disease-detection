"""Evaluate a trained checkpoint on the canonical test split.

This script loads a requested architecture, restores trained weights, runs
inference on ``test.csv``, computes summary metrics, and saves both JSON
metrics and a confusion-matrix PNG. The outputs are used in the written report
and model-comparison materials.
"""

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
    """Instantiate the requested architecture for evaluation.

    Args:
        name: Model architecture identifier.
        num_classes: Number of classifier outputs.

    Returns:
        Uninitialized model instance matching the requested architecture.
    """
    normalized_name = name.lower()
    if normalized_name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if normalized_name == "mobilenet":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)


def main() -> None:
    """Run checkpoint evaluation and export metrics artifacts.

    Side Effects:
        Writes a metrics JSON file and a confusion-matrix PNG.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test split and export metrics + confusion matrix."
    )
    parser.add_argument("--model", choices=["cnn", "mobilenet", "hybrid"], required=True, help="Model architecture to instantiate.")
    parser.add_argument("--weights", required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--split-dir", default="ml/splits", help="Directory containing test.csv and classes.txt.")
    parser.add_argument("--image-size", type=int, default=384, help="Input image size after evaluation transforms.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=2, help="PyTorch DataLoader workers.")
    parser.add_argument("--out-dir", default="ml/weights", help="Output directory for JSON metrics and confusion matrix PNG.")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    class_names = [line.strip() for line in (split_dir / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = PlantVillageSplitDataset(
        str(split_dir / "test.csv"),
        transform=build_eval_transforms(args.image_size),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            input_batch = input_batch.to(device)
            logits = model(input_batch)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(target_batch.tolist())

    metrics = compute_metrics(y_true, y_pred)
    out_dir = Path(args.out_dir)
    out_json = out_dir / f"{args.model}_test_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_true, y_pred, class_names, str(out_dir / f"{args.model}_confusion_matrix.png"))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
