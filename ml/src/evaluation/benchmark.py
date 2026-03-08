"""Benchmark inference latency and checkpoint size for a model artifact."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.models.cnn_baseline import SimpleCNN
from ml.src.models.hybrid_model import HybridPlantDiseaseModel
from ml.src.models.mobilenet_baseline import build_mobilenet_v2


def build_model(name: str, num_classes: int = 38):
    """Instantiate model used for latency benchmarking."""
    if name == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "mobilenet":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)


def main():
    """CLI entrypoint for average per-image inference timing benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark model inference latency and report checkpoint size."
    )
    parser.add_argument("--model", choices=["cnn", "mobilenet", "hybrid"], required=True, help="Model architecture to benchmark.")
    parser.add_argument("--weights", required=True, help="Path to trained model checkpoint (.pt).")
    parser.add_argument("--image-size", type=int, default=384, help="Synthetic input size for latency timing.")
    parser.add_argument("--iters", type=int, default=100, help="Number of timed inference iterations.")
    parser.add_argument("--num-classes", type=int, default=38, help="Classifier output classes used when instantiating model.")
    parser.add_argument("--out-dir", default="ml/weights", help="Output directory for benchmark JSON.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model, num_classes=args.num_classes).to(device)

    state = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            _ = model(x)
    elapsed = (time.perf_counter() - t0) / args.iters

    model_size_mb = Path(args.weights).stat().st_size / (1024 * 1024)
    payload = {
        "model": args.model,
        "avg_inference_ms": round(elapsed * 1000, 3),
        "model_size_mb": round(model_size_mb, 2),
        "device": device,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_benchmark.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
