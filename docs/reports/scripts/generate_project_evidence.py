"""Generate runtime evidence artifacts for submission docs and slides.

This script automates everything reproducible and skips model-dependent steps when
weights are missing. It does not fabricate outputs.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.gradcam import GradCAMService
from app.services.image_validation import LeafValidationService
from app.services.inference import InferenceService
from app.services.segmentation import SegmentationService
from ml.src.data.augmentations import build_eval_transforms
from ml.src.data.dataset import PlantVillageSplitDataset
from ml.src.models.cnn_baseline import SimpleCNN
from ml.src.models.hybrid_model import HybridPlantDiseaseModel
from ml.src.models.mobilenet_baseline import build_mobilenet_v2


WEIGHTS_DIR = ROOT / "ml" / "weights"
SPLITS_DIR = ROOT / "ml" / "splits"
METRICS_DIR = ROOT / "docs" / "reports" / "metrics"
FIGURES_DIR = ROOT / "docs" / "reports" / "figures"

MODELS = {
    "cnn": "cnn_best.pt",
    "mobilenet": "mobilenet_best.pt",
    "hybrid": "hybrid_best.pt",
}


@dataclass
class Artifacts:
    metrics: list[str]
    benchmarks: list[str]
    confusion: list[str]
    gradcam: list[str]
    demo: list[str]
    missing_weights: list[str]


def run_cmd(cmd: list[str], allow_failure: bool = False) -> bool:
    """Run subprocess command from repo root.

    Returns True on success. When `allow_failure=True`, failures are logged and
    False is returned instead of raising.
    """
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode == 0:
        return True
    if allow_failure:
        print(f"[warn] command failed ({proc.returncode}), continuing.")
        return False
    raise subprocess.CalledProcessError(proc.returncode, cmd)


def load_class_names() -> list[str]:
    """Read class order from split artifact."""
    classes_path = SPLITS_DIR / "classes.txt"
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_evaluate_and_benchmark(model: str, weights: Path) -> tuple[Path, Path] | tuple[None, None]:
    """Execute standard evaluation and benchmark scripts for one model."""
    if not weights.exists():
        return None, None

    eval_ok = run_cmd(
        [
            sys.executable,
            "-m",
            "ml.src.evaluation.evaluate",
            "--model",
            model,
            "--weights",
            str(weights),
            "--out-dir",
            str(METRICS_DIR),
            "--split-dir",
            str(SPLITS_DIR),
            "--num-workers",
            "0",
        ]
        ,
        allow_failure=True,
    )
    bench_ok = run_cmd(
        [
            sys.executable,
            "-m",
            "ml.src.evaluation.benchmark",
            "--model",
            model,
            "--weights",
            str(weights),
            "--out-dir",
            str(METRICS_DIR),
        ]
        ,
        allow_failure=True,
    )

    metrics_src = METRICS_DIR / f"{model}_test_metrics.json"
    metrics_dst = METRICS_DIR / f"{model}_metrics.json"
    if eval_ok and metrics_src.exists():
        metrics_dst.write_bytes(metrics_src.read_bytes())

    bench_src = METRICS_DIR / f"{model}_benchmark.json"
    return metrics_dst if metrics_dst.exists() else None, bench_src if bench_ok and bench_src.exists() else None


def build_model(model: str, num_classes: int) -> torch.nn.Module:
    """Instantiate architecture for normalized confusion matrix generation."""
    if model == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if model == "mobilenet":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)


def generate_confusion_pair(model: str, weights: Path, class_names: list[str], image_size: int = 384) -> tuple[Path, Path] | tuple[None, None]:
    """Generate raw and normalized confusion matrix images in figures directory."""
    if not weights.exists():
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PlantVillageSplitDataset(str(SPLITS_DIR / "test.csv"), transform=build_eval_transforms(image_size))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    m = build_model(model, num_classes=len(class_names)).to(device)
    state = torch.load(str(weights), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = m(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred)
    cmn = confusion_matrix(y_true, y_pred, normalize="true")

    raw_path = FIGURES_DIR / f"{model}_confusion_matrix.png"
    norm_path = FIGURES_DIR / f"{model}_confusion_matrix_normalized.png"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title(f"{model.upper()} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(raw_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"{model.upper()} Confusion Matrix (Normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(norm_path, dpi=180)
    plt.close(fig)

    return raw_path, norm_path


def generate_class_distribution_plots() -> list[Path]:
    """Generate train/validation/test class distribution figures."""
    paths = []
    split_map = {
        "train": "class_distribution_train.png",
        "val": "class_distribution_validation.png",
        "test": "class_distribution_test.png",
    }
    for split, out_name in split_map.items():
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        counts = df["class_name"].value_counts().sort_index()

        plt.figure(figsize=(16, 6))
        counts.plot(kind="bar", color="#2d6a4f")
        plt.title(f"Class Distribution ({split})")
        plt.xlabel("Class")
        plt.ylabel("Image Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        out_path = FIGURES_DIR / out_name
        plt.savefig(out_path, dpi=200)
        plt.close()
        paths.append(out_path)
    return paths


def select_sample_images() -> tuple[Path | None, Path | None]:
    """Pick one healthy and one diseased image from test split."""
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    healthy = df[df["class_name"].str.contains("healthy", case=False, na=False)]
    diseased = df[~df["class_name"].str.contains("healthy", case=False, na=False)]
    h = Path(healthy.iloc[0]["image_path"]) if not healthy.empty else None
    d = Path(diseased.iloc[0]["image_path"]) if not diseased.empty else None
    return h, d


def make_multi_leaf_image(healthy: np.ndarray, diseased: np.ndarray) -> np.ndarray:
    """Build synthetic multi-leaf image by placing two leaves on white canvas."""
    canvas = np.full((max(healthy.shape[0], diseased.shape[0]) + 40, healthy.shape[1] + diseased.shape[1] + 80, 3), 255, dtype=np.uint8)
    canvas[20:20 + healthy.shape[0], 20:20 + healthy.shape[1]] = healthy
    x2 = 40 + healthy.shape[1]
    canvas[20:20 + diseased.shape[0], x2:x2 + diseased.shape[1]] = diseased
    return canvas


def render_demo_image(image: np.ndarray, title: str, lines: list[str], out_path: Path) -> None:
    """Render a simple annotated demo panel image."""
    panel = image.copy()
    y = 24
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for line in lines:
        y += 26
        cv2.putText(panel, line[:90], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.imwrite(str(out_path), panel)


def run_inference_pipeline(image: np.ndarray, infer: InferenceService, gradcam: GradCAMService, threshold: float = 0.70) -> dict:
    """Execute validation->segmentation->classification->gradcam like API path."""
    validator = LeafValidationService()
    segmenter = SegmentationService(max_leaves=5)

    val = validator.validate(image)
    if not val.is_valid:
        return {"status": "invalid", "message": "No leaf detected. Please retake the photo.", "results": []}

    segs = segmenter.segment_leaves(image)
    if not segs:
        return {"status": "invalid", "message": "No leaf detected. Please retake the photo.", "results": []}

    results = []
    for seg in segs:
        pred, tensor = infer.predict(seg.crop_bgr)
        if pred.confidence < threshold:
            continue
        overlay = gradcam.generate_overlay(tensor, seg.crop_bgr, class_idx=pred.class_index)
        results.append(
            {
                "leaf_id": seg.leaf_id,
                "crop_name": pred.crop_name,
                "disease_name": pred.disease_name,
                "confidence": float(pred.confidence),
                "healthy_or_diseased": pred.healthy_or_diseased,
                "short_description": pred.short_description,
                "bbox": seg.bbox,
                "overlay": overlay,
            }
        )

    if not results:
        return {"status": "low_confidence", "message": "Low confidence predictions. Please retake the photo.", "results": []}

    return {"status": "ok", "message": "Diagnosis completed.", "results": results}


def generate_gradcam_and_demo_outputs(mobilenet_weights: Path) -> tuple[list[Path], list[Path], Path | None]:
    """Generate Grad-CAM and demo output figures for required scenarios."""
    if not mobilenet_weights.exists():
        return [], [], None

    class_names = WEIGHTS_DIR / "classes.json"
    if not class_names.exists():
        print("[warn] classes.json missing; skipping Grad-CAM/demo evidence.")
        return [], [], None

    infer = InferenceService(
        model_name="mobilenet",
        weights_path=str(mobilenet_weights),
        class_names_path=str(class_names),
        image_size=384,
        strict_loading=True,
    )
    gradcam = GradCAMService(infer.model, infer.get_gradcam_target_layer())

    healthy_path, diseased_path = select_sample_images()
    if healthy_path is None or diseased_path is None:
        print("[warn] could not find healthy/diseased test samples; skipping Grad-CAM/demo evidence.")
        return [], [], None

    healthy = cv2.imread(str(healthy_path))
    diseased = cv2.imread(str(diseased_path))
    if healthy is None or diseased is None:
        print("[warn] sample image load failed; skipping Grad-CAM/demo evidence.")
        return [], [], None

    multi = make_multi_leaf_image(healthy, diseased)
    invalid = np.full((384, 384, 3), 180, dtype=np.uint8)

    gradcam_paths = []
    demo_paths = []
    responses = {}

    # Healthy
    r_h = run_inference_pipeline(healthy, infer, gradcam)
    responses["healthy"] = {k: v for k, v in r_h.items() if k != "results"}
    responses["healthy"]["results"] = [{kk: vv for kk, vv in item.items() if kk not in {"overlay"}} for item in r_h["results"]]
    if r_h["results"]:
        cv2.imwrite(str(FIGURES_DIR / "gradcam_healthy.png"), r_h["results"][0]["overlay"])
        gradcam_paths.append(FIGURES_DIR / "gradcam_healthy.png")
    render_demo_image(healthy, "Healthy Leaf Demo", [r_h["status"], r_h["message"]], FIGURES_DIR / "demo_healthy_output.png")
    demo_paths.append(FIGURES_DIR / "demo_healthy_output.png")

    # Diseased
    r_d = run_inference_pipeline(diseased, infer, gradcam)
    responses["diseased"] = {k: v for k, v in r_d.items() if k != "results"}
    responses["diseased"]["results"] = [{kk: vv for kk, vv in item.items() if kk not in {"overlay"}} for item in r_d["results"]]
    if r_d["results"]:
        cv2.imwrite(str(FIGURES_DIR / "gradcam_disease.png"), r_d["results"][0]["overlay"])
        gradcam_paths.append(FIGURES_DIR / "gradcam_disease.png")
    render_demo_image(diseased, "Diseased Leaf Demo", [r_d["status"], r_d["message"]], FIGURES_DIR / "demo_disease_output.png")
    demo_paths.append(FIGURES_DIR / "demo_disease_output.png")

    # Multi leaf
    r_m = run_inference_pipeline(multi, infer, gradcam)
    responses["multi_leaf"] = {k: v for k, v in r_m.items() if k != "results"}
    responses["multi_leaf"]["results"] = [{kk: vv for kk, vv in item.items() if kk not in {"overlay"}} for item in r_m["results"]]
    if r_m["results"]:
        # place first leaf overlay back into image for a clear multi-leaf Grad-CAM evidence figure
        first = r_m["results"][0]
        x0, y0, x1, y1 = first["bbox"]
        vis = multi.copy()
        vis[y0:y1, x0:x1] = first["overlay"]
        cv2.imwrite(str(FIGURES_DIR / "gradcam_multi_leaf.png"), vis)
        gradcam_paths.append(FIGURES_DIR / "gradcam_multi_leaf.png")
    render_demo_image(multi, "Multi-Leaf Demo", [r_m["status"], r_m["message"]], FIGURES_DIR / "demo_multi_leaf_output.png")
    demo_paths.append(FIGURES_DIR / "demo_multi_leaf_output.png")

    # Invalid
    r_i = run_inference_pipeline(invalid, infer, gradcam)
    responses["invalid"] = r_i
    render_demo_image(invalid, "Invalid Input Demo", [r_i["status"], r_i["message"]], FIGURES_DIR / "demo_invalid_input.png")
    demo_paths.append(FIGURES_DIR / "demo_invalid_input.png")
    cv2.imwrite(str(FIGURES_DIR / "invalid_input.png"), invalid)
    gradcam_paths.append(FIGURES_DIR / "invalid_input.png")

    api_path = FIGURES_DIR / "demo_api_response.json"
    api_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")
    return gradcam_paths, demo_paths, api_path


def write_model_summary_csv(metrics_files: dict[str, Path | None], bench_files: dict[str, Path | None], weight_files: dict[str, Path]) -> Path:
    """Create model comparison CSV from generated metrics and benchmark files."""
    out = ROOT / "docs" / "reports" / "model_metrics_summary.csv"
    rows = []
    for model in ["cnn", "mobilenet", "hybrid"]:
        m_path = metrics_files.get(model)
        b_path = bench_files.get(model)
        metrics = {}
        bench = {}
        if m_path and m_path.exists():
            metrics = json.loads(m_path.read_text(encoding="utf-8"))
        if b_path and b_path.exists():
            bench = json.loads(b_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "Model": model,
                "Accuracy": metrics.get("accuracy", ""),
                "Precision": metrics.get("precision_macro", ""),
                "Recall": metrics.get("recall_macro", ""),
                "F1": metrics.get("f1_macro", ""),
                "InferenceTime": bench.get("avg_inference_ms", ""),
                "ModelSize": round(weight_files[model].stat().st_size / (1024 * 1024), 2) if weight_files[model].exists() else "",
            }
        )

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1", "InferenceTime", "ModelSize"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_figure_index() -> Path:
    """Create a markdown index of current figure files and descriptions."""
    descriptions = {
        "cnn_confusion_matrix.png": "confusion matrix for CNN baseline",
        "mobilenet_confusion_matrix.png": "confusion matrix for MobileNetV2",
        "hybrid_confusion_matrix.png": "confusion matrix for Hybrid model",
        "mobilenet_confusion_matrix_normalized.png": "normalized confusion matrix for MobileNetV2",
        "class_distribution_train.png": "class distribution for training split",
        "class_distribution_validation.png": "class distribution for validation split",
        "class_distribution_test.png": "class distribution for test split",
        "gradcam_healthy.png": "Grad-CAM highlighting healthy-leaf decision regions",
        "gradcam_disease.png": "Grad-CAM highlighting diseased area",
        "gradcam_multi_leaf.png": "Grad-CAM overlay on multi-leaf example",
        "demo_multi_leaf_output.png": "multiple leaves inference output",
    }

    lines = [
        "# Figure Index",
        "",
        "| Figure | Description |",
        "|---|---|",
    ]

    for p in sorted(FIGURES_DIR.glob("*.png")):
        lines.append(f"| {p.name} | {descriptions.get(p.name, 'generated evidence figure')} |")

    out = FIGURES_DIR / "figure_index.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    missing_weights = []
    metrics_files: dict[str, Path | None] = {}
    bench_files: dict[str, Path | None] = {}
    confusion_paths: list[Path] = []

    class_names = load_class_names()

    for model, fname in MODELS.items():
        weights = WEIGHTS_DIR / fname
        if not weights.exists():
            missing_weights.append(str(weights))

        m_out, b_out = run_evaluate_and_benchmark(model, weights)
        metrics_files[model] = m_out
        bench_files[model] = b_out

        raw_cm, norm_cm = generate_confusion_pair(model, weights, class_names)
        if raw_cm:
            confusion_paths.append(raw_cm)
        if norm_cm:
            confusion_paths.append(norm_cm)

    # Required warning behavior for missing weights
    if missing_weights:
        print("Model weights not found. Please train models before generating evidence.")
        for p in missing_weights:
            print("-", p)

    # distribution plots
    dist_paths = generate_class_distribution_plots()

    # Grad-CAM + demo outputs from MobileNetV2 only
    gradcam_paths, demo_paths, api_path = generate_gradcam_and_demo_outputs(WEIGHTS_DIR / "mobilenet_best.pt")

    # summary CSV and index
    csv_path = write_model_summary_csv(metrics_files, bench_files, {k: WEIGHTS_DIR / v for k, v in MODELS.items()})
    index_path = write_figure_index()

    # Figure count check
    fig_count = len(list(FIGURES_DIR.glob("*.png")))

    print("\nExecution summary")
    print("Artifacts generated:")
    print("- metrics files:", [str(p) for p in METRICS_DIR.glob("*_metrics.json")])
    print("- benchmark files:", [str(p) for p in METRICS_DIR.glob("*_benchmark.json")])
    print("- confusion matrices:", [str(p) for p in confusion_paths])
    print("- Grad-CAM figures:", [str(p) for p in gradcam_paths])
    print("- demo outputs:", [str(p) for p in demo_paths], "api:", str(api_path) if api_path else "not generated")
    print("- class distributions:", [str(p) for p in dist_paths])
    print("- model summary CSV:", str(csv_path))
    print("- figure index:", str(index_path))
    print(f"- figures count: {fig_count}")
    if fig_count < 10:
        print("[warn] Figure count is below 10. Generate missing model/runtime evidence to complete package.")


if __name__ == "__main__":
    main()
