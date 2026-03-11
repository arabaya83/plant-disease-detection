"""Generate reproducible evidence artifacts for reports and submission assets.

This script is the most complete evidence-generation workflow in the
repository. It runs evaluation and benchmark scripts, renders confusion
matrices, creates class-distribution plots, executes the runtime inference
pipeline on representative examples, and writes summary indices for the
documentation package. It deliberately skips artifacts that cannot be produced
honestly without the required model weights or source images.
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
    """Structured record of the artifacts produced by this script.

    Attributes:
        metrics: Paths to generated metrics JSON files.
        benchmarks: Paths to benchmark JSON files.
        confusion: Paths to confusion-matrix figures.
        gradcam: Paths to Grad-CAM evidence figures.
        demo: Paths to rendered demo-output figures.
        missing_weights: Checkpoint files that were expected but unavailable.
    """
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
    """Read the canonical class order from the split artifacts.

    Returns:
        Ordered list of class labels from ``ml/splits/classes.txt``.
    """
    classes_path = SPLITS_DIR / "classes.txt"
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_evaluate_and_benchmark(model: str, weights: Path) -> tuple[Path, Path] | tuple[None, None]:
    """Run evaluation and benchmark scripts for one model.

    Args:
        model: Model architecture identifier understood by the CLI tools.
        weights: Path to the checkpoint file for that model.

    Returns:
        Tuple of paths to the metrics and benchmark JSON files when generated,
        or ``(None, None)`` if the required weights are missing.
    """
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
    """Instantiate a model architecture for local artifact generation.

    Args:
        model: Architecture identifier.
        num_classes: Number of output classes.

    Returns:
        Torch model instance matching the requested architecture.
    """
    if model == "cnn":
        return SimpleCNN(num_classes=num_classes)
    if model == "mobilenet":
        return build_mobilenet_v2(num_classes=num_classes, pretrained=False)
    return HybridPlantDiseaseModel(num_classes=num_classes, pretrained_backbone=False)


def generate_confusion_pair(model: str, weights: Path, class_names: list[str], image_size: int = 384) -> tuple[Path, Path] | tuple[None, None]:
    """Generate raw and normalized confusion matrices for one checkpoint.

    Args:
        model: Architecture identifier.
        weights: Path to the trained checkpoint.
        class_names: Ordered class labels aligned with model outputs.
        image_size: Input resolution for evaluation preprocessing.

    Returns:
        Tuple of ``(raw_path, normalized_path)`` when generation succeeds, or
        ``(None, None)`` if the weights are missing.
    """
    if not weights.exists():
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = PlantVillageSplitDataset(
        str(SPLITS_DIR / "test.csv"),
        transform=build_eval_transforms(image_size),
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model_instance = build_model(model, num_classes=len(class_names)).to(device)
    state = torch.load(str(weights), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_instance.load_state_dict(state, strict=False)
    model_instance.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            input_batch = input_batch.to(device)
            logits = model_instance(input_batch)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(predictions)
            y_true.extend(target_batch.numpy().tolist())

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
    """Generate class-distribution figures for all data splits.

    Returns:
        List of paths to the generated PNG files.
    """
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
    """Pick one healthy and one diseased example image from the test split.

    Returns:
        Tuple of ``(healthy_image_path, diseased_image_path)``. Either value can
        be ``None`` if no suitable sample is available.
    """
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    healthy_rows = test_df[test_df["class_name"].str.contains("healthy", case=False, na=False)]
    diseased_rows = test_df[~test_df["class_name"].str.contains("healthy", case=False, na=False)]
    healthy_path = Path(healthy_rows.iloc[0]["image_path"]) if not healthy_rows.empty else None
    diseased_path = Path(diseased_rows.iloc[0]["image_path"]) if not diseased_rows.empty else None
    return healthy_path, diseased_path


def make_multi_leaf_image(healthy: np.ndarray, diseased: np.ndarray) -> np.ndarray:
    """Build a synthetic two-leaf image for multi-leaf demo evidence.

    Args:
        healthy: Healthy-leaf image.
        diseased: Diseased-leaf image.

    Returns:
        White canvas containing both leaves side by side.
    """
    canvas = np.full(
        (max(healthy.shape[0], diseased.shape[0]) + 40, healthy.shape[1] + diseased.shape[1] + 80, 3),
        255,
        dtype=np.uint8,
    )
    canvas[20:20 + healthy.shape[0], 20:20 + healthy.shape[1]] = healthy
    x2 = 40 + healthy.shape[1]
    canvas[20:20 + diseased.shape[0], x2:x2 + diseased.shape[1]] = diseased
    return canvas


def render_demo_image(image: np.ndarray, title: str, lines: list[str], out_path: Path) -> None:
    """Render a simple annotated panel summarizing one demo scenario.

    Args:
        image: Base image to annotate.
        title: Title placed at the top of the panel.
        lines: Status or explanation lines rendered below the title.
        out_path: Destination path for the rendered image.
    """
    panel = image.copy()
    y = 24
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for line in lines:
        y += 26
        cv2.putText(panel, line[:90], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.imwrite(str(out_path), panel)


def run_inference_pipeline(image: np.ndarray, infer: InferenceService, gradcam: GradCAMService, threshold: float = 0.70) -> dict:
    """Execute the same validation-to-Grad-CAM flow used by the API.

    Args:
        image: Input image to process.
        infer: Reusable inference service.
        gradcam: Reusable Grad-CAM service.
        threshold: Confidence threshold for accepting predictions.

    Returns:
        Dictionary mirroring the key response structure of the deployed API.
    """
    validator = LeafValidationService()
    segmenter = SegmentationService(max_leaves=5)

    validation = validator.validate(image)
    if not validation.is_valid:
        return {"status": "invalid", "message": "No leaf detected. Please retake the photo.", "results": []}

    segments = segmenter.segment_leaves(image)
    if not segments:
        return {"status": "invalid", "message": "No leaf detected. Please retake the photo.", "results": []}

    results = []
    for segment in segments:
        prediction, input_tensor = infer.predict(segment.crop_bgr)
        if prediction.confidence < threshold:
            continue
        overlay = gradcam.generate_overlay(
            input_tensor,
            segment.crop_bgr,
            class_idx=prediction.class_index,
        )
        results.append(
            {
                "leaf_id": segment.leaf_id,
                "crop_name": prediction.crop_name,
                "disease_name": prediction.disease_name,
                "confidence": float(prediction.confidence),
                "healthy_or_diseased": prediction.healthy_or_diseased,
                "short_description": prediction.short_description,
                "bbox": segment.bbox,
                "overlay": overlay,
            }
        )

    if not results:
        return {"status": "low_confidence", "message": "Low confidence predictions. Please retake the photo.", "results": []}

    return {"status": "ok", "message": "Diagnosis completed.", "results": results}


def generate_gradcam_and_demo_outputs(mobilenet_weights: Path) -> tuple[list[Path], list[Path], Path | None]:
    """Generate demo evidence figures using the deployed MobileNetV2 model.

    Args:
        mobilenet_weights: Path to the deployment checkpoint.

    Returns:
        Tuple containing generated Grad-CAM figure paths, demo figure paths,
        and the saved API-response JSON path.
    """
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
    """Create a compact model-comparison CSV from generated artifacts.

    Args:
        metrics_files: Mapping of model name to metrics JSON path.
        bench_files: Mapping of model name to benchmark JSON path.
        weight_files: Mapping of model name to checkpoint path.

    Returns:
        Path to the generated CSV summary file.
    """
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
    """Create a markdown index describing the generated figure files.

    Returns:
        Path to the generated markdown index.
    """
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
    """Run the full evidence-generation workflow.

    Side Effects:
        Writes metrics files, benchmark files, figures, demo JSON, and summary
        index artifacts into the documentation directories.
    """
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
