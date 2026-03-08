"""Submission evidence orchestrator.

Automates reproducible evidence generation without fabricating manual runtime captures.

Actions:
1. Run evaluate/benchmark for available models (if weights exist).
2. Copy confusion matrices and mobilenet summary artifacts into figures folder.
3. Generate distribution figures and TODO placeholders via existing script.
4. Print a final status report of missing manual evidence captures.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "ml" / "weights"
FIGURES = ROOT / "docs" / "reports" / "figures"


def run(cmd: list[str]) -> None:
    """Execute command and stream output; fail fast on non-zero status."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def maybe_run_model_artifacts(model: str, weight_name: str) -> None:
    """Run evaluation and benchmark for a model if corresponding weights exist."""
    weight_path = WEIGHTS / weight_name
    if not weight_path.exists():
        print(f"[skip] Missing weights for {model}: {weight_path}")
        return

    run(
        [
            sys.executable,
            "-m",
            "ml.src.evaluation.evaluate",
            "--model",
            model,
            "--weights",
            str(weight_path),
            "--out-dir",
            str(WEIGHTS),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "ml.src.evaluation.benchmark",
            "--model",
            model,
            "--weights",
            str(weight_path),
            "--out-dir",
            str(WEIGHTS),
        ]
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    """Copy a file if it exists, else print an explicit missing notice."""
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[ok] {src} -> {dst}")
    else:
        print(f"[missing] {src}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generate model metrics and benchmarks (if weights exist)")
    maybe_run_model_artifacts("mobilenet", "mobilenet_best.pt")
    maybe_run_model_artifacts("cnn", "cnn_best.pt")
    maybe_run_model_artifacts("hybrid", "hybrid_best.pt")

    print("[2/4] Generate/copy report figures")
    run(
        [
            sys.executable,
            str(ROOT / "docs" / "reports" / "scripts" / "generate_submission_figures.py"),
            "--splits-dir",
            str(ROOT / "ml" / "splits"),
            "--weights-dir",
            str(WEIGHTS),
            "--output-dir",
            str(FIGURES),
        ]
    )

    print("[3/4] Copy canonical submission filenames")
    copy_if_exists(WEIGHTS / "cnn_confusion_matrix.png", FIGURES / "cnn_confusion_matrix.png")
    copy_if_exists(WEIGHTS / "mobilenet_confusion_matrix.png", FIGURES / "mobilenet_confusion_matrix.png")
    copy_if_exists(WEIGHTS / "hybrid_confusion_matrix.png", FIGURES / "hybrid_confusion_matrix.png")
    copy_if_exists(WEIGHTS / "mobilenet_test_metrics.json", FIGURES / "mobilenet_metrics.json")
    copy_if_exists(WEIGHTS / "mobilenet_benchmark.json", FIGURES / "mobilenet_benchmark.json")

    print("[4/4] Manual evidence status")
    manual_required = [
        "gradcam_healthy.png",
        "gradcam_disease.png",
        "gradcam_multi_leaf.png",
        "invalid_input.png",
        "inference_healthy.json",
        "inference_diseased.json",
        "inference_multi_leaf.json",
        "inference_invalid.json",
    ]
    status = {}
    for name in manual_required:
        exists = (FIGURES / name).exists()
        status[name] = "ok" if exists else "missing"
        print(f"[{status[name]}] {FIGURES / name}")

    (FIGURES / "evidence_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote status summary: {FIGURES / 'evidence_status.json'}")


if __name__ == "__main__":
    main()
