#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

WEIGHTS_DIR="ml/weights"
SPLITS_DIR="ml/splits"
FIGURES_DIR="docs/reports/figures"

mkdir -p "$FIGURES_DIR"

echo "[1/5] Checking expected weight files..."
required_weights=(
  "$WEIGHTS_DIR/mobilenet_best.pt"
  "$WEIGHTS_DIR/cnn_best.pt"
  "$WEIGHTS_DIR/hybrid_best.pt"
)
missing=()
for f in "${required_weights[@]}"; do
  if [[ ! -f "$f" ]]; then
    missing+=("$f")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing weight files:"
  printf '  - %s\n' "${missing[@]}"
  echo "Skipping evaluation/benchmark until weights are available."
else
  echo "[2/5] Running evaluation for all models..."
  python ml/src/evaluation/evaluate.py --model mobilenet --weights "$WEIGHTS_DIR/mobilenet_best.pt" --out-dir "$WEIGHTS_DIR"
  python ml/src/evaluation/evaluate.py --model cnn --weights "$WEIGHTS_DIR/cnn_best.pt" --out-dir "$WEIGHTS_DIR"
  python ml/src/evaluation/evaluate.py --model hybrid --weights "$WEIGHTS_DIR/hybrid_best.pt" --out-dir "$WEIGHTS_DIR"

  echo "[3/5] Running benchmark for all models..."
  python ml/src/evaluation/benchmark.py --model mobilenet --weights "$WEIGHTS_DIR/mobilenet_best.pt" --out-dir "$WEIGHTS_DIR"
  python ml/src/evaluation/benchmark.py --model cnn --weights "$WEIGHTS_DIR/cnn_best.pt" --out-dir "$WEIGHTS_DIR"
  python ml/src/evaluation/benchmark.py --model hybrid --weights "$WEIGHTS_DIR/hybrid_best.pt" --out-dir "$WEIGHTS_DIR"
fi

echo "[4/5] Generating/copying report figures..."
python docs/reports/scripts/generate_submission_figures.py --splits-dir "$SPLITS_DIR" --weights-dir "$WEIGHTS_DIR" --output-dir "$FIGURES_DIR"

echo "[5/5] Manual capture status"
manual_targets=(
  "$FIGURES_DIR/sample_gradcam_healthy.png"
  "$FIGURES_DIR/sample_gradcam_diseased.png"
  "$FIGURES_DIR/sample_inference_healthy.json"
  "$FIGURES_DIR/sample_inference_diseased.json"
  "$FIGURES_DIR/sample_inference_multileaf.json"
  "$FIGURES_DIR/sample_inference_invalid.json"
)
for f in "${manual_targets[@]}"; do
  if [[ -f "$f" ]]; then
    echo "OK: $f"
  else
    echo "MISSING (manual capture required): $f"
  fi
done

echo "Done. See docs/reports/figures/evidence_capture_checklist.md for full instructions."
