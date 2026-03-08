# Evidence Capture Checklist

This checklist defines exact commands and output destinations for submission evidence.

## A. Generate Metrics, Benchmarks, and Confusion Matrices
Run from repo root after weights are available:

```bash
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/evaluate.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights

python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights
```

Expected outputs in `ml/weights/`:
- `mobilenet_test_metrics.json`, `cnn_test_metrics.json`, `hybrid_test_metrics.json`
- `mobilenet_benchmark.json`, `cnn_benchmark.json`, `hybrid_benchmark.json`
- `mobilenet_confusion_matrix.png`, `cnn_confusion_matrix.png`, `hybrid_confusion_matrix.png`

## B. Generate Documentation Figure Package
```bash
python docs/reports/scripts/generate_submission_figures.py \
  --splits-dir ml/splits \
  --weights-dir ml/weights \
  --output-dir docs/reports/figures
```

Expected outputs in `docs/reports/figures/`:
- `class_distribution_train.png`
- `class_distribution_all_splits.png`
- copied confusion matrices (if present)
- TODO placeholders for manual runtime captures

## C. Capture Manual Runtime Evidence (Required)
Run the app:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then execute and save these artifacts into `docs/reports/figures/`:
1. Healthy case
- `sample_inference_healthy.json`
- `sample_gradcam_healthy.png`

2. Diseased case
- `sample_inference_diseased.json`
- `sample_gradcam_diseased.png`

3. Multi-leaf mixed case
- `sample_inference_multileaf.json`

4. Invalid/no-leaf case
- `sample_inference_invalid.json`

## D. Example API Capture Command
```bash
curl -s -X POST http://localhost:8000/infer \
  -F "file=@path/to/image.jpg" \
  -o docs/reports/figures/sample_inference_example.json
```

## E. Completion Check
`docs/reports/figures/` should contain final artifacts above and no unresolved
`.TODO.txt` placeholders for required evidence files.
