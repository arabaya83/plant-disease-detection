# Evidence Capture Checklist

Use this checklist to produce final submission evidence without fabricating outputs.

## 1) Auto-Generated Evidence (Code-Driven)
Run from repository root:

```bash
python docs/reports/scripts/run_submission_evidence.py
```

This runs evaluation/benchmark for available models and copies expected artifacts
into `docs/reports/figures/` when available.

### Expected copied outputs
- `docs/reports/figures/cnn_confusion_matrix.png`
- `docs/reports/figures/mobilenet_confusion_matrix.png`
- `docs/reports/figures/hybrid_confusion_matrix.png`
- `docs/reports/figures/mobilenet_metrics.json`
- `docs/reports/figures/mobilenet_benchmark.json`

## 2) Manual Runtime Evidence (Required)
Start app:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Capture and save these files in `docs/reports/figures/`:
- `gradcam_healthy.png`
- `gradcam_disease.png`
- `gradcam_multi_leaf.png`
- `invalid_input.png`

Also save inference JSON payloads for submission traceability:
- `inference_healthy.json`
- `inference_diseased.json`
- `inference_multi_leaf.json`
- `inference_invalid.json`

## 3) Suggested API Capture Command
```bash
curl -s -X POST http://localhost:8000/infer \
  -F "file=@path/to/image.jpg" \
  -o docs/reports/figures/inference_healthy.json
```

## 4) Completion Check
Before submission, ensure all required files above exist and no placeholder
`.TODO.txt` evidence files remain unresolved.
