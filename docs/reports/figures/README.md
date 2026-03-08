# Figures Package

This folder stores submission visual evidence and API-output captures.

## Auto-Generated Artifacts
Generated with:
```bash
python docs/reports/scripts/run_submission_evidence.py
```

Expected auto artifacts:
- `class_distribution_train.png`
- `class_distribution_all_splits.png`
- `cnn_confusion_matrix.png` (if weights available)
- `mobilenet_confusion_matrix.png` (if weights available)
- `hybrid_confusion_matrix.png` (if weights available)
- `mobilenet_metrics.json` (if weights available)
- `mobilenet_benchmark.json` (if weights available)

## Manual Runtime Captures (Required)
Save these after live inference runs:
- `gradcam_healthy.png`
- `gradcam_disease.png`
- `gradcam_multi_leaf.png`
- `invalid_input.png`
- `inference_healthy.json`
- `inference_diseased.json`
- `inference_multi_leaf.json`
- `inference_invalid.json`

For exact commands and verification, use:
- `docs/reports/figures/evidence_capture_checklist.md`
