# Figures Package

This folder contains visual evidence used in the report, demo, and presentation.

## Expected Figures
1. `class_distribution_train.png` - class counts in training split
2. `class_distribution_all_splits.png` - aggregated counts by split
3. `cnn_confusion_matrix.png` - copied from `ml/weights/` if available
4. `mobilenet_confusion_matrix.png` - copied from `ml/weights/` if available
5. `hybrid_confusion_matrix.png` - copied from `ml/weights/` if available
6. `sample_gradcam_healthy.png` - manual capture from inference output
7. `sample_gradcam_diseased.png` - manual capture from inference output
8. `sample_inference_healthy.json` - expected API response sample
9. `sample_inference_diseased.json` - expected API response sample
10. `sample_inference_multileaf.json` - expected API response sample
11. `sample_inference_invalid.json` - expected API response sample

## Reproducible Generation
Run from repository root:

```bash
python docs/reports/scripts/generate_submission_figures.py \
  --splits-dir ml/splits \
  --weights-dir ml/weights \
  --output-dir docs/reports/figures
```

What this script does:
- builds class distribution plots from split CSV files
- copies confusion matrices from `ml/weights/` if present
- creates clearly-marked placeholders for missing manual evidence

## Manual Evidence Capture (Required Human Step)
For final submission visuals, replace placeholder files with real artifacts:
1. Run app and execute demo scenarios.
2. Save Grad-CAM output images into this folder.
3. Save representative `/infer` JSON payloads for all four required scenarios.
