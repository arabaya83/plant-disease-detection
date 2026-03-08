# Final Artifacts Manifest

This manifest lists final-project deliverables and exact expected paths.

## In-Repo Artifacts
- Project entry docs:
  - `README.md`
- Architecture diagrams:
  - `docs/architecture/system_flow.md`
  - `docs/architecture/system_architecture.md`
- Model evidence:
  - `docs/reports/model_comparison_summary.md`
  - `docs/reports/selected_model_rationale.md`
- Figure package:
  - `docs/reports/figures/README.md`
  - `docs/reports/figures/evidence_capture_checklist.md`
  - `docs/reports/scripts/generate_submission_figures.py`
  - `docs/reports/scripts/run_submission_evidence.py`
- Submission docs:
  - `docs/submission/assignment_compliance_report.md`
  - `docs/submission/technical_synopsis.md`
  - `docs/submission/demo_video_script.md`
  - `docs/submission/presentation/management_presentation_content.md`
  - `docs/submission/pre_submission_checklist.md`
  - `docs/submission/published_links.md`

## Generated Artifacts (Expected After Run)
- `ml/weights/*_best.pt`
- `ml/weights/*_test_metrics.json`
- `ml/weights/*_benchmark.json`
- `ml/weights/*_confusion_matrix.png`
- `docs/reports/figures/class_distribution_train.png`
- `docs/reports/figures/class_distribution_all_splits.png`
- `docs/reports/figures/mobilenet_metrics.json`
- `docs/reports/figures/mobilenet_benchmark.json`

## Human-Export Artifacts
- Technical synopsis PDF:
  - `docs/submission/technical_synopsis.pdf`
- Final presentation deck:
  - `docs/submission/presentation/management_presentation.pptx`
- Published Jupyter Book URL:
  - `docs/submission/published_links.md`
- Demo video URL:
  - `docs/submission/published_links.md`

## Manual Runtime Evidence (Replace Placeholders)
Destination folder: `docs/reports/figures/`
- `gradcam_healthy.png`
- `gradcam_disease.png`
- `gradcam_multi_leaf.png`
- `invalid_input.png`
- `inference_healthy.json`
- `inference_diseased.json`
- `inference_multi_leaf.json`
- `inference_invalid.json`
