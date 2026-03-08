# Pre-Submission Checklist (Final Ops)

## Code and Model Artifacts
- [ ] `ml/weights/mobilenet_best.pt` available
- [ ] `ml/weights/cnn_best.pt` and `ml/weights/hybrid_best.pt` available (comparison evidence)
- [ ] `ml/weights/*_test_metrics.json` generated
- [ ] `ml/weights/*_benchmark.json` generated
- [ ] `ml/weights/*_confusion_matrix.png` generated

## Evidence Package
- [ ] `docs/reports/scripts/run_submission_evidence.sh` executed
- [ ] `docs/reports/figures/class_distribution_train.png` exists
- [ ] `docs/reports/figures/class_distribution_all_splits.png` exists
- [ ] Runtime placeholders replaced with real captures:
  - [ ] healthy inference JSON + Grad-CAM
  - [ ] diseased inference JSON + Grad-CAM
  - [ ] multi-leaf inference JSON
  - [ ] invalid/no-leaf inference JSON

## Documentation and Submission Files
- [ ] `README.md` reviewed for final consistency
- [ ] `docs/reports/selected_model_rationale.md` matches README and slides
- [ ] `docs/submission/assignment_compliance_report.md` reviewed
- [ ] `docs/submission/final_artifacts_manifest.md` reviewed

## Human Export and Publish Steps
- [ ] `docs/submission/technical_synopsis.pdf` exported
- [ ] `docs/submission/presentation/management_presentation.pptx` exported
- [ ] Jupyter Book published and URL recorded
- [ ] Demo video recorded/posted and URL recorded
- [ ] `docs/submission/published_links.md` fully filled

## Final Sanity Check
- [ ] Default deployment model is MobileNetV2 across docs/code
- [ ] No unresolved TODO placeholders remain in final submission bundle
