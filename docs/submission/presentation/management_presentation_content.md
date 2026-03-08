# Management Presentation (15 Slides)

## Slide 1 - Executive Summary
- Business problem: crop loss from delayed disease diagnosis.
- Solution: mobile web AI diagnosis with per-leaf analysis and explainability.
- Key result: high classification performance on PlantVillage; deployable prototype.
- Business value: faster intervention, reduced losses, scalable advisory workflow.

## Slide 2 - Team Introduction
- Team member names and roles.
- Role split: data/ML, backend/API, frontend/demo, analytics/documentation.
- Presentation flow ownership (who presents which section).

## Slide 3 - Problem Statement
- Current farmer workflow pain points (time, expertise access, uncertainty).
- Why this matters operationally (yield, pesticide efficiency, cost).

## Slide 4 - Problem Context and Objectives
- Target users: small farmers with smartphones.
- v1 objectives and constraints (PlantVillage only, English only, <=5 leaves/image).

## Slide 5 - Data Profile
- Source: PlantVillage.
- Class granularity: Crop + Disease.
- Data format: folder-per-class image dataset.
- Split strategy: stratified 80/10/10.

## Slide 6 - Exploratory Data Analysis
- Class distribution summary (include histogram).
- Sample visual variability by class.
- Note on healthy vs diseased patterns.

## Slide 7 - EDA Insights and Data Decisions
- Imbalance across classes -> weighted loss.
- Need for robust preprocessing and mild augmentation.
- Expected domain shift from lab-like to field-like imagery.

## Slide 8 - Technology Platform and Architecture
- Stack: FastAPI, PyTorch, OpenCV, HTML/CSS/JS.
- Hybrid browser/server workflow.
- Show architecture diagram from docs.

## Slide 9 - Methodology (CV Techniques)
- Leaf validation (HSV-based).
- Classical segmentation (threshold + contours).
- Per-leaf classification.
- Grad-CAM explainability.

## Slide 10 - Model Strategy
- Baselines: Simple CNN and MobileNetV2.
- Main candidate: Hybrid model.
- Training policy: transfer learning, early stopping, class weights.

## Slide 11 - Literature Review and References
- Short review of transfer learning for plant disease tasks.
- Explainability relevance (Grad-CAM in medical/agri vision).
- Positioning against typical single-leaf academic setups.

## Slide 12 - Results and Evaluation
- Accuracy/Precision/Recall/F1 per model (table).
- Confusion matrix summary and key error classes.
- Selected model justification.

## Slide 13 - Deployment Demo Outcomes
- Show sample outputs: healthy, diseased, multi-leaf mixed, invalid case.
- Show confidence threshold behavior and retake prompts.
- Show example Grad-CAM overlays.

## Slide 14 - Limitations and Risk Management
- Domain gap risk (PlantVillage vs in-field conditions).
- Segmentation failure cases in clutter/shadows.
- Mitigations: retake guidance, confidence threshold, roadmap for field data.

## Slide 15 - Next Steps and Path to MVP
- Immediate next steps: field validation set, calibration, model compression.
- MVP path: pilot with extension officers, feedback loop, monitoring.
- Resource/timeline estimate and scale considerations.

## Appendix A (Optional)
- Detailed hyperparameters and training curves.

## Appendix B (Optional)
- API schema and analytics examples.
