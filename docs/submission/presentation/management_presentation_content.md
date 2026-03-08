# Management Presentation Source Package (12-15 Slides)

Audience: non-technical stakeholders (business leaders / senior management).  
Recommended deck length: 14 slides + optional appendix.

## Slide 1 — Executive Summary: Why This Matters
**Key points**
- Business challenge: delayed disease diagnosis drives avoidable crop loss.
- Solution: mobile web AI diagnosis with explainable per-leaf outputs.
- Result: strong PlantVillage performance, deployable prototype architecture.
- Business impact: faster triage, better treatment timing, scalable advisory workflow.
**Visual placeholder**: `[[V1: one-slide value proposition graphic]]`
**Speaker notes**
- Start with farmer pain point and business impact.
- Keep technical details minimal; focus on outcomes.

## Slide 2 — Team and Delivery Roles
**Key points**
- Team members: Dr. Amish Jain, Ayman Rabaya, Shannon Coutinho, Aayush Sharma.
- Role ownership by stream: ML, backend/API, frontend/demo, documentation/reporting.
- Presentation speaking order by section.
**Visual placeholder**: `[[V2: team roles matrix]]`
**Speaker notes**
- Confirm each member presents at least one section.

## Slide 3 — Problem Definition in Business Terms
**Key points**
- Expert access is limited in-field.
- Delays increase treatment cost and yield loss.
- Need: near-immediate, explainable diagnosis from smartphone photos.
**Visual placeholder**: `[[V3: current-state pain flow]]`
**Speaker notes**
- Translate technical problem into operational and financial risk.

## Slide 4 — Scope and Success Criteria
**Key points**
- Version-1 scope: PlantVillage-only, English-only, web-based prototype.
- Must support camera capture + upload fallback.
- Must support multi-leaf analysis and confidence-based retake guidance.
**Visual placeholder**: `[[V4: scope in/out table]]`
**Speaker notes**
- Clarify this is MVP-oriented, not final production release.

## Slide 5 — Data Profile
**Key points**
- Source: PlantVillage dataset.
- Label granularity: Crop + Disease.
- Split strategy: stratified 80/10/10.
**Visual placeholder**: `[[V5: dataset summary table]]`
**Speaker notes**
- Mention controlled dataset conditions and implications.

## Slide 6 — EDA Insights That Drove Modeling Choices
**Key points**
- Class imbalance observed -> weighted loss used.
- Augmentation kept realistic and conservative.
- Domain shift risk identified early.
**Visual placeholder**: `[[V6: class distribution chart from docs/reports/figures/class_distribution_train.png]]`
**Speaker notes**
- Emphasize data-driven rationale, not arbitrary model choices.

## Slide 7 — Technology Platform and System Architecture
**Key points**
- Client: mobile browser capture/preview/compression.
- Backend: FastAPI orchestration.
- CV/ML services: validation, segmentation, classification, Grad-CAM.
- Logging/analytics for operational observability.
**Visual placeholder**: `[[V7: architecture diagram from docs/architecture/system_architecture.md]]`
**Speaker notes**
- Keep it high-level; avoid implementation overload.

## Slide 8 — Methodology: End-to-End CV Pipeline
**Key points**
- Leaf validation (HSV heuristic).
- Classical segmentation (threshold + contours) up to 5 leaves.
- Per-leaf classification with confidence threshold.
- Grad-CAM for explainability.
**Visual placeholder**: `[[V8: flow diagram from docs/architecture/system_flow.md]]`
**Speaker notes**
- Stress integration of multiple CV techniques in one product workflow.

## Slide 9 — Model Strategy and Training Approach
**Key points**
- Compared three models: CNN, MobileNetV2, Hybrid.
- Training: Adam + early stopping + best-checkpoint restore.
- Evaluation: Accuracy, Precision, Recall, F1, confusion matrix, speed, size.
**Visual placeholder**: `[[V9: model family diagram]]`
**Speaker notes**
- Explain why transfer learning is practical for MVP quality.

## Slide 10 — Literature and Benchmark Positioning
**Key points**
- Transfer learning is standard in plant disease CV tasks.
- Explainability is needed for practical trust in advisory settings.
- Our contribution: deployable multi-leaf + explainability + API workflow.
**Visual placeholder**: `[[V10: references and comparison snapshot]]`
**Speaker notes**
- Keep citations concise and relevance-focused.

## Slide 11 — Results and Model Selection
**Key points**
- MobileNetV2 achieved best Accuracy/F1 in observed comparison.
- Hybrid was competitive but larger and slower.
- Deployment decision for v1: MobileNetV2.
**Visual placeholder**: `[[V11: results table from docs/reports/model_comparison_summary.md]]`
**Speaker notes**
- Tie decision to business constraints (quality + deployability).

## Slide 12 — Demo Evidence: Real Inference Outputs
**Key points**
- Show healthy, diseased, mixed multi-leaf, and invalid/no-leaf cases.
- Show Grad-CAM overlays and confidence behavior.
- Show clear retake guidance for poor inputs.
**Visual placeholder**:
- `[[V12A: sample output cards]]`
- `[[V12B: Grad-CAM examples]]`
**Speaker notes**
- This slide should mirror the recorded demo narrative.

## Slide 13 — Risks, Limitations, and Mitigations
**Key points**
- PlantVillage-to-field domain shift remains primary risk.
- Classical segmentation can fail in clutter/harsh lighting.
- Mitigations: thresholding, retake guidance, field-data roadmap.
**Visual placeholder**: `[[V13: risk heatmap]]`
**Speaker notes**
- Present limitations candidly and with mitigation actions.

## Slide 14 — Path to MVP and Implementation Plan
**Key points**
- Next steps: field validation dataset, calibration, model optimization.
- Pilot plan: extension officers + monitored rollout.
- Resource ask: data collection support, deployment hardening effort.
**Visual placeholder**: `[[V14: roadmap timeline]]`
**Speaker notes**
- End with actionable plan and ownership.

## Optional Appendix A — Technical Backup
- Hyperparameters, training curves, confusion matrix details.

## Optional Appendix B — API and Analytics Backup
- `/infer` response schema examples
- `/analytics/summary` interpretation
