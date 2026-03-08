# Assignment Compliance Report (Submission Mapping)

This report maps each course requirement to concrete repository evidence and identifies remaining human-deliverable tasks.

## Status Legend
- `Complete`: evidence exists in repository now
- `Pending Human Action`: requires manual export/record/publish step
- `Conditional`: complete once expected generated artifacts are produced from scripts

## Requirement-to-Evidence Matrix
| Requirement | Evidence Path(s) | Status | Notes |
|---|---|---|---|
| Problem definition and objective framing | `README.md`, `docs/submission/technical_synopsis.md` | Complete | Problem, users, constraints, and goals documented |
| Data loading and preprocessing pipeline | `ml/src/data/split_data.py`, `ml/src/data/dataset.py`, `ml/src/data/augmentations.py`, `ml/src/data/class_weights.py` | Complete | Stratified splits + augmentation + class weighting |
| Multi-technique CV integration | `app/services/image_validation.py`, `app/services/segmentation.py`, `app/services/inference.py`, `app/services/gradcam.py` | Complete | Validation + segmentation + classification + explainability |
| Model development (multiple architectures) | `ml/src/models/cnn_baseline.py`, `ml/src/models/mobilenet_baseline.py`, `ml/src/models/hybrid_model.py` | Complete | 3-model comparison implemented |
| Training pipeline reproducibility | `ml/src/training/train_cnn.py`, `ml/src/training/train_mobilenet.py`, `ml/src/training/train_hybrid.py`, `ml/src/training/utils.py` | Complete | Configurable CLI args + deterministic output paths |
| Evaluation and benchmarking | `ml/src/evaluation/evaluate.py`, `ml/src/evaluation/benchmark.py`, `docs/reports/model_comparison_summary.md` | Conditional | Requires regenerated weight-based artifacts in `ml/weights/` |
| Inference API and deployment-oriented prototype | `app/main.py`, `app/api/routes_infer.py`, `app/api/routes_health.py`, `app/templates/index.html`, `app/static/js/app.js` | Complete | Supports health/infer/analytics and mobile demo flow |
| Logging and analytics | `app/services/analytics.py`, `docs/architecture/system_flow.md` | Complete | Event logging + summary endpoint |
| Architecture documentation | `docs/architecture/system_flow.md`, `docs/architecture/system_architecture.md` | Complete | Includes flow and layered diagrams |
| Model selection rationale visibility | `README.md`, `docs/reports/model_comparison_summary.md`, `docs/submission/technical_synopsis.md` | Complete | MobileNetV2 selected consistently for v1 deployment |
| Figure evidence package | `docs/reports/figures/README.md`, `docs/reports/scripts/generate_submission_figures.py`, `docs/reports/figures/*` | Conditional | Auto-generated charts exist; manual demo captures still required |
| Technical synopsis source (2-page target) | `docs/submission/technical_synopsis.md` | Complete | Polished source ready for PDF export |
| Technical synopsis PDF | `docs/submission/technical_synopsis.pdf` | Pending Human Action | Export from markdown (not auto-generated in repo) |
| Demo video runbook | `docs/submission/demo_video_script.md` | Complete | Includes 5–8 minute run-of-show and scenario checklist |
| Demo video artifact and posting | external link in submission notes | Pending Human Action | Must be recorded and uploaded by team |
| Management presentation source package | `docs/submission/presentation/management_presentation_content.md` | Complete | Executive-ready 12–15 slide outline + speaker notes |
| Final presentation deck file | `docs/submission/presentation/management_presentation.pptx` | Pending Human Action | Export from source outline |
| Jupyter Book source | `docs/submission/jupyter_book/` | Complete | Includes `_config.yml`, `_toc.yml`, chapters |
| Published Jupyter Book URL | external published link | Pending Human Action | Build + host required |

## End-to-End CV Proof Checklist
- Leaf validation implemented and wired in `/infer`: `Complete`
- Classical segmentation implemented and capped at 5 leaves: `Complete`
- Per-leaf classification with confidence thresholding: `Complete`
- Grad-CAM explainability integrated in response pipeline: `Complete`
- Mobile-friendly frontend for capture/upload and result rendering: `Complete`
- Deployment-oriented API + logging + analytics summary: `Complete`

## Final Human Actions Before Submission
1. Regenerate `ml/weights` evaluation artifacts from trained checkpoints (if not present).
2. Export `technical_synopsis.pdf` from `docs/submission/technical_synopsis.md`.
3. Build/publish Jupyter Book and record URL.
4. Export presentation `.pptx` from `management_presentation_content.md`.
5. Record and post demo video link.
