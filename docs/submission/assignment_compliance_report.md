# Assignment Compliance Report (Submission Mapping)

This report maps course requirements to repository evidence and clearly separates
in-repo completion from human-deliverable finalization steps.

## Status Legend
- `Complete`: evidence exists in repository now.
- `Conditional`: complete once generated artifacts are recreated from scripts.
- `Pending Human Action`: requires export/record/publish by the team.

## Requirement Mapping

### 1) Technical Documentation and Code Repository
- Requirement: complete pipeline from data to inference/deployment.
- Evidence:
  - `README.md`
  - `app/`
  - `ml/src/`
  - `docs/architecture/*`
- Status: `Complete`

### 2) Preprocessing, Training, Evaluation, Inference Coverage
- Evidence:
  - Preprocessing: `ml/src/data/*`
  - Training: `ml/src/training/*`
  - Evaluation/Benchmark: `ml/src/evaluation/*`
  - Inference API: `app/api/routes_infer.py`
- Status: `Complete` (code) / `Conditional` (requires regenerated weight outputs)

### 3) Multi-Technique CV Integration
- Evidence:
  - Leaf validation: `app/services/image_validation.py`
  - Classical segmentation: `app/services/segmentation.py`
  - Classification: `app/services/inference.py`
  - Grad-CAM explainability: `app/services/gradcam.py`
- Status: `Complete`

### 4) Deployment-Oriented Prototype Story
- Evidence:
  - Backend: `app/main.py`, `app/api/routes_health.py`, `app/api/routes_infer.py`
  - Frontend: `app/templates/index.html`, `app/static/js/app.js`
  - Analytics: `app/services/analytics.py`
- Status: `Complete`

### 5) Jupyter Book
- Evidence:
  - Source: `docs/submission/jupyter_book/`
- Status: `Complete` (source), `Pending Human Action` (publish URL)

### 6) Technical Synopsis (Strict 2-Page PDF Source)
- Evidence:
  - Source: `docs/submission/technical_synopsis.md`
- Status: `Complete` (source), `Pending Human Action` (PDF export)

### 7) Demo Video Support
- Evidence:
  - Script/runbook: `docs/submission/demo_video_script.md`
- Status: `Complete` (support docs), `Pending Human Action` (record + post)

### 8) Management Presentation Support
- Evidence:
  - Deck source package: `docs/submission/presentation/management_presentation_content.md`
- Status: `Complete` (source docs), `Pending Human Action` (export `.pptx/.pdf`)

### 9) Model Comparison and Selection Visibility
- Evidence:
  - `docs/reports/model_comparison_summary.md`
  - `docs/reports/selected_model_rationale.md`
  - `README.md`
- Status: `Complete`

## End-to-End CV Proof Checklist
- [x] No-leaf validation behavior documented and implemented
- [x] Segmentation up to 5 leaves documented and implemented
- [x] Per-leaf classification with confidence threshold documented and implemented
- [x] Grad-CAM explainability integrated and documented
- [x] API + frontend + analytics workflow documented and implemented

## Final Human Actions Before Submission
1. Generate/evaluate model artifacts in `ml/weights/` if missing.
2. Export `docs/submission/technical_synopsis.pdf`.
3. Export presentation deck (`.pptx`) from source package.
4. Build/publish Jupyter Book and add URL to `docs/submission/published_links.md`.
5. Record/upload demo video and add URL to `docs/submission/published_links.md`.
6. Replace figure placeholder TODO files with runtime captures.
