# AI-Powered Plant Disease Detection (PlantVillage)

Submission-ready computer vision final project for mobile-first plant disease diagnosis. The system demonstrates an end-to-end deployable workflow from data preparation and model training to explainable multi-leaf inference through a FastAPI + browser interface.

Developer-oriented documentation:
- `docs/DEVELOPER_GUIDE.md`

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://localhost:8000`

Quick health checks:
- `GET /health`
- `GET /analytics/summary`

Runtime requirements:
- `ml/weights/classes.json`
- `ml/weights/mobilenet_best.pt`

If `ml/weights/mobilenet_best.pt` is missing, the app will attempt to download
the default MobileNetV2 checkpoint from the configured Hugging Face model repo
at startup.

## 1. Project Overview
For each uploaded/captured image, the system:
1. validates leaf presence,
2. segments up to 5 leaves using classical CV,
3. classifies each leaf as `Crop + Disease`,
4. applies confidence thresholding (default 70%),
5. generates Grad-CAM overlays,
6. returns per-leaf results and logs analytics.

## 2. Integrated CV Techniques (End-to-End)
- Leaf validation: `app/services/image_validation.py`
- Classical segmentation: `app/services/segmentation.py`
- Per-leaf classification: `app/services/inference.py`
- Explainability (Grad-CAM): `app/services/gradcam.py`
- Inference API + analytics: `app/api/routes_infer.py`, `app/api/routes_health.py`
- Mobile-friendly UI (camera + upload): `app/templates/index.html`, `app/static/js/app.js`

## 3. Technology Stack
- Backend: FastAPI
- ML/CV: PyTorch, torchvision, OpenCV, NumPy, scikit-learn
- Frontend: HTML/CSS/JavaScript (mobile-friendly, no React)
- Dataset framing: PlantVillage only

## 4. Repository Structure
```text
app/                    # FastAPI app, inference services, frontend templates/static
ml/src/                 # Data, models, training, evaluation, explainability code
ml/splits/              # Stratified split CSVs + class metadata
ml/weights/             # Expected model artifacts (weights/metrics/benchmarks)
docs/architecture/      # System flow and architecture diagrams
docs/reports/           # Model comparison and figures/evidence package
docs/submission/        # Compliance + synopsis + demo + presentation package
```

## 5. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Recommended first-run verification:
1. Confirm `ml/weights/classes.json` exists.
2. Start the API with `uvicorn app.main:app --reload`.
3. Open `http://localhost:8000`.
4. Check `http://localhost:8000/health`.
5. Upload a valid leaf image through the browser UI.

Expected dataset layout:
```text
ml/datasets/plantvillage/<Crop___Disease>/*.jpg
```

## 6. Data Preparation Workflow
```bash
python -m ml.src.data.split_data --data-dir ml/datasets/plantvillage --out-dir ml/splits
python -m ml.src.data.class_weights --train-csv ml/splits/train.csv --out-json ml/splits/class_weights.json
python -m ml.src.data.export_classes --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

Expected artifact path:
- `ml/weights/classes.json`

If missing, regenerate with:
```bash
python -m ml.src.data.export_classes --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

## 7. Training Workflow
Train all comparison models:
```bash
python -m ml.src.training.train_mobilenet
python -m ml.src.training.train_cnn
python -m ml.src.training.train_hybrid
```

Expected deployment artifact:
- `ml/weights/mobilenet_best.pt`

If missing, regenerate with:
```bash
python -m ml.src.training.train_mobilenet --out-weights ml/weights/mobilenet_best.pt --out-history ml/weights/mobilenet_history.json
```

## 8. Evaluation Workflow
### Primary Evaluation (Deployment Model: MobileNetV2)
```bash
python -m ml.src.evaluation.evaluate --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python -m ml.src.evaluation.benchmark --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

### Model Comparison Experiments
```bash
python -m ml.src.evaluation.evaluate --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python -m ml.src.evaluation.evaluate --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python -m ml.src.evaluation.evaluate --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights

python -m ml.src.evaluation.benchmark --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python -m ml.src.evaluation.benchmark --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python -m ml.src.evaluation.benchmark --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights
```

Expected artifacts after evaluation:
- `ml/weights/*_test_metrics.json`
- `ml/weights/*_benchmark.json`
- `ml/weights/*_confusion_matrix.png`

If missing, run the commands above.

## 9. Inference Workflow
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open: `http://localhost:8000`

API endpoints:
- `GET /health`
- `POST /infer`
- `GET /analytics/summary`

Runtime directories used by the deployed app:
- `app/static/uploads` for original uploaded images
- `app/static/outputs` for Grad-CAM overlays
- `logs/analytics.jsonl` for inference analytics events
- `logs/app.log` for application logs

## 10. Model Comparison Summary (Observed Run)
See:
- `docs/reports/model_comparison_summary.md`
- `docs/reports/selected_model_rationale.md`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Avg Inference (ms) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 |

### Deployment Model
`MobileNetV2` is the default deployment model for v1.

Submission config defaults:
- `MODEL_NAME=mobilenet`
- `MODEL_WEIGHTS_PATH=ml/weights/mobilenet_best.pt`

## 11. Reproducibility Notes
Large model/data artifacts are intentionally excluded from public git history.

Reproducibility references:
- `docs/architecture/data_pipeline.md`
- `docs/architecture/code_documentation.md`
- `docs/reports/figures/evidence_capture_checklist.md`
- `docs/reports/scripts/run_submission_evidence.py`

## 12. Limitations
- PlantVillage-to-field domain shift may reduce real-world performance.
- Classical segmentation can degrade in heavy shadows/clutter.
- Thresholded inference may increase retake prompts on uncertain samples.

## 13. Troubleshooting
- Startup fails with model/class errors:
  Check that `ml/weights/classes.json` exists and that the configured weights
  path is valid.
- `POST /infer` returns `"status": "invalid"` for a seemingly good image:
  The HSV validation or contour-based segmentation may be rejecting the sample.
- `POST /infer` returns `"status": "low_confidence"`:
  The model did not produce any predictions above the confidence threshold.
- UI loads but no heatmap appears:
  Check `app/static/outputs/` and `logs/app.log`.

## 14. Final Deliverables Map
- Compliance report: `docs/submission/assignment_compliance_report.md`
- Technical synopsis source: `docs/submission/technical_synopsis.md`
- Demo runbook: `docs/submission/demo_video_script.md`
- Presentation source package: `docs/submission/presentation/management_presentation_content.md`
- Final artifacts manifest: `docs/submission/final_artifacts_manifest.md`
- Pre-submission checklist: `docs/submission/pre_submission_checklist.md`
- Published links tracker: `docs/submission/published_links.md`
- Completion status board: `docs/submission/completion_status.md`
```
