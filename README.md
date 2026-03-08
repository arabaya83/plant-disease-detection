# AI-Powered Plant Disease Detection (PlantVillage)

Submission-ready computer vision final project for mobile-first plant disease diagnosis. The system demonstrates an end-to-end deployable workflow from data preparation and model training to explainable multi-leaf inference through a FastAPI + browser interface.

## Submission Status
### In-Repo Complete Artifacts
- Core codebase: backend, frontend, CV/ML services (`app/`, `ml/src/`)
- Architecture docs: `docs/architecture/system_flow.md`, `docs/architecture/system_architecture.md`
- Model selection evidence: `docs/reports/model_comparison_summary.md`
- Submission package docs: `docs/submission/*`

### Generated Artifacts Expected After Training/Evaluation
- Weight files: `ml/weights/cnn_best.pt`, `ml/weights/mobilenet_best.pt`, `ml/weights/hybrid_best.pt`
- Metrics/benchmark: `ml/weights/*_test_metrics.json`, `ml/weights/*_benchmark.json`
- Confusion matrices: `ml/weights/*_confusion_matrix.png`

### Human-Export Artifacts Required Before Final Submission
- `docs/submission/technical_synopsis.pdf`
- `docs/submission/presentation/management_presentation.pptx` (optional PDF backup)
- Published Jupyter Book URL
- Demo video URL

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

Expected dataset layout:
```text
ml/datasets/plantvillage/<Crop___Disease>/*.jpg
```

## 6. Data Preparation Workflow
```bash
python ml/src/data/split_data.py \
  --data-dir ml/datasets/plantvillage \
  --out-dir ml/splits

python ml/src/data/class_weights.py \
  --train-csv ml/splits/train.csv \
  --out-json ml/splits/class_weights.json

python ml/src/data/export_classes.py \
  --classes-txt ml/splits/classes.txt \
  --out-json ml/weights/classes.json
```

Expected artifact path:
- `ml/weights/classes.json`

If not present, regenerate with:
```bash
python ml/src/data/export_classes.py --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

## 7. Training Workflow
Train all comparison models:
```bash
python ml/src/training/train_cnn.py
python ml/src/training/train_mobilenet.py
python ml/src/training/train_hybrid.py
```

## 8. Evaluation and Benchmark Workflow
### Primary Deployment Model (MobileNetV2)
```bash
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

### Comparison Models (CNN and Hybrid)
```bash
python ml/src/evaluation/evaluate.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights

python ml/src/evaluation/benchmark.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights
```

Expected artifact paths after evaluation:
- `ml/weights/*_test_metrics.json`
- `ml/weights/*_benchmark.json`
- `ml/weights/*_confusion_matrix.png`

If not present, run the commands above.

## 9. Inference Workflow
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open: `http://localhost:8000`

API endpoints:
- `GET /health`
- `POST /infer`
- `GET /analytics/summary`

## 10. Model Comparison Summary (Observed Run)
See: `docs/reports/model_comparison_summary.md`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Avg Inference (ms) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 |

### Selected Deployment Model (v1)
`MobileNetV2` is the default deployment model for v1.

Canonical rationale:
- `docs/reports/selected_model_rationale.md`

Submission config defaults:
- `MODEL_NAME=mobilenet`
- `MODEL_WEIGHTS_PATH=ml/weights/mobilenet_best.pt`

## 11. Reproducibility Notes
Large model/data artifacts are intentionally excluded from public git history. Reproducible regeneration is fully documented in this README and:
- `docs/architecture/data_pipeline.md`
- `docs/reports/figures/evidence_capture_checklist.md`
- `docs/reports/scripts/run_submission_evidence.sh`

## 12. Limitations
- PlantVillage-to-field domain shift may reduce real-world performance.
- Classical segmentation can degrade in heavy shadows/clutter.
- Thresholded inference may increase retake prompts on uncertain samples.

## 13. Final Deliverables Map
- Compliance report: `docs/submission/assignment_compliance_report.md`
- Technical synopsis source: `docs/submission/technical_synopsis.md`
- Demo runbook: `docs/submission/demo_video_script.md`
- Presentation source package: `docs/submission/presentation/management_presentation_content.md`
- Final artifacts manifest: `docs/submission/final_artifacts_manifest.md`
- Pre-submission checklist: `docs/submission/pre_submission_checklist.md`
- Published links tracker: `docs/submission/published_links.md`
