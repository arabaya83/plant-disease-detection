# AI-Powered Plant Disease Detection (PlantVillage)

Submission-ready computer vision final project for mobile-first plant disease diagnosis. The system provides a deployable web workflow (camera/upload in browser + FastAPI inference backend) and demonstrates complete CV integration from preprocessing and model training to explainable inference.

## 1. Project Overview
This project targets small farmers using smartphones in the field. A user captures or uploads a plant image, and the system:
1. validates leaf presence,
2. segments up to 5 leaves with classical computer vision,
3. classifies each leaf as `Crop + Disease`,
4. applies a confidence threshold (default 70%),
5. generates Grad-CAM overlays,
6. returns structured per-leaf results and logs analytics.

## 2. Integrated CV Techniques (End-to-End Proof)
- Leaf validation: HSV green-content heuristic (`app/services/image_validation.py`)
- Classical segmentation: thresholding + contour filtering (`app/services/segmentation.py`)
- Per-leaf classification: CNN/MobileNetV2/Hybrid (`ml/src/models/*` + `app/services/inference.py`)
- Explainability: Grad-CAM overlays (`app/services/gradcam.py`, `ml/src/explainability/gradcam_utils.py`)
- Deployment API: FastAPI `/infer` + `/health` + `/analytics/summary` (`app/api/*`)
- Mobile frontend demo: camera capture + upload fallback (`app/templates/index.html`, `app/static/js/app.js`)

## 3. Technology Stack
- Backend: FastAPI
- ML/CV: PyTorch, torchvision, OpenCV, NumPy, scikit-learn
- Frontend: HTML/CSS/JavaScript (mobile-friendly, no React)
- Dataset: PlantVillage only

## 4. Repository Structure
```text
app/                    # FastAPI app, inference services, frontend templates/static
ml/src/                 # Data, models, training, evaluation, explainability code
ml/splits/              # Stratified split CSVs + class metadata
ml/weights/             # Expected model artifacts (weights/metrics/benchmarks)
docs/architecture/      # System flow and architecture diagrams
docs/reports/           # Model comparison and figure-generation assets
docs/submission/        # Compliance mapping + synopsis + demo + presentation package
```

## 5. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Expected PlantVillage layout:
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

## 7. Training Workflow
All training scripts support configurable core arguments (`--split-dir`, `--weights-json`, `--image-size`, `--batch-size`, `--max-epochs`, `--patience`, `--lr`) and output destination controls.

```bash
python ml/src/training/train_cnn.py
python ml/src/training/train_mobilenet.py
python ml/src/training/train_hybrid.py
```

## 8. Evaluation Workflow
```bash
python ml/src/evaluation/evaluate.py --model cnn --weights ml/weights/cnn_best.pt
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt

python ml/src/evaluation/benchmark.py --model cnn --weights ml/weights/cnn_best.pt
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt
```

## 9. Inference Workflow
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000`

API endpoints:
- `GET /health`
- `POST /infer`
- `GET /analytics/summary`

## 10. Model Comparison Summary (Observed Run)
Source of comparison record: `docs/reports/model_comparison_summary.md`.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Avg Inference (ms) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 |

### Deployment Model Selection (v1)
`MobileNetV2` is the selected deployment model for v1 because it achieved the best test Accuracy/F1 while staying smaller and slightly faster than Hybrid in the benchmarked environment.

Submission configuration defaults:
- `MODEL_NAME=mobilenet`
- `MODEL_WEIGHTS_PATH=ml/weights/mobilenet_best.pt`

## 11. Reproducibility Notes
Large artifacts are intentionally excluded from the public repository. Expected runtime artifacts:
- `ml/weights/cnn_best.pt`
- `ml/weights/mobilenet_best.pt`
- `ml/weights/hybrid_best.pt`
- `ml/weights/*_test_metrics.json`
- `ml/weights/*_benchmark.json`
- `ml/weights/*_confusion_matrix.png`

To regenerate, run training + evaluation commands in Sections 7 and 8.

`classes.json` must exist at:
- `ml/weights/classes.json`
Generated by:
```bash
python ml/src/data/export_classes.py --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

## 12. Limitations
- PlantVillage-to-field domain shift may reduce real-world performance.
- Classical segmentation can degrade in heavy shadows/clutter.
- Inference confidence thresholding prevents forced predictions but may increase retake prompts.

## 13. Final Deliverables Map
- Technical docs + code: `README.md`, `app/`, `ml/src/`
- Architecture docs: `docs/architecture/system_flow.md`, `docs/architecture/system_architecture.md`
- Model comparison evidence: `docs/reports/model_comparison_summary.md`
- Figures package: `docs/reports/figures/README.md`
- Compliance mapping: `docs/submission/assignment_compliance_report.md`
- Technical synopsis source: `docs/submission/technical_synopsis.md`
- Demo package: `docs/submission/demo_video_script.md`
- Management presentation package: `docs/submission/presentation/management_presentation_content.md`
- Final artifact manifest: `docs/submission/final_artifacts_manifest.md`

## 14. Human-Action Items (Cannot Be Auto-Fabricated)
- Export `docs/submission/technical_synopsis.pdf` from synopsis markdown.
- Publish Jupyter Book from `docs/submission/jupyter_book/`.
- Record and upload 5–8 minute demo video.
- Export final management deck (`.pptx` / `.pdf`) from presentation content package.
