# Code Documentation Guide

This guide explains the codebase module-by-module so reviewers can quickly
understand the end-to-end implementation.

## Runtime Application (`app/`)

### Entrypoint
- `app/main.py`
  - Creates FastAPI app.
  - Instantiates shared services (storage, validation, segmentation, inference, analytics).
  - Binds services into routers.

### API Layer
- `app/api/routes_health.py`
  - `GET /health`: service health check.
  - `GET /analytics/summary`: aggregated usage and quality metrics.
- `app/api/routes_infer.py`
  - `GET /`: serves capture/upload web page.
  - `POST /infer`: full inference pipeline:
    1. save upload
    2. validate leaf presence
    3. classical multi-leaf segmentation
    4. per-leaf classification
    5. confidence threshold filtering
    6. Grad-CAM generation and save
    7. analytics logging

### Core Services
- `app/services/image_validation.py`
  - HSV green-ratio heuristic.
  - Returns invalid when leaf-like content is insufficient.
- `app/services/segmentation.py`
  - Classical OpenCV segmentation with contour filtering.
  - Returns up to `max_leaves` crops with bounding boxes.
- `app/services/inference.py`
  - Loads selected architecture (`cnn`, `mobilenet`, `hybrid`).
  - Loads class labels and model weights.
  - Produces `Prediction` object with crop-level metadata.
- `app/services/gradcam.py`
  - Hooks model target layer and computes Grad-CAM overlays.
- `app/services/storage.py`
  - Persists uploads and deterministic overlay output paths.
- `app/services/analytics.py`
  - Appends JSONL inference events.
  - Computes summary stats for dashboard/reporting.
- `app/services/disease_info.py`
  - Label parsing and short disease description mapping.

### Schemas
- `app/schemas/request_schemas.py`
- `app/schemas/response_schemas.py`
  - Typed request/response contracts used by FastAPI and frontend.

### Frontend
- `app/templates/index.html`
  - Mobile-first UI for camera capture + file upload.
- `app/static/js/app.js`
  - Capture, preview, submit, loading state, result rendering.
- `app/static/css/`
  - Responsive styles for field-device usage.

## ML Pipeline (`ml/src/`)

### Data Preparation
- `ml/src/data/split_data.py`
  - Stratified 80/10/10 split.
  - Exports `train.csv`, `val.csv`, `test.csv`, `classes.txt`.
- `ml/src/data/class_weights.py`
  - Computes loss weights from training distribution.
- `ml/src/data/export_classes.py`
  - Converts class list into `ml/weights/classes.json`.
- `ml/src/data/dataset.py`
  - CSV-backed dataset loader used by train/eval scripts.
- `ml/src/data/augmentations.py`
  - Standard augmentation policy for v1.

### Models
- `ml/src/models/cnn_baseline.py`
  - Simple CNN baseline from scratch.
- `ml/src/models/mobilenet_baseline.py`
  - MobileNetV2 fine-tuning baseline (ImageNet-pretrained option).
- `ml/src/models/hybrid_model.py`
  - MobileNet branch + residual branch + feature fusion head.

### Training
- `ml/src/training/train_cnn.py`
- `ml/src/training/train_mobilenet.py`
- `ml/src/training/train_hybrid.py`
  - Per-model training entrypoints with configurable CLI args.
- `ml/src/training/utils.py`
  - Shared epoch loop, macro metrics, early stopping, history export.

### Evaluation
- `ml/src/evaluation/evaluate.py`
  - Test metrics + confusion matrix export.
- `ml/src/evaluation/benchmark.py`
  - Inference latency and model-size benchmark export.
- `ml/src/evaluation/metrics.py`
  - Accuracy / Precision / Recall / F1 utilities.
- `ml/src/evaluation/confusion_matrix.py`
  - Confusion matrix visualization helper.

## Artifacts and Reports

- `ml/splits/`: split CSVs and class metadata.
- `ml/weights/`: model weights and evaluation outputs.
- `docs/reports/`: model comparison and evidence package.
- `docs/submission/`: compliance mapping, synopsis source, demo script, presentation source.

## Typical End-to-End Command Flow

```bash
# 1) Data split and metadata
python -m ml.src.data.split_data --data-dir ml/datasets/plantvillage --out-dir ml/splits
python -m ml.src.data.class_weights --train-csv ml/splits/train.csv --out-json ml/splits/class_weights.json
python -m ml.src.data.export_classes --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json

# 2) Train deployment model
python -m ml.src.training.train_mobilenet --split-dir ml/splits --weights-json ml/splits/class_weights.json

# 3) Evaluate + benchmark
python -m ml.src.evaluation.evaluate --model mobilenet --weights ml/weights/mobilenet_best.pt --split-dir ml/splits
python -m ml.src.evaluation.benchmark --model mobilenet --weights ml/weights/mobilenet_best.pt

# 4) Run application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Notes for Reviewers

- Default deployment model for v1 is **MobileNetV2**.
- Dataset is intentionally excluded from Git tracking.
- Large generated artifacts may be managed via Git LFS or external links.
