# Developer Guide — AI-Powered Plant Disease Detection System

## 1. Project Overview

This repository implements an end-to-end computer vision prototype for plant
disease detection using the PlantVillage dataset. The system is designed as a
mobile-first academic prototype: a user captures or uploads a leaf image in a
browser, the backend validates the image, segments leaf regions, classifies each
leaf, generates Grad-CAM explanations, and returns a structured diagnosis.

The real-world problem addressed here is delayed disease identification in crop
leaves. In practice, growers often need a quick first-pass diagnosis before
consulting an agronomist. This project demonstrates how that workflow can be
supported with a deployable AI-assisted interface.

Major capabilities:

- Browser-based image capture and upload
- Leaf-content validation before inference
- Multi-leaf segmentation with classical OpenCV
- Per-leaf crop+disease classification
- Confidence-thresholded responses
- Grad-CAM explainability overlays
- Lightweight analytics logging and summary reporting

High-level workflow:

```text
User
  -> Image Capture / Upload
  -> Leaf Validation
  -> Leaf Segmentation
  -> Classification
  -> Grad-CAM
  -> Structured JSON Response + Rendered UI
```

## 2. System Architecture

The project is split into an online inference stack and an offline ML
experiment stack.

### High-level architecture

```text
User Image
   |
   v
FastAPI API
   |
   +--> Leaf Validation
   +--> Leaf Segmentation
   +--> Image Preprocessing
   +--> Model Inference
   +--> Grad-CAM Generation
   +--> Analytics Logging
   |
   v
Structured JSON Response
   |
   v
Browser UI Rendering
```

### Subsystems

| Subsystem | Purpose | Main Files |
|---|---|---|
| Web/client interface | Camera capture, upload fallback, preview, result rendering | `app/templates/index.html`, `app/static/js/app.js`, `app/static/css/styles.css` |
| FastAPI backend | Route handling, dependency wiring, static asset serving | `app/main.py`, `app/api/routes_infer.py`, `app/api/routes_health.py` |
| Image preprocessing | Resize/normalize for model input | `app/services/inference.py`, `ml/src/data/augmentations.py` |
| Leaf validation | Reject non-leaf images early using HSV green-ratio heuristic | `app/services/image_validation.py` |
| Leaf segmentation | Extract up to 5 leaf crops using masking + contour filtering | `app/services/segmentation.py` |
| Disease classification | Run CNN, MobileNetV2, or hybrid classifier | `app/services/inference.py`, `ml/src/models/*.py` |
| Grad-CAM explainability | Produce per-leaf saliency overlays | `app/services/gradcam.py`, `ml/src/explainability/gradcam_utils.py` |
| Analytics logging | Persist JSONL events and compute summaries | `app/services/analytics.py` |

### Runtime data flow

1. The browser submits an image to `POST /infer`.
2. The backend saves the original upload to `app/static/uploads`.
3. The image is validated for sufficient leaf-like content.
4. If valid, segmentation extracts candidate leaf crops.
5. Each crop is normalized and passed through the selected classifier.
6. Predictions below the configured confidence threshold are discarded.
7. Accepted predictions generate Grad-CAM overlays saved under
   `app/static/outputs`.
8. A structured `InferResponse` JSON payload is returned to the browser.
9. An analytics JSONL event is appended for later summary reporting.

## 3. Repository Structure

### Top-level view

```text
app/
  api/
  core/
  schemas/
  services/
  static/
  templates/

ml/
  src/
    data/
    models/
    training/
    evaluation/
    explainability/
  splits/
  weights/

docs/
  architecture/
  reports/
  submission/

requirements.txt
README.md
Dockerfile
```

### Folder responsibilities

| Folder | Responsibility | Notes |
|---|---|---|
| `app/` | Deployed inference application | FastAPI, services, templates, static assets |
| `app/api/` | HTTP routes | `/`, `/infer`, `/health`, `/analytics/summary` |
| `app/core/` | Shared infrastructure | Config and logging |
| `app/schemas/` | API contracts | Pydantic models for request/response payloads |
| `app/services/` | Runtime pipeline logic | Validation, segmentation, inference, Grad-CAM, analytics, storage |
| `ml/src/data/` | Dataset preparation and transforms | Splits, class weights, classes export, dataset wrapper |
| `ml/src/models/` | Model definitions | CNN, MobileNetV2, hybrid |
| `ml/src/training/` | Model training scripts | Shared training loop and model-specific CLIs |
| `ml/src/evaluation/` | Metrics and benchmarking | Evaluation JSON, confusion matrices, latency benchmarks |
| `ml/src/explainability/` | Offline Grad-CAM helpers | Experimental/notebook explainability |
| `ml/splits/` | Generated split artifacts | `train.csv`, `val.csv`, `test.csv`, `classes.txt` |
| `ml/weights/` | Model artifacts | Checkpoints, classes JSON, training history |
| `docs/architecture/` | Architecture writeups | System flow and pipeline documentation |
| `docs/reports/` | Evaluation reports and scripts | Metrics, figures, evidence generation |
| `docs/submission/` | Submission package | Synopsis, slides, Jupyter Book, checklists |

## 4. Data Pipeline

The offline data pipeline follows this sequence:

```text
PlantVillage Dataset
  -> Split Generation
  -> Class Weight Export
  -> Class Name Export
  -> Model Training
  -> Test Evaluation
  -> Deployment Artifact Selection
```

### Dataset

- Dataset: PlantVillage
- Expected layout:

```text
ml/datasets/plantvillage/<Crop___Disease>/*.jpg
```

- Label format: `Crop___Disease`
  - Example: `Tomato___Late_blight`
  - Healthy examples use `___healthy`

### Split policy

The repository uses stratified `80/10/10` train/validation/test splits:

- Train: 80%
- Validation: 10%
- Test: 10%

Split generation is implemented in `ml/src/data/split_data.py`.

### Preprocessing and augmentation

Training transforms in `ml/src/data/augmentations.py`:

- Resize to square `image_size`
- Random horizontal flip
- Random rotation (`20` degrees)
- Random affine scaling
- Brightness jitter
- Tensor conversion
- ImageNet normalization

Evaluation/inference transforms:

- Resize to square `image_size`
- Tensor conversion
- ImageNet normalization

Normalization constants:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

## 5. Machine Learning Models

Three classifier families are implemented.

### 1. CNN baseline

- File: `ml/src/models/cnn_baseline.py`
- Purpose: simple from-scratch baseline for comparison
- Architecture: stacked conv blocks + adaptive pooling + MLP classifier
- Strengths:
  - Small and easy to understand
  - Fastest of the compared models
- Limitations:
  - Lower overall predictive quality than MobileNetV2

### 2. MobileNetV2

- File: `ml/src/models/mobilenet_baseline.py`
- Purpose: transfer-learning baseline and deployment model
- Architecture: torchvision MobileNetV2 with replaced classifier head
- Strengths:
  - Best reported accuracy and macro-F1 in this repository
  - Strong tradeoff between quality and deployability
  - Standard pretrained backbone
- Limitations:
  - Larger and slower than the simple CNN baseline

### 3. Hybrid model

- File: `ml/src/models/hybrid_model.py`
- Purpose: experimental fusion of transfer-learned and custom CNN features
- Architecture:
  - MobileNetV2 feature branch
  - custom residual branch
  - concatenated fused head
- Strengths:
  - Competitive accuracy
  - Tests whether fused features improve classification
- Limitations:
  - Largest footprint
  - Slightly slower than MobileNetV2
  - More complex to maintain

### Why MobileNetV2 was selected for deployment

According to `docs/reports/model_comparison_summary.md`, MobileNetV2 achieved
the best reported accuracy and macro-F1 among the compared models while keeping
the runtime and size manageable for a browser-backed prototype. It is therefore
the default model loaded by the deployed API.

## 6. Training Pipeline

Training entrypoints:

- `python -m ml.src.training.train_cnn`
- `python -m ml.src.training.train_mobilenet`
- `python -m ml.src.training.train_hybrid`

Shared training logic lives in `ml/src/training/utils.py`.

### Core training behavior

- Loss: weighted cross-entropy
- Optimizer: Adam
- Early stopping: validation-loss based
- Checkpoint policy: save best model state after early stopping tracking
- History output: JSON file containing per-epoch train/validation metrics

### Important hyperparameters

These are configured per script:

- `image_size`
- `batch_size`
- `max_epochs`
- `patience`
- `lr`
- `num_workers`

### Output artifacts

Saved under `ml/weights/`, for example:

- `mobilenet_best.pt`
- `mobilenet_history.json`
- `cnn_best.pt`
- `hybrid_best.pt`

### Example command

```bash
python ml/src/training/train_mobilenet.py
```

### Resuming or running new experiments

There is no dedicated resume flag in the current scripts. The intended workflow
is to:

1. prepare split artifacts,
2. choose output checkpoint/history paths,
3. rerun the desired training script.

To add explicit resume support, the cleanest extension point is
`ml/src/training/utils.py`.

## 7. Evaluation and Benchmarking

Evaluation scripts:

- `ml/src/evaluation/evaluate.py`
- `ml/src/evaluation/benchmark.py`
- `ml/src/evaluation/confusion_matrix.py`
- `ml/src/evaluation/metrics.py`

### Metrics used

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix
- Average per-image inference time
- Model checkpoint size

### Outputs

Typically written to `ml/weights/` or `docs/reports/metrics/`:

- `*_test_metrics.json`
- `*_benchmark.json`
- `*_confusion_matrix.png`

### Example commands

```bash
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

## 8. Inference Pipeline

The deployed inference path is implemented in `app/api/routes_infer.py` and
`app/services/*`.

### Step-by-step flow

1. Receive uploaded image.
2. Save original image to `app/static/uploads`.
3. Validate that the image contains leaf-like content.
4. Segment up to 5 leaf regions.
5. Resize and normalize each crop.
6. Run classifier for each crop.
7. Apply confidence threshold.
8. Generate Grad-CAM explanation for accepted predictions.
9. Return structured JSON response.

### Multiple leaves per image

Multi-leaf handling is classical-CV based:

- segmentation returns multiple `LeafSegment` objects,
- each segment is classified independently,
- each accepted segment gets its own `LeafResult`.

The response includes:

- `total_leaves_detected`
- zero or more accepted `results`

### Edge cases

#### No leaf detected

If validation fails, or segmentation produces no valid contours:

- response status: `invalid`
- message: `"No leaf detected. Please retake the photo."`

#### Low-confidence predictions

If all segmented leaves fall below the confidence threshold:

- response status: `low_confidence`
- message: `"Low confidence predictions. Please retake the photo."`

This avoids force-labeling uncertain samples.

## 9. Explainability (Grad-CAM)

Grad-CAM highlights image regions that contributed most strongly to the chosen
prediction.

### Why it is used

- improves interpretability for demos and reviews
- makes the model output easier to inspect visually
- supports educational or audit-oriented explanations

### Where it is implemented

- Runtime API path: `app/services/gradcam.py`
- Offline/experimental helper: `ml/src/explainability/gradcam_utils.py`

### Generation flow

1. Run a forward pass on the input tensor.
2. Select the predicted class (or requested class).
3. Backpropagate that class score.
4. Average gradients over channels.
5. Weight target-layer activations.
6. Apply ReLU, normalize, resize, and overlay on the original crop.

## 10. API Endpoints

### `GET /`

Serves the browser UI.

### `GET /health`

Simple liveness probe.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /analytics/summary`

Returns aggregate diagnostics summary from the analytics log.

### `POST /infer`

Runs the full diagnosis pipeline on one uploaded image.

#### Request format

- Content type: `multipart/form-data`
- Field: `file`

#### Example response

```json
{
  "status": "ok",
  "message": "Diagnosis completed.",
  "image_id": "abc123",
  "latency_ms": 183.42,
  "total_leaves_detected": 2,
  "results": [
    {
      "leaf_id": 1,
      "crop_name": "Tomato",
      "disease_name": "Late blight",
      "confidence": 0.9821,
      "healthy_or_diseased": "Diseased",
      "short_description": "Serious oomycete disease with dark lesions and rapid spread.",
      "heatmap_path": "/app/static/outputs/abc123_leaf_1_gradcam.jpg"
    }
  ]
}
```

#### Error handling

- invalid image decode -> HTTP `400`
- non-leaf or unusable input -> valid JSON response with `status="invalid"`
- uncertain predictions -> valid JSON response with `status="low_confidence"`

## 11. Analytics and Logging

### Analytics

File: `app/services/analytics.py`

Each inference request appends one JSON event to:

- `logs/analytics.jsonl`

Logged information includes:

- timestamp
- image ID
- pipeline status
- response message
- accepted results
- latency

### Logging

File: `app/core/logging_config.py`

Application logs are written to:

- console
- rotating log file (`logs/app.log`)

### Why analytics matter

Analytics help answer questions such as:

- how often users upload invalid images
- which diseases are most frequently predicted
- what average confidence and latency look like

This is useful for future error analysis, UX improvements, and prioritizing
dataset/model work.

## 12. Running the System Locally

This section is written as an operator runbook: it focuses on the minimum
steps needed to bring the system up, confirm that required artifacts exist, and
verify that inference is working.

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare dataset

Place PlantVillage images under:

```text
ml/datasets/plantvillage/<Crop___Disease>/*.jpg
```

### 3. Generate splits and metadata

```bash
python -m ml.src.data.split_data --data-dir ml/datasets/plantvillage --out-dir ml/splits
python -m ml.src.data.class_weights --train-csv ml/splits/train.csv --out-json ml/splits/class_weights.json
python -m ml.src.data.export_classes --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

### 4. Train a model (optional if using existing weights)

```bash
python -m ml.src.training.train_mobilenet
```

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000
```

### 6. Test inference

- use the browser UI, or
- send a multipart request to `POST /infer`

### Deployment prerequisites checklist

Before starting the API, confirm the following files exist:

| Artifact | Expected Path | Why it matters |
|---|---|---|
| Deployment checkpoint | `ml/weights/mobilenet_best.pt` | Required by runtime inference |
| Class mapping | `ml/weights/classes.json` | Required to decode model outputs |
| Static upload dir | `app/static/uploads/` | Stores original user uploads |
| Static output dir | `app/static/outputs/` | Stores Grad-CAM images |
| Log dir | `logs/` | Stores app and analytics logs |

If `ml/weights/mobilenet_best.pt` is missing, `app/main.py` attempts to fetch
it from the configured Hugging Face repository during startup. If you want a
fully local/offline run, place the checkpoint in `ml/weights/` before launching
the API.

### Recommended startup verification

After launching `uvicorn`, verify in this order:

1. `GET /health` returns `{"status":"ok"}`.
2. `GET /` loads the browser UI.
3. Upload a known valid leaf image and confirm:
   - `status="ok"`
   - at least one result item
   - a Grad-CAM image appears under `app/static/outputs/`
4. Upload a clearly invalid image and confirm:
   - `status="invalid"`
   - no result items
5. Check that `logs/analytics.jsonl` has a new event row.

### Minimal API smoke test

Example with `curl`:

```bash
curl -X POST http://localhost:8000/infer \
  -F "file=@/path/to/leaf.jpg"
```

### What to inspect when something fails

| Symptom | Likely cause | Where to inspect |
|---|---|---|
| App fails on startup | Missing or unreadable weights/classes file | `app/main.py`, `app/services/inference.py`, `logs/app.log` |
| `/infer` returns HTTP 400 | Uploaded file is not a decodable image | `app/api/routes_infer.py` |
| `status="invalid"` for good images | Validation or segmentation heuristic too strict | `app/services/image_validation.py`, `app/services/segmentation.py` |
| `status="low_confidence"` repeatedly | Model confidence threshold too high or poor sample quality | `app/core/config.py`, `app/services/inference.py` |
| No Grad-CAM images saved | Output directory issue or Grad-CAM failure | `app/services/gradcam.py`, `app/services/storage.py` |
| UI loads but images do not display | Static file path mismatch | `app/main.py`, `app/api/routes_infer.py`, `app/static/js/app.js` |

### Runtime configuration used by deployment

The deployed app reads environment-backed defaults from `app/core/config.py`.
The most relevant settings for runbook use are:

| Setting | Default | Purpose |
|---|---:|---|
| `MODEL_NAME` | `mobilenet` | Selects deployed architecture |
| `MODEL_WEIGHTS_PATH` | `ml/weights/mobilenet_best.pt` | Checkpoint path |
| `CLASS_NAMES_PATH` | `ml/weights/classes.json` | Class metadata path |
| `CONFIDENCE_THRESHOLD` | `0.70` | Filters weak predictions |
| `MAX_LEAVES` | `5` | Caps segmented leaves per image |
| `TARGET_IMAGE_SIZE` | `384` | Runtime input size |
| `UPLOAD_DIR` | `app/static/uploads` | Original image storage |
| `OUTPUT_DIR` | `app/static/outputs` | Grad-CAM storage |

### Demo-day run sequence

For a stable local demo, use this sequence:

1. Confirm checkpoint and `classes.json` are present.
2. Start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
3. Open `http://localhost:8000`.
4. Test one valid healthy image.
5. Test one diseased image.
6. Test one invalid non-leaf image.
7. Confirm analytics summary via `GET /analytics/summary`.

This sequence validates the three most important runtime paths:

- successful diagnosis
- confidence/validation failure handling
- analytics logging

## 13. Reproducibility

Reproducibility in this repository depends on keeping the full artifact chain
consistent:

- fixed split generation procedure
- explicit `classes.txt` ordering
- exported class-weight JSON
- saved checkpoint and history artifacts
- explicit evaluation and benchmark scripts

### Reproducible experiment flow

```bash
python -m ml.src.data.split_data --data-dir ml/datasets/plantvillage --out-dir ml/splits
python -m ml.src.data.class_weights --train-csv ml/splits/train.csv --out-json ml/splits/class_weights.json
python -m ml.src.data.export_classes --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
python -m ml.src.training.train_mobilenet
python -m ml.src.evaluation.evaluate --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python -m ml.src.evaluation.benchmark --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

### Result locations

- splits: `ml/splits/`
- model artifacts: `ml/weights/`
- evaluation summaries: `docs/reports/metrics/`, `docs/reports/`

Note: the repository does not expose a full seed-control training framework in
the same way a production experiment platform would. For strict reproducibility
beyond artifact traceability, seed-setting and environment capture could be
extended in future work.

## 14. Deployment Notes

### Runtime lifecycle

At startup, the app performs these steps:

1. ensure model weights are present,
2. load environment-backed settings,
3. initialize logging,
4. create storage/validation/segmentation/inference/analytics services,
5. bind those services into the FastAPI routers.

This means most deployment issues appear immediately at startup rather than on
the first request.

### Files written during runtime

| File or directory | Written by | Purpose |
|---|---|---|
| `app/static/uploads/*` | `StorageService` | Original uploaded images |
| `app/static/outputs/*` | `GradCAMService` | Saved Grad-CAM overlays |
| `logs/analytics.jsonl` | `AnalyticsService` | One event per inference request |
| `logs/app.log` | logging config | Application log stream |

### Safe operational modifications

Changes that are usually safe for deployment tuning:

- adjusting `confidence_threshold`
- adjusting `max_leaves`
- changing model artifact paths
- replacing the checkpoint with a compatible model of the same architecture

Changes that require more care:

- modifying response schema fields
- changing label format or class ordering
- changing segmentation output assumptions
- swapping in a different architecture without updating `InferenceService`

### Containerization

The repository includes a `Dockerfile`, but the primary documented deployment
path is local `uvicorn` execution. If you containerize the app, make sure the
container image includes:

- Python dependencies from `requirements.txt`
- model artifacts under `ml/weights/`, or equivalent mounted volume
- writable directories for `app/static/uploads`, `app/static/outputs`, and
  `logs`

## 15. How to Extend the System

### Add a new model

- add the model under `ml/src/models/`
- update training scripts or add a new one under `ml/src/training/`
- update `ml/src/evaluation/evaluate.py`
- update `app/services/inference.py` model factory

### Add a new dataset

- create a compatible class-folder layout or a new data-loader path
- update `ml/src/data/split_data.py` if label extraction changes
- regenerate `classes.txt`, splits, and class weights

### Improve segmentation

- modify `app/services/segmentation.py`
- keep return contract compatible with `LeafSegment`
- retest multi-leaf and invalid-image behavior

### Improve leaf validation

- modify thresholds or logic in `app/services/image_validation.py`
- keep failure mode compatible with current API response handling

### Add ONNX or alternate deployment backend

- add a new inference backend behind `app/services/inference.py`
- keep output shape, label parsing, and response schema unchanged

### Add mobile deployment or packaged client

- preserve current API contract in `app/schemas/response_schemas.py`
- treat the FastAPI service as the stable backend boundary

## 16. Contribution Guidelines

For contributors working on this repository:

- keep code readable and documented
- preserve current API response structure unless a coordinated change is needed
- prefer adding focused modules over embedding complex logic in route handlers
- keep training/evaluation scripts reproducible and CLI-driven
- update documentation when behavior or artifacts change

Suggested workflow:

1. understand the affected layer (`app/` vs `ml/src/`)
2. make the smallest coherent change
3. validate the change locally
4. update docs or reports if artifacts/usage changed

When modifying the inference path, verify:

- `mom`-style schema concerns do not apply here; the stable contract is
  `InferResponse`
- image validation and segmentation failures still return compatible responses
- Grad-CAM output paths remain browser-accessible

## 17. Known Limitations

Current limitations include:

- strong domain dependence on PlantVillage imagery
- likely field-performance drop under clutter, shadows, and uncontrolled scenes
- classical segmentation heuristics may fail on complex backgrounds
- confidence thresholding can trade off recall for safer UX
- no production auth, queueing, or large-scale deployment hardening
- no explicit training resume workflow

## 18. Future Roadmap

Realistic next steps for this codebase:

- train and evaluate on real field datasets
- replace heuristic segmentation with learned segmentation/detection
- export optimized inference paths such as ONNX or TensorRT
- improve runtime benchmarking and deployment packaging
- add richer analytics dashboards
- support multilingual guidance and treatment recommendations
- add stronger experiment reproducibility controls, including seed capture and
  environment manifests

## Useful Entry Points

For quick orientation, start with these files:

- `README.md`
- `app/main.py`
- `app/api/routes_infer.py`
- `app/services/inference.py`
- `app/services/segmentation.py`
- `ml/src/training/train_mobilenet.py`
- `ml/src/evaluation/evaluate.py`
- `docs/architecture/system_architecture.md`
- `docs/reports/model_comparison_summary.md`
