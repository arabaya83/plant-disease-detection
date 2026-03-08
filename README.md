# AI-Powered Plant Disease Detection Prototype (PlantVillage)

End-to-end web prototype for mobile field usage by small farmers. Supports camera capture/upload, leaf validation, multi-leaf segmentation, per-leaf classification, Grad-CAM explainability, and analytics.

## Tech Stack
- Backend: FastAPI (Python)
- ML/CV: PyTorch, torchvision, OpenCV, NumPy, scikit-learn
- Frontend: HTML/CSS/JavaScript (mobile-friendly, no React)
- Dataset: PlantVillage only

## Repository Structure
See requested structure under `app/`, `ml/`, `docs/`, and `logs/`.

## Quick Start
1. Create environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy env file:
```bash
cp .env.example .env
```

3. Place PlantVillage data:
- Expected format: `ml/datasets/plantvillage/<class_name>/*.jpg`

4. Build splits:
```bash
python ml/src/data/split_data.py \
  --data-dir ml/datasets/plantvillage \
  --out-dir ml/splits
```

5. Compute class weights:
```bash
python ml/src/data/class_weights.py \
  --train-csv ml/splits/train.csv \
  --out-json ml/splits/class_weights.json
```

6. Train models:
```bash
python ml/src/training/train_cnn.py
python ml/src/training/train_mobilenet.py
python ml/src/training/train_hybrid.py
```

7. Export class names for inference:
```bash
python ml/src/data/export_classes.py --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

8. Evaluate:
```bash
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt
```

9. Run app:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000`

## API
- `GET /health`
- `POST /infer` (multipart file upload)
- `GET /analytics/summary`

## Workflow
Browser handles capture/preview/optional resize. Server handles validation, segmentation (classical CV), inference, Grad-CAM, storage, logging, analytics.

## Notes
- Confidence threshold default is 70% (`CONFIDENCE_THRESHOLD=0.70`).
- If no valid leaf is detected: `No leaf detected. Please retake the photo.`
- If all predictions are below threshold: asks user to retake photo.
- Up to 5 leaves processed per image.
- TODO: Train and export final production weights into `ml/weights/hybrid_best.pt` with full PlantVillage classes for accurate field-ready predictions.
- TODO: Populate `ml/weights/classes.json` from `ml/splits/classes.txt` after split/training.

## Limitations
- Accuracy depends on trained weights availability.
- Classical segmentation may degrade under severe lighting/background clutter.
- PlantVillage domain shift expected for real field photos.

## Future Roadmap
- Better domain adaptation for in-field images.
- On-device model optimization and PWA/mobile app packaging.
- Personalized advisory and multilingual support.
