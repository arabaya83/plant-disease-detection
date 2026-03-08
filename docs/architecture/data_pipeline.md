# Data and Experiment Pipeline

## 1) Dataset Ingestion
Input directory format:
`ml/datasets/plantvillage/<Crop___Disease>/*.jpg`

## 2) Split Generation (Stratified 80/10/10)
```bash
python ml/src/data/split_data.py --data-dir ml/datasets/plantvillage --out-dir ml/splits
```
Outputs:
- `ml/splits/train.csv`
- `ml/splits/val.csv`
- `ml/splits/test.csv`
- `ml/splits/classes.txt`

## 3) Class Weights and Class Export
```bash
python ml/src/data/class_weights.py --train-csv ml/splits/train.csv --out-json ml/splits/class_weights.json
python ml/src/data/export_classes.py --classes-txt ml/splits/classes.txt --out-json ml/weights/classes.json
```

## 4) Model Training (Reproducible CLI)
Examples:
```bash
python ml/src/training/train_cnn.py --split-dir ml/splits --weights-json ml/splits/class_weights.json --out-weights ml/weights/cnn_best.pt --out-history ml/weights/cnn_history.json
python ml/src/training/train_mobilenet.py --split-dir ml/splits --weights-json ml/splits/class_weights.json --out-weights ml/weights/mobilenet_best.pt --out-history ml/weights/mobilenet_history.json
python ml/src/training/train_hybrid.py --split-dir ml/splits --weights-json ml/splits/class_weights.json --out-weights ml/weights/hybrid_best.pt --out-history ml/weights/hybrid_history.json
```

## 5) Evaluation and Benchmark
```bash
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

## 6) Deployment Inference
FastAPI loads model artifacts from `.env`:
- `MODEL_NAME`
- `MODEL_WEIGHTS_PATH`
- `CLASS_NAMES_PATH`

These must match available artifacts in `ml/weights/`.
