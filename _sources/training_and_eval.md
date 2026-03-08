# Training and Evaluation

## Training
Scripts:
- `ml/src/training/train_cnn.py`
- `ml/src/training/train_mobilenet.py`
- `ml/src/training/train_hybrid.py`

Shared utility: `ml/src/training/utils.py` implements early stopping and best-checkpoint restore.

## Evaluation
Scripts:
- `ml/src/evaluation/evaluate.py`
- `ml/src/evaluation/benchmark.py`

Metrics:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Inference speed and model size
