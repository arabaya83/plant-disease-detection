# Selected Model Rationale (Single Source of Truth)

## Final Deployment Model (v1)
**MobileNetV2**

## Comparison Snapshot
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Inference (ms) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 |

## Why MobileNetV2 Was Selected
1. Highest observed Accuracy and macro-F1 among compared models.
2. Better deployability than Hybrid due to smaller size.
3. Slightly faster benchmarked inference than Hybrid in recorded runs.
4. Better production trade-off than CNN, which is smaller/faster but less accurate.

## How It Differs from Other Models
- Versus CNN baseline:
  - stronger predictive performance,
  - moderate size increase with significant quality gain.
- Versus Hybrid:
  - comparable family performance,
  - lower model size and better speed in observed benchmark,
  - simpler deployment profile.

## Artifact Paths for MobileNetV2
Expected paths:
- Weights: `ml/weights/mobilenet_best.pt`
- Test metrics: `ml/weights/mobilenet_test_metrics.json`
- Benchmark: `ml/weights/mobilenet_benchmark.json`
- Confusion matrix: `ml/weights/mobilenet_confusion_matrix.png`

If not present, regenerate with:
```bash
python ml/src/training/train_mobilenet.py --out-weights ml/weights/mobilenet_best.pt --out-history ml/weights/mobilenet_history.json
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
```

## Config Alignment
Deployment defaults are aligned in `app/core/config.py`:
- `model_name = mobilenet`
- `model_weights_path = ml/weights/mobilenet_best.pt`
