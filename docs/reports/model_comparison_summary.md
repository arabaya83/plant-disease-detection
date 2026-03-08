# Model Comparison Summary (Submission)

This document provides the consolidated model-selection evidence for the final project submission.

## Scope
Compared models:
- Simple CNN baseline
- MobileNetV2 baseline (ImageNet pretrained)
- Hybrid model (MobileNetV2 branch + custom residual branch)

## Results Table
| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Inference Speed (ms) | Model Size (MB) | Selection Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 | Baseline only |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 | **Selected for v1 deployment** |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 | Not selected |

## Interpretation
- MobileNetV2 delivered the highest test Accuracy and macro-F1 in the recorded experiment set.
- Hybrid remained competitive but was larger and slightly slower in benchmarked inference.
- CNN provided a strong small-model baseline but lower overall predictive performance than MobileNetV2.

## Final Selection Rationale
`MobileNetV2` is selected for version 1 deployment because it offers the best practical balance of:
1. Predictive quality (Accuracy + F1)
2. Deployment footprint
3. Runtime responsiveness

## Evidence Regeneration and Storage
Expected regenerated artifacts in `ml/weights/`:
- `cnn_test_metrics.json`, `mobilenet_test_metrics.json`, `hybrid_test_metrics.json`
- `cnn_benchmark.json`, `mobilenet_benchmark.json`, `hybrid_benchmark.json`
- `cnn_confusion_matrix.png`, `mobilenet_confusion_matrix.png`, `hybrid_confusion_matrix.png`

Recompute commands:
```bash
python ml/src/evaluation/evaluate.py --model cnn --weights ml/weights/cnn_best.pt
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt

python ml/src/evaluation/benchmark.py --model cnn --weights ml/weights/cnn_best.pt
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt
```
