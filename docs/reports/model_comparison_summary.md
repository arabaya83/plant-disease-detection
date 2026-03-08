# Model Comparison Summary (Submission)

This report consolidates model performance evidence and the deployment decision.

## Compared Models
- Simple CNN baseline
- MobileNetV2 baseline (ImageNet pretrained)
- Hybrid model (MobileNetV2 branch + custom residual branch)

## Results
| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) | Inference Speed (ms) | Model Size (MB) | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 | Baseline only |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 | **Selected for v1 deployment** |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 | Comparison model |

## Interpretation
- MobileNetV2 achieved the best Accuracy and macro-F1 in the observed run.
- Hybrid remained competitive but had larger footprint and slightly slower speed.
- CNN is compact and fast but lower overall predictive quality than MobileNetV2.

## Selection Decision
Final v1 deployment model: **MobileNetV2**.

Canonical rationale and usage paths:
- `docs/reports/selected_model_rationale.md`

## Regeneration Commands
```bash
python ml/src/evaluation/evaluate.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model mobilenet --weights ml/weights/mobilenet_best.pt --out-dir ml/weights

python ml/src/evaluation/evaluate.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/evaluate.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model cnn --weights ml/weights/cnn_best.pt --out-dir ml/weights
python ml/src/evaluation/benchmark.py --model hybrid --weights ml/weights/hybrid_best.pt --out-dir ml/weights
```
