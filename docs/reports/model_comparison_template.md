# Model Comparison Summary

## Models
- Simple CNN baseline
- MobileNetV2 baseline (ImageNet pretrained)
- Hybrid model (MobileNetV2 branch + custom residual branch)

## Metrics Table
| Model | Accuracy | Precision | Recall | F1 | Avg Inference (ms) | Model Size (MB) |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 0.9878 | 0.9844 | 0.9882 | 0.9861 | 127.754 | 1.79 |
| MobileNetV2 | 0.9976 | 0.9938 | 0.9965 | 0.9950 | 201.414 | 8.90 |
| Hybrid | 0.9948 | 0.9899 | 0.9939 | 0.9917 | 206.399 | 18.35 |

## Final Selection Rule
Choose the model with the best balance of:
1. Accuracy / F1
2. Inference speed
3. Model size and deployability

## Selected Model
`MobileNetV2` selected for version 1 deployment because it has the best test accuracy/F1 while running slightly faster and much smaller than Hybrid on the benchmarked CPU setup.
