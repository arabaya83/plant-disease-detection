# AI-Powered Plant Disease Detection for Small Farmers

## 1. Problem and Objective
Smallholder farmers often face delayed disease diagnosis due to limited in-field
expert access. Delayed action can increase crop-loss risk and treatment cost.
This project delivers a mobile-first web prototype for explainable diagnosis
from leaf images.

Version-1 objectives:
- support camera capture and browser upload,
- validate leaf presence,
- segment up to 5 leaves per image,
- classify each leaf as `Crop + Disease`,
- return retake guidance for invalid/low-confidence inputs,
- provide Grad-CAM visual explanations,
- expose inference through FastAPI with analytics logging.

## 2. Dataset and Preprocessing
Dataset: PlantVillage (38 classes in this setup), folder-per-class format.

Target labels use `Crop + Disease` naming
(e.g., `Tomato___Late_blight`, `Apple___healthy`).
Healthy outputs are presented as `Healthy - No disease detected`.

Split policy:
- 80% train, 10% validation, 10% test,
- stratified by class.

Preprocessing:
- resize to 384x384 (configurable),
- ImageNet normalization,
- standard augmentation only (flip, rotation, scale/zoom, brightness).

Class imbalance is handled by class-weighted cross-entropy.

## 3. Methods and Mathematical Basis
Compared classifiers:
1. Simple CNN baseline,
2. MobileNetV2 baseline (ImageNet pretrained),
3. Hybrid model (MobileNetV2 branch + residual branch with feature fusion).

For input `x`, logits `z` produce class probability:

`p(y=k|x) = softmax(z)_k`

Weighted loss:

`L = - w_y log p(y|x)`

Grad-CAM explanation for class `c`:

`L^c = ReLU(sum_k alpha_k^c A^k)`

where `A^k` are target-layer feature maps and `alpha_k^c` are pooled gradients.

## 4. Training Policy
- Optimizer: Adam
- Max epochs: 50
- Early stopping on validation loss
- Best-checkpoint restore

Representative hyperparameters:
- image size 384x384,
- batch size 16 (CNN/MobileNetV2), 12 (Hybrid),
- learning rate 1e-3 (CNN), 1e-4 (MobileNetV2/Hybrid),
- patience 8.

## 5. Evaluation and Results
Metrics:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix
- Inference speed (ms)
- Model size (MB)

Observed comparison record:
- CNN: 0.9878 Acc, 0.9844 Prec, 0.9882 Rec, 0.9861 F1, 127.754 ms, 1.79 MB
- MobileNetV2: 0.9976 Acc, 0.9938 Prec, 0.9965 Rec, 0.9950 F1, 201.414 ms, 8.90 MB
- Hybrid: 0.9948 Acc, 0.9899 Prec, 0.9939 Rec, 0.9917 F1, 206.399 ms, 18.35 MB

Deployment selection for v1: **MobileNetV2**.
Rationale: best Accuracy/F1 with smaller footprint and slightly better speed than Hybrid.
Canonical decision record: `docs/reports/selected_model_rationale.md`.

## 6. Inference and Deployment Behavior
`POST /infer` pipeline:
1. save uploaded image,
2. validate leaf content,
3. segment leaves with classical OpenCV processing,
4. classify each leaf independently,
5. suppress predictions below confidence threshold (0.70 default),
6. generate Grad-CAM overlays for accepted leaves,
7. return structured per-leaf response,
8. log status/confidence/classes/latency.

Edge handling:
- no leaf detected -> `No leaf detected. Please retake the photo.`
- all low-confidence -> retake guidance.

## 7. Challenges and Mitigations
1. Domain shift (PlantVillage vs field images): mitigated with confidence thresholding and retake guidance.
2. Multi-leaf photos: mitigated with segmentation + per-leaf inference.
3. Explainability need: mitigated with Grad-CAM overlays.
4. Operational traceability: mitigated with analytics logging and summary endpoint.

## 8. Conclusion
The project demonstrates full CV integration from data pipeline to deployment-oriented
inference. It provides a practical MVP foundation and clear next steps for
field-data adaptation, calibration, and mobile optimization.
