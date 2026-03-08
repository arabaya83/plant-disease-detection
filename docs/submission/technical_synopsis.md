# AI-Powered Plant Disease Detection for Small Farmers

## 1. Problem and Objective
Smallholder farmers often face delayed disease diagnosis because agronomy support is not immediately available in the field. Delayed decisions can increase yield loss and treatment cost. This project delivers a mobile-first web prototype that performs explainable plant disease diagnosis from leaf images.

Version-1 objectives:
- support camera capture and image upload in browser,
- validate leaf presence,
- segment up to 5 leaves in one image,
- classify each leaf as `Crop + Disease`,
- return retake guidance for invalid/low-confidence inputs,
- provide Grad-CAM visual explanations,
- expose inference through FastAPI with logging/analytics.

## 2. Dataset and Preprocessing
Dataset: PlantVillage (38 classes in this setup), folder-per-class format.

Label target: `Crop + Disease` (e.g., `Tomato___Late_blight`, `Apple___healthy`).
Healthy predictions are rendered as `Healthy - No disease detected`.

Data split policy:
- 80% train, 10% validation, 10% test,
- stratified by class.

Preprocessing:
- resize to configurable size (default 384x384),
- ImageNet normalization,
- standard augmentation only (horizontal flip, rotation, scale/zoom, brightness).

Class imbalance is handled with class-weighted cross-entropy (weights computed from train split).

## 3. Methods and Mathematical Basis
Three classifiers were implemented and compared:
1. Simple CNN baseline,
2. MobileNetV2 baseline (ImageNet pretrained),
3. Hybrid model (MobileNetV2 branch + custom residual branch, feature concatenation).

For input image `x`, model outputs logits `z`, with class probability:

`p(y=k|x) = softmax(z)_k`.

Training loss:

`L = - w_y log p(y|x)`

where `w_y` is the class-specific weight for true class `y`.

Explainability uses Grad-CAM:

`L^c = ReLU(sum_k alpha_k^c A^k)`

where `A^k` are target-layer feature maps and `alpha_k^c` are pooled gradients for class `c`.

## 4. Training Policy
- Optimizer: Adam
- Max epochs: up to 50
- Early stopping on validation loss
- Best-checkpoint restore

Representative hyperparameters:
- image size 384x384,
- batch size 16 (CNN/MobileNetV2), 12 (Hybrid),
- learning rate 1e-3 (CNN), 1e-4 (MobileNetV2/Hybrid),
- early-stopping patience 8.

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
- CNN: Accuracy 0.9878, Precision 0.9844, Recall 0.9882, F1 0.9861, 127.754 ms, 1.79 MB
- MobileNetV2: Accuracy 0.9976, Precision 0.9938, Recall 0.9965, F1 0.9950, 201.414 ms, 8.90 MB
- Hybrid: Accuracy 0.9948, Precision 0.9899, Recall 0.9939, F1 0.9917, 206.399 ms, 18.35 MB

Selection decision (v1 deployment): `MobileNetV2`.
Rationale: best Accuracy/F1 with better size and slightly better speed than Hybrid in the benchmarked setup.

## 6. Inference and Deployment Behavior
`POST /infer` pipeline:
1. save uploaded image,
2. validate leaf presence (HSV heuristic),
3. segment leaves with classical OpenCV processing,
4. classify each leaf independently,
5. suppress predictions below confidence threshold (default 0.70),
6. generate Grad-CAM overlays for accepted leaves,
7. return structured per-leaf response,
8. log timestamp, status, confidence, predicted classes, and latency.

Edge-case handling:
- no leaf detected -> `No leaf detected. Please retake the photo.`
- all low-confidence predictions -> retake guidance message.

## 7. Challenges and Mitigations
1. Domain shift (PlantVillage vs real-field images): mitigated with conservative confidence thresholding and explicit retake messaging.
2. Multi-leaf field photos: handled by segmentation + per-leaf inference.
3. Trust and interpretability: addressed with Grad-CAM heatmap overlays.
4. Operational traceability: addressed with analytics logging and summary endpoint.

## 8. Conclusion
The project demonstrates full computer vision integration from data preparation to deployment-oriented inference. It includes multi-model training, quantitative comparison, explainability, API deployment, and mobile demo workflow. The current system is suitable as an MVP foundation, with next steps focused on real-field data adaptation, calibration, and mobile optimization.
