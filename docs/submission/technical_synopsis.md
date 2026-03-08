# AI-Powered Plant Disease Detection for Small Farmers

## 1. Problem Statement and Objectives
Plant disease identification in smallholder farms is frequently delayed because access to agronomy experts is limited in-field. Late diagnosis can reduce yield and increase treatment cost. This project targets a practical web prototype that lets farmers capture leaf images using a mobile browser and obtain rapid disease assessment with explainability.

Objectives:
- Build an end-to-end computer vision system using PlantVillage as the training source.
- Support image upload and direct camera capture in a mobile-friendly interface.
- Handle multi-leaf images by classical segmentation and per-leaf classification.
- Provide explainable outputs with Grad-CAM overlays.
- Return robust user guidance when no leaf is found or predictions are low confidence.

## 2. Dataset and Preprocessing
Dataset: PlantVillage (folder-per-class format, 38 classes in this project setup).

Target label: `Crop + Disease` (e.g., `Tomato___Late_blight`, `Apple___healthy`). Healthy labels are displayed as `Healthy - No disease detected`.

Data splitting strategy:
- Train/Validation/Test = 80/10/10
- Stratified by class to preserve class proportions.

Preprocessing and augmentation:
- Resize to configurable input size (default 384x384)
- Normalization to ImageNet statistics
- Standard augmentations only: horizontal flip, rotation, scale/zoom, brightness jitter

Class imbalance handling:
- Class weights are computed from the training split and used in cross-entropy loss.

## 3. Mathematical and Modeling Foundations
The project compares three classifiers:
1. **Simple CNN baseline**: stacked conv-batchnorm-ReLU blocks with pooling and MLP head.
2. **MobileNetV2 baseline**: ImageNet-pretrained backbone, fine-tuned for PlantVillage classes.
3. **Hybrid model**: dual-branch architecture combining MobileNetV2 features with a custom residual branch, followed by feature concatenation and classification head.

Given input image `x`, each model outputs logits `z`. Predicted class probability is:

`p(y=k|x) = softmax(z)_k = exp(z_k) / sum_j exp(z_j)`

Training objective is weighted cross-entropy:

`L = - w_y log p(y|x)`

where `w_y` is the class weight for true class `y`.

Explainability uses Grad-CAM. For class `c`, Grad-CAM map is:

`L^c = ReLU(sum_k alpha_k^c A^k)`

where `A^k` are final convolution feature maps and `alpha_k^c` are gradients pooled over spatial dimensions.

## 4. Training Methodology and Hyperparameters
Training policy:
- Optimizer: Adam
- Max epochs: up to 50
- Early stopping on validation loss
- Best checkpoint restoration

Representative hyperparameters:
- Input size: 384x384
- Batch size: 16 (CNN/MobileNet), 12 (Hybrid)
- Learning rate: 1e-3 (CNN), 1e-4 (MobileNet/Hybrid)
- Patience: 8 epochs

Model artifacts saved under `ml/weights/`, including best checkpoints and training history JSON files.

## 5. Evaluation Metrics and Detailed Results
Metrics implemented:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix
- Inference speed (ms per image)
- Model size (MB)

Observed test/benchmark outcomes:
- CNN: Accuracy 0.9878, F1 0.9861, 127.754 ms, 1.79 MB
- MobileNetV2: Accuracy 0.9976, F1 0.9950, 201.414 ms, 8.90 MB
- Hybrid: Accuracy 0.9948, F1 0.9917, 206.399 ms, 18.35 MB

Model selection rationale:
- MobileNetV2 provided the best accuracy/F1 while remaining smaller and slightly faster than the hybrid model in benchmark conditions, so it was selected as deployment default for v1.

## 6. End-to-End Inference and Deployment Behavior
Pipeline behavior in FastAPI `/infer`:
1. Save uploaded image.
2. Validate leaf presence via HSV green-content heuristic.
3. Segment up to 5 leaf candidates using classical CV (thresholding + contour filtering).
4. Run per-leaf classification.
5. Apply confidence threshold (0.70 default): low-confidence leaves are not forced.
6. Generate Grad-CAM overlay for accepted predictions.
7. Return structured JSON response with per-leaf disease info.
8. Log analytics (status, confidence, latency, predicted diseases).

User-facing edge-case handling:
- No valid leaf: `No leaf detected. Please retake the photo.`
- All predictions below threshold: ask user to retake photo.

## 7. Baseline Comparison and Practical Trade-offs
Compared to the simple CNN baseline, transfer learning significantly improves robustness and macro-F1. The hybrid model increases complexity and model size but does not outperform MobileNetV2 in this setup. For field prototypes where latency and deployability matter, MobileNetV2 provides the best balance.

## 8. Challenges and Solutions
1. **Domain gap (PlantVillage vs field images)**
- Challenge: controlled backgrounds in PlantVillage may not match in-field environments.
- Solution: conservative confidence threshold, explicit retake messaging, Grad-CAM transparency.

2. **Multi-leaf photos in real usage**
- Challenge: single-image classifiers assume one subject.
- Solution: classical segmentation + per-leaf inference to support mixed outcomes.

3. **Interpretability requirements**
- Challenge: end users need trust signals.
- Solution: Grad-CAM heatmap overlays returned per leaf.

4. **Operational monitoring**
- Challenge: prototype quality must be measurable.
- Solution: analytics logs and summary endpoint for invalid rate, common diseases, confidence, and latency.

## 9. Conclusion
The project demonstrates an integrated computer vision solution covering data engineering, multi-model training, evaluation, explainability, backend inference, and a mobile-oriented frontend. The implementation is production-oriented in structure and suitable as an MVP foundation, with next steps focused on domain adaptation, stronger field-validation datasets, and packaging for mobile deployment.
