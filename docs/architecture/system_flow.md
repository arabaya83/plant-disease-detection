# System Flow Diagram

This flow documents the complete user-to-result inference path used in the deployed prototype.

```mermaid
flowchart TD
  U[Farmer on Mobile Browser] --> C1[Capture Photo or Upload from Gallery]
  C1 --> C2[Client Preview + Resize/Compression]
  C2 --> API1[POST /infer]

  API1 --> S1[Store Original Image]
  API1 --> V1[Leaf Validation Service]

  V1 -->|No leaf-like content| R1[Response: No leaf detected. Please retake the photo]
  V1 -->|Valid leaf-like content| SEG1[Classical Segmentation Service]

  SEG1 -->|0 segments| R1
  SEG1 -->|1 to 5 segments| INF1[Per-leaf Classification Service]

  INF1 --> TH1{Confidence >= 0.70?}
  TH1 -->|No| R2[Retake guidance if all leaves low-confidence]
  TH1 -->|Yes| CAM1[Generate Grad-CAM Overlay]

  CAM1 --> FMT1[Format per-leaf response payload]
  FMT1 --> LOG1[Log status, confidence, latency, predicted classes]
  FMT1 --> UI1[Render crop, disease, confidence, description, heatmap]

  API1 --> A1[GET /analytics/summary data source]
```

## Runtime Behavior Notes
- Max leaves processed per image: 5
- Low-confidence predictions are not force-assigned
- Mixed outcomes are supported (healthy and diseased leaves in one image)
- Response includes per-leaf explainability artifacts (Grad-CAM heatmaps)
