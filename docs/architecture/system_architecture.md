# Full System Architecture

This architecture summarizes the production-oriented prototype layers and interfaces.

```mermaid
flowchart LR
  subgraph ClientLayer[Client Layer]
    M1[Mobile Browser UI\nHTML/CSS/JS]
    M2[Camera Capture + Upload Fallback]
    M3[Image Preview + Resize/Compression]
  end

  subgraph APILayer[FastAPI Layer]
    A0[app/main.py]
    A1[GET /health]
    A2[POST /infer]
    A3[GET /analytics/summary]
  end

  subgraph CVMLLayer[CV / ML Services Layer]
    S1[Image Validation\nHSV Heuristic]
    S2[Leaf Segmentation\nOpenCV Threshold + Contours]
    S3[Inference Service\nCNN/MobileNetV2/Hybrid]
    S4[Grad-CAM Service]
    S5[Disease Info Mapping]
  end

  subgraph StorageLayer[Storage and Logging Layer]
    ST1[(Uploads Directory)]
    ST2[(Grad-CAM Outputs Directory)]
    ST3[(Analytics JSONL Log)]
    ST4[(Application Log)]
  end

  subgraph OfflineAssets[Training and Evaluation Assets]
    T1[Data Split + Weights Scripts]
    T2[Training Scripts]
    T3[Evaluation + Benchmark Scripts]
    T4[(Model Weights + Classes JSON)]
  end

  M1 --> M2 --> M3 --> A2
  A0 --> A1
  A0 --> A2
  A0 --> A3

  A2 --> S1 --> S2 --> S3 --> S4
  S3 --> S5

  A2 --> ST1
  S4 --> ST2
  A2 --> ST3
  A0 --> ST4

  T1 --> T2 --> T3 --> T4 --> S3
```

## Layer Responsibilities
- Client Layer: acquisition and lightweight preprocessing for bandwidth efficiency.
- FastAPI Layer: endpoint orchestration and response formatting.
- CV/ML Layer: validation, segmentation, classification, and explainability.
- Storage/Logging Layer: traceability of inputs, outputs, and runtime analytics.
- Offline Assets: reproducible training/evaluation pipeline and deployable model artifacts.
