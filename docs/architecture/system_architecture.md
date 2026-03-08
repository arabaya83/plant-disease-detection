# System Architecture

## Components
- Mobile Browser Frontend (camera capture, upload fallback, preview, compression)
- FastAPI Backend (validation, segmentation, inference, Grad-CAM, analytics)
- ML Layer (PyTorch models and training/evaluation scripts)
- Storage Layer (`app/static/uploads`, `app/static/outputs`, `logs/analytics.jsonl`)

## Architecture Diagram
```mermaid
flowchart LR
  A[Mobile Browser] -->|Capture/Upload| B[FastAPI /infer]
  B --> C[Image Validation]
  C --> D[Classical Segmentation]
  D --> E[Per-Leaf Classification]
  E --> F[Grad-CAM Overlay]
  F --> G[JSON Response]
  B --> H[(Uploads/Outputs Storage)]
  B --> I[(Analytics Log)]
  J[Training Pipeline] --> K[(Weights)]
  K --> E
```
