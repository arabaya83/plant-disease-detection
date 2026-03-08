# API Usage

## Health
```bash
curl -X GET http://localhost:8000/health
```

## Inference
```bash
curl -X POST http://localhost:8000/infer \
  -F "file=@sample_leaf.jpg"
```

Response shape:
- `status`
- `message`
- `image_id`
- `latency_ms`
- `total_leaves_detected`
- `results[]` with per-leaf crop/disease/confidence/health/description/heatmap

## Analytics Summary
```bash
curl -X GET http://localhost:8000/analytics/summary
```
