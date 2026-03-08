# Inference and Deployment

## Inference Pipeline
`POST /infer` performs:
1. image storage,
2. leaf validation,
3. classical segmentation,
4. per-leaf classification,
5. confidence filtering,
6. Grad-CAM generation,
7. structured response + analytics logging.

## API Endpoints
- `GET /health`
- `POST /infer`
- `GET /analytics/summary`

## Frontend
Mobile-friendly browser UI supports camera capture and upload fallback (`app/templates/index.html`, `app/static/js/app.js`).
