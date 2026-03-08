"""Frontend and inference API routes.

This router coordinates the full prediction workflow:
upload -> validate -> segment -> classify -> Grad-CAM -> response + analytics.
"""

import logging
import time
from datetime import datetime, timezone

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.schemas.response_schemas import InferResponse, LeafResult
from app.services.gradcam import GradCAMService

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


def bind_services(storage, validator, segmenter, inference, analytics, settings) -> None:
    """Attach shared service instances to router state at application startup."""
    router.storage = storage
    router.validator = validator
    router.segmenter = segmenter
    router.inference = inference
    router.analytics = analytics
    router.settings = settings
    router.gradcam = GradCAMService(inference.model, inference.get_gradcam_target_layer())


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the mobile-friendly capture/upload frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    """Run full multi-leaf diagnosis pipeline on an uploaded image."""
    start = time.perf_counter()
    image_id, image_path = await router.storage.save_upload(file)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    validation = router.validator.validate(image_bgr)
    if not validation.is_valid:
        # Fail fast when the frame does not contain enough leaf-like content.
        latency_ms = (time.perf_counter() - start) * 1000
        response = InferResponse(
            status="invalid",
            message="No leaf detected. Please retake the photo.",
            image_id=image_id,
            latency_ms=latency_ms,
            total_leaves_detected=0,
            results=[],
        )
        router.analytics.log_event(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_id": image_id,
                "status": response.status,
                "message": response.message,
                "results": [],
                "latency_ms": latency_ms,
            }
        )
        return response

    segments = router.segmenter.segment_leaves(image_bgr)
    if not segments:
        # Validation passed but no robust contour survived segmentation filters.
        latency_ms = (time.perf_counter() - start) * 1000
        response = InferResponse(
            status="invalid",
            message="No leaf detected. Please retake the photo.",
            image_id=image_id,
            latency_ms=latency_ms,
            total_leaves_detected=0,
            results=[],
        )
        router.analytics.log_event(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_id": image_id,
                "status": response.status,
                "message": response.message,
                "results": [],
                "latency_ms": latency_ms,
            }
        )
        return response

    leaf_results = []
    accepted = 0
    for seg in segments:
        pred, input_tensor = router.inference.predict(seg.crop_bgr)

        if pred.confidence < router.settings.confidence_threshold:
            # Do not force low-confidence predictions.
            continue

        overlay = router.gradcam.generate_overlay(input_tensor, seg.crop_bgr, class_idx=pred.class_index)
        out_path = router.storage.output_path(image_id=image_id, leaf_idx=seg.leaf_id)
        router.gradcam.save_overlay(overlay, out_path)

        leaf_results.append(
            LeafResult(
                leaf_id=seg.leaf_id,
                crop_name=pred.crop_name,
                disease_name=pred.disease_name,
                confidence=round(pred.confidence, 4),
                healthy_or_diseased=pred.healthy_or_diseased,
                short_description=pred.short_description,
                heatmap_path=f"/{out_path.as_posix()}",
            )
        )
        accepted += 1

    latency_ms = (time.perf_counter() - start) * 1000

    if accepted == 0:
        response = InferResponse(
            status="low_confidence",
            message="Low confidence predictions. Please retake the photo.",
            image_id=image_id,
            latency_ms=latency_ms,
            total_leaves_detected=len(segments),
            results=[],
        )
    else:
        response = InferResponse(
            status="ok",
            message="Diagnosis completed.",
            image_id=image_id,
            latency_ms=latency_ms,
            total_leaves_detected=len(segments),
            results=leaf_results,
        )

    router.analytics.log_event(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": image_id,
            "status": response.status,
            "message": response.message,
            "results": [r.model_dump() for r in response.results],
            "latency_ms": latency_ms,
        }
    )
    logger.info("inference image_id=%s status=%s leaves=%d latency_ms=%.2f", image_id, response.status, len(segments), latency_ms)

    return response
