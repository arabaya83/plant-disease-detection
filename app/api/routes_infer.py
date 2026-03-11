"""Frontend page and inference endpoint for the deployed diagnosis pipeline.

This module owns the main user-facing request flow. The route handlers receive
uploads from the browser UI, invoke the shared service layer in the correct
order, and shape the final response contract returned to the frontend.
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
    """Attach startup-created services to the router.

    Args:
        storage: Service responsible for saving uploads and derived images.
        validator: Service that rejects clearly invalid non-leaf inputs.
        segmenter: Service that extracts candidate leaf crops.
        inference: Service that runs the trained classifier.
        analytics: Service that logs request outcomes.
        settings: Application settings object shared across routes.

    Side Effects:
        Stores the shared services on the router object and creates the
        Grad-CAM wrapper for the active model.
    """
    router.storage = storage
    router.validator = validator
    router.segmenter = segmenter
    router.inference = inference
    router.analytics = analytics
    router.settings = settings
    router.gradcam = GradCAMService(inference.model, inference.get_gradcam_target_layer())


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    """Serve the mobile-friendly browser UI.

    Args:
        request: FastAPI request object required by Jinja templates.

    Returns:
        Rendered HTML page for camera capture and file upload.
    """
    return templates.TemplateResponse("index.html", {"request": request})


def _build_invalid_response(image_id: str, latency_ms: float) -> InferResponse:
    """Create the standard response used for invalid or non-leaf images."""
    return InferResponse(
        status="invalid",
        message="No leaf detected. Please retake the photo.",
        image_id=image_id,
        latency_ms=latency_ms,
        total_leaves_detected=0,
        results=[],
    )


def _log_inference_event(response: InferResponse) -> None:
    """Write a normalized analytics record for an inference response.

    Args:
        response: API response object produced by the inference pipeline.

    Side Effects:
        Appends a record to the analytics JSONL log.
    """
    router.analytics.log_event(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": response.image_id,
            "status": response.status,
            "message": response.message,
            "results": [result.model_dump() for result in response.results],
            "latency_ms": response.latency_ms,
        }
    )


@router.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    """Run the full image-to-diagnosis pipeline on one upload.

    The flow is intentionally linear so maintainers can trace behavior:
    save upload -> validate leaf content -> segment leaves -> classify each
    segment -> apply confidence filter -> generate Grad-CAM -> log analytics.

    Args:
        file: Uploaded image file from the browser client.

    Returns:
        Structured inference response containing overall status, latency, and
        zero or more accepted leaf-level predictions.

    Raises:
        HTTPException: If the uploaded file cannot be decoded as an image.
    """
    start = time.perf_counter()
    image_id, image_path = await router.storage.save_upload(file)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    validation = router.validator.validate(image_bgr)
    if not validation.is_valid:
        latency_ms = (time.perf_counter() - start) * 1000
        response = _build_invalid_response(image_id=image_id, latency_ms=latency_ms)
        _log_inference_event(response)
        return response

    segments = router.segmenter.segment_leaves(image_bgr)
    if not segments:
        latency_ms = (time.perf_counter() - start) * 1000
        # Segmentation can still fail after validation if the green regions are
        # too small or too noisy to survive contour filtering.
        response = _build_invalid_response(image_id=image_id, latency_ms=latency_ms)
        _log_inference_event(response)
        return response

    leaf_results = []
    accepted_count = 0
    for segment in segments:
        prediction, input_tensor = router.inference.predict(segment.crop_bgr)

        # Low-confidence predictions are suppressed rather than force-labeled so
        # the UI can ask for a retake instead of presenting misleading results.
        if prediction.confidence < router.settings.confidence_threshold:
            continue

        overlay = router.gradcam.generate_overlay(
            input_tensor,
            segment.crop_bgr,
            class_idx=prediction.class_index,
        )
        out_path = router.storage.output_path(image_id=image_id, leaf_idx=segment.leaf_id)
        router.gradcam.save_overlay(overlay, out_path)

        leaf_results.append(
            LeafResult(
                leaf_id=segment.leaf_id,
                crop_name=prediction.crop_name,
                disease_name=prediction.disease_name,
                confidence=round(prediction.confidence, 4),
                healthy_or_diseased=prediction.healthy_or_diseased,
                short_description=prediction.short_description,
                heatmap_path=f"/{out_path.as_posix()}",
            )
        )
        accepted_count += 1

    latency_ms = (time.perf_counter() - start) * 1000

    if accepted_count == 0:
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

    _log_inference_event(response)
    logger.info("inference image_id=%s status=%s leaves=%d latency_ms=%.2f", image_id, response.status, len(segments), latency_ms)

    return response
