"""FastAPI application entrypoint for the deployed diagnosis service.

This module constructs the production-style runtime graph used by the browser
demo: configuration, logging, storage, validation, segmentation, model
inference, Grad-CAM generation, analytics logging, and route registration.
It is intentionally import-driven so ``uvicorn app.main:app`` can boot the
entire service without extra wiring code.
"""

import os

from huggingface_hub import hf_hub_download
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes_health import bind_analytics
from app.api.routes_health import router as health_router
from app.api.routes_infer import bind_services
from app.api.routes_infer import router as infer_router
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.services.analytics import AnalyticsService
from app.services.image_validation import LeafValidationService
from app.services.inference import InferenceService
from app.services.segmentation import SegmentationService
from app.services.storage import StorageService


DEFAULT_WEIGHTS_PATH = "ml/weights/mobilenet_best.pt"
HF_MODEL_REPO_ID = "rabaya/plant-disease-detection"
HF_MODEL_FILENAME = "mobilenet_best.pt"


def ensure_default_weights_present(weights_path: str = DEFAULT_WEIGHTS_PATH) -> None:
    """Download the default deployment checkpoint if it is not present locally.

    The repository intentionally does not store large binary checkpoints in git.
    This helper keeps the deployed app runnable by pulling the canonical
    MobileNetV2 weights from Hugging Face when needed.

    Args:
        weights_path: Local path where the default checkpoint is expected.

    Side Effects:
        Creates ``ml/weights`` if needed and may download model weights from
        the project's Hugging Face repository.
    """
    if os.path.exists(weights_path):
        return

    os.makedirs("ml/weights", exist_ok=True)
    hf_hub_download(
        repo_id=HF_MODEL_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model",
        local_dir="ml/weights",
    )


ensure_default_weights_present()
settings = get_settings()
setup_logging(settings.app_log)

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

storage = StorageService(settings.upload_dir, settings.output_dir)
validator = LeafValidationService()
segmenter = SegmentationService(max_leaves=settings.max_leaves)
inference = InferenceService(
    model_name=settings.model_name,
    weights_path=settings.model_weights_path,
    class_names_path=settings.class_names_path,
    image_size=settings.target_image_size,
    strict_loading=settings.strict_model_loading,
)
analytics = AnalyticsService(settings.analytics_log)

bind_services(storage, validator, segmenter, inference, analytics, settings)
bind_analytics(analytics)

app.include_router(health_router)
app.include_router(infer_router)
