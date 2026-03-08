"""Application entrypoint.

Initializes shared services (storage, CV pipeline, inference, analytics),
binds them to API routers, and exposes static frontend assets.
"""

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
