"""Health and analytics routes."""

from fastapi import APIRouter

from app.services.analytics import AnalyticsService

router = APIRouter()


def bind_analytics(service: AnalyticsService) -> None:
    """Attach analytics dependency to router state."""
    router.analytics_service = service


@router.get("/health")
def health() -> dict:
    """Liveness probe endpoint."""
    return {"status": "ok"}


@router.get("/analytics/summary")
def analytics_summary() -> dict:
    """Return aggregate diagnostic usage metrics."""
    return router.analytics_service.summary()
