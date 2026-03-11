"""Liveness and analytics endpoints for the deployed API.

These routes intentionally stay simple: ``/health`` provides a lightweight
probe for deployment checks, while ``/analytics/summary`` exposes aggregate
statistics computed from the JSONL analytics log written during inference.
"""

from fastapi import APIRouter

from app.services.analytics import AnalyticsService

router = APIRouter()


def bind_analytics(service: AnalyticsService) -> None:
    """Attach the shared analytics service to the router.

    Args:
        service: Application-wide analytics service created at startup.

    Side Effects:
        Stores the service on the router object so route handlers can access it
        without re-instantiating dependencies.
    """
    router.analytics_service = service


@router.get("/health")
def health() -> dict:
    """Return a minimal liveness payload for uptime checks.

    Returns:
        A small dictionary confirming that the API process is running.
    """
    return {"status": "ok"}


@router.get("/analytics/summary")
def analytics_summary() -> dict:
    """Return aggregate inference usage metrics from the analytics log.

    Returns:
        A dictionary of summary statistics such as total diagnoses, average
        confidence, common diseases, and mean response time.
    """
    return router.analytics_service.summary()
