"""Health check endpoints."""

from fastapi import APIRouter
from app.config import settings
from app.schemas.common import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the health status, version, and running environment of the application.",
)
async def get_health() -> HealthResponse:
    """Return application health status."""
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        services={
            "api": "healthy",
            "calendar": "healthy",
        },
    )
