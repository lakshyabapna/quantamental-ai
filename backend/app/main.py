"""Quantamental AI - FastAPI Application Entry Point.

Configures application lifecycle, middleware, exception handling, and routing.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.v1.router import api_v1_router
from app.config import settings
from app.core.exceptions import QuantamentalException
from app.core.logger import get_logger, setup_logging
from app.schemas.common import HealthResponse

logger = get_logger("quantamental.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    setup_logging(settings.LOG_LEVEL)
    logger.info(
        "Starting %s v%s in %s mode",
        settings.PROJECT_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Quantitative finance research and decision-support platform "
            "for Indian NSE equities."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(QuantamentalException)
    async def quantamental_exception_handler(
        request: Request, exc: QuantamentalException
    ) -> JSONResponse:
        logger.warning(
            "QuantamentalException handled: [%s] %s (Path: %s)",
            exc.error_code,
            exc.message,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled internal exception on %s: %s",
            request.url.path,
            str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected internal server error occurred.",
                "details": {"type": type(exc).__name__} if settings.DEBUG else {},
            },
        )

    # Root informational endpoint
    @app.get(
        "/",
        summary="Platform Root",
        description="Returns basic platform status and metadata.",
    )
    async def root() -> Dict[str, str]:
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "operational",
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
        }

    # Root-level health endpoint (conforms to standard observability checks)
    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health",
        description="Observability health endpoint returning application and service status.",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            services={
                "api": "healthy",
                "calendar": "healthy",
            },
        )

    # Mount API v1 router
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    return app


# Application instance for ASGI servers (e.g. uvicorn app.main:app)
app = create_app()
