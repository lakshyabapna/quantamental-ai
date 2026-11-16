"""Quantamental AI - Common Pydantic Schemas.

Defines schemas for health checks, generic API responses, and standard error envelopes.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Overall service status")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Running environment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check",
    )
    services: Dict[str, str] = Field(
        default_factory=lambda: {
            "api": "healthy",
            "calendar": "healthy",
        },
        description="Status of individual sub-services",
    )


class APIResponse(BaseModel, Generic[DataT]):
    """Standard generic API response envelope."""

    success: bool = Field(default=True, description="Whether the operation succeeded")
    data: Optional[DataT] = Field(default=None, description="Payload data")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the response",
    )


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error_code: str = Field(..., description="Standardized machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured diagnostics or contextual details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the error occurrence",
    )
