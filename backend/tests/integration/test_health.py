"""Integration tests for application health and status endpoints."""

from datetime import datetime, timezone
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test root endpoint '/' returns operational metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Quantamental AI"
    assert "version" in data
    assert data["status"] == "operational"
    assert data["docs"] == "/docs"


def test_root_health_endpoint(client: TestClient):
    """Test '/health' endpoint returns status ok and service health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert data["services"]["api"] == "healthy"
    assert data["services"]["calendar"] == "healthy"

    # Verify timestamp is ISO parseable
    parsed_ts = datetime.fromisoformat(data["timestamp"])
    assert parsed_ts is not None


def test_api_v1_health_endpoint(client: TestClient):
    """Test '/api/v1/health' endpoint returns consistent health response."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["services"]["api"] == "healthy"
    assert data["services"]["calendar"] == "healthy"


def test_custom_exception_handler(app):
    """Test global QuantamentalException handler formats JSON with error_code."""
    from fastapi import APIRouter
    from app.core.exceptions import CalendarDateRangeError

    # Dynamically mount a test route that triggers a custom exception
    test_router = APIRouter()

    @test_router.get("/test-error-trigger")
    def trigger_error():
        raise CalendarDateRangeError(
            message="Test range error",
            details={"param": "start_date"},
        )

    app.include_router(test_router)

    with TestClient(app) as test_client:
        response = test_client.get("/test-error-trigger")
        assert response.status_code == 400
        data = response.json()
        assert data["error_code"] == "CALENDAR_DATE_RANGE_ERROR"
        assert data["message"] == "Test range error"
        assert data["details"]["param"] == "start_date"
