"""Quantamental AI - PyTest Configuration and Fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.data_engine.calendar import NSECalendar
from app.main import create_app


@pytest.fixture(scope="session")
def app():
    """Create a FastAPI application instance for the test session."""
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Provide a TestClient for testing HTTP endpoints."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def calendar():
    """Provide a fresh instance of NSECalendar for unit testing."""
    return NSECalendar(load_builtin=True)
