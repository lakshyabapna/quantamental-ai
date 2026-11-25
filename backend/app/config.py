"""Quantamental AI - Application Configuration.

Centralized configuration using Pydantic Settings.
Environment variables override default configuration values.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core Application Settings
    PROJECT_NAME: str = "Quantamental AI"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"  # development, staging, production, test
    DEBUG: bool = False

    # Security & CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    # Data Engine & Storage Settings
    PARQUET_STORAGE_DIR: str = "data/parquet"
    CACHE_DIR: str = "data/cache"
    MODEL_ARTIFACT_DIR: str = "data/models"
    DEFAULT_ROLLING_WINDOW_YEARS: int = 5

    # Quantitative Financial Defaults
    DEFAULT_RISK_FREE_RATE: float = 0.065  # 6.5% annual — Indian 91-day GoI T-Bill benchmark rate (illustrative default; recalibrate before use)
    TRADING_DAYS_PER_YEAR: int = 252  # Standard NSE annualization factor
    DEFAULT_OPTIMIZER_FALLBACK: str = "RAISE_EXCEPTION"  # Approved alternatives: RAISE_EXCEPTION | EQUAL_WEIGHT

    # Numerical Calibration & Tolerances
    MIN_EIGENVALUE_FLOOR: float = 1e-7  # Eigenvalue floor for PD regularization
    MAX_CONDITION_NUMBER: float = 1e6  # Maximum allowable condition number for covariance matrices

    # Logging
    LOG_LEVEL: str = "INFO"


# Singleton instance
settings = Settings()
