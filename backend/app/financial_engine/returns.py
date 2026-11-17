"""Quantamental AI - Financial Engine Returns Module (Re-exporting from portfolio_engine.returns)."""

from app.portfolio_engine.returns import (
    DEFAULT_TRADING_DAYS_PER_YEAR,
    ReturnsResult,
    calculate_cagr,
    calculate_expected_returns,
    calculate_log_returns,
    calculate_simple_returns,
    compute_returns_summary,
)

__all__ = [
    "DEFAULT_TRADING_DAYS_PER_YEAR",
    "ReturnsResult",
    "calculate_cagr",
    "calculate_expected_returns",
    "calculate_log_returns",
    "calculate_simple_returns",
    "compute_returns_summary",
]
