"""Quantamental AI - Financial Engine Module A (Re-exporting from portfolio_engine)."""

from app.portfolio_engine import (
    ReturnsResult,
    calculate_cagr,
    calculate_expected_returns,
    calculate_log_returns,
    calculate_simple_returns,
    calculate_sample_covariance,
    calculate_ledoit_wolf_covariance,
    fit_ledoit_wolf,
    regularize_covariance_higham,
    compute_returns_summary,
    OptimizationResult,
    optimize_global_minimum_variance,
)

__all__ = [
    "ReturnsResult",
    "calculate_cagr",
    "calculate_expected_returns",
    "calculate_log_returns",
    "calculate_simple_returns",
    "calculate_sample_covariance",
    "calculate_ledoit_wolf_covariance",
    "fit_ledoit_wolf",
    "regularize_covariance_higham",
    "compute_returns_summary",
    "OptimizationResult",
    "optimize_global_minimum_variance",
]
