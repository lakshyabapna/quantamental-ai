"""Quantamental AI - Portfolio Engine.

Module A: Classical Financial Mathematics & Portfolio Optimization Core.
Implements deterministic, audited quantitative finance models:
- Returns calculation & expected returns (arithmetic annualization & geometric CAGR)
- Covariance estimation & shrinkage
- Positive-definite (PD) projection & Higham regularization
- Constrained SLSQP portfolio optimization & dual-solver validation
- Monte Carlo frontier simulation & downside risk analytics
"""

from app.portfolio_engine.covariance import calculate_sample_covariance
from app.portfolio_engine.returns import (
    ReturnsResult,
    calculate_cagr,
    calculate_expected_returns,
    calculate_log_returns,
    calculate_simple_returns,
    compute_returns_summary,
)

__all__ = [
    # Returns (Phase 2.1)
    "ReturnsResult",
    "calculate_cagr",
    "calculate_expected_returns",
    "calculate_log_returns",
    "calculate_simple_returns",
    "compute_returns_summary",
    # Covariance (Phase 2.2 Step 1)
    "calculate_sample_covariance",
]
