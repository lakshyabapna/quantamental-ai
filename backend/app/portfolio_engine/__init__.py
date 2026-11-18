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
from app.portfolio_engine.optimizer import (
    OptimizationResult,
    optimize_global_minimum_variance,
)
from app.portfolio_engine.regularization import regularize_covariance_higham
from app.portfolio_engine.returns import (
    ReturnsResult,
    calculate_cagr,
    calculate_expected_returns,
    calculate_log_returns,
    calculate_simple_returns,
    compute_returns_summary,
)
from app.portfolio_engine.shrinkage import (
    calculate_ledoit_wolf_covariance,
    fit_ledoit_wolf,
)
from app.portfolio_engine.tangency import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
    optimize_tangency_portfolio,
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
    # Shrinkage (Phase 2.2 Step 2)
    "calculate_ledoit_wolf_covariance",
    "fit_ledoit_wolf",
    # Regularization (Phase 2.2 Step 3)
    "regularize_covariance_higham",
    # Optimization (Phase 2.3 Step 1 & Step 2)
    "OptimizationResult",
    "optimize_global_minimum_variance",
    "TangencyOptimizationResult",
    "optimize_maximum_sharpe_ratio",
    "optimize_tangency_portfolio",
]
