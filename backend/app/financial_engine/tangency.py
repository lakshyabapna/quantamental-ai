"""Quantamental AI - Financial Engine Module A (Re-exporting from portfolio_engine.tangency)."""

from app.portfolio_engine.tangency import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
    optimize_tangency_portfolio,
)

__all__ = [
    "TangencyOptimizationResult",
    "optimize_maximum_sharpe_ratio",
    "optimize_tangency_portfolio",
]
