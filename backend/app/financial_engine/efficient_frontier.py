"""Quantamental AI - Efficient Frontier (compatibility re-export shim).

Re-exports from app.portfolio_engine.efficient_frontier for callers
using the financial_engine namespace.
"""

from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    FrontierPortfolio,
    compute_efficient_frontier,
)

__all__ = [
    "EfficientFrontierResult",
    "FrontierPortfolio",
    "compute_efficient_frontier",
]
