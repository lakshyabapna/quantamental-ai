"""Quantamental AI - PyPortfolioOpt Shadow Validator (compatibility re-export shim).

Re-exports shadow validator symbols from app.portfolio_engine.shadow_validator
for callers using the financial_engine namespace.
"""

from app.portfolio_engine.shadow_validator import (
    DEFAULT_VALIDATION_TOLERANCES,
    FrontierPointValidation,
    FrontierValidationResult,
    GMVValidationResult,
    PortfolioMetrics,
    PyPortfolioOptShadowValidator,
    ShadowValidator,
    TangencyValidationResult,
    ValidationDifferences,
    ValidationStatus,
    ValidationTolerances,
    validate_efficient_frontier,
    validate_gmv,
    validate_tangency,
)

__all__ = [
    "DEFAULT_VALIDATION_TOLERANCES",
    "FrontierPointValidation",
    "FrontierValidationResult",
    "GMVValidationResult",
    "PortfolioMetrics",
    "PyPortfolioOptShadowValidator",
    "ShadowValidator",
    "TangencyValidationResult",
    "ValidationDifferences",
    "ValidationStatus",
    "ValidationTolerances",
    "validate_efficient_frontier",
    "validate_gmv",
    "validate_tangency",
]
