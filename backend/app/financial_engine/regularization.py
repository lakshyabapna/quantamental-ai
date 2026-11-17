"""Quantamental AI - Financial Engine Regularization Re-export.

Re-exports Higham positive-definite covariance regularization from
portfolio_engine for consumers of the financial_engine public namespace.
"""

from app.portfolio_engine.regularization import (
    DEFAULT_EIGENVALUE_FLOOR,
    DEFAULT_MAX_CONDITION_NUMBER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    DEFAULT_VARIANCE_PRESERVATION_TOL,
    regularize_covariance_higham,
)

__all__ = [
    "DEFAULT_EIGENVALUE_FLOOR",
    "DEFAULT_MAX_CONDITION_NUMBER",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "DEFAULT_VARIANCE_PRESERVATION_TOL",
    "regularize_covariance_higham",
]
