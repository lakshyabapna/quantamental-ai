"""Quantamental AI - Financial Engine Covariance Re-export.

Re-exports sample covariance estimation from portfolio_engine for
consumers of the financial_engine public namespace.
"""

from app.portfolio_engine.covariance import calculate_sample_covariance

__all__ = ["calculate_sample_covariance"]
