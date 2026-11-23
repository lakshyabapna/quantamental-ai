"""Quantamental AI - Downside & Tail Risk Analytics (compatibility re-export shim).

Re-exports downside risk metrics symbols from app.portfolio_engine.risk_metrics
for callers using the financial_engine namespace.
"""

from app.portfolio_engine.risk_metrics import (
    ALLOWED_QUANTILE_METHODS,
    DEFAULT_CONFIDENCE_LEVELS,
    CornishFisherExpansionResult,
    RiskMetricsEngine,
    RiskMetricsResult,
    calculate_cornish_fisher_expansion,
    calculate_cornish_fisher_var,
    calculate_cvar,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_risk_metrics,
)

__all__ = [
    "ALLOWED_QUANTILE_METHODS",
    "DEFAULT_CONFIDENCE_LEVELS",
    "CornishFisherExpansionResult",
    "RiskMetricsEngine",
    "RiskMetricsResult",
    "calculate_cornish_fisher_expansion",
    "calculate_cornish_fisher_var",
    "calculate_cvar",
    "calculate_historical_var",
    "calculate_parametric_var",
    "calculate_risk_metrics",
]
