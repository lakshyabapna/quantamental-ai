"""Quantamental AI - Monte Carlo Portfolio Simulator (compatibility re-export shim).

Re-exports Monte Carlo simulation symbols from app.portfolio_engine.monte_carlo
for callers using the financial_engine namespace.
"""

from app.portfolio_engine.monte_carlo import (
    DEFAULT_NUM_PORTFOLIOS,
    DEFAULT_RISK_FREE_RATE,
    MonteCarloSimulationResult,
    MonteCarloSimulator,
    MonteCarloValidationResult,
    SimulatedLandmarkPortfolio,
    simulate_monte_carlo_portfolios,
    validate_monte_carlo_against_frontier,
)

__all__ = [
    "DEFAULT_NUM_PORTFOLIOS",
    "DEFAULT_RISK_FREE_RATE",
    "MonteCarloSimulationResult",
    "MonteCarloSimulator",
    "MonteCarloValidationResult",
    "SimulatedLandmarkPortfolio",
    "simulate_monte_carlo_portfolios",
    "validate_monte_carlo_against_frontier",
]
