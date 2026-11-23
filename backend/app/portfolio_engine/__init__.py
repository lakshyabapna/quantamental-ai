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
from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    FrontierPortfolio,
    compute_efficient_frontier,
)
from app.portfolio_engine.cal import (
    CALCoordinate,
    CALPoint,
    CALResult,
    CapitalAllocationLineResult,
    calculate_cal_return,
    calculate_cal_slope,
    generate_cal_coordinates,
    generate_capital_allocation_line,
)
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
    # Optimization (Phase 2.3 Step 1 & Step 2 & Step 3 & Step 4)
    "OptimizationResult",
    "optimize_global_minimum_variance",
    "TangencyOptimizationResult",
    "optimize_maximum_sharpe_ratio",
    "optimize_tangency_portfolio",
    "EfficientFrontierResult",
    "FrontierPortfolio",
    "compute_efficient_frontier",
    # Capital Allocation Line (Phase 2.3 Step 4)
    "CALCoordinate",
    "CALPoint",
    "CALResult",
    "CapitalAllocationLineResult",
    "calculate_cal_return",
    "calculate_cal_slope",
    "generate_cal_coordinates",
    "generate_capital_allocation_line",
    # Shadow Validation (Phase 2.4 Step 1)
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
    # Monte Carlo Simulation (Phase 2.5)
    "DEFAULT_NUM_PORTFOLIOS",
    "DEFAULT_RISK_FREE_RATE",
    "MonteCarloSimulationResult",
    "MonteCarloSimulator",
    "MonteCarloValidationResult",
    "SimulatedLandmarkPortfolio",
    "simulate_monte_carlo_portfolios",
    "validate_monte_carlo_against_frontier",
]
