"""Quantamental AI - Maximum Sharpe Ratio (Tangency Portfolio) Optimization.

Implements Phase 2.3, Step 2: Deterministic Maximum Sharpe Ratio / Tangency
portfolio optimization using Sequential Least Squares Programming (SLSQP).

Mathematical Reference (FINANCIAL_METHODOLOGY.md §5.3 & ARCHITECTURE.md §4.2):
    Objective Function:
        max_w  SR(w) = (w.T @ mu - r_f) / sqrt(w.T @ Sigma_ann @ w)
    Equivalent SLSQP Minimization Objective:
        min_w  f(w) = - (w.T @ mu - r_f) / sigma(w)
        where e(w) = w.T @ mu - r_f and sigma(w) = sqrt(w.T @ Sigma_ann @ w)
    Subject to:
        sum_{i=1}^N w_i = 1.0  (Fully invested budget constraint)
        0.0 <= w_i <= 1.0      (Long-only constraint, no short selling / leverage)
    Analytical Gradient:
        nabla_w f(w) = - mu / sigma(w) + (e(w) * Sigma_ann @ w) / (sigma(w)^3)

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §5.1, §5.3, §5.4, §5.5
    - ARCHITECTURE.md §4.2: Deterministic Financial Engine optimization pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize

from app.config import settings
from app.core.exceptions import (
    FinancialCalculationError,
    IllConditionedMatrixError,
    InsufficientDataError,
    NonPositiveDefiniteMatrixError,
    OptimizationException,
)
from app.portfolio_engine.optimizer import (
    ALLOWED_FALLBACK_POLICIES,
    DEFAULT_INPUT_SYMMETRY_TOLERANCE,
    DEFAULT_MAX_CONDITION_NUMBER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OPTIMIZER_FALLBACK,
    DEFAULT_OPTIMIZER_TOLERANCE,
    _contains_boolean,
    _validate_initial_weights,
    _validate_input_covariance,
    _validate_optimization_parameters,
)

logger = logging.getLogger("quantamental.tangency")

# Numerical boundaries for constraint satisfaction checks
_WEIGHT_SUM_TOLERANCE: float = 1e-5
_WEIGHT_BOUND_TOLERANCE: float = 1e-7


@dataclass(frozen=True)
class TangencyOptimizationResult:
    """Immutable container for Maximum Sharpe Ratio (Tangency) portfolio optimization results.

    Attributes:
        weights: Optimal asset allocation weights (pd.Series indexed by asset labels).
        expected_return: Annualized expected portfolio return (w.T @ mu).
        variance: Annualized portfolio variance (w.T @ Sigma @ w).
        volatility: Annualized portfolio volatility (sqrt(variance)).
        risk_free_rate: Annualized benchmark risk-free rate used for optimization.
        sharpe_ratio: Annualized portfolio Sharpe ratio ((expected_return - risk_free_rate) / volatility).
        success: Whether the numerical solver converged successfully.
        status: Numerical solver exit status code.
        message: Solver termination message or description.
        iterations: Number of iterations executed by the solver.
        metadata: Dictionary containing diagnostic details (condition number, eigenvalues, etc.).
        fallback_executed: Whether a fallback policy was executed instead of solver output.
    """

    weights: pd.Series
    expected_return: float
    variance: float
    volatility: float
    risk_free_rate: float
    sharpe_ratio: float
    success: bool
    status: int
    message: str
    iterations: int
    metadata: Dict[str, Any]
    fallback_executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary representation."""
        return {
            "weights": self.weights.to_dict(),
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "risk_free_rate": self.risk_free_rate,
            "sharpe_ratio": self.sharpe_ratio,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "metadata": self.metadata,
            "fallback_executed": self.fallback_executed,
        }


def _validate_expected_returns(
    expected_returns: object,
    asset_labels: pd.Index,
) -> np.ndarray:
    """Validate expected returns Series structure, dtypes, and label alignment."""
    if expected_returns is None:
        raise FinancialCalculationError(
            "expected_returns must be a pandas Series, got None.",
            details={"type": "NoneType"},
        )
    if not isinstance(expected_returns, pd.Series):
        raise FinancialCalculationError(
            f"expected_returns must be a pandas Series, got {type(expected_returns).__name__}.",
            details={"type": type(expected_returns).__name__},
        )
    if expected_returns.empty:
        raise InsufficientDataError(
            "expected_returns Series is empty; at least 2 assets are required.",
            details={"len": 0},
        )
    if len(expected_returns) < 2:
        raise InsufficientDataError(
            f"expected_returns requires at least 2 assets; got {len(expected_returns)}.",
            details={"len": len(expected_returns)},
        )
    if expected_returns.index.duplicated().any():
        dupes = expected_returns.index[expected_returns.index.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "expected_returns Series contains duplicate asset labels.",
            details={"duplicate_labels": dupes},
        )
    if set(expected_returns.index) != set(asset_labels):
        raise FinancialCalculationError(
            "expected_returns asset labels do not match covariance asset labels.",
            details={
                "expected_returns_labels": list(expected_returns.index),
                "covariance_labels": list(asset_labels),
            },
        )

    # Reindex by covariance column order to ensure exact label and index alignment
    mu_series = expected_returns.reindex(asset_labels)

    if pd.api.types.is_bool_dtype(mu_series.dtype) or _contains_boolean(mu_series):
        raise FinancialCalculationError(
            "expected_returns contains boolean values; must be numeric.",
            details={"dtype": str(mu_series.dtype)},
        )
    if not pd.api.types.is_numeric_dtype(mu_series):
        raise FinancialCalculationError(
            "expected_returns contains non-numeric values.",
            details={"dtype": str(mu_series.dtype)},
        )
    if pd.api.types.is_complex_dtype(mu_series):
        raise FinancialCalculationError(
            "expected_returns contains complex values.",
            details={"dtype": str(mu_series.dtype)},
        )
    if mu_series.isnull().values.any():
        nan_assets = [str(col) for col, is_nan in mu_series.isnull().items() if is_nan]
        raise FinancialCalculationError(
            "expected_returns contains NaN or missing values.",
            details={"nan_assets": nan_assets},
        )

    try:
        mu_f64: np.ndarray = mu_series.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"Could not convert expected_returns to float64 array: {exc}.",
            details={"error": str(exc)},
        ) from exc

    if not np.all(np.isfinite(mu_f64)):
        raise FinancialCalculationError(
            "expected_returns contains infinite values.",
            details={"values": [None if not np.isfinite(v) else float(v) for v in mu_f64]},
        )

    return mu_f64


def _validate_risk_free_rate(risk_free_rate: object) -> float:
    """Validate that risk_free_rate is a finite real numeric value, strictly rejecting booleans and complex types."""
    if isinstance(risk_free_rate, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a finite numeric float, got bool ({risk_free_rate!r}).",
            details={"type": "bool", "value": risk_free_rate},
        )
    if isinstance(risk_free_rate, (complex, np.complexfloating)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a real numeric value, got complex ({risk_free_rate!r}).",
            details={"type": "complex", "value": str(risk_free_rate)},
        )
    if not isinstance(risk_free_rate, (int, float, np.integer, np.floating)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a numeric float or int, got {type(risk_free_rate).__name__}.",
            details={"type": type(risk_free_rate).__name__},
        )
    rf_float = float(risk_free_rate)
    if not np.isfinite(rf_float):
        raise FinancialCalculationError(
            f"risk_free_rate must be finite, got {risk_free_rate!r}.",
            details={"value": risk_free_rate},
        )
    return rf_float


def _negative_sharpe_objective(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> float:
    """Compute negative Sharpe ratio objective for SLSQP minimization.

    f(w) = - (w.T @ mu - rf) / sqrt(w.T @ cov @ w)
    """
    var = float(w @ cov @ w)
    if var <= 0.0 or not np.isfinite(var):
        raise FinancialCalculationError(
            f"Non-positive or non-finite portfolio variance encountered during optimization: {var:.2e}.",
            details={"variance": var},
        )
    vol = np.sqrt(var)
    excess_ret = float(w @ mu) - rf
    return -float(excess_ret / vol)


def _negative_sharpe_gradient(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> np.ndarray:
    """Compute analytical gradient of negative Sharpe ratio objective.

    nabla f(w) = - mu / sigma(w) + (e(w) * cov @ w) / (sigma(w)^3)
    where e(w) = w.T @ mu - rf and sigma(w) = sqrt(w.T @ cov @ w).
    """
    var = float(w @ cov @ w)
    if var <= 0.0 or not np.isfinite(var):
        raise FinancialCalculationError(
            f"Non-positive or non-finite portfolio variance encountered in gradient: {var:.2e}.",
            details={"variance": var},
        )
    vol = np.sqrt(var)
    vol_cubed = vol * var
    excess_ret = float(w @ mu) - rf
    cov_w = cov @ w
    return (-mu / vol) + ((excess_ret * cov_w) / vol_cubed)


def optimize_maximum_sharpe_ratio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = settings.DEFAULT_RISK_FREE_RATE,
    initial_weights: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    *,
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
    input_symmetry_tolerance: float = DEFAULT_INPUT_SYMMETRY_TOLERANCE,
    tolerance: float = DEFAULT_OPTIMIZER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    fallback: str = DEFAULT_OPTIMIZER_FALLBACK,
) -> TangencyOptimizationResult:
    """Compute the long-only Maximum Sharpe Ratio (Tangency) portfolio.

    Solves the non-linear optimization problem:
        min_w  - (w.T @ mu - r_f) / sqrt(w.T @ Sigma @ w)
    Subject to:
        sum_i w_i = 1.0   (Fully invested budget constraint)
        0.0 <= w_i <= 1.0 (Long-only non-negativity constraint)

    Supplies the analytical gradient:
        nabla_w f(w) = - mu / sigma(w) + ((w.T @ mu - r_f) * Sigma @ w) / (sigma(w)^3)

    Parameters
    ----------
    expected_returns:
        Labelled pandas Series of annualized arithmetic expected returns.
        Asset labels must match covariance row and column labels.
    covariance:
        Labelled square DataFrame representing the annualized covariance matrix.
        Row and column asset labels must match identically in order.
        Must be symmetric, finite, positive diagonal, and strictly positive definite.
    risk_free_rate:
        Annualized benchmark risk-free rate (e.g. 0.065 for 6.5%).
        Defaults to settings.DEFAULT_RISK_FREE_RATE.
    initial_weights:
        Optional initial guess vector w0. Can be a pd.Series, np.ndarray, or sequence.
        If None, defaults to the equal-weight vector (1/N, ..., 1/N).
    max_condition_number:
        Maximum allowable condition number (lambda_max / lambda_min).
        Defaults to settings.MAX_CONDITION_NUMBER (1e6).
    input_symmetry_tolerance:
        Maximum allowable asymmetry |Sigma - Sigma.T|_max before rejecting input.
        Defaults to 1e-8.
    tolerance:
        SLSQP termination tolerance (ftol). Defaults to 1e-9.
    max_iterations:
        Maximum number of SLSQP iterations. Defaults to 1000.
    fallback:
        Optimizer fallback policy on solver non-convergence:
        - "RAISE_EXCEPTION" (approved default): raises OptimizationException with diagnostics.
        - "EQUAL_WEIGHT": allocates w_i = 1/N, setting fallback_executed=True.

    Returns
    -------
    TangencyOptimizationResult
        Container with optimal weights pd.Series, annualized expected return,
        annualized variance, volatility, risk-free rate, Sharpe ratio, solver status,
        iterations, and diagnostic metadata.

    Raises
    ------
    FinancialCalculationError:
        If expected_returns or covariance is invalid, contains non-numeric/NaN/Inf/complex entries,
        is materially asymmetric, or initial_weights violate constraints.
    InsufficientDataError:
        If inputs are empty or have fewer than 2 assets.
    NonPositiveDefiniteMatrixError:
        If covariance is not strictly positive definite (min_eigenvalue <= 0).
    IllConditionedMatrixError:
        If covariance condition number exceeds max_condition_number.
    OptimizationException:
        If SLSQP optimization fails and fallback policy is "RAISE_EXCEPTION".
    """
    # ------------------------------------------------------------------
    # Step 0: Validate configuration parameters
    # ------------------------------------------------------------------
    _validate_optimization_parameters(
        max_condition_number=max_condition_number,
        input_symmetry_tolerance=input_symmetry_tolerance,
        tolerance=tolerance,
        max_iterations=max_iterations,
        fallback=fallback,
    )

    # ------------------------------------------------------------------
    # Step 1: Validate input covariance matrix
    # ------------------------------------------------------------------
    cov_f64, asset_labels, min_eig, max_eig, condition_number = _validate_input_covariance(
        covariance=covariance,
        input_symmetry_tolerance=input_symmetry_tolerance,
        max_condition_number=max_condition_number,
    )
    n_assets = len(asset_labels)

    # ------------------------------------------------------------------
    # Step 2: Validate expected returns & risk-free rate
    # ------------------------------------------------------------------
    mu_f64 = _validate_expected_returns(
        expected_returns=expected_returns,
        asset_labels=asset_labels,
    )
    rf_float = _validate_risk_free_rate(risk_free_rate)

    # ------------------------------------------------------------------
    # Step 3: Validate initial weights vector
    # ------------------------------------------------------------------
    w0 = _validate_initial_weights(
        initial_weights=initial_weights,
        asset_labels=asset_labels,
    )

    # ------------------------------------------------------------------
    # Step 4: Setup objective function and analytical gradient
    # ------------------------------------------------------------------
    objective = functools.partial(
        _negative_sharpe_objective,
        mu=mu_f64,
        cov=cov_f64,
        rf=rf_float,
    )
    gradient = functools.partial(
        _negative_sharpe_gradient,
        mu=mu_f64,
        cov=cov_f64,
        rf=rf_float,
    )

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w),
        }
    ]

    # ------------------------------------------------------------------
    # Step 5: Execute SLSQP optimization
    # ------------------------------------------------------------------
    logger.debug(
        "Executing Tangency SLSQP optimization: n_assets=%d, cond_num=%.2e, ftol=%.2e, maxiter=%d",
        n_assets,
        condition_number,
        tolerance,
        max_iterations,
    )

    res = scipy.optimize.minimize(
        fun=objective,
        x0=w0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": max_iterations},
    )

    base_metadata: Dict[str, Any] = {
        "condition_number": condition_number,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "n_assets": n_assets,
        "method": "SLSQP",
        "fallback_policy": fallback.upper(),
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "input_symmetry_tolerance": input_symmetry_tolerance,
    }

    # ------------------------------------------------------------------
    # Step 6: Post-optimization evaluation & fallback handling
    # ------------------------------------------------------------------
    is_valid_solution = False
    if res.success:
        w_opt = res.x
        if (
            np.all(np.isfinite(w_opt))
            and np.all(w_opt >= -_WEIGHT_BOUND_TOLERANCE)
            and np.all(w_opt <= 1.0 + _WEIGHT_BOUND_TOLERANCE)
            and abs(float(np.sum(w_opt)) - 1.0) <= _WEIGHT_SUM_TOLERANCE
        ):
            is_valid_solution = True

    if not is_valid_solution:
        failure_msg = res.message if not res.success else "Solver converged but violated portfolio constraints"
        logger.warning(
            "SLSQP Tangency optimization failure (status=%d, msg=%s). Fallback policy: %s",
            res.status,
            failure_msg,
            fallback.upper(),
        )

        if fallback.upper() == "RAISE_EXCEPTION":
            raise OptimizationException(
                f"SLSQP Tangency optimization failed: {failure_msg}.",
                details={
                    "status": int(res.status),
                    "message": str(failure_msg),
                    "iterations": int(getattr(res, "nit", 0)),
                    "condition_number": condition_number,
                    "min_eigenvalue": min_eig,
                    "max_eigenvalue": max_eig,
                },
            )
        elif fallback.upper() == "EQUAL_WEIGHT":
            w_eq = np.ones(n_assets, dtype=np.float64) / float(n_assets)
            exp_ret_eq = float(w_eq @ mu_f64)
            var_eq = float(w_eq @ cov_f64 @ w_eq)
            vol_eq = float(np.sqrt(max(0.0, var_eq)))
            sharpe_eq = float((exp_ret_eq - rf_float) / vol_eq) if vol_eq > 0.0 else 0.0
            weights_series = pd.Series(w_eq, index=asset_labels, name="weight")

            return TangencyOptimizationResult(
                weights=weights_series,
                expected_return=exp_ret_eq,
                variance=var_eq,
                volatility=vol_eq,
                risk_free_rate=rf_float,
                sharpe_ratio=sharpe_eq,
                success=False,
                status=int(res.status),
                message=f"Fallback to EQUAL_WEIGHT executed due to solver failure: {failure_msg}",
                iterations=int(getattr(res, "nit", 0)),
                metadata=base_metadata,
                fallback_executed=True,
            )

    # ------------------------------------------------------------------
    # Step 7: Numerical hygiene and final result construction
    # ------------------------------------------------------------------
    w_clean = np.clip(res.x, 0.0, 1.0)
    sum_w = float(np.sum(w_clean))
    if sum_w <= 0.0:
        w_clean = np.ones(n_assets, dtype=np.float64) / float(n_assets)
    else:
        w_clean = w_clean / sum_w

    exp_ret = float(w_clean @ mu_f64)
    variance = float(w_clean @ cov_f64 @ w_clean)
    volatility = float(np.sqrt(max(0.0, variance)))

    if volatility <= 0.0 or not np.isfinite(volatility):
        raise FinancialCalculationError(
            f"Optimized portfolio volatility is zero or non-finite: {volatility:.2e}.",
            details={"volatility": volatility, "variance": variance},
        )

    sharpe_ratio = float((exp_ret - rf_float) / volatility)
    weights_series = pd.Series(w_clean, index=asset_labels, name="weight")

    logger.info(
        "Tangency optimization converged successfully: iters=%d, Sharpe=%.4f, ret=%.4f, vol=%.4f",
        getattr(res, "nit", 0),
        sharpe_ratio,
        exp_ret,
        volatility,
    )

    return TangencyOptimizationResult(
        weights=weights_series,
        expected_return=exp_ret,
        variance=variance,
        volatility=volatility,
        risk_free_rate=rf_float,
        sharpe_ratio=sharpe_ratio,
        success=True,
        status=int(res.status),
        message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        metadata=base_metadata,
        fallback_executed=False,
    )


# Alias for explicit semantic naming
optimize_tangency_portfolio = optimize_maximum_sharpe_ratio
