"""Quantamental AI - Portfolio Optimization Engine.

Implements Phase 2.3, Step 1: Deterministic Global Minimum Variance (GMV)
portfolio optimization using Sequential Least Squares Programming (SLSQP).

Mathematical Reference (FINANCIAL_METHODOLOGY.md §5.2 & ARCHITECTURE.md §4.2):
    Objective Function:
        min_w  f(w) = 0.5 * w.T @ Sigma_ann @ w
    Subject to:
        sum_{i=1}^N w_i = 1.0  (Fully invested budget constraint)
        0.0 <= w_i <= 1.0      (Long-only constraint, no short selling / leverage)
    Analytical Gradient:
        nabla_w f(w) = Sigma_ann @ w

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §5.1, §5.2, §5.4, §5.5
    - ARCHITECTURE.md §4.2: Deterministic Financial Engine optimization pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
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

logger = logging.getLogger("quantamental.optimizer")

# ---------------------------------------------------------------------------
# Module Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_SYMMETRY_TOLERANCE: float = 1e-8
DEFAULT_OPTIMIZER_TOLERANCE: float = 1e-9
DEFAULT_MAX_ITERATIONS: int = 1000
DEFAULT_MAX_CONDITION_NUMBER: float = settings.MAX_CONDITION_NUMBER  # 1e6
DEFAULT_OPTIMIZER_FALLBACK: str = settings.DEFAULT_OPTIMIZER_FALLBACK  # RAISE_EXCEPTION

ALLOWED_FALLBACK_POLICIES = frozenset({"RAISE_EXCEPTION", "EQUAL_WEIGHT"})

# Numerical boundaries for constraint satisfaction checks
_WEIGHT_SUM_TOLERANCE: float = 1e-5
_WEIGHT_BOUND_TOLERANCE: float = 1e-7


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable container for portfolio optimization results.

    Attributes:
        weights: Optimal asset allocation weights (pd.Series indexed by asset labels).
        variance: Annualized portfolio variance (w.T @ Sigma @ w).
        volatility: Annualized portfolio volatility (sqrt(variance)).
        objective_value: Final objective function value from solver (0.5 * w.T @ Sigma @ w).
        success: Whether the numerical solver converged successfully.
        status: Numerical solver exit status code.
        message: Solver termination message or description.
        iterations: Number of iterations executed by the solver.
        metadata: Dictionary containing diagnostic details (condition number, eigenvalues, etc.).
        fallback_executed: Whether a fallback policy was executed instead of solver output.
    """

    weights: pd.Series
    variance: float
    volatility: float
    objective_value: float
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
            "variance": self.variance,
            "volatility": self.volatility,
            "objective_value": self.objective_value,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "metadata": self.metadata,
            "fallback_executed": self.fallback_executed,
        }


def _validate_optimization_parameters(
    max_condition_number: object,
    input_symmetry_tolerance: object,
    tolerance: object,
    max_iterations: object,
    fallback: object,
) -> None:
    """Validate numerical configuration arguments strictly."""
    if (
        isinstance(max_condition_number, bool)
        or not isinstance(max_condition_number, (int, float))
        or not np.isfinite(max_condition_number)
        or max_condition_number <= 0.0
    ):
        raise FinancialCalculationError(
            f"max_condition_number must be a finite positive float, got {max_condition_number!r}.",
            details={"max_condition_number": max_condition_number},
        )

    if (
        isinstance(input_symmetry_tolerance, bool)
        or not isinstance(input_symmetry_tolerance, (int, float))
        or not np.isfinite(input_symmetry_tolerance)
        or input_symmetry_tolerance <= 0.0
    ):
        raise FinancialCalculationError(
            f"input_symmetry_tolerance must be a finite positive float, got {input_symmetry_tolerance!r}.",
            details={"input_symmetry_tolerance": input_symmetry_tolerance},
        )

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise FinancialCalculationError(
            f"tolerance must be a finite positive float, got {tolerance!r}.",
            details={"tolerance": tolerance},
        )

    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise FinancialCalculationError(
            f"max_iterations must be a positive integer, got {max_iterations!r}.",
            details={"max_iterations": max_iterations},
        )

    if (
        not isinstance(fallback, str)
        or fallback.upper() not in ALLOWED_FALLBACK_POLICIES
    ):
        raise FinancialCalculationError(
            f"Unsupported fallback policy: {fallback!r}. "
            f"Must be one of {sorted(ALLOWED_FALLBACK_POLICIES)}.",
            details={"fallback": fallback, "allowed_policies": sorted(ALLOWED_FALLBACK_POLICIES)},
        )


def _validate_input_covariance(
    covariance: object,
    input_symmetry_tolerance: float,
    max_condition_number: float,
) -> Tuple[np.ndarray, pd.Index, float, float, float]:
    """Validate covariance DataFrame structure, symmetry, positive definiteness, and condition number."""
    if covariance is None:
        raise FinancialCalculationError(
            "covariance must be a pandas DataFrame, got None.",
            details={"type": "NoneType"},
        )
    if not isinstance(covariance, pd.DataFrame):
        raise FinancialCalculationError(
            f"covariance must be a pandas DataFrame, got {type(covariance).__name__}.",
            details={"type": type(covariance).__name__},
        )
    if covariance.empty:
        raise InsufficientDataError(
            "covariance DataFrame is empty; at least 2 assets are required.",
            details={"shape": list(covariance.shape)},
        )

    n_rows, n_cols = covariance.shape
    if n_rows != n_cols:
        raise FinancialCalculationError(
            f"covariance matrix must be square, got shape ({n_rows}, {n_cols}).",
            details={"shape": list(covariance.shape)},
        )
    if n_rows < 2:
        raise InsufficientDataError(
            f"covariance matrix requires at least 2 assets; got {n_rows}.",
            details={"n_assets": n_rows},
        )

    # Check for duplicate labels
    if covariance.columns.duplicated().any():
        dupes_col = covariance.columns[covariance.columns.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "covariance DataFrame contains duplicate column labels.",
            details={"duplicate_columns": dupes_col},
        )
    if covariance.index.duplicated().any():
        dupes_idx = covariance.index[covariance.index.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "covariance DataFrame contains duplicate index labels.",
            details={"duplicate_index": dupes_idx},
        )

    # Identical index and column labels and order
    if list(covariance.index) != list(covariance.columns):
        raise FinancialCalculationError(
            "covariance row and column labels must be identical and in the exact same order.",
            details={
                "index_labels": list(covariance.index),
                "column_labels": list(covariance.columns),
            },
        )

    # Dtype checks
    non_numeric_cols = [
        col
        for col in covariance.columns
        if not pd.api.types.is_numeric_dtype(covariance[col])
    ]
    if non_numeric_cols:
        raise FinancialCalculationError(
            "covariance DataFrame contains non-numeric columns.",
            details={"non_numeric_columns": non_numeric_cols},
        )

    complex_cols = [
        col
        for col in covariance.columns
        if pd.api.types.is_complex_dtype(covariance[col])
    ]
    if complex_cols:
        raise FinancialCalculationError(
            "covariance DataFrame contains complex-valued columns.",
            details={"complex_columns": complex_cols},
        )

    # Missing values check
    if covariance.isnull().values.any():
        nan_counts = covariance.isnull().sum()
        nan_assets = nan_counts[nan_counts > 0].to_dict()
        raise FinancialCalculationError(
            "covariance DataFrame contains NaN or missing values.",
            details={"nan_counts_by_asset": {str(k): int(v) for k, v in nan_assets.items()}},
        )

    # Convert to NumPy float64
    try:
        cov_f64: np.ndarray = covariance.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"Could not convert covariance DataFrame to float64 array: {exc}.",
            details={"error": str(exc)},
        ) from exc

    if not np.all(np.isfinite(cov_f64)):
        raise FinancialCalculationError(
            "covariance matrix contains infinite values.",
            details={"shape": list(cov_f64.shape)},
        )

    # Diagonal variance positivity check
    variances = np.diag(cov_f64)
    if not np.all(variances > 0.0):
        invalid_vars = {
            str(col): float(val)
            for col, val in zip(covariance.columns, variances)
            if val <= 0.0 or not np.isfinite(val)
        }
        raise FinancialCalculationError(
            "All diagonal variance elements must be finite and strictly positive (> 0).",
            details={"invalid_diagonal_variances": invalid_vars},
        )

    # Symmetry check
    asymmetry = float(np.max(np.abs(cov_f64 - cov_f64.T)))
    if asymmetry > input_symmetry_tolerance:
        raise FinancialCalculationError(
            f"covariance matrix is materially asymmetric "
            f"(max asymmetry={asymmetry:.2e} > tolerance={input_symmetry_tolerance:.2e}).",
            details={
                "max_asymmetry": asymmetry,
                "input_symmetry_tolerance": input_symmetry_tolerance,
            },
        )

    # Symmetrize slightly within tolerance to eliminate floating-point rounding noise
    cov_f64 = (cov_f64 + cov_f64.T) / 2.0

    # Spectral analysis for positive definiteness & conditioning
    eigenvalues = np.linalg.eigvalsh(cov_f64)
    min_eig = float(eigenvalues[0])
    max_eig = float(eigenvalues[-1])

    if min_eig <= 0.0:
        raise NonPositiveDefiniteMatrixError(
            f"Covariance matrix is not strictly positive definite "
            f"(min_eigenvalue={min_eig:.2e} <= 0). "
            "Apply Higham regularization (regularize_covariance_higham) before optimization.",
            details={
                "min_eigenvalue": min_eig,
                "max_eigenvalue": max_eig,
                "n_assets": n_rows,
            },
        )

    condition_number = max_eig / min_eig
    if condition_number >= max_condition_number:
        raise IllConditionedMatrixError(
            f"Covariance matrix condition number ({condition_number:.2e}) "
            f"exceeds maximum allowable threshold ({max_condition_number:.2e}). "
            "Matrix is ill-conditioned for numerical optimization.",
            details={
                "condition_number": condition_number,
                "max_condition_number": max_condition_number,
                "min_eigenvalue": min_eig,
                "max_eigenvalue": max_eig,
            },
        )

    return cov_f64, covariance.columns, min_eig, max_eig, condition_number


def _contains_boolean(weights: object) -> bool:
    """Check if weights container contains any boolean values."""
    if isinstance(weights, pd.Series):
        if pd.api.types.is_bool_dtype(weights.dtype):
            return True
        return any(isinstance(val, (bool, np.bool_)) for val in weights)
    if isinstance(weights, np.ndarray):
        if weights.dtype == bool or np.issubdtype(weights.dtype, np.bool_):
            return True
        if weights.dtype == object:
            return any(isinstance(val, (bool, np.bool_)) for val in weights.flat)
        return False
    if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
        return any(isinstance(val, (bool, np.bool_)) for val in weights)
    return False


def _validate_initial_weights(
    initial_weights: object,
    asset_labels: pd.Index,
) -> np.ndarray:
    """Validate and normalize user-provided initial weights vector."""
    n_assets = len(asset_labels)

    if initial_weights is None:
        return np.ones(n_assets, dtype=np.float64) / float(n_assets)

    if isinstance(initial_weights, (bool, np.bool_)):
        raise FinancialCalculationError(
            "initial_weights must be a pd.Series, np.ndarray, or sequence of floats, got bool.",
            details={"type": "bool"},
        )

    if _contains_boolean(initial_weights):
        raise FinancialCalculationError(
            "initial_weights contains boolean values; weights must be numeric (int or float).",
            details={"type": type(initial_weights).__name__},
        )

    if isinstance(initial_weights, pd.Series):
        if initial_weights.index.duplicated().any():
            dupes = initial_weights.index[initial_weights.index.duplicated(keep=False)].tolist()
            raise FinancialCalculationError(
                "initial_weights Series contains duplicate index labels.",
                details={"duplicate_labels": dupes},
            )
        if set(initial_weights.index) != set(asset_labels):
            raise FinancialCalculationError(
                "initial_weights index labels do not match covariance asset labels.",
                details={
                    "weights_labels": list(initial_weights.index),
                    "covariance_labels": list(asset_labels),
                },
            )
        # Reindex to ensure identical asset ordering
        initial_weights_series = initial_weights.reindex(asset_labels)
        if not pd.api.types.is_numeric_dtype(initial_weights_series):
            raise FinancialCalculationError(
                "initial_weights contains non-numeric values.",
                details={"dtype": str(initial_weights_series.dtype)},
            )
        if pd.api.types.is_complex_dtype(initial_weights_series):
            raise FinancialCalculationError(
                "initial_weights contains complex values.",
                details={"dtype": str(initial_weights_series.dtype)},
            )
        try:
            w0 = initial_weights_series.to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise FinancialCalculationError(
                f"Could not convert initial_weights to float64 array: {exc}.",
                details={"error": str(exc)},
            ) from exc
    elif isinstance(initial_weights, (np.ndarray, Sequence)):
        if isinstance(initial_weights, (str, bytes)):
            raise FinancialCalculationError(
                f"initial_weights must be a numeric sequence, got {type(initial_weights).__name__}.",
                details={"type": type(initial_weights).__name__},
            )
        if len(initial_weights) != n_assets:
            raise FinancialCalculationError(
                f"initial_weights length ({len(initial_weights)}) does not match asset count ({n_assets}).",
                details={"len_weights": len(initial_weights), "n_assets": n_assets},
            )
        try:
            w0 = np.asarray(initial_weights, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FinancialCalculationError(
                f"Could not convert initial_weights to float64 array: {exc}.",
                details={"error": str(exc)},
            ) from exc
    else:
        raise FinancialCalculationError(
            f"initial_weights must be a pd.Series, np.ndarray, or sequence of floats, got {type(initial_weights).__name__}.",
            details={"type": type(initial_weights).__name__},
        )

    if not np.all(np.isfinite(w0)):
        raise FinancialCalculationError(
            "initial_weights contains NaN or non-finite values.",
            details={"values": [None if not np.isfinite(v) else float(v) for v in w0]},
        )

    if np.any(w0 < -_WEIGHT_BOUND_TOLERANCE) or np.any(w0 > 1.0 + _WEIGHT_BOUND_TOLERANCE):
        invalid_elements = {
            str(asset): float(val)
            for asset, val in zip(asset_labels, w0)
            if val < -_WEIGHT_BOUND_TOLERANCE or val > 1.0 + _WEIGHT_BOUND_TOLERANCE
        }
        raise FinancialCalculationError(
            "initial_weights elements must lie within [0.0, 1.0] (long-only constraint).",
            details={"invalid_weights": invalid_elements},
        )

    weight_sum = float(np.sum(w0))
    if abs(weight_sum - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise FinancialCalculationError(
            f"initial_weights must sum to 1.0 within tolerance "
            f"(sum={weight_sum:.6f}, discrepancy={abs(weight_sum - 1.0):.2e} > {_WEIGHT_SUM_TOLERANCE:.2e}).",
            details={"sum": weight_sum, "tolerance": _WEIGHT_SUM_TOLERANCE},
        )

    # Normalize slightly within tolerance
    w0 = np.clip(w0, 0.0, 1.0)
    w0 = w0 / np.sum(w0)

    return w0


def optimize_global_minimum_variance(
    covariance: pd.DataFrame,
    initial_weights: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    *,
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
    input_symmetry_tolerance: float = DEFAULT_INPUT_SYMMETRY_TOLERANCE,
    tolerance: float = DEFAULT_OPTIMIZER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    fallback: str = DEFAULT_OPTIMIZER_FALLBACK,
) -> OptimizationResult:
    """Compute the long-only Global Minimum Variance (GMV) portfolio.

    Solves the quadratic programming problem:
        min_w  0.5 * w.T @ Sigma @ w
    Subject to:
        sum_i w_i = 1.0   (Fully invested budget constraint)
        0.0 <= w_i <= 1.0 (Long-only non-negativity constraint)

    Supplies the analytical gradient:
        nabla_w f(w) = Sigma @ w

    Parameters
    ----------
    covariance:
        Labelled square DataFrame representing the annualized covariance matrix.
        Row and column asset labels must match identically in order.
        Must be symmetric, finite, positive diagonal, and strictly positive definite.
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
    OptimizationResult
        Container with optimal weights pd.Series, annualized variance, volatility,
        objective value, solver status, iterations, and diagnostic metadata.

    Raises
    ------
    FinancialCalculationError:
        If covariance is invalid, not square, contains non-numeric/NaN/Inf/complex entries,
        is materially asymmetric, or initial_weights violate constraints.
    InsufficientDataError:
        If covariance is empty or has fewer than 2 assets.
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
    # Step 2: Validate initial weights vector
    # ------------------------------------------------------------------
    w0 = _validate_initial_weights(
        initial_weights=initial_weights,
        asset_labels=asset_labels,
    )

    # ------------------------------------------------------------------
    # Step 3: Setup objective function and analytical gradient
    # ------------------------------------------------------------------
    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ cov_f64 @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return cov_f64 @ w

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w),
        }
    ]

    # ------------------------------------------------------------------
    # Step 4: Execute SLSQP optimization
    # ------------------------------------------------------------------
    logger.debug(
        "Executing GMV SLSQP optimization: n_assets=%d, cond_num=%.2e, ftol=%.2e, maxiter=%d",
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
    # Step 5: Post-optimization evaluation & fallback handling
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
            "SLSQP GMV optimization failure (status=%d, msg=%s). Fallback policy: %s",
            res.status,
            failure_msg,
            fallback.upper(),
        )

        if fallback.upper() == "RAISE_EXCEPTION":
            raise OptimizationException(
                f"SLSQP GMV optimization failed: {failure_msg}.",
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
            var_eq = float(w_eq @ cov_f64 @ w_eq)
            vol_eq = float(np.sqrt(max(0.0, var_eq)))
            weights_series = pd.Series(w_eq, index=asset_labels, name="weight")

            return OptimizationResult(
                weights=weights_series,
                variance=var_eq,
                volatility=vol_eq,
                objective_value=0.5 * var_eq,
                success=False,
                status=int(res.status),
                message=f"Fallback to EQUAL_WEIGHT executed due to solver failure: {failure_msg}",
                iterations=int(getattr(res, "nit", 0)),
                metadata=base_metadata,
                fallback_executed=True,
            )

    # ------------------------------------------------------------------
    # Step 6: Numerical hygiene and final result construction
    # ------------------------------------------------------------------
    w_clean = np.clip(res.x, 0.0, 1.0)
    sum_w = float(np.sum(w_clean))
    if sum_w <= 0.0:
        # Extreme numerical failure guard
        w_clean = np.ones(n_assets, dtype=np.float64) / float(n_assets)
    else:
        w_clean = w_clean / sum_w

    variance = float(w_clean @ cov_f64 @ w_clean)
    volatility = float(np.sqrt(max(0.0, variance)))
    obj_val = float(0.5 * variance)
    weights_series = pd.Series(w_clean, index=asset_labels, name="weight")

    logger.info(
        "GMV optimization converged successfully: iters=%d, variance=%.6e, volatility=%.4f",
        getattr(res, "nit", 0),
        variance,
        volatility,
    )

    return OptimizationResult(
        weights=weights_series,
        variance=variance,
        volatility=volatility,
        objective_value=obj_val,
        success=True,
        status=int(res.status),
        message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        metadata=base_metadata,
        fallback_executed=False,
    )
