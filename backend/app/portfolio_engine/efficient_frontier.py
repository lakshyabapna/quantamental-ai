"""Quantamental AI - Efficient Frontier Optimization Engine.

Implements Phase 2.3, Step 3: Efficient Frontier computation via constrained
minimum-variance optimization across a range of target expected returns.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §5.4 & ARCHITECTURE.md §4.2):

    For each target return mu_p, solve:

        min_w  f(w) = 0.5 * w.T @ Sigma_ann @ w

    Subject to:
        sum_{i=1}^N w_i = 1.0          (Fully invested budget constraint)
        mu.T @ w = mu_p                (Target expected return constraint)
        0.0 <= w_i <= 1.0             (Long-only constraint)

    Analytical Gradient:
        nabla_w f(w) = Sigma_ann @ w

Feasible return range (long-only):
    mu_min = min(mu)   -- smallest individual asset expected return
    mu_max = max(mu)   -- largest individual asset expected return

Efficient vs. Inefficient frontier:
    - GMV portfolio return = mu_gmv = mu.T @ w_gmv
    - Efficient branch: mu_p >= mu_gmv  (minimum variance for a given return level)
    - Inefficient branch: mu_p < mu_gmv (redundant; a higher-return portfolio exists
      with equal or lower variance, since the frontier is convex below the GMV)

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §5.1, §5.2, §5.3, §5.4, §5.5
    - ARCHITECTURE.md §4.2: Deterministic Financial Engine optimization pipeline.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    OptimizationResult,
    _validate_input_covariance,
    _validate_optimization_parameters,
    optimize_global_minimum_variance,
)

logger = logging.getLogger("quantamental.efficient_frontier")

# ---------------------------------------------------------------------------
# Module Constants
# ---------------------------------------------------------------------------
DEFAULT_FRONTIER_POINTS: int = 50

# Tolerances inherited from optimizer conventions
_WEIGHT_SUM_TOLERANCE: float = 1e-5
_WEIGHT_BOUND_TOLERANCE: float = 1e-7
_TARGET_RETURN_TOLERANCE: float = 1e-6   # achieved return must match target within this
_FEASIBILITY_TOLERANCE: float = 1e-8     # allowed breach of [mu_min, mu_max] boundary
_DUPLICATE_TARGET_TOLERANCE: float = 1e-8  # duplicate target returns rejected within this


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class FrontierPortfolio:
    """Immutable container for a single minimum-variance portfolio on the frontier.

    Defensively immutable: internal Series cannot be mutated via the weights property
    or by mutating the original input Series.

    Attributes:
        target_return: Annualized target expected return passed to the optimizer.
        expected_return: Annualized expected return achieved by the solution (w.T @ mu).
        variance: Annualized portfolio variance (w.T @ Sigma @ w).
        volatility: Annualized portfolio volatility (sqrt(variance)).
        weights: Optimal asset allocation weights (pd.Series indexed by asset labels).
        is_efficient: True if target_return >= GMV expected return (efficient branch),
            False if strictly below GMV expected return (inefficient branch),
            or None if GMV reference optimization failed or this portfolio is a
            fallback / unconverged point.
        success: Whether the numerical solver converged successfully.
        solver_status: Numerical solver exit status code.
        solver_message: Solver termination message.
        iterations: Number of solver iterations executed.
        fallback_executed: True if a fallback policy was executed instead of solver output.
        classification_valid: True if efficient/inefficient classification is mathematically
            valid (requires GMV reference optimizer success without fallback).
    """

    target_return: float
    expected_return: float
    variance: float
    volatility: float
    weights: pd.Series
    is_efficient: Optional[bool]
    success: bool
    solver_status: int
    solver_message: str
    iterations: int
    fallback_executed: bool = False
    classification_valid: bool = True
    _weights: pd.Series = field(repr=False)

    def __init__(
        self,
        target_return: float,
        expected_return: float,
        variance: float,
        volatility: float,
        weights: pd.Series,
        is_efficient: Optional[bool],
        success: bool,
        solver_status: int,
        solver_message: str,
        iterations: int,
        fallback_executed: bool = False,
        classification_valid: bool = True,
    ):
        object.__setattr__(self, "target_return", float(target_return))
        object.__setattr__(self, "expected_return", float(expected_return))
        object.__setattr__(self, "variance", float(variance))
        object.__setattr__(self, "volatility", float(volatility))
        object.__setattr__(
            self,
            "is_efficient",
            bool(is_efficient) if is_efficient is not None else None,
        )
        object.__setattr__(self, "success", bool(success))
        object.__setattr__(self, "solver_status", int(solver_status))
        object.__setattr__(self, "solver_message", str(solver_message))
        object.__setattr__(self, "iterations", int(iterations))
        object.__setattr__(self, "fallback_executed", bool(fallback_executed))
        object.__setattr__(self, "classification_valid", bool(classification_valid))
        object.__setattr__(self, "_weights", weights.copy())

    @property
    def weights(self) -> pd.Series:
        """Return a defensive copy of the portfolio weights Series."""
        return self._weights.copy()

    @property
    def is_classification_valid(self) -> bool:
        """Alias for classification_valid."""
        return self.classification_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "target_return": self.target_return,
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "weights": self.weights.to_dict(),
            "is_efficient": self.is_efficient,
            "success": self.success,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
            "iterations": self.iterations,
            "fallback_executed": self.fallback_executed,
            "classification_valid": self.classification_valid,
        }


@dataclass(frozen=True, init=False)
class EfficientFrontierResult:
    """Immutable container for a computed efficient frontier.

    Defensively immutable: internal tuples, metadata dictionary, and nested
    GMV portfolio weights/metadata cannot be modified by callers.

    Attributes:
        portfolios: All frontier portfolios in ascending target-return order.
        efficient_portfolios: Subset at or above GMV expected return (empty if
            classification_valid is False).
        inefficient_portfolios: Subset strictly below GMV expected return (empty if
            classification_valid is False).
        minimum_feasible_return: min(mu) for the given long-only universe.
        maximum_feasible_return: max(mu) for the given long-only universe.
        gmv_portfolio: Global minimum variance portfolio (from existing GMV optimizer).
        number_of_points: Number of portfolios successfully computed.
        success: True if all requested points converged successfully AND the GMV
            reference optimizer converged without executing a fallback.
        metadata: Diagnostic dictionary (condition number, eigenvalues, etc.).
        fallback_executed: True if any frontier point or GMV optimizer executed a fallback.
        classification_valid: True if efficient/inefficient classification is mathematically
            valid (requires GMV reference optimizer success without fallback).
    """

    portfolios: Tuple[FrontierPortfolio, ...]
    efficient_portfolios: Tuple[FrontierPortfolio, ...]
    inefficient_portfolios: Tuple[FrontierPortfolio, ...]
    minimum_feasible_return: float
    maximum_feasible_return: float
    gmv_portfolio: OptimizationResult
    number_of_points: int
    success: bool
    metadata: Dict[str, Any]
    fallback_executed: bool = False
    classification_valid: bool = True
    _portfolios: Tuple[FrontierPortfolio, ...] = field(repr=False)
    _efficient_portfolios: Tuple[FrontierPortfolio, ...] = field(repr=False)
    _inefficient_portfolios: Tuple[FrontierPortfolio, ...] = field(repr=False)
    _gmv_portfolio: OptimizationResult = field(repr=False)
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        portfolios: Sequence[FrontierPortfolio],
        efficient_portfolios: Sequence[FrontierPortfolio],
        inefficient_portfolios: Sequence[FrontierPortfolio],
        minimum_feasible_return: float,
        maximum_feasible_return: float,
        gmv_portfolio: OptimizationResult,
        number_of_points: int,
        success: bool,
        metadata: Dict[str, Any],
        fallback_executed: bool = False,
        classification_valid: bool = True,
    ):
        object.__setattr__(self, "minimum_feasible_return", float(minimum_feasible_return))
        object.__setattr__(self, "maximum_feasible_return", float(maximum_feasible_return))
        object.__setattr__(self, "number_of_points", int(number_of_points))
        object.__setattr__(self, "success", bool(success))
        object.__setattr__(self, "fallback_executed", bool(fallback_executed))
        object.__setattr__(self, "classification_valid", bool(classification_valid))
        object.__setattr__(self, "_portfolios", tuple(portfolios))
        object.__setattr__(self, "_efficient_portfolios", tuple(efficient_portfolios))
        object.__setattr__(self, "_inefficient_portfolios", tuple(inefficient_portfolios))
        object.__setattr__(
            self,
            "_gmv_portfolio",
            dataclasses.replace(
                gmv_portfolio,
                weights=gmv_portfolio.weights.copy(),
                metadata=copy.deepcopy(gmv_portfolio.metadata),
            ),
        )
        object.__setattr__(self, "_metadata", copy.deepcopy(metadata))

    @property
    def portfolios(self) -> Tuple[FrontierPortfolio, ...]:
        return self._portfolios

    @property
    def efficient_portfolios(self) -> Tuple[FrontierPortfolio, ...]:
        return self._efficient_portfolios

    @property
    def inefficient_portfolios(self) -> Tuple[FrontierPortfolio, ...]:
        return self._inefficient_portfolios

    @property
    def gmv_portfolio(self) -> OptimizationResult:
        """Return a defensive copy of the GMV optimization result."""
        return dataclasses.replace(
            self._gmv_portfolio,
            weights=self._gmv_portfolio.weights.copy(),
            metadata=copy.deepcopy(self._gmv_portfolio.metadata),
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a defensive deep copy of diagnostic metadata."""
        return copy.deepcopy(self._metadata)

    @property
    def is_classification_valid(self) -> bool:
        """Alias for classification_valid."""
        return self.classification_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "portfolios": [p.to_dict() for p in self.portfolios],
            "efficient_portfolios": [p.to_dict() for p in self.efficient_portfolios],
            "inefficient_portfolios": [p.to_dict() for p in self.inefficient_portfolios],
            "minimum_feasible_return": self.minimum_feasible_return,
            "maximum_feasible_return": self.maximum_feasible_return,
            "gmv_portfolio": self.gmv_portfolio.to_dict(),
            "number_of_points": self.number_of_points,
            "success": self.success,
            "fallback_executed": self.fallback_executed,
            "classification_valid": self.classification_valid,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Input Validation Helpers
# ---------------------------------------------------------------------------

def _validate_expected_returns_frontier(
    expected_returns: object,
    asset_labels: pd.Index,
) -> np.ndarray:
    """Validate expected returns vector for frontier computation.

    Parameters
    ----------
    expected_returns:
        Labelled pandas Series of annualized arithmetic expected returns.
    asset_labels:
        Asset labels from the validated covariance matrix (defines ordering).

    Returns
    -------
    np.ndarray
        Float64 array of expected returns aligned to asset_labels order.
    """
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

    # Reindex to covariance column ordering for exact alignment
    mu_series = expected_returns.reindex(asset_labels)

    # Boolean check before dtype check (object Series may contain booleans)
    if pd.api.types.is_bool_dtype(mu_series.dtype) or any(
        isinstance(v, (bool, np.bool_)) for v in mu_series
    ):
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


def _validate_target_returns(
    target_returns: object,
    mu_min: float,
    mu_max: float,
) -> np.ndarray:
    """Validate an explicitly supplied array of target returns.

    Parameters
    ----------
    target_returns:
        Caller-supplied sequence of annualized target returns.
    mu_min:
        Minimum feasible return (min of asset expected returns).
    mu_max:
        Maximum feasible return (max of asset expected returns).

    Returns
    -------
    np.ndarray
        Sorted float64 array of validated target returns.
    """
    if isinstance(target_returns, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"target_returns must be a sequence of real numbers, got bool ({target_returns!r}).",
            details={"type": "bool"},
        )
    if isinstance(target_returns, (str, bytes)):
        raise FinancialCalculationError(
            f"target_returns must be a sequence of real numbers, got {type(target_returns).__name__}.",
            details={"type": type(target_returns).__name__},
        )
    if not isinstance(target_returns, (list, tuple, np.ndarray, pd.Series)):
        raise FinancialCalculationError(
            f"target_returns must be a sequence (list, tuple, np.ndarray, pd.Series), "
            f"got {type(target_returns).__name__}.",
            details={"type": type(target_returns).__name__},
        )

    if len(target_returns) == 0:
        raise FinancialCalculationError(
            "target_returns is empty; at least one target return is required.",
            details={"len": 0},
        )

    # Check for boolean, complex, non-numeric strings across container types
    if isinstance(target_returns, pd.Series):
        if pd.api.types.is_bool_dtype(target_returns.dtype) or any(
            isinstance(x, (bool, np.bool_)) for x in target_returns
        ):
            raise FinancialCalculationError(
                "target_returns contains boolean values; must be numeric.",
                details={"dtype": str(target_returns.dtype)},
            )
        if pd.api.types.is_complex_dtype(target_returns.dtype) or any(
            isinstance(x, (complex, np.complexfloating)) for x in target_returns
        ):
            raise FinancialCalculationError(
                "target_returns contains complex values; must be real numeric.",
                details={"dtype": str(target_returns.dtype)},
            )
        if any(isinstance(x, (str, bytes)) for x in target_returns) or not pd.api.types.is_numeric_dtype(target_returns):
            raise FinancialCalculationError(
                "target_returns contains non-numeric values (such as strings).",
                details={"dtype": str(target_returns.dtype)},
            )
    elif isinstance(target_returns, np.ndarray):
        if target_returns.dtype == bool or np.issubdtype(target_returns.dtype, np.bool_):
            raise FinancialCalculationError(
                "target_returns contains boolean values; must be numeric.",
                details={"dtype": str(target_returns.dtype)},
            )
        if np.iscomplexobj(target_returns) or np.issubdtype(target_returns.dtype, np.complexfloating):
            raise FinancialCalculationError(
                "target_returns contains complex values; must be real numeric.",
                details={"dtype": str(target_returns.dtype)},
            )
        if target_returns.dtype.kind in ("U", "S"):
            raise FinancialCalculationError(
                "target_returns contains string values; must be real numeric.",
                details={"dtype": str(target_returns.dtype)},
            )
        if target_returns.dtype == object:
            for x in target_returns.flat:
                if isinstance(x, (bool, np.bool_)):
                    raise FinancialCalculationError(
                        "target_returns contains boolean values; must be numeric.",
                        details={"element": x},
                    )
                if isinstance(x, (complex, np.complexfloating)):
                    raise FinancialCalculationError(
                        "target_returns contains complex values; must be real numeric.",
                        details={"element": str(x)},
                    )
                if not isinstance(x, (int, float, np.integer, np.floating)):
                    raise FinancialCalculationError(
                        f"target_returns contains non-numeric values (got {type(x).__name__}).",
                        details={"element": str(x), "type": type(x).__name__},
                    )
    else:  # list, tuple
        for x in target_returns:
            if isinstance(x, (bool, np.bool_)):
                raise FinancialCalculationError(
                    "target_returns contains boolean values; must be numeric.",
                    details={"element": x},
                )
            if isinstance(x, (complex, np.complexfloating)):
                raise FinancialCalculationError(
                    "target_returns contains complex values; must be real numeric.",
                    details={"element": str(x)},
                )
            if not isinstance(x, (int, float, np.integer, np.floating)):
                raise FinancialCalculationError(
                    f"target_returns contains non-numeric values (got {type(x).__name__}).",
                    details={"element": str(x), "type": type(x).__name__},
                )

    # Convert to float64 1-D numpy array
    try:
        tr_arr = np.asarray(target_returns, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"Could not convert target_returns to float64 array: {exc}.",
            details={"error": str(exc)},
        ) from exc

    if tr_arr.ndim != 1:
        raise FinancialCalculationError(
            f"target_returns must be a 1-D sequence, got shape {tr_arr.shape}.",
            details={"shape": list(tr_arr.shape)},
        )

    if not np.all(np.isfinite(tr_arr)):
        raise FinancialCalculationError(
            "target_returns contains NaN or infinite values.",
            details={"values": [None if not np.isfinite(v) else float(v) for v in tr_arr]},
        )

    # Feasibility check with tolerance
    out_of_range = [
        float(r)
        for r in tr_arr
        if r < mu_min - _FEASIBILITY_TOLERANCE or r > mu_max + _FEASIBILITY_TOLERANCE
    ]
    if out_of_range:
        raise FinancialCalculationError(
            f"target_returns contains values outside the feasible range "
            f"[{mu_min:.6f}, {mu_max:.6f}] (tolerance={_FEASIBILITY_TOLERANCE:.2e}).",
            details={
                "out_of_range": out_of_range,
                "mu_min": mu_min,
                "mu_max": mu_max,
                "feasibility_tolerance": _FEASIBILITY_TOLERANCE,
            },
        )

    sorted_tr = np.sort(tr_arr)

    # Duplicate check: reject duplicates within numerical tolerance
    if len(sorted_tr) > 1:
        diffs = np.diff(sorted_tr)
        if np.any(diffs <= _DUPLICATE_TARGET_TOLERANCE):
            dup_indices = np.where(diffs <= _DUPLICATE_TARGET_TOLERANCE)[0]
            duplicate_pairs = [(float(sorted_tr[i]), float(sorted_tr[i + 1])) for i in dup_indices]
            raise FinancialCalculationError(
                f"target_returns contains duplicate or near-duplicate target returns "
                f"within tolerance {_DUPLICATE_TARGET_TOLERANCE:.2e}: {duplicate_pairs}.",
                details={
                    "duplicate_pairs": duplicate_pairs,
                    "duplicate_tolerance": _DUPLICATE_TARGET_TOLERANCE,
                },
            )

    return sorted_tr


def _validate_n_points(n_points: object) -> int:
    """Validate the number of frontier grid points."""
    if isinstance(n_points, bool):
        raise FinancialCalculationError(
            f"n_points must be a positive integer, got bool ({n_points!r}).",
            details={"type": "bool"},
        )
    if not isinstance(n_points, int) or n_points <= 0:
        raise FinancialCalculationError(
            f"n_points must be a strictly positive integer, got {n_points!r}.",
            details={"n_points": n_points},
        )
    return n_points


# ---------------------------------------------------------------------------
# Warm-Start Feasible Initial Weight Helper
# ---------------------------------------------------------------------------

def _feasible_initial_weights(
    target_return: float,
    mu: np.ndarray,
    mu_min: float,
    mu_max: float,
    n_assets: int,
    prev_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a feasible initial weight vector that satisfies the target-return constraint.

    Strategy:
    1. If prev_weights satisfies the target-return constraint within tolerance, use it.
    2. Otherwise interpolate between the min-return asset (w=1) and max-return asset (w=1)
       to find a convex combination that hits the target return exactly.

    The resulting vector always satisfies:
        sum(w) = 1, w_i >= 0, w.T @ mu == target_return (within floating-point precision).
    """
    # Check if previous solution is already feasible for this target
    if prev_weights is not None:
        prev_ret = float(prev_weights @ mu)
        if abs(prev_ret - target_return) <= _TARGET_RETURN_TOLERANCE:
            return prev_weights.copy()

    # Find indices of min and max return assets (use first occurrence on tie)
    idx_min = int(np.argmin(mu))
    idx_max = int(np.argmax(mu))

    mu_lo = mu[idx_min]
    mu_hi = mu[idx_max]

    # Clip target return to [mu_lo, mu_hi] for safety (already validated before call)
    target_clipped = np.clip(target_return, mu_lo, mu_hi)

    if abs(mu_hi - mu_lo) < _FEASIBILITY_TOLERANCE:
        # All assets have identical expected returns; equal weight is feasible
        return np.ones(n_assets, dtype=np.float64) / float(n_assets)

    # Convex combination: alpha * e_max + (1 - alpha) * e_min achieves target_clipped
    # alpha * mu_hi + (1-alpha) * mu_lo = target_clipped
    # alpha = (target_clipped - mu_lo) / (mu_hi - mu_lo)
    alpha = float((target_clipped - mu_lo) / (mu_hi - mu_lo))
    alpha = np.clip(alpha, 0.0, 1.0)

    w0 = np.zeros(n_assets, dtype=np.float64)
    if idx_min == idx_max:
        w0[idx_min] = 1.0
    else:
        w0[idx_max] = alpha
        w0[idx_min] = 1.0 - alpha

    # Normalize (should already sum to 1 by construction)
    total = float(np.sum(w0))
    if total > 0.0:
        w0 = w0 / total

    return w0


# ---------------------------------------------------------------------------
# Single-Point Minimum-Variance Solver
# ---------------------------------------------------------------------------

def _solve_minimum_variance_for_target(
    target_return: float,
    mu: np.ndarray,
    cov: np.ndarray,
    asset_labels: pd.Index,
    gmv_expected_return: Optional[float],
    w0: np.ndarray,
    tolerance: float,
    max_iterations: int,
    fallback: str,
) -> FrontierPortfolio:
    """Solve the minimum-variance problem for a single target expected return.

    Minimizes:
        f(w) = 0.5 * w.T @ Sigma @ w
    Subject to:
        sum(w) = 1
        mu.T @ w = target_return
        0 <= w_i <= 1

    Parameters
    ----------
    target_return:
        Annualized target expected portfolio return.
    mu:
        Float64 array of annualized expected returns (aligned to asset_labels).
    cov:
        Float64 covariance matrix (symmetric, PD).
    asset_labels:
        Asset label index.
    gmv_expected_return:
        Expected return of the GMV portfolio (for efficient/inefficient classification),
        or None if GMV reference optimization failed / fallback executed.
    w0:
        Feasible initial weight vector (must satisfy all constraints within tolerance).
    tolerance:
        SLSQP ftol termination tolerance.
    max_iterations:
        Maximum SLSQP iterations.
    fallback:
        Fallback policy on solver failure.

    Returns
    -------
    FrontierPortfolio
        Minimum-variance portfolio at the requested target return.
    """
    n_assets = len(asset_labels)

    # Detect degenerate case: all assets have identical expected returns.
    # The target-return equality constraint is trivially satisfied by any
    # long-only fully-invested portfolio (mu is a constant scalar times 1),
    # but its Jacobian (mu vector) causes SLSQP numerical issues because the
    # constraint gradient is the same as the budget-constraint gradient scaled
    # by a constant, making the KKT system singular.
    # In this case, solve as a pure GMV (drop the target-return constraint).
    mu_range = float(np.max(mu) - np.min(mu))
    _degenerate = mu_range < _FEASIBILITY_TOLERANCE

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ cov @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return cov @ w

    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    if _degenerate:
        # Only budget constraint; target-return constraint is redundant
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            },
        ]
    else:
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            },
            {
                "type": "eq",
                "fun": lambda w, mu=mu, tr=target_return: float(mu @ w - tr),
                "jac": lambda w, mu=mu: mu.copy(),
            },
        ]

    logger.debug(
        "Frontier SLSQP: target_return=%.6f, n_assets=%d, ftol=%.2e",
        target_return,
        n_assets,
        tolerance,
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

    # ------------------------------------------------------------------
    # Step 5: Post-optimization evaluation & independent revalidation
    # ------------------------------------------------------------------
    is_valid_solution = False
    failure_msg = ""

    if not res.success:
        failure_msg = str(res.message)
    else:
        # Check raw solver solution
        w_raw = res.x
        raw_return = float(mu @ w_raw)
        raw_return_ok = _degenerate or abs(raw_return - target_return) <= _TARGET_RETURN_TOLERANCE
        if not (
            np.all(np.isfinite(w_raw))
            and np.all(w_raw >= -_WEIGHT_BOUND_TOLERANCE)
            and np.all(w_raw <= 1.0 + _WEIGHT_BOUND_TOLERANCE)
            and abs(float(np.sum(w_raw)) - 1.0) <= _WEIGHT_SUM_TOLERANCE
            and raw_return_ok
        ):
            failure_msg = "Raw solver solution violated portfolio constraints or target return"
        else:
            # Clean and normalize weights
            w_clean = np.clip(w_raw, 0.0, 1.0)
            sum_w = float(np.sum(w_clean))
            if sum_w <= 0.0:
                failure_msg = "Cleaned weights sum to zero"
            else:
                w_clean = w_clean / sum_w

                # Independently revalidate cleaned weights and financial metrics
                exp_ret = float(mu @ w_clean)
                clean_return_ok = _degenerate or abs(exp_ret - target_return) <= _TARGET_RETURN_TOLERANCE
                variance = float(w_clean @ cov @ w_clean)
                volatility = float(np.sqrt(max(0.0, variance)))

                if not (
                    np.all(np.isfinite(w_clean))
                    and np.all(w_clean >= 0.0)
                    and np.all(w_clean <= 1.0)
                    and abs(float(np.sum(w_clean)) - 1.0) <= _WEIGHT_SUM_TOLERANCE
                    and clean_return_ok
                    and np.isfinite(variance)
                    and variance >= 0.0
                    and np.isfinite(volatility)
                    and abs(volatility - np.sqrt(max(0.0, variance))) <= 1e-12
                ):
                    failure_msg = "Cleaned portfolio violated constraints, target return, or variance positivity"
                else:
                    is_valid_solution = True

    if not is_valid_solution:
        logger.warning(
            "Frontier SLSQP failure at target_return=%.6f (status=%d, msg=%s). Fallback: %s",
            target_return,
            res.status,
            failure_msg,
            fallback.upper(),
        )

        if fallback.upper() == "RAISE_EXCEPTION":
            raise OptimizationException(
                f"SLSQP Efficient Frontier optimization failed at target_return={target_return:.6f}: "
                f"{failure_msg}.",
                details={
                    "target_return": target_return,
                    "status": int(res.status),
                    "message": str(failure_msg),
                    "iterations": int(getattr(res, "nit", 0)),
                },
            )
        elif fallback.upper() == "EQUAL_WEIGHT":
            # EQUAL_WEIGHT fallback: explicitly flagged with fallback_executed=True and success=False.
            # is_efficient is strictly None because this fallback is not a verified solution
            # to the requested target-return optimization.
            w_eq = np.ones(n_assets, dtype=np.float64) / float(n_assets)
            eq_ret = float(mu @ w_eq)
            eq_var = float(w_eq @ cov @ w_eq)
            eq_vol = float(np.sqrt(max(0.0, eq_var)))
            classification_valid = gmv_expected_return is not None
            return FrontierPortfolio(
                target_return=target_return,
                expected_return=eq_ret,
                variance=eq_var,
                volatility=eq_vol,
                weights=pd.Series(w_eq, index=asset_labels, name="weight"),
                is_efficient=None,
                success=False,
                solver_status=int(res.status),
                solver_message=f"Fallback to EQUAL_WEIGHT: {failure_msg}",
                iterations=int(getattr(res, "nit", 0)),
                fallback_executed=True,
                classification_valid=classification_valid,
            )

    # Valid solution
    classification_valid = gmv_expected_return is not None
    is_eff = bool(target_return >= gmv_expected_return) if classification_valid else None
    return FrontierPortfolio(
        target_return=target_return,
        expected_return=exp_ret,
        variance=variance,
        volatility=volatility,
        weights=pd.Series(w_clean, index=asset_labels, name="weight"),
        is_efficient=is_eff,
        success=True,
        solver_status=int(res.status),
        solver_message=str(res.message),
        iterations=int(getattr(res, "nit", 0)),
        fallback_executed=False,
        classification_valid=classification_valid,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    n_points: int = DEFAULT_FRONTIER_POINTS,
    target_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER,
    input_symmetry_tolerance: float = DEFAULT_INPUT_SYMMETRY_TOLERANCE,
    tolerance: float = DEFAULT_OPTIMIZER_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    fallback: str = DEFAULT_OPTIMIZER_FALLBACK,
) -> EfficientFrontierResult:
    """Compute the minimum-variance efficient frontier for a long-only portfolio.

    For each target return in a grid spanning [min(mu), max(mu)], solves:

        min_w  0.5 * w.T @ Sigma @ w
        s.t.   sum(w) = 1,  mu.T @ w = target_return,  0 <= w_i <= 1

    Uses warm-starting with feasible initial weight vectors to improve stability.
    Classifies portfolios as efficient (target >= GMV return) or inefficient (below).

    Parameters
    ----------
    expected_returns:
        Labelled pandas Series of annualized arithmetic expected returns.
        Asset labels must match covariance row and column labels.
    covariance:
        Labelled square DataFrame of the annualized covariance matrix.
        Must be symmetric, finite, positive diagonal, and strictly positive definite.
    n_points:
        Number of target return grid points to generate (default 50). Ignored if
        target_returns is supplied explicitly. Must be a strictly positive integer.
    target_returns:
        Optional explicit sequence of target annualized returns. If provided,
        n_points is ignored and these values define the frontier grid. All values
        must lie in [min(mu), max(mu)] within tolerance.
    max_condition_number:
        Maximum allowable condition number. Defaults to settings.MAX_CONDITION_NUMBER.
    input_symmetry_tolerance:
        Maximum allowable asymmetry |Sigma - Sigma.T|_max. Defaults to 1e-8.
    tolerance:
        SLSQP termination tolerance (ftol). Defaults to 1e-9.
    max_iterations:
        Maximum number of SLSQP iterations per frontier point. Defaults to 1000.
    fallback:
        Solver failure fallback policy:
        - "RAISE_EXCEPTION" (default): raises OptimizationException on any failure.
        - "EQUAL_WEIGHT": returns equal-weight portfolio, marks success=False.

    Returns
    -------
    EfficientFrontierResult
        Container with all frontier portfolios, efficient and inefficient subsets,
        GMV portfolio, feasible return range, and diagnostic metadata.

    Raises
    ------
    FinancialCalculationError:
        If inputs are invalid (type, structure, NaN, Inf, complex, booleans, labels).
    InsufficientDataError:
        If inputs are empty or have fewer than 2 assets.
    NonPositiveDefiniteMatrixError:
        If covariance is not strictly positive definite.
    IllConditionedMatrixError:
        If covariance condition number exceeds max_condition_number.
    OptimizationException:
        If any frontier point fails and fallback is "RAISE_EXCEPTION".
    """
    # ------------------------------------------------------------------
    # Step 0: Validate solver configuration parameters
    # ------------------------------------------------------------------
    _validate_optimization_parameters(
        max_condition_number=max_condition_number,
        input_symmetry_tolerance=input_symmetry_tolerance,
        tolerance=tolerance,
        max_iterations=max_iterations,
        fallback=fallback,
    )

    # ------------------------------------------------------------------
    # Step 1: Validate covariance matrix
    # ------------------------------------------------------------------
    cov_f64, asset_labels, min_eig, max_eig, condition_number = _validate_input_covariance(
        covariance=covariance,
        input_symmetry_tolerance=input_symmetry_tolerance,
        max_condition_number=max_condition_number,
    )
    n_assets = len(asset_labels)

    # ------------------------------------------------------------------
    # Step 2: Validate expected returns
    # ------------------------------------------------------------------
    mu_f64 = _validate_expected_returns_frontier(
        expected_returns=expected_returns,
        asset_labels=asset_labels,
    )

    # Feasible return range (long-only constraint)
    mu_min = float(np.min(mu_f64))
    mu_max = float(np.max(mu_f64))

    # ------------------------------------------------------------------
    # Step 3: Build target-return grid
    # ------------------------------------------------------------------
    if target_returns is not None:
        tr_grid = _validate_target_returns(target_returns, mu_min, mu_max)
        # Clip to exact boundaries within tolerance
        tr_grid = np.clip(tr_grid, mu_min, mu_max)
    else:
        _validate_n_points(n_points)
        if abs(mu_max - mu_min) < _FEASIBILITY_TOLERANCE:
            # All assets have identical expected returns; single point suffices
            tr_grid = np.array([mu_min])
        else:
            tr_grid = np.linspace(mu_min, mu_max, n_points)

    # ------------------------------------------------------------------
    # Step 4: Compute GMV portfolio (using existing optimizer)
    # ------------------------------------------------------------------
    # GMV reference calculation uses at least DEFAULT_MAX_ITERATIONS to ensure
    # reliable reference point convergence even if the caller configures tight
    # iterations for the frontier points.
    gmv_result = optimize_global_minimum_variance(
        covariance=covariance,
        max_condition_number=max_condition_number,
        input_symmetry_tolerance=input_symmetry_tolerance,
        tolerance=tolerance,
        max_iterations=max(max_iterations, DEFAULT_MAX_ITERATIONS),
        fallback=fallback,
    )

    # Inspect GMV result: never silently use fallback weights as a verified GMV reference
    gmv_is_valid = bool(gmv_result.success and not gmv_result.fallback_executed)
    if gmv_is_valid:
        gmv_expected_return: Optional[float] = float(
            mu_f64 @ gmv_result.weights.reindex(asset_labels).to_numpy()
        )
        classification_valid = True
    else:
        logger.warning(
            "GMV reference optimization failed or executed fallback (success=%s, fallback_executed=%s). "
            "Equal-weight fallback is not verified GMV; frontier classification disabled.",
            gmv_result.success,
            gmv_result.fallback_executed,
        )
        gmv_expected_return = None
        classification_valid = False

    logger.debug(
        "Frontier setup: n_assets=%d, mu_min=%.6f, mu_max=%.6f, gmv_ret=%s, "
        "classification_valid=%s, n_grid_points=%d, cond_num=%.2e",
        n_assets,
        mu_min,
        mu_max,
        f"{gmv_expected_return:.6f}" if gmv_expected_return is not None else "None",
        classification_valid,
        len(tr_grid),
        condition_number,
    )

    # ------------------------------------------------------------------
    # Step 5: Solve minimum-variance problem for each target return
    # ------------------------------------------------------------------
    portfolios: List[FrontierPortfolio] = []
    prev_weights: Optional[np.ndarray] = None
    all_succeeded = True

    for target in tr_grid:
        # Build a feasible warm-start initial weight vector
        w0 = _feasible_initial_weights(
            target_return=float(target),
            mu=mu_f64,
            mu_min=mu_min,
            mu_max=mu_max,
            n_assets=n_assets,
            prev_weights=prev_weights,
        )

        portfolio = _solve_minimum_variance_for_target(
            target_return=float(target),
            mu=mu_f64,
            cov=cov_f64,
            asset_labels=asset_labels,
            gmv_expected_return=gmv_expected_return,
            w0=w0,
            tolerance=tolerance,
            max_iterations=max_iterations,
            fallback=fallback,
        )
        portfolios.append(portfolio)

        if portfolio.success:
            # Update warm-start for next iteration
            prev_weights = portfolio.weights.to_numpy()
        else:
            all_succeeded = False
            prev_weights = None  # Don't warm-start from a failed solution

    # ------------------------------------------------------------------
    # Step 6: Sort by target return (linspace already sorted, but explicit sort
    # handles caller-supplied unsorted target_returns)
    # ------------------------------------------------------------------
    portfolios_sorted = tuple(sorted(portfolios, key=lambda p: p.target_return))

    if classification_valid:
        efficient = tuple(
            p for p in portfolios_sorted
            if p.success and not p.fallback_executed and p.is_efficient is True
        )
        inefficient = tuple(
            p for p in portfolios_sorted
            if p.success and not p.fallback_executed and p.is_efficient is False
        )
    else:
        efficient = ()
        inefficient = ()

    frontier_fallback_executed = any(p.fallback_executed for p in portfolios_sorted)
    overall_fallback_executed = frontier_fallback_executed or gmv_result.fallback_executed

    # Overall success semantics: must NOT report success=True when the GMV reference
    # optimization failed or used a fallback, or when any frontier point failed.
    overall_success = all_succeeded and gmv_is_valid and not overall_fallback_executed

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
        "gmv_expected_return": gmv_expected_return,
        "gmv_variance": gmv_result.variance,
        "gmv_volatility": gmv_result.volatility,
        "gmv_success": gmv_result.success,
        "gmv_fallback_executed": gmv_result.fallback_executed,
        "frontier_fallback_executed": frontier_fallback_executed,
        "classification_valid": classification_valid,
        "n_efficient_portfolios": len(efficient),
        "n_inefficient_portfolios": len(inefficient),
        "feasibility_tolerance": _FEASIBILITY_TOLERANCE,
        "target_return_tolerance": _TARGET_RETURN_TOLERANCE,
        "duplicate_target_tolerance": _DUPLICATE_TARGET_TOLERANCE,
    }
    if not classification_valid:
        base_metadata["classification_error"] = (
            "GMV reference optimization failed or executed fallback; "
            "efficient/inefficient frontier classification is unavailable."
        )

    logger.info(
        "Efficient frontier computed: %d points (%d efficient, %d inefficient), "
        "all_succeeded=%s, overall_success=%s, fallback_executed=%s, classification_valid=%s, cond_num=%.2e",
        len(portfolios_sorted),
        len(efficient),
        len(inefficient),
        all_succeeded,
        overall_success,
        overall_fallback_executed,
        classification_valid,
        condition_number,
    )

    return EfficientFrontierResult(
        portfolios=portfolios_sorted,
        efficient_portfolios=efficient,
        inefficient_portfolios=inefficient,
        minimum_feasible_return=mu_min,
        maximum_feasible_return=mu_max,
        gmv_portfolio=gmv_result,
        number_of_points=len(portfolios_sorted),
        success=overall_success,
        metadata=base_metadata,
        fallback_executed=overall_fallback_executed,
        classification_valid=classification_valid,
    )
