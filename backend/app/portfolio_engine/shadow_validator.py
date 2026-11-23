"""Quantamental AI - PyPortfolioOpt Shadow Validator Engine.

Implements Phase 2.4, Step 1: Independent PyPortfolioOpt-based shadow validation
to cross-check deterministic portfolio optimization engine outputs.

Mathematical & Validation Principles:
    - Shadow validation is purely diagnostic and verification-only.
    - Production optimization MUST NEVER be routed through or replaced by PyPortfolioOpt.
    - Identical annualized expected returns and covariance matrices are fed to both solvers.
    - Long-only, fully invested constraints (w_i >= 0, sum(w_i) = 1.0) are strictly preserved.
    - Asset ordering is rigorously aligned before comparison; assets are never silently dropped or reordered.
    - Numerical differences are reported rather than assuming solvers must return identical weights.
    - Recognizes that multiple numerically equivalent optimal solutions may exist for flat
      optima or highly collinear assets. Metric parity (variance, volatility, expected return,
      Sharpe ratio) is the primary optimality criterion.

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §5
    - ARCHITECTURE.md §4.2
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, exceptions as pypf_exceptions

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    OptimizationException,
)
from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    FrontierPortfolio,
    compute_efficient_frontier,
)
from app.portfolio_engine.optimizer import (
    OptimizationResult,
    optimize_global_minimum_variance,
)
from app.portfolio_engine.tangency import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
)

logger = logging.getLogger("quantamental.shadow_validator")


# ---------------------------------------------------------------------------
# Status & Tolerance Containers
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    """Validation outcome status code."""

    PASSED = "PASSED"
    EQUIVALENT_OPTIMUM = "EQUIVALENT_OPTIMUM"
    WARNING = "WARNING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ValidationTolerances:
    """Configurable numerical tolerances for shadow validation comparison.

    Attributes:
        weight_tol: Maximum allowable absolute difference in individual asset weights (default 1e-3).
        variance_tol: Maximum allowable absolute difference in annualized variance (default 1e-4).
        volatility_tol: Maximum allowable absolute difference in annualized volatility (default 1e-4).
        return_tol: Maximum allowable absolute difference in annualized expected return (default 1e-4).
        sharpe_tol: Maximum allowable absolute difference in Sharpe ratio (default 1e-3).
        weight_sum_tol: Maximum allowable deviation from sum(w) == 1.0 (default 1e-5).
        relative_tol: Maximum allowable relative difference for metrics when absolute difference is small (default 1e-2).
    """

    weight_tol: float = 1e-3
    variance_tol: float = 1e-4
    volatility_tol: float = 1e-4
    return_tol: float = 1e-4
    sharpe_tol: float = 1e-3
    weight_sum_tol: float = 1e-5
    relative_tol: float = 1e-2

    def to_dict(self) -> Dict[str, float]:
        """Convert tolerance configuration to dictionary."""
        return {
            "weight_tol": self.weight_tol,
            "variance_tol": self.variance_tol,
            "volatility_tol": self.volatility_tol,
            "return_tol": self.return_tol,
            "sharpe_tol": self.sharpe_tol,
            "weight_sum_tol": self.weight_sum_tol,
            "relative_tol": self.relative_tol,
        }


DEFAULT_VALIDATION_TOLERANCES = ValidationTolerances()


# ---------------------------------------------------------------------------
# Metric & Difference Containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class PortfolioMetrics:
    """Immutable snapshot of portfolio weights and performance metrics.

    Defensively immutable: internal Series cannot be modified by callers.
    """

    weights: pd.Series
    variance: float
    volatility: float
    expected_return: Optional[float]
    sharpe_ratio: Optional[float]
    _weights: pd.Series = field(repr=False)

    def __init__(
        self,
        weights: pd.Series,
        variance: float,
        volatility: float,
        expected_return: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
    ):
        object.__setattr__(self, "_weights", weights.copy())
        object.__setattr__(self, "variance", float(variance))
        object.__setattr__(self, "volatility", float(volatility))
        object.__setattr__(
            self,
            "expected_return",
            float(expected_return) if expected_return is not None else None,
        )
        object.__setattr__(
            self,
            "sharpe_ratio",
            float(sharpe_ratio) if sharpe_ratio is not None else None,
        )

    @property
    def weights(self) -> pd.Series:
        """Return a defensive copy of the weights series."""
        return self._weights.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "weights": {str(k): float(v) for k, v in self._weights.items()},
            "variance": self.variance,
            "volatility": self.volatility,
            "expected_return": self.expected_return,
            "sharpe_ratio": self.sharpe_ratio,
        }


@dataclass(frozen=True, init=False)
class ValidationDifferences:
    """Comparison deltas between reference optimizer and shadow optimizer outputs."""

    weight_differences: pd.Series
    max_weight_diff: float
    abs_diff_variance: float
    abs_diff_volatility: float
    abs_diff_return: Optional[float]
    abs_diff_sharpe: Optional[float]
    rel_diff_variance: float
    rel_diff_volatility: float
    rel_diff_return: Optional[float]
    rel_diff_sharpe: Optional[float]
    _weight_differences: pd.Series = field(repr=False)

    def __init__(
        self,
        weight_differences: pd.Series,
        max_weight_diff: float,
        abs_diff_variance: float,
        abs_diff_volatility: float,
        abs_diff_return: Optional[float] = None,
        abs_diff_sharpe: Optional[float] = None,
        rel_diff_variance: float = 0.0,
        rel_diff_volatility: float = 0.0,
        rel_diff_return: Optional[float] = None,
        rel_diff_sharpe: Optional[float] = None,
    ):
        object.__setattr__(self, "_weight_differences", weight_differences.copy())
        object.__setattr__(self, "max_weight_diff", float(max_weight_diff))
        object.__setattr__(self, "abs_diff_variance", float(abs_diff_variance))
        object.__setattr__(self, "abs_diff_volatility", float(abs_diff_volatility))
        object.__setattr__(
            self,
            "abs_diff_return",
            float(abs_diff_return) if abs_diff_return is not None else None,
        )
        object.__setattr__(
            self,
            "abs_diff_sharpe",
            float(abs_diff_sharpe) if abs_diff_sharpe is not None else None,
        )
        object.__setattr__(self, "rel_diff_variance", float(rel_diff_variance))
        object.__setattr__(self, "rel_diff_volatility", float(rel_diff_volatility))
        object.__setattr__(
            self,
            "rel_diff_return",
            float(rel_diff_return) if rel_diff_return is not None else None,
        )
        object.__setattr__(
            self,
            "rel_diff_sharpe",
            float(rel_diff_sharpe) if rel_diff_sharpe is not None else None,
        )

    @property
    def weight_differences(self) -> pd.Series:
        """Return defensive copy of asset weight differences."""
        return self._weight_differences.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert differences to JSON-serializable dictionary."""
        return {
            "weight_differences": {str(k): float(v) for k, v in self._weight_differences.items()},
            "max_weight_diff": self.max_weight_diff,
            "abs_diff_variance": self.abs_diff_variance,
            "abs_diff_volatility": self.abs_diff_volatility,
            "abs_diff_return": self.abs_diff_return,
            "abs_diff_sharpe": self.abs_diff_sharpe,
            "rel_diff_variance": self.rel_diff_variance,
            "rel_diff_volatility": self.rel_diff_volatility,
            "rel_diff_return": self.rel_diff_return,
            "rel_diff_sharpe": self.rel_diff_sharpe,
        }


# ---------------------------------------------------------------------------
# Validation Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class GMVValidationResult:
    """Immutable validation result for Global Minimum Variance (GMV) portfolio."""

    is_valid: bool
    status: str
    metric_valid: bool
    weight_valid: bool
    reference_metrics: Optional[PortfolioMetrics]
    shadow_metrics: Optional[PortfolioMetrics]
    differences: Optional[ValidationDifferences]
    tolerances: ValidationTolerances
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    metadata: Dict[str, Any]
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        is_valid: bool,
        status: Union[ValidationStatus, str],
        metric_valid: bool,
        weight_valid: bool,
        reference_metrics: Optional[PortfolioMetrics],
        shadow_metrics: Optional[PortfolioMetrics],
        differences: Optional[ValidationDifferences],
        tolerances: ValidationTolerances,
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "is_valid", bool(is_valid))
        object.__setattr__(
            self,
            "status",
            status.value if isinstance(status, ValidationStatus) else str(status),
        )
        object.__setattr__(self, "metric_valid", bool(metric_valid))
        object.__setattr__(self, "weight_valid", bool(weight_valid))
        object.__setattr__(self, "reference_metrics", reference_metrics)
        object.__setattr__(self, "shadow_metrics", shadow_metrics)
        object.__setattr__(self, "differences", differences)
        object.__setattr__(self, "tolerances", tolerances)
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "errors", tuple(errors))
        object.__setattr__(self, "_metadata", copy.deepcopy(metadata or {}))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return defensive deep copy of metadata."""
        return copy.deepcopy(self._metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "metric_valid": self.metric_valid,
            "weight_valid": self.weight_valid,
            "reference_metrics": self.reference_metrics.to_dict() if self.reference_metrics else None,
            "shadow_metrics": self.shadow_metrics.to_dict() if self.shadow_metrics else None,
            "differences": self.differences.to_dict() if self.differences else None,
            "tolerances": self.tolerances.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": self.metadata,
        }


@dataclass(frozen=True, init=False)
class TangencyValidationResult:
    """Immutable validation result for Maximum Sharpe Ratio (Tangency) portfolio."""

    is_valid: bool
    status: str
    metric_valid: bool
    weight_valid: bool
    risk_free_rate: float
    reference_metrics: Optional[PortfolioMetrics]
    shadow_metrics: Optional[PortfolioMetrics]
    differences: Optional[ValidationDifferences]
    tolerances: ValidationTolerances
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    metadata: Dict[str, Any]
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        is_valid: bool,
        status: Union[ValidationStatus, str],
        metric_valid: bool,
        weight_valid: bool,
        risk_free_rate: float,
        reference_metrics: Optional[PortfolioMetrics],
        shadow_metrics: Optional[PortfolioMetrics],
        differences: Optional[ValidationDifferences],
        tolerances: ValidationTolerances,
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "is_valid", bool(is_valid))
        object.__setattr__(
            self,
            "status",
            status.value if isinstance(status, ValidationStatus) else str(status),
        )
        object.__setattr__(self, "metric_valid", bool(metric_valid))
        object.__setattr__(self, "weight_valid", bool(weight_valid))
        object.__setattr__(self, "risk_free_rate", float(risk_free_rate))
        object.__setattr__(self, "reference_metrics", reference_metrics)
        object.__setattr__(self, "shadow_metrics", shadow_metrics)
        object.__setattr__(self, "differences", differences)
        object.__setattr__(self, "tolerances", tolerances)
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "errors", tuple(errors))
        object.__setattr__(self, "_metadata", copy.deepcopy(metadata or {}))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return defensive deep copy of metadata."""
        return copy.deepcopy(self._metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "metric_valid": self.metric_valid,
            "weight_valid": self.weight_valid,
            "risk_free_rate": self.risk_free_rate,
            "reference_metrics": self.reference_metrics.to_dict() if self.reference_metrics else None,
            "shadow_metrics": self.shadow_metrics.to_dict() if self.shadow_metrics else None,
            "differences": self.differences.to_dict() if self.differences else None,
            "tolerances": self.tolerances.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class FrontierPointValidation:
    """Validation report for an individual target-return portfolio on the Efficient Frontier."""

    target_return: float
    is_valid: bool
    status: str
    metric_valid: bool
    weight_valid: bool
    reference_metrics: Optional[PortfolioMetrics]
    shadow_metrics: Optional[PortfolioMetrics]
    differences: Optional[ValidationDifferences]
    warnings: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Convert point validation to dictionary."""
        return {
            "target_return": self.target_return,
            "is_valid": self.is_valid,
            "status": self.status,
            "metric_valid": self.metric_valid,
            "weight_valid": self.weight_valid,
            "reference_metrics": self.reference_metrics.to_dict() if self.reference_metrics else None,
            "shadow_metrics": self.shadow_metrics.to_dict() if self.shadow_metrics else None,
            "differences": self.differences.to_dict() if self.differences else None,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, init=False)
class FrontierValidationResult:
    """Immutable validation result for Efficient Frontier portfolios."""

    is_valid: bool
    status: str
    validated_points: Tuple[FrontierPointValidation, ...]
    n_points_validated: int
    n_points_passed: int
    tolerances: ValidationTolerances
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    metadata: Dict[str, Any]
    _metadata: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        is_valid: bool,
        status: Union[ValidationStatus, str],
        validated_points: Sequence[FrontierPointValidation],
        n_points_validated: int,
        n_points_passed: int,
        tolerances: ValidationTolerances,
        warnings: Sequence[str] = (),
        errors: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "is_valid", bool(is_valid))
        object.__setattr__(
            self,
            "status",
            status.value if isinstance(status, ValidationStatus) else str(status),
        )
        object.__setattr__(self, "validated_points", tuple(validated_points))
        object.__setattr__(self, "n_points_validated", int(n_points_validated))
        object.__setattr__(self, "n_points_passed", int(n_points_passed))
        object.__setattr__(self, "tolerances", tolerances)
        object.__setattr__(self, "warnings", tuple(warnings))
        object.__setattr__(self, "errors", tuple(errors))
        object.__setattr__(self, "_metadata", copy.deepcopy(metadata or {}))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return defensive deep copy of metadata."""
        return copy.deepcopy(self._metadata)

    def __iter__(self):
        """Iterate over individual validated frontier points."""
        return iter(self.validated_points)

    def __len__(self) -> int:
        """Return number of validated points."""
        return len(self.validated_points)

    def to_dict(self) -> Dict[str, Any]:
        """Convert frontier validation result to JSON-serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "validated_points": [p.to_dict() for p in self.validated_points],
            "n_points_validated": self.n_points_validated,
            "n_points_passed": self.n_points_passed,
            "tolerances": self.tolerances.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Input Normalization & Alignment Helpers
# ---------------------------------------------------------------------------

def _validate_and_normalize_covariance(
    covariance: object,
) -> Tuple[pd.DataFrame, pd.Index]:
    """Validate covariance matrix structure, square shape, finiteness, and asset labels."""
    if covariance is None:
        raise FinancialCalculationError(
            "covariance matrix cannot be None.",
            details={"type": "NoneType"},
        )

    if isinstance(covariance, pd.DataFrame):
        if covariance.empty:
            raise InsufficientDataError(
                "covariance DataFrame is empty.",
                details={"shape": covariance.shape},
            )
        if covariance.shape[0] != covariance.shape[1]:
            raise FinancialCalculationError(
                f"covariance DataFrame must be square (N x N), got shape {covariance.shape}.",
                details={"shape": covariance.shape},
            )
        if len(covariance) < 2:
            raise InsufficientDataError(
                f"covariance matrix must contain at least 2 assets, got {len(covariance)}.",
                details={"n_assets": len(covariance)},
            )

        # Check column-index alignment
        if not covariance.index.equals(covariance.columns):
            raise FinancialCalculationError(
                "covariance DataFrame index and columns must match identically.",
                details={"index": list(covariance.index), "columns": list(covariance.columns)},
            )

        # Verify all elements are numeric, not boolean, not complex, and finite
        raw_vals = covariance.values
        if np.issubdtype(raw_vals.dtype, np.complexfloating):
            raise FinancialCalculationError(
                f"covariance contains non-numeric data with dtype {raw_vals.dtype} (complex numbers are not permitted).",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.issubdtype(raw_vals.dtype, np.number) or np.issubdtype(raw_vals.dtype, np.bool_):
            raise FinancialCalculationError(
                f"covariance contains non-numeric data with dtype {raw_vals.dtype}.",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.all(np.isfinite(raw_vals)):
            raise FinancialCalculationError(
                "covariance matrix contains NaN, infinite, or non-finite values.",
                details={"shape": covariance.shape},
            )

        # Symmetry check
        asymmetry = float(np.max(np.abs(raw_vals - raw_vals.T)))
        if asymmetry > 1e-8:
            raise FinancialCalculationError(
                f"covariance matrix is materially asymmetric (max asymmetry={asymmetry:.2e} > tolerance=1e-08).",
                details={"max_asymmetry": asymmetry, "input_symmetry_tolerance": 1e-8},
            )

        # Diagonal variance positivity check (variances must be strictly positive)
        variances = np.diag(raw_vals)
        if not np.all(variances > 0.0):
            invalid_vars = {
                str(col): float(val)
                for col, val in zip(covariance.columns, variances)
                if val <= 0.0 or not np.isfinite(val)
            }
            raise FinancialCalculationError(
                "All diagonal variance elements in covariance matrix must be finite and strictly positive (> 0). Negative or zero variance is not permitted.",
                details={"invalid_diagonal_variances": invalid_vars},
            )

        asset_labels = covariance.columns
        cov_df = covariance.astype(float)
        return cov_df, asset_labels

    elif isinstance(covariance, np.ndarray):
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise FinancialCalculationError(
                f"covariance array must be 2D square (N x N), got shape {covariance.shape}.",
                details={"shape": covariance.shape},
            )
        if covariance.shape[0] < 2:
            raise InsufficientDataError(
                f"covariance array must contain at least 2 assets, got {covariance.shape[0]}.",
                details={"n_assets": covariance.shape[0]},
            )
        if np.issubdtype(covariance.dtype, np.complexfloating):
            raise FinancialCalculationError(
                "covariance contains complex numbers.",
                details={"dtype": str(covariance.dtype)},
            )
        if not np.issubdtype(covariance.dtype, np.number) or np.issubdtype(covariance.dtype, np.bool_):
            raise FinancialCalculationError(
                f"covariance array contains non-numeric data with dtype {covariance.dtype}.",
                details={"dtype": str(covariance.dtype)},
            )
        if not np.all(np.isfinite(covariance)):
            raise FinancialCalculationError(
                "covariance array contains NaN, infinite, or non-finite values.",
                details={"shape": covariance.shape},
            )

        # Symmetry check
        asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        if asymmetry > 1e-8:
            raise FinancialCalculationError(
                f"covariance array is materially asymmetric (max asymmetry={asymmetry:.2e} > tolerance=1e-08).",
                details={"max_asymmetry": asymmetry, "input_symmetry_tolerance": 1e-8},
            )

        # Diagonal variance positivity check
        variances = np.diag(covariance)
        if not np.all(variances > 0.0):
            invalid_vars = {
                f"asset_{i}": float(val)
                for i, val in enumerate(variances)
                if val <= 0.0 or not np.isfinite(val)
            }
            raise FinancialCalculationError(
                "All diagonal variance elements in covariance array must be finite and strictly positive (> 0). Negative or zero variance is not permitted.",
                details={"invalid_diagonal_variances": invalid_vars},
            )

        n = covariance.shape[0]
        asset_labels = pd.Index([f"asset_{i}" for i in range(n)])
        cov_df = pd.DataFrame(covariance.astype(float), index=asset_labels, columns=asset_labels)
        return cov_df, asset_labels

    else:
        raise FinancialCalculationError(
            f"covariance must be a pandas DataFrame or 2D numpy array, got {type(covariance).__name__}.",
            details={"type": type(covariance).__name__},
        )


def _validate_and_normalize_expected_returns(
    expected_returns: object,
    asset_labels: pd.Index,
) -> pd.Series:
    """Validate expected returns vector and align with asset labels."""
    if expected_returns is None:
        raise FinancialCalculationError(
            "expected_returns vector cannot be None.",
            details={"type": "NoneType"},
        )

    if isinstance(expected_returns, pd.Series):
        if expected_returns.empty:
            raise InsufficientDataError(
                "expected_returns Series is empty.",
                details={"len": 0},
            )
        if len(expected_returns) != len(asset_labels):
            raise FinancialCalculationError(
                f"expected_returns length ({len(expected_returns)}) does not match asset count ({len(asset_labels)}).",
                details={"expected_returns_len": len(expected_returns), "asset_count": len(asset_labels)},
            )
        if not set(asset_labels).issubset(set(expected_returns.index)):
            missing = sorted(set(asset_labels) - set(expected_returns.index))
            raise FinancialCalculationError(
                f"expected_returns Series is missing asset labels: {missing}.",
                details={"missing": missing},
            )
        # Verify numeric and finite
        raw_vals = expected_returns.values
        if not np.issubdtype(raw_vals.dtype, np.number) or np.issubdtype(raw_vals.dtype, np.bool_):
            raise FinancialCalculationError(
                f"expected_returns contains non-numeric data with dtype {raw_vals.dtype}.",
                details={"dtype": str(raw_vals.dtype)},
            )
        if not np.all(np.isfinite(raw_vals)):
            raise FinancialCalculationError(
                "expected_returns Series contains NaN, infinite, or non-finite values.",
                details={"values": expected_returns.to_dict()},
            )

        # Strictly align with asset_labels
        aligned = expected_returns.loc[asset_labels].astype(float)
        return aligned

    elif isinstance(expected_returns, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_returns)
        if arr.ndim != 1 or len(arr) != len(asset_labels):
            raise FinancialCalculationError(
                f"expected_returns sequence must be 1D with length {len(asset_labels)}, got shape {arr.shape}.",
                details={"shape": arr.shape, "expected_len": len(asset_labels)},
            )
        if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            raise FinancialCalculationError(
                f"expected_returns sequence contains non-numeric data with dtype {arr.dtype}.",
                details={"dtype": str(arr.dtype)},
            )
        if not np.all(np.isfinite(arr)):
            raise FinancialCalculationError(
                "expected_returns sequence contains NaN, infinite, or non-finite values.",
                details={"len": len(arr)},
            )

        return pd.Series(arr.astype(float), index=asset_labels)

    else:
        raise FinancialCalculationError(
            f"expected_returns must be a pandas Series or 1D array/sequence, got {type(expected_returns).__name__}.",
            details={"type": type(expected_returns).__name__},
        )


def _validate_risk_free_rate(risk_free_rate: object) -> float:
    """Validate that risk-free rate is a finite real numeric scalar."""
    if isinstance(risk_free_rate, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"risk_free_rate must be a numeric float, got bool ({risk_free_rate!r}).",
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
    rf = float(risk_free_rate)
    if not np.isfinite(rf):
        raise FinancialCalculationError(
            f"risk_free_rate must be finite, got {risk_free_rate!r}.",
            details={"value": risk_free_rate},
        )
    return rf


def _validate_reference_weights(
    weights: object,
    asset_labels: pd.Index,
    weight_sum_tol: float,
    name: str = "reference_result",
) -> None:
    """Validate reference portfolio weights for exact label match, finiteness, bounds, and sum.

    Raises
    ------
    FinancialCalculationError
        If weights are not a pandas Series, asset labels do not match universe exactly,
        weights are non-finite, violate long-only bounds [0, 1] within 1e-7 tolerance,
        or weight sum deviates from 1.0 beyond weight_sum_tol.
    """
    if not isinstance(weights, pd.Series):
        raise FinancialCalculationError(
            f"{name} weights must be a pandas Series, got {type(weights).__name__}.",
            details={"type": type(weights).__name__},
        )
    if set(weights.index) != set(asset_labels):
        extra = sorted(set(weights.index) - set(asset_labels))
        missing = sorted(set(asset_labels) - set(weights.index))
        raise FinancialCalculationError(
            f"{name} weights asset labels do not match covariance universe exactly: extra={extra}, missing={missing}.",
            details={"extra": extra, "missing": missing},
        )
    raw = weights.values
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise FinancialCalculationError(
            f"{name} weights contain non-numeric data with dtype {raw.dtype}.",
            details={"dtype": str(raw.dtype)},
        )
    if not np.all(np.isfinite(raw)):
        raise FinancialCalculationError(
            f"{name} weights contain NaN, infinite, or non-finite values.",
            details={"weights": weights.to_dict()},
        )
    if (raw < -1e-7).any() or (raw > 1.0 + 1e-7).any():
        raise FinancialCalculationError(
            f"{name} weights violate long-only bounds [0, 1]: min={float(np.min(raw)):.6f}, max={float(np.max(raw)):.6f}.",
            details={"min_weight": float(np.min(raw)), "max_weight": float(np.max(raw))},
        )
    sum_w = float(np.sum(raw))
    if abs(sum_w - 1.0) > weight_sum_tol:
        raise FinancialCalculationError(
            f"{name} weights sum ({sum_w:.6f}) deviates from 1.0 beyond weight_sum_tol={weight_sum_tol:.2e}.",
            details={"weight_sum": sum_w, "weight_sum_tol": weight_sum_tol},
        )


def _check_shadow_weight_feasibility(
    shadow_w: pd.Series,
    asset_labels: pd.Index,
    tol: ValidationTolerances,
) -> List[str]:
    """Verify shadow portfolio weights feasibility.

    Returns
    -------
    List[str]
        List of feasibility error descriptions. Empty list indicates feasible weights.
    """
    errors: List[str] = []
    if set(shadow_w.index) != set(asset_labels):
        extra = sorted(set(shadow_w.index) - set(asset_labels))
        missing = sorted(set(asset_labels) - set(shadow_w.index))
        errors.append(
            f"Shadow weights asset labels do not match covariance universe: extra={extra}, missing={missing}."
        )
    raw = shadow_w.values
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        errors.append(f"Shadow weights contain non-numeric data with dtype {raw.dtype}.")
        return errors
    if not np.all(np.isfinite(raw)):
        errors.append("Shadow weights contain NaN, infinite, or non-finite values.")
        return errors
    if (raw < -1e-7).any() or (raw > 1.0 + 1e-7).any():
        errors.append(
            f"Shadow weights violate long-only bounds [0, 1]: min={float(np.min(raw)):.6f}, max={float(np.max(raw)):.6f}."
        )
    sum_w = float(np.sum(raw))
    if abs(sum_w - 1.0) > tol.weight_sum_tol:
        errors.append(
            f"Shadow weights sum ({sum_w:.6f}) deviates from 1.0 beyond weight_sum_tol={tol.weight_sum_tol:.2e}."
        )
    return errors


def _compute_validation_differences(
    ref_metrics: PortfolioMetrics,
    shadow_metrics: PortfolioMetrics,
    asset_labels: pd.Index,
) -> ValidationDifferences:
    """Calculate absolute and relative differences between reference and shadow metrics."""
    ref_w = ref_metrics.weights.reindex(asset_labels)
    shadow_w = shadow_metrics.weights.reindex(asset_labels)

    weight_diffs = ref_w - shadow_w
    max_w_diff = float(np.max(np.abs(weight_diffs.values)))

    abs_var = abs(ref_metrics.variance - shadow_metrics.variance)
    rel_var = abs_var / max(abs(shadow_metrics.variance), 1e-8)

    abs_vol = abs(ref_metrics.volatility - shadow_metrics.volatility)
    rel_vol = abs_vol / max(abs(shadow_metrics.volatility), 1e-8)

    abs_ret: Optional[float] = None
    rel_ret: Optional[float] = None
    if ref_metrics.expected_return is not None and shadow_metrics.expected_return is not None:
        abs_ret = abs(ref_metrics.expected_return - shadow_metrics.expected_return)
        rel_ret = abs_ret / max(abs(shadow_metrics.expected_return), 1e-8)

    abs_sr: Optional[float] = None
    rel_sr: Optional[float] = None
    if ref_metrics.sharpe_ratio is not None and shadow_metrics.sharpe_ratio is not None:
        abs_sr = abs(ref_metrics.sharpe_ratio - shadow_metrics.sharpe_ratio)
        rel_sr = abs_sr / max(abs(shadow_metrics.sharpe_ratio), 1e-8)

    return ValidationDifferences(
        weight_differences=weight_diffs,
        max_weight_diff=max_w_diff,
        abs_diff_variance=abs_var,
        abs_diff_volatility=abs_vol,
        abs_diff_return=abs_ret,
        abs_diff_sharpe=abs_sr,
        rel_diff_variance=rel_var,
        rel_diff_volatility=rel_vol,
        rel_diff_return=rel_ret,
        rel_diff_sharpe=rel_sr,
    )


# ---------------------------------------------------------------------------
# Shadow Validation Operations
# ---------------------------------------------------------------------------

def validate_gmv(
    covariance: Union[pd.DataFrame, np.ndarray],
    reference_result: Optional[OptimizationResult] = None,
    expected_returns: Optional[Union[pd.Series, Sequence[float], np.ndarray]] = None,
    *,
    tolerances: Optional[ValidationTolerances] = None,
) -> GMVValidationResult:
    """Validate Global Minimum Variance (GMV) portfolio against PyPortfolioOpt shadow solver.

    Compares:
        - Portfolio weights (elementwise alignment and maximum absolute delta)
        - Annualized portfolio variance (w.T @ Sigma @ w)
        - Annualized portfolio volatility (sqrt(variance))
        - Annualized expected return (w.T @ mu, if expected_returns is provided)

    Parameters
    ----------
    covariance:
        Annualized covariance matrix (pd.DataFrame or square np.ndarray).
    reference_result:
        Optional production OptimizationResult from optimize_global_minimum_variance.
        If None, the reference optimizer is executed automatically.
    expected_returns:
        Optional annualized expected returns vector to compare expected return metrics.
    tolerances:
        Optional ValidationTolerances configuration (defaults to DEFAULT_VALIDATION_TOLERANCES).

    Returns
    -------
    GMVValidationResult
        Detailed comparison container with metric parity, weight deltas, and validation status.
    """
    tol = tolerances or DEFAULT_VALIDATION_TOLERANCES
    cov_df, asset_labels = _validate_and_normalize_covariance(covariance)

    exp_ret_series: Optional[pd.Series] = None
    if expected_returns is not None:
        exp_ret_series = _validate_and_normalize_expected_returns(expected_returns, asset_labels)

    warnings: List[str] = []
    errors: List[str] = []

    # 1. Resolve or execute reference optimizer
    ref_res: OptimizationResult
    if reference_result is not None:
        if not isinstance(reference_result, OptimizationResult):
            raise FinancialCalculationError(
                f"reference_result must be an OptimizationResult, got {type(reference_result).__name__}.",
                details={"type": type(reference_result).__name__},
            )
        if (
            not np.isfinite(reference_result.variance)
            or reference_result.variance < 0.0
            or not np.isfinite(reference_result.volatility)
            or reference_result.volatility < 0.0
        ):
            raise FinancialCalculationError(
                f"reference_result contains non-finite or negative variance/volatility: "
                f"var={reference_result.variance}, vol={reference_result.volatility}.",
                details={"variance": reference_result.variance, "volatility": reference_result.volatility},
            )
        ref_res = reference_result
    else:
        ref_res = optimize_global_minimum_variance(covariance=cov_df)

    # Check reference solver success
    if not ref_res.success or ref_res.fallback_executed:
        errors.append(
            f"Reference optimizer did not converge successfully or used fallback: "
            f"success={ref_res.success}, fallback_executed={ref_res.fallback_executed}, message={ref_res.message}."
        )
        return GMVValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            metric_valid=False,
            weight_valid=False,
            reference_metrics=None,
            shadow_metrics=None,
            differences=None,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "reference_optimizer"},
        )

    # Validate reference result weights (both supplied and generated)
    _validate_reference_weights(
        ref_res.weights,
        asset_labels,
        tol.weight_sum_tol,
        name="reference_result",
    )

    ref_w = ref_res.weights.loc[asset_labels].astype(float)
    ref_ret = float(ref_w.values @ exp_ret_series.values) if exp_ret_series is not None else None
    ref_metrics = PortfolioMetrics(
        weights=ref_w,
        variance=ref_res.variance,
        volatility=ref_res.volatility,
        expected_return=ref_ret,
    )

    # 2. Execute PyPortfolioOpt shadow optimization
    try:
        ef = EfficientFrontier(None, cov_df, weight_bounds=(0, 1))
        raw_shadow_w = ef.min_volatility()
        shadow_w = pd.Series(raw_shadow_w).loc[asset_labels].astype(float)
    except Exception as exc:
        errors.append(f"PyPortfolioOpt shadow GMV optimization failed: {type(exc).__name__}: {exc}")
        return GMVValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            metric_valid=False,
            weight_valid=False,
            reference_metrics=ref_metrics,
            shadow_metrics=None,
            differences=None,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "shadow_optimizer", "exception": str(exc)},
        )

    # Verify shadow weight constraints (hard gate)
    feasibility_errors = _check_shadow_weight_feasibility(shadow_w, asset_labels, tol)
    errors.extend(feasibility_errors)
    has_feasibility_violation = len(feasibility_errors) > 0

    shadow_var = float(shadow_w.values @ cov_df.values @ shadow_w.values)
    shadow_vol = np.sqrt(shadow_var)
    shadow_ret = float(shadow_w.values @ exp_ret_series.values) if exp_ret_series is not None else None
    shadow_metrics = PortfolioMetrics(
        weights=shadow_w,
        variance=shadow_var,
        volatility=shadow_vol,
        expected_return=shadow_ret,
    )

    # 3. Calculate differences
    diffs = _compute_validation_differences(ref_metrics, shadow_metrics, asset_labels)

    # 4. Determine validity
    metric_valid = (
        (diffs.abs_diff_variance <= tol.variance_tol or diffs.rel_diff_variance <= tol.relative_tol)
        and (diffs.abs_diff_volatility <= tol.volatility_tol or diffs.rel_diff_volatility <= tol.relative_tol)
    )
    if exp_ret_series is not None and diffs.abs_diff_return is not None and diffs.rel_diff_return is not None:
        metric_valid = metric_valid and (
            diffs.abs_diff_return <= tol.return_tol or diffs.rel_diff_return <= tol.relative_tol
        )

    weight_valid = (diffs.max_weight_diff <= tol.weight_tol) and not has_feasibility_violation

    if has_feasibility_violation or not metric_valid:
        status = ValidationStatus.FAILED
        is_valid = False
        if not metric_valid and not has_feasibility_violation:
            errors.append(
                f"Metric difference exceeds tolerance: var_diff={diffs.abs_diff_variance:.2e} (tol={tol.variance_tol:.2e}), "
                f"vol_diff={diffs.abs_diff_volatility:.2e} (tol={tol.volatility_tol:.2e})."
            )
    elif metric_valid and weight_valid:
        status = ValidationStatus.PASSED
        is_valid = True
    else:
        # Metric parity satisfied, but weights differ (e.g. collinear assets / multiple numerical optima)
        status = ValidationStatus.EQUIVALENT_OPTIMUM
        is_valid = True
        warnings.append(
            f"Metric parity verified (variance diff={diffs.abs_diff_variance:.2e} <= {tol.variance_tol:.2e}), "
            f"but max weight delta ({diffs.max_weight_diff:.2e}) > weight_tol ({tol.weight_tol:.2e}). "
            "Solvers identified numerically equivalent optimal allocations."
        )

    return GMVValidationResult(
        is_valid=is_valid,
        status=status,
        metric_valid=metric_valid,
        weight_valid=weight_valid,
        reference_metrics=ref_metrics,
        shadow_metrics=shadow_metrics,
        differences=diffs,
        tolerances=tol,
        warnings=warnings,
        errors=errors,
        metadata={
            "n_assets": len(asset_labels),
            "has_expected_returns": exp_ret_series is not None,
            "reference_iterations": ref_res.iterations,
        },
    )


def validate_tangency(
    expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
    covariance: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float,
    reference_result: Optional[TangencyOptimizationResult] = None,
    *,
    tolerances: Optional[ValidationTolerances] = None,
) -> TangencyValidationResult:
    """Validate Maximum Sharpe Ratio (Tangency) portfolio against PyPortfolioOpt shadow solver.

    Compares:
        - Portfolio weights (elementwise alignment and maximum absolute delta)
        - Expected annualized return (w.T @ mu)
        - Annualized volatility (sqrt(w.T @ Sigma @ w))
        - Annualized Sharpe ratio ((expected_return - risk_free_rate) / volatility)

    Parameters
    ----------
    expected_returns:
        Annualized expected returns vector.
    covariance:
        Annualized covariance matrix.
    risk_free_rate:
        Annualized risk-free rate used for Sharpe calculation.
    reference_result:
        Optional production TangencyOptimizationResult from optimize_maximum_sharpe_ratio.
        If None, the reference optimizer is executed automatically.
    tolerances:
        Optional ValidationTolerances configuration.

    Returns
    -------
    TangencyValidationResult
        Detailed comparison container with metric parity, weight deltas, and validation status.
    """
    tol = tolerances or DEFAULT_VALIDATION_TOLERANCES
    cov_df, asset_labels = _validate_and_normalize_covariance(covariance)
    exp_ret_series = _validate_and_normalize_expected_returns(expected_returns, asset_labels)
    rf = _validate_risk_free_rate(risk_free_rate)

    warnings: List[str] = []
    errors: List[str] = []

    # 1. Resolve or execute reference optimizer
    ref_res: TangencyOptimizationResult
    if reference_result is not None:
        if not isinstance(reference_result, TangencyOptimizationResult):
            raise FinancialCalculationError(
                f"reference_result must be a TangencyOptimizationResult, got {type(reference_result).__name__}.",
                details={"type": type(reference_result).__name__},
            )
        if (
            not np.isfinite(reference_result.variance)
            or reference_result.variance < 0.0
            or not np.isfinite(reference_result.volatility)
            or reference_result.volatility < 0.0
            or not np.isfinite(reference_result.expected_return)
            or not np.isfinite(reference_result.sharpe_ratio)
        ):
            raise FinancialCalculationError(
                f"reference_result contains non-finite metrics: "
                f"var={reference_result.variance}, vol={reference_result.volatility}, "
                f"ret={reference_result.expected_return}, sharpe={reference_result.sharpe_ratio}.",
                details={
                    "variance": reference_result.variance,
                    "volatility": reference_result.volatility,
                    "expected_return": reference_result.expected_return,
                    "sharpe_ratio": reference_result.sharpe_ratio,
                },
            )
        if abs(reference_result.risk_free_rate - rf) > 1e-6:
            raise FinancialCalculationError(
                f"Supplied risk_free_rate ({rf:.6f}) does not match reference_result.risk_free_rate ({reference_result.risk_free_rate:.6f}).",
                details={"supplied_rf": rf, "reference_rf": reference_result.risk_free_rate},
            )
        ref_res = reference_result
    else:
        ref_res = optimize_maximum_sharpe_ratio(
            expected_returns=exp_ret_series,
            covariance=cov_df,
            risk_free_rate=rf,
        )

    # Check reference solver success
    if not ref_res.success or ref_res.fallback_executed:
        errors.append(
            f"Reference tangency optimizer did not converge successfully or used fallback: "
            f"success={ref_res.success}, fallback_executed={ref_res.fallback_executed}, message={ref_res.message}."
        )
        return TangencyValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            metric_valid=False,
            weight_valid=False,
            risk_free_rate=rf,
            reference_metrics=None,
            shadow_metrics=None,
            differences=None,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "reference_optimizer"},
        )

    # Validate reference result weights (both supplied and generated)
    _validate_reference_weights(
        ref_res.weights,
        asset_labels,
        tol.weight_sum_tol,
        name="reference_result",
    )

    ref_w = ref_res.weights.loc[asset_labels].astype(float)
    ref_metrics = PortfolioMetrics(
        weights=ref_w,
        variance=ref_res.variance,
        volatility=ref_res.volatility,
        expected_return=ref_res.expected_return,
        sharpe_ratio=ref_res.sharpe_ratio,
    )

    # 2. Execute PyPortfolioOpt shadow optimization
    try:
        ef = EfficientFrontier(exp_ret_series, cov_df, weight_bounds=(0, 1))
        raw_shadow_w = ef.max_sharpe(risk_free_rate=rf)
        shadow_w = pd.Series(raw_shadow_w).loc[asset_labels].astype(float)
        shadow_ret, shadow_vol, shadow_sr = ef.portfolio_performance(risk_free_rate=rf)
        shadow_var = shadow_vol ** 2
    except Exception as exc:
        errors.append(f"PyPortfolioOpt shadow Tangency optimization failed: {type(exc).__name__}: {exc}")
        return TangencyValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            metric_valid=False,
            weight_valid=False,
            risk_free_rate=rf,
            reference_metrics=ref_metrics,
            shadow_metrics=None,
            differences=None,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "shadow_optimizer", "exception": str(exc)},
        )

    # Verify shadow weight constraints (hard gate)
    feasibility_errors = _check_shadow_weight_feasibility(shadow_w, asset_labels, tol)
    errors.extend(feasibility_errors)
    has_feasibility_violation = len(feasibility_errors) > 0

    shadow_metrics = PortfolioMetrics(
        weights=shadow_w,
        variance=shadow_var,
        volatility=shadow_vol,
        expected_return=shadow_ret,
        sharpe_ratio=shadow_sr,
    )

    # 3. Calculate differences
    diffs = _compute_validation_differences(ref_metrics, shadow_metrics, asset_labels)

    # 4. Determine validity
    metric_valid = (
        (diffs.abs_diff_variance <= tol.variance_tol or diffs.rel_diff_variance <= tol.relative_tol)
        and (diffs.abs_diff_volatility <= tol.volatility_tol or diffs.rel_diff_volatility <= tol.relative_tol)
        and (diffs.abs_diff_return is not None and (diffs.abs_diff_return <= tol.return_tol or (diffs.rel_diff_return or 0.0) <= tol.relative_tol))
        and (diffs.abs_diff_sharpe is not None and (diffs.abs_diff_sharpe <= tol.sharpe_tol or (diffs.rel_diff_sharpe or 0.0) <= tol.relative_tol))
    )

    weight_valid = (diffs.max_weight_diff <= tol.weight_tol) and not has_feasibility_violation

    if has_feasibility_violation or not metric_valid:
        status = ValidationStatus.FAILED
        is_valid = False
        if not metric_valid and not has_feasibility_violation:
            errors.append(
                f"Metric difference exceeds tolerance: ret_diff={diffs.abs_diff_return:.2e} (tol={tol.return_tol:.2e}), "
                f"vol_diff={diffs.abs_diff_volatility:.2e} (tol={tol.volatility_tol:.2e}), "
                f"sharpe_diff={diffs.abs_diff_sharpe:.2e} (tol={tol.sharpe_tol:.2e})."
            )
    elif metric_valid and weight_valid:
        status = ValidationStatus.PASSED
        is_valid = True
    else:
        status = ValidationStatus.EQUIVALENT_OPTIMUM
        is_valid = True
        warnings.append(
            f"Metric parity verified (Sharpe diff={diffs.abs_diff_sharpe:.2e} <= {tol.sharpe_tol:.2e}), "
            f"but max weight delta ({diffs.max_weight_diff:.2e}) > weight_tol ({tol.weight_tol:.2e}). "
            "Solvers identified numerically equivalent optimal allocations."
        )

    return TangencyValidationResult(
        is_valid=is_valid,
        status=status,
        metric_valid=metric_valid,
        weight_valid=weight_valid,
        risk_free_rate=rf,
        reference_metrics=ref_metrics,
        shadow_metrics=shadow_metrics,
        differences=diffs,
        tolerances=tol,
        warnings=warnings,
        errors=errors,
        metadata={
            "n_assets": len(asset_labels),
            "reference_iterations": ref_res.iterations,
        },
    )


def validate_efficient_frontier(
    expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
    covariance: Union[pd.DataFrame, np.ndarray],
    target_returns: Optional[Sequence[float]] = None,
    reference_result: Optional[EfficientFrontierResult] = None,
    *,
    tolerances: Optional[ValidationTolerances] = None,
) -> FrontierValidationResult:
    """Validate selected feasible target-return portfolios against PyPortfolioOpt shadow solver.

    Mathematical Scope:
        PyPortfolioOpt's `efficient_return` minimizes variance subject to `ret >= target_return`.
        Therefore, comparison is mathematically consistent and strictly valid for target returns
        on the efficient upper branch (target_return >= GMV return).
        For any target returns strictly below GMV return, PyPortfolioOpt solves the inequality
        returning GMV rather than the exact lower branch, which is explicitly noted and skipped
        or warned.

    Parameters
    ----------
    expected_returns:
        Annualized expected returns vector.
    covariance:
        Annualized covariance matrix.
    target_returns:
        Optional sequence of target returns to validate.
        If None, target returns are extracted from reference_result or generated.
    reference_result:
        Optional production EfficientFrontierResult from compute_efficient_frontier.
        If None, compute_efficient_frontier is executed automatically.
    tolerances:
        Optional ValidationTolerances configuration.

    Returns
    -------
    FrontierValidationResult
        Detailed container holding validation reports for all compared target returns.
    """
    tol = tolerances or DEFAULT_VALIDATION_TOLERANCES
    cov_df, asset_labels = _validate_and_normalize_covariance(covariance)
    exp_ret_series = _validate_and_normalize_expected_returns(expected_returns, asset_labels)

    warnings: List[str] = []
    errors: List[str] = []

    # 1. Resolve or execute reference optimizer
    ref_res: EfficientFrontierResult
    if reference_result is not None:
        if not isinstance(reference_result, EfficientFrontierResult):
            raise FinancialCalculationError(
                f"reference_result must be an EfficientFrontierResult, got {type(reference_result).__name__}.",
                details={"type": type(reference_result).__name__},
            )
        ref_res = reference_result
    else:
        ref_res = compute_efficient_frontier(
            expected_returns=exp_ret_series,
            covariance=cov_df,
            target_returns=target_returns,
        )

    if not ref_res.success:
        errors.append(f"Reference efficient frontier computation did not succeed: message={ref_res.message}.")
        return FrontierValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            validated_points=(),
            n_points_validated=0,
            n_points_passed=0,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "reference_frontier"},
        )

    # Extract target portfolios to validate
    # Note: Only compare successfully converged portfolios on the efficient branch (target >= GMV).
    # gmv_portfolio is an OptimizationResult (no expected_return field); derive it from weights @ mu.
    if ref_res.gmv_portfolio is None:
        errors.append(
            "Reference EfficientFrontierResult has no GMV portfolio; cannot classify efficient/inefficient branch."
        )
        return FrontierValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            validated_points=(),
            n_points_validated=0,
            n_points_passed=0,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "missing_gmv_reference"},
        )

    if not ref_res.gmv_portfolio.success or ref_res.gmv_portfolio.fallback_executed:
        errors.append(
            f"Reference GMV portfolio did not converge successfully or used fallback "
            f"(success={ref_res.gmv_portfolio.success}, fallback_executed={ref_res.gmv_portfolio.fallback_executed}); "
            f"cannot reliably classify frontier branch."
        )
        return FrontierValidationResult(
            is_valid=False,
            status=ValidationStatus.FAILED,
            validated_points=(),
            n_points_validated=0,
            n_points_passed=0,
            tolerances=tol,
            warnings=warnings,
            errors=errors,
            metadata={"solver_failure": "unreliable_gmv_reference"},
        )

    _validate_reference_weights(
        ref_res.gmv_portfolio.weights,
        asset_labels,
        tol.weight_sum_tol,
        name="reference GMV portfolio",
    )

    gmv_w = ref_res.gmv_portfolio.weights.loc[asset_labels].astype(float)
    gmv_ret = float(gmv_w.values @ exp_ret_series.values)

    candidate_portfolios: List[FrontierPortfolio] = []
    if target_returns is not None:
        target_set = set(float(t) for t in target_returns)
        for p in ref_res.portfolios:
            if any(abs(p.target_return - t) < 1e-7 for t in target_set):
                candidate_portfolios.append(p)
    else:
        # Default to efficient portfolios that converged successfully
        candidate_portfolios = [p for p in ref_res.portfolios if p.success and not p.fallback_executed]

    validated_points: List[FrontierPointValidation] = []
    n_passed = 0

    for port in candidate_portfolios:
        target_ret = port.target_return
        point_warnings: List[str] = []
        point_errors: List[str] = []

        # Check whether target is on the efficient branch
        if target_ret < gmv_ret - 1e-6:
            msg = (
                f"Target return {target_ret:.4f} is strictly below GMV return {gmv_ret:.4f} (inefficient branch). "
                "PyPortfolioOpt solves ret >= target, which returns GMV rather than the exact lower branch. "
                "Skipping shadow comparison for this point to prevent false mismatch."
            )
            warnings.append(msg)
            continue

        # Validate reference frontier portfolio weights (both supplied and generated)
        _validate_reference_weights(
            port.weights,
            asset_labels,
            tol.weight_sum_tol,
            name=f"reference frontier portfolio (target={target_ret:.4f})",
        )

        ref_w = port.weights.loc[asset_labels].astype(float)
        ref_metrics = PortfolioMetrics(
            weights=ref_w,
            variance=port.variance,
            volatility=port.volatility,
            expected_return=port.expected_return,
        )

        try:
            ef = EfficientFrontier(exp_ret_series, cov_df, weight_bounds=(0, 1))
            raw_shadow_w = ef.efficient_return(target_return=float(target_ret))
            shadow_w = pd.Series(raw_shadow_w).loc[asset_labels].astype(float)
            shadow_ret, shadow_vol, _ = ef.portfolio_performance()
            shadow_var = shadow_vol ** 2
        except Exception as exc:
            point_errors.append(f"PyPortfolioOpt efficient_return({target_ret:.4f}) failed: {type(exc).__name__}: {exc}")
            validated_points.append(
                FrontierPointValidation(
                    target_return=target_ret,
                    is_valid=False,
                    status=ValidationStatus.FAILED.value,
                    metric_valid=False,
                    weight_valid=False,
                    reference_metrics=ref_metrics,
                    shadow_metrics=None,
                    differences=None,
                    warnings=tuple(point_warnings),
                    errors=tuple(point_errors),
                )
            )
            continue

        # Verify shadow weight constraints (hard gate)
        pt_feasibility_errors = _check_shadow_weight_feasibility(shadow_w, asset_labels, tol)
        point_errors.extend(pt_feasibility_errors)
        pt_has_feasibility_violation = len(pt_feasibility_errors) > 0

        shadow_metrics = PortfolioMetrics(
            weights=shadow_w,
            variance=shadow_var,
            volatility=shadow_vol,
            expected_return=shadow_ret,
        )

        diffs = _compute_validation_differences(ref_metrics, shadow_metrics, asset_labels)

        metric_valid = (
            (diffs.abs_diff_variance <= tol.variance_tol or diffs.rel_diff_variance <= tol.relative_tol)
            and (diffs.abs_diff_volatility <= tol.volatility_tol or diffs.rel_diff_volatility <= tol.relative_tol)
            and (diffs.abs_diff_return is not None and (diffs.abs_diff_return <= tol.return_tol or (diffs.rel_diff_return or 0.0) <= tol.relative_tol))
        )
        weight_valid = (diffs.max_weight_diff <= tol.weight_tol) and not pt_has_feasibility_violation

        if pt_has_feasibility_violation or not metric_valid:
            status = ValidationStatus.FAILED
            is_pt_valid = False
            if not metric_valid and not pt_has_feasibility_violation:
                point_errors.append(
                    f"Metric mismatch at target {target_ret:.4f}: vol_diff={diffs.abs_diff_volatility:.2e}."
                )
        elif metric_valid and weight_valid:
            status = ValidationStatus.PASSED
            is_pt_valid = True
            n_passed += 1
        else:
            status = ValidationStatus.EQUIVALENT_OPTIMUM
            is_pt_valid = True
            n_passed += 1
            point_warnings.append(
                f"Equivalent optimum at target {target_ret:.4f}: max weight delta {diffs.max_weight_diff:.2e} > {tol.weight_tol:.2e}."
            )

        validated_points.append(
            FrontierPointValidation(
                target_return=target_ret,
                is_valid=is_pt_valid,
                status=status.value,
                metric_valid=metric_valid,
                weight_valid=weight_valid,
                reference_metrics=ref_metrics,
                shadow_metrics=shadow_metrics,
                differences=diffs,
                warnings=tuple(point_warnings),
                errors=tuple(point_errors),
            )
        )

    all_passed = len(validated_points) > 0 and all(p.is_valid for p in validated_points)
    overall_status = (
        ValidationStatus.PASSED
        if all_passed and all(p.status == ValidationStatus.PASSED.value for p in validated_points)
        else ValidationStatus.EQUIVALENT_OPTIMUM
        if all_passed
        else ValidationStatus.FAILED
    )

    return FrontierValidationResult(
        is_valid=all_passed,
        status=overall_status,
        validated_points=validated_points,
        n_points_validated=len(validated_points),
        n_points_passed=n_passed,
        tolerances=tol,
        warnings=warnings,
        errors=errors,
        metadata={
            "n_assets": len(asset_labels),
            "gmv_expected_return": gmv_ret,
        },
    )


# ---------------------------------------------------------------------------
# Object-Oriented Shadow Validator Class
# ---------------------------------------------------------------------------

class PyPortfolioOptShadowValidator:
    """Independent shadow validation engine powered by PyPortfolioOpt.

    Designed exclusively for test-suite and audit-grade verification of
    Quantamental AI's production deterministic optimization solvers.

    Usage
    -----
    >>> validator = PyPortfolioOptShadowValidator()
    >>> gmv_val = validator.validate_gmv(covariance)
    >>> tang_val = validator.validate_tangency(expected_returns, covariance, risk_free_rate)
    >>> ef_val = validator.validate_efficient_frontier(expected_returns, covariance)
    """

    def __init__(self, tolerances: Optional[ValidationTolerances] = None):
        self.tolerances = tolerances or DEFAULT_VALIDATION_TOLERANCES

    def validate_gmv(
        self,
        covariance: Union[pd.DataFrame, np.ndarray],
        reference_result: Optional[OptimizationResult] = None,
        expected_returns: Optional[Union[pd.Series, Sequence[float], np.ndarray]] = None,
    ) -> GMVValidationResult:
        """Validate GMV portfolio against shadow solver."""
        return validate_gmv(
            covariance=covariance,
            reference_result=reference_result,
            expected_returns=expected_returns,
            tolerances=self.tolerances,
        )

    def validate_tangency(
        self,
        expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
        covariance: Union[pd.DataFrame, np.ndarray],
        risk_free_rate: float,
        reference_result: Optional[TangencyOptimizationResult] = None,
    ) -> TangencyValidationResult:
        """Validate Tangency portfolio against shadow solver."""
        return validate_tangency(
            expected_returns=expected_returns,
            covariance=covariance,
            risk_free_rate=risk_free_rate,
            reference_result=reference_result,
            tolerances=self.tolerances,
        )

    def validate_efficient_frontier(
        self,
        expected_returns: Union[pd.Series, Sequence[float], np.ndarray],
        covariance: Union[pd.DataFrame, np.ndarray],
        target_returns: Optional[Sequence[float]] = None,
        reference_result: Optional[EfficientFrontierResult] = None,
    ) -> FrontierValidationResult:
        """Validate Efficient Frontier portfolios against shadow solver."""
        return validate_efficient_frontier(
            expected_returns=expected_returns,
            covariance=covariance,
            target_returns=target_returns,
            reference_result=reference_result,
            tolerances=self.tolerances,
        )


# Convenience aliases
ShadowValidator = PyPortfolioOptShadowValidator
