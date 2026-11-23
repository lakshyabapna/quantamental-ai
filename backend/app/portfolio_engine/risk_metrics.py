"""Quantamental AI - Downside & Tail Risk Analytics Engine.

Module A / Phase 2 Task 7: Deterministic, production-quality risk metrics engine
implementing Parametric Value-at-Risk (VaR), Historical VaR, Modified Cornish-Fisher
VaR, and Conditional Value-at-Risk (CVaR / Expected Shortfall).

Mathematical Formulations & Sign Conventions:
    Under standard financial risk conventions (FINANCIAL_METHODOLOGY.md §6):
    - Positive VaR and CVaR values represent POSITIVE LOSSES.
    - Negative VaR and CVaR values represent GAINS.
    - For confidence level c in (0, 1) (e.g. 0.95 or 0.99), alpha = 1 - c (e.g. 0.05 or 0.01).

1. Parametric VaR (Gaussian):
    VaR_alpha^{param} = - ( mu_p + z_alpha * sigma_p )
    where z_alpha = Phi^{-1}(alpha) (e.g. z_{0.05} = -1.64485..., z_{0.01} = -2.32635...).

2. Historical VaR (Non-Parametric):
    VaR_alpha^{hist} = - Quantile_alpha({ R_{p, 1}, ..., R_{p, T} })

3. Modified Cornish-Fisher VaR:
    Adjusts the standard normal quantile z_alpha for empirical skewness (S) and excess kurtosis (K):
    z_tilde_alpha = z_alpha + (1/6)*(z_alpha^2 - 1)*S + (1/24)*(z_alpha^3 - 3*z_alpha)*K - (1/36)*(2*z_alpha^3 - 5*z_alpha)*S^2
    VaR_alpha^{CF} = - ( mu_p + z_tilde_alpha * sigma_p )

4. Conditional Value-at-Risk (CVaR / Expected Shortfall):
    CVaR_alpha = - (1 / sum_{t=1}^T I(R_{p, t} <= -VaR_alpha)) * sum_{t: R_{p, t} <= -VaR_alpha} R_{p, t}
    where tail returns are all observations R_{p, t} <= Quantile_alpha(R_p).
    Guarantees CVaR_alpha >= VaR_alpha^{hist}.

Canonical design references:
    - FINANCIAL_METHODOLOGY.md §6.1, §6.2
    - PROJECT_PLAN.md Phase 2 Task 7
    - ARCHITECTURE.md §4.2
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
)

logger = logging.getLogger("quantamental.risk_metrics")

# ---------------------------------------------------------------------------
# Module Constants & Numerical Tolerances
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_LEVELS: Tuple[float, float] = (0.95, 0.99)
MIN_SAMPLE_SIZE_BASIC: int = 2          # Mean and standard deviation require T >= 2
MIN_SAMPLE_SIZE_SKEW: int = 3           # Sample skewness requires T >= 3
MIN_SAMPLE_SIZE_KURTOSIS: int = 4       # Sample excess kurtosis requires T >= 4
_ZERO_VOLATILITY_THRESHOLD: float = 1e-12

ALLOWED_QUANTILE_METHODS: frozenset[str] = frozenset({
    "linear",
    "lower",
    "higher",
    "midpoint",
    "nearest",
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "median_unbiased",
    "normal_unbiased",
})


# ---------------------------------------------------------------------------
# Result Containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CornishFisherExpansionResult:
    """Immutable diagnostic breakdown of a Cornish-Fisher quantile expansion.

    Attributes:
        confidence_level: Confidence level c in (0, 1) (e.g. 0.95 or 0.99).
        alpha: Lower tail significance level alpha = 1 - c.
        standard_z: Standard Gaussian quantile Phi^{-1}(alpha).
        adjusted_z: Skewness/kurtosis adjusted Cornish-Fisher quantile z_tilde_alpha.
        skewness: Empirical skewness S used in expansion.
        excess_kurtosis: Empirical excess kurtosis K used in expansion (Normal = 0).
        skewness_linear_term: (1/6) * (z^2 - 1) * S.
        kurtosis_term: (1/24) * (z^3 - 3z) * K.
        skewness_quadratic_term: - (1/36) * (2z^3 - 5z) * S^2.
    """

    confidence_level: float
    alpha: float
    standard_z: float
    adjusted_z: float
    skewness: float
    excess_kurtosis: float
    skewness_linear_term: float
    kurtosis_term: float
    skewness_quadratic_term: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert expansion diagnostics to JSON-serializable dictionary."""
        return {
            "confidence_level": self.confidence_level,
            "alpha": self.alpha,
            "standard_z": self.standard_z,
            "adjusted_z": self.adjusted_z,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "skewness_linear_term": self.skewness_linear_term,
            "kurtosis_term": self.kurtosis_term,
            "skewness_quadratic_term": self.skewness_quadratic_term,
        }


@dataclass(frozen=True, init=False)
class RiskMetricsResult:
    """Immutable container for comprehensive portfolio downside risk analytics.

    Defensively immutable: internal details dictionary is deep-copied on construction
    and deep-copied on property access.

    Sign Convention:
        Positive values represent potential losses (e.g. 0.05 = 5.0% loss).
        Negative values represent gains (e.g. -0.02 = 2.0% gain).

    Attributes:
        parametric_var_95: Parametric Gaussian VaR at 95% confidence (alpha=0.05).
        parametric_var_99: Parametric Gaussian VaR at 99% confidence (alpha=0.01).
        historical_var_95: Historical non-parametric VaR at 95% confidence (alpha=0.05).
        historical_var_99: Historical non-parametric VaR at 99% confidence (alpha=0.01).
        cornish_fisher_var_95: Cornish-Fisher VaR at 95% confidence (alpha=0.05).
        cornish_fisher_var_99: Cornish-Fisher VaR at 99% confidence (alpha=0.01).
        cvar_95: Conditional Value-at-Risk (Expected Shortfall) at 95% confidence.
        cvar_99: Conditional Value-at-Risk (Expected Shortfall) at 99% confidence.
        mean_return: Empirical mean of the return series.
        volatility: Empirical sample standard deviation (ddof=1).
        skewness: Empirical sample skewness S.
        excess_kurtosis: Empirical sample excess kurtosis K (Normal = 0).
        sample_size: Number of observations T in the validated return series.
        details: Diagnostic dictionary containing quantile expansions and metadata.
    """

    parametric_var_95: float
    parametric_var_99: float
    historical_var_95: float
    historical_var_99: float
    cornish_fisher_var_95: float
    cornish_fisher_var_99: float
    cvar_95: float
    cvar_99: float
    mean_return: float
    volatility: float
    skewness: float
    excess_kurtosis: float
    sample_size: int
    details: Dict[str, Any]
    _details: Dict[str, Any] = field(repr=False)

    def __init__(
        self,
        parametric_var_95: float,
        parametric_var_99: float,
        historical_var_95: float,
        historical_var_99: float,
        cornish_fisher_var_95: float,
        cornish_fisher_var_99: float,
        cvar_95: float,
        cvar_99: float,
        mean_return: float,
        volatility: float,
        skewness: float,
        excess_kurtosis: float,
        sample_size: int,
        details: Optional[Dict[str, Any]] = None,
    ):
        object.__setattr__(self, "parametric_var_95", float(parametric_var_95))
        object.__setattr__(self, "parametric_var_99", float(parametric_var_99))
        object.__setattr__(self, "historical_var_95", float(historical_var_95))
        object.__setattr__(self, "historical_var_99", float(historical_var_99))
        object.__setattr__(self, "cornish_fisher_var_95", float(cornish_fisher_var_95))
        object.__setattr__(self, "cornish_fisher_var_99", float(cornish_fisher_var_99))
        object.__setattr__(self, "cvar_95", float(cvar_95))
        object.__setattr__(self, "cvar_99", float(cvar_99))
        object.__setattr__(self, "mean_return", float(mean_return))
        object.__setattr__(self, "volatility", float(volatility))
        object.__setattr__(self, "skewness", float(skewness))
        object.__setattr__(self, "excess_kurtosis", float(excess_kurtosis))
        object.__setattr__(self, "sample_size", int(sample_size))
        object.__setattr__(self, "_details", copy.deepcopy(details or {}))

    @property
    def details(self) -> Dict[str, Any]:
        """Return a defensive deep copy of the diagnostic details dictionary."""
        return copy.deepcopy(self._details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to JSON-serializable dictionary."""
        return {
            "parametric_var_95": self.parametric_var_95,
            "parametric_var_99": self.parametric_var_99,
            "historical_var_95": self.historical_var_95,
            "historical_var_99": self.historical_var_99,
            "cornish_fisher_var_95": self.cornish_fisher_var_95,
            "cornish_fisher_var_99": self.cornish_fisher_var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "sample_size": self.sample_size,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Input Validation & Normalization Helpers
# ---------------------------------------------------------------------------

def _validate_and_normalize_returns(
    returns: object,
    *,
    min_observations: int = MIN_SAMPLE_SIZE_BASIC,
) -> np.ndarray:
    """Validate return series finiteness, real numeric type, and minimum sample size.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio or asset returns (pd.Series, np.ndarray, list).
    min_observations:
        Minimum number of observations required (default 2).

    Returns
    -------
    np.ndarray
        1-D contiguous float64 array of clean returns.

    Raises
    ------
    FinancialCalculationError
        If returns contain boolean, complex, non-numeric, NaN, infinite values,
        or multidimensional data.
    InsufficientDataError
        If number of observations is strictly less than min_observations.
    """
    if returns is None:
        raise FinancialCalculationError(
            "returns input cannot be None.",
            details={"type": "NoneType"},
        )

    # Reject boolean or complex containers upfront
    if isinstance(returns, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"returns input must be a numeric sequence, got bool ({returns!r}).",
            details={"type": "bool"},
        )

    if isinstance(returns, pd.Series):
        if returns.dtype == bool or np.issubdtype(returns.dtype, np.bool_):
            raise FinancialCalculationError(
                "returns Series contains boolean data.",
                details={"dtype": str(returns.dtype)},
            )
        if np.issubdtype(returns.dtype, np.complexfloating):
            raise FinancialCalculationError(
                "returns Series contains complex numbers.",
                details={"dtype": str(returns.dtype)},
            )
        arr = returns.to_numpy()
    elif isinstance(returns, np.ndarray):
        if arr_dtype := returns.dtype:
            if arr_dtype == bool or np.issubdtype(arr_dtype, np.bool_):
                raise FinancialCalculationError(
                    "returns array contains boolean data.",
                    details={"dtype": str(arr_dtype)},
                )
            if np.issubdtype(arr_dtype, np.complexfloating):
                raise FinancialCalculationError(
                    "returns array contains complex numbers.",
                    details={"dtype": str(arr_dtype)},
                )
        arr = returns
    elif isinstance(returns, (list, tuple)):
        # Inspect elements for booleans and non-numerics
        for idx, val in enumerate(returns):
            if isinstance(val, (bool, np.bool_)):
                raise FinancialCalculationError(
                    f"returns contains boolean value at index {idx} ({val!r}).",
                    details={"index": idx, "value": val},
                )
            if isinstance(val, (complex, np.complexfloating)):
                raise FinancialCalculationError(
                    f"returns contains complex number at index {idx} ({val!r}).",
                    details={"index": idx, "value": str(val)},
                )
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise FinancialCalculationError(
                    f"returns contains non-numeric value at index {idx} ({type(val).__name__}).",
                    details={"index": idx, "type": type(val).__name__},
                )
        arr = np.asarray(returns)
    else:
        raise FinancialCalculationError(
            f"returns must be a pandas Series, numpy array, list, or tuple, got {type(returns).__name__}.",
            details={"type": type(returns).__name__},
        )

    if arr.ndim != 1:
        raise FinancialCalculationError(
            f"returns sequence must be 1-dimensional, got shape {arr.shape}.",
            details={"shape": list(arr.shape)},
        )

    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        raise FinancialCalculationError(
            f"returns array contains non-numeric data with dtype {arr.dtype}.",
            details={"dtype": str(arr.dtype)},
        )

    # Sample size check
    n_obs = len(arr)
    if n_obs < min_observations:
        raise InsufficientDataError(
            f"Insufficient return observations: got {n_obs}, required minimum {min_observations}.",
            details={"sample_size": n_obs, "min_required": min_observations},
        )

    # Finiteness check
    try:
        arr_f64 = arr.astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise FinancialCalculationError(
            f"Could not convert returns to float64: {exc}.",
            details={"error": str(exc)},
        ) from exc

    if not np.all(np.isfinite(arr_f64)):
        nan_count = int(np.sum(np.isnan(arr_f64)))
        inf_count = int(np.sum(np.isinf(arr_f64)))
        raise FinancialCalculationError(
            f"returns contains non-finite values (NaN count: {nan_count}, Inf count: {inf_count}).",
            details={"nan_count": nan_count, "inf_count": inf_count},
        )

    return arr_f64


def _validate_confidence_level(confidence_level: object) -> float:
    """Validate that confidence level is a real numeric float strictly in (0.0, 1.0).

    Parameters
    ----------
    confidence_level:
        Confidence level c (e.g. 0.95 or 0.99).

    Returns
    -------
    float
        Validated confidence level.

    Raises
    ------
    FinancialCalculationError
        If confidence_level is non-numeric, boolean, non-finite, or outside (0, 1).
    """
    if isinstance(confidence_level, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"confidence_level must be a numeric float, got bool ({confidence_level!r}).",
            details={"type": "bool", "value": confidence_level},
        )
    if not isinstance(confidence_level, (int, float, np.integer, np.floating)):
        raise FinancialCalculationError(
            f"confidence_level must be a numeric float, got {type(confidence_level).__name__}.",
            details={"type": type(confidence_level).__name__},
        )

    c = float(confidence_level)
    if not np.isfinite(c):
        raise FinancialCalculationError(
            f"confidence_level must be finite, got {confidence_level!r}.",
            details={"value": confidence_level},
        )

    if c <= 0.0 or c >= 1.0:
        raise FinancialCalculationError(
            f"confidence_level must be strictly within (0.0, 1.0), got {c}.",
            details={"confidence_level": c},
        )

    return c


def _validate_confidence_levels(confidence_levels: object) -> Tuple[float, ...]:
    """Validate that confidence_levels is a non-empty sequence of real floats in (0.0, 1.0).

    Parameters
    ----------
    confidence_levels:
        Sequence of confidence levels c (e.g. (0.95, 0.99)).

    Returns
    -------
    Tuple[float, ...]
        Tuple of validated confidence levels.

    Raises
    ------
    FinancialCalculationError
        If confidence_levels is empty, not a sequence, or contains invalid elements.
    """
    if confidence_levels is None or isinstance(confidence_levels, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"confidence_levels must be a sequence of floats in (0, 1), got {type(confidence_levels).__name__}.",
            details={"type": type(confidence_levels).__name__},
        )
    if not isinstance(confidence_levels, (list, tuple, np.ndarray, Sequence)) or isinstance(confidence_levels, str):
        raise FinancialCalculationError(
            f"confidence_levels must be a sequence, got {type(confidence_levels).__name__}.",
            details={"type": type(confidence_levels).__name__},
        )
    if len(confidence_levels) == 0:
        raise FinancialCalculationError(
            "confidence_levels sequence cannot be empty.",
            details={"len": 0},
        )
    validated: List[float] = []
    seen: set[float] = set()
    for c in confidence_levels:
        val = _validate_confidence_level(c)
        if val in seen or any(abs(val - existing) < 1e-12 for existing in validated):
            raise FinancialCalculationError(
                f"Duplicate confidence level detected: {val}. All confidence levels must be unique.",
                details={"duplicate": val},
            )
        seen.add(val)
        validated.append(val)
    return tuple(validated)


def _format_confidence_key(c: float) -> str:
    """Format confidence level into a deterministic, human-readable dictionary key.

    Only genuinely exact whole-percentage levels (where c == k / 100.0 exactly)
    receive canonical keys such as '90%', '95%', and '99%'. Fractional, custom,
    or near-canonical levels preserve full decimal precision without trailing zeros
    (e.g. 0.975 -> '97.5%', 0.950000000000001 -> '95.0000000000001%').
    """
    k_int = int(round(c * 100.0))
    if c == k_int / 100.0:
        return f"{k_int}%"
    pct = c * 100.0
    formatted = str(pct)
    return f"{formatted}%"


def _validate_bias(bias: object) -> bool:
    """Validate that bias is an actual boolean without silent coercion.

    Parameters
    ----------
    bias:
        Boolean flag (True or False).

    Returns
    -------
    bool
        Validated boolean value.

    Raises
    ------
    FinancialCalculationError
        If bias is not an instance of bool.
    """
    if not isinstance(bias, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"bias must be a boolean (True or False), got {type(bias).__name__} ({bias!r}).",
            details={"type": type(bias).__name__, "value": str(bias)},
        )
    return bool(bias)


def _validate_quantile_method(method: object) -> str:
    """Validate that quantile interpolation method is an allowed string.

    Parameters
    ----------
    method:
        Quantile method string (e.g. 'linear').

    Returns
    -------
    str
        Validated lowercased method string.

    Raises
    ------
    FinancialCalculationError
        If method is non-string or not in ALLOWED_QUANTILE_METHODS.
    """
    if isinstance(method, (bool, np.bool_)):
        raise FinancialCalculationError(
            f"quantile method must be a string, got bool ({method!r}).",
            details={"type": "bool", "value": method},
        )
    if not isinstance(method, str):
        raise FinancialCalculationError(
            f"quantile method must be a string, got {type(method).__name__} ({method!r}).",
            details={"type": type(method).__name__, "value": str(method)},
        )
    m_clean = method.lower().strip()
    if m_clean not in ALLOWED_QUANTILE_METHODS:
        raise FinancialCalculationError(
            f"Unsupported quantile method {method!r}. Must be one of: {sorted(ALLOWED_QUANTILE_METHODS)}.",
            details={"method": method, "allowed_methods": sorted(ALLOWED_QUANTILE_METHODS)},
        )
    return m_clean


# ---------------------------------------------------------------------------
# Individual Metric Calculators
# ---------------------------------------------------------------------------

def calculate_parametric_var(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_level: float = 0.95,
) -> float:
    """Calculate Parametric Value-at-Risk under a Gaussian assumption.

    Formula (FINANCIAL_METHODOLOGY.md §6.1.1):
        VaR_alpha^{param} = - ( mu_p + z_alpha * sigma_p )
        where alpha = 1 - confidence_level, z_alpha = Phi^{-1}(alpha),
        mu_p is the sample mean, and sigma_p is the sample standard deviation (ddof=1).

    Sign Convention:
        Positive VaR represents a potential loss (e.g. +0.045 = 4.5% loss).
        Negative VaR represents a potential gain.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns.
    confidence_level:
        Confidence level c in (0, 1) (default 0.95).

    Returns
    -------
    float
        Annualized or single-period Parametric VaR matching the frequency of returns.

    Raises
    ------
    InsufficientDataError
        If fewer than 2 observations are provided.
    FinancialCalculationError
        If returns contain NaN/Inf or confidence_level is invalid.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_BASIC)
    c = _validate_confidence_level(confidence_level)
    alpha = 1.0 - c

    mu_p = float(np.mean(arr))
    sigma_p = float(np.std(arr, ddof=1))

    # Standard normal quantile
    z_alpha = float(stats.norm.ppf(alpha))

    # If distribution is degenerate (zero volatility), VaR collapses strictly to -mu_p
    if sigma_p <= _ZERO_VOLATILITY_THRESHOLD:
        return -mu_p

    var_val = -(mu_p + z_alpha * sigma_p)
    return float(var_val)


def calculate_historical_var(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_level: float = 0.95,
    method: str = "linear",
) -> float:
    """Calculate Historical Value-at-Risk (non-parametric empirical quantile).

    Formula (FINANCIAL_METHODOLOGY.md §6.1.2):
        VaR_alpha^{hist} = - Quantile_alpha({ R_{p, 1}, R_{p, 2}, ..., R_{p, T} })
        where alpha = 1 - confidence_level.

    Sign Convention:
        Positive VaR represents a potential loss (e.g. +0.05 = 5.0% loss).
        Negative VaR represents a potential gain.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns.
    confidence_level:
        Confidence level c in (0, 1) (default 0.95).
    method:
        Quantile estimation method supported by numpy (default 'linear').

    Returns
    -------
    float
        Historical VaR matching the frequency of returns.

    Raises
    ------
    InsufficientDataError
        If fewer than 2 observations are provided.
    FinancialCalculationError
        If returns contain NaN/Inf, confidence_level is invalid, or method is unsupported.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_BASIC)
    c = _validate_confidence_level(confidence_level)
    m = _validate_quantile_method(method)
    alpha = 1.0 - c

    try:
        q_alpha = float(np.quantile(arr, alpha, method=m))
    except (ValueError, TypeError) as exc:
        raise FinancialCalculationError(
            f"Invalid quantile calculation with method={method!r}: {exc}.",
            details={"method": method, "error": str(exc)},
        ) from exc

    var_val = -q_alpha
    return float(var_val)


def calculate_cornish_fisher_expansion(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_level: float = 0.95,
    bias: bool = False,
) -> CornishFisherExpansionResult:
    """Compute the detailed Cornish-Fisher expansion terms and adjusted quantile.

    Formula (FINANCIAL_METHODOLOGY.md §6.1.3):
        z_tilde = z + (1/6)*(z^2 - 1)*S + (1/24)*(z^3 - 3z)*K - (1/36)*(2z^3 - 5z)*S^2
        where z = Phi^{-1}(alpha), S is skewness, and K is excess kurtosis.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns.
    confidence_level:
        Confidence level c in (0, 1) (default 0.95).
    bias:
        Whether to use biased sample moments (default False = sample-unbiased).

    Returns
    -------
    CornishFisherExpansionResult
        Detailed breakdown of standard z, adjusted z, and individual expansion terms.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_KURTOSIS)
    c = _validate_confidence_level(confidence_level)
    b = _validate_bias(bias)
    alpha = 1.0 - c

    z = float(stats.norm.ppf(alpha))
    sigma_p = float(np.std(arr, ddof=1))

    if sigma_p <= _ZERO_VOLATILITY_THRESHOLD:
        s = 0.0
        k = 0.0
    else:
        s = float(stats.skew(arr, bias=b))
        k = float(stats.kurtosis(arr, bias=b, fisher=True))

    # Terms of the Cornish-Fisher expansion
    term1 = (1.0 / 6.0) * (z**2 - 1.0) * s
    term2 = (1.0 / 24.0) * (z**3 - 3.0 * z) * k
    term3 = -(1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)

    z_tilde = z + term1 + term2 + term3

    return CornishFisherExpansionResult(
        confidence_level=c,
        alpha=alpha,
        standard_z=z,
        adjusted_z=float(z_tilde),
        skewness=s,
        excess_kurtosis=k,
        skewness_linear_term=float(term1),
        kurtosis_term=float(term2),
        skewness_quadratic_term=float(term3),
    )


def calculate_cornish_fisher_var(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_level: float = 0.95,
    bias: bool = False,
) -> float:
    """Calculate Modified Cornish-Fisher Value-at-Risk adjusted for skewness and kurtosis.

    Formula (FINANCIAL_METHODOLOGY.md §6.1.3):
        z_tilde = z + (1/6)*(z^2 - 1)*S + (1/24)*(z^3 - 3z)*K - (1/36)*(2z^3 - 5z)*S^2
        VaR_alpha^{CF} = - ( mu_p + z_tilde * sigma_p )

    Sign Convention:
        Positive VaR represents a potential loss (e.g. +0.052 = 5.2% loss).
        Negative VaR represents a potential gain.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns (requires T >= 4).
    confidence_level:
        Confidence level c in (0, 1) (default 0.95).
    bias:
        Whether to use biased sample moments (default False = sample-unbiased).

    Returns
    -------
    float
        Cornish-Fisher VaR matching the frequency of returns.

    Raises
    ------
    InsufficientDataError
        If fewer than 4 observations are provided.
    FinancialCalculationError
        If returns contain NaN/Inf or confidence_level is invalid.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_KURTOSIS)
    b = _validate_bias(bias)
    cf_res = calculate_cornish_fisher_expansion(arr, confidence_level=confidence_level, bias=b)

    mu_p = float(np.mean(arr))
    sigma_p = float(np.std(arr, ddof=1))

    if sigma_p <= _ZERO_VOLATILITY_THRESHOLD:
        return -mu_p

    var_cf = -(mu_p + cf_res.adjusted_z * sigma_p)
    return float(var_cf)


def calculate_cvar(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_level: float = 0.95,
    var_threshold: Optional[float] = None,
    method: str = "linear",
) -> float:
    """Calculate Conditional Value-at-Risk (CVaR / Expected Shortfall).

    Formula (FINANCIAL_METHODOLOGY.md §6.2):
        CVaR_alpha = - (1 / |T_tail|) * sum_{t in T_tail} R_{p, t}
        where T_tail = { t | R_{p, t} <= -VaR_alpha }.

    When var_threshold is None (canonical default), -VaR_alpha is the empirical
    alpha-quantile: -VaR_alpha = Quantile_alpha(R_p).
    Consequently, CVaR is the negative of the expected return in the lower alpha tail.

    Sign Convention:
        Positive CVaR represents a potential loss (e.g. +0.07 = 7.0% tail loss).
        Negative CVaR represents a gain.

    Mathematical Invariant:
        When var_threshold is None, CVaR_alpha >= VaR_alpha^{hist}.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns.
    confidence_level:
        Confidence level c in (0, 1) (default 0.95).
    var_threshold:
        Optional precomputed VaR threshold (in positive loss units). If provided,
        the tail threshold return is -var_threshold. If None, uses Historical VaR.
    method:
        Quantile estimation method for Historical VaR threshold (default 'linear').

    Returns
    -------
    float
        Conditional Value-at-Risk (Expected Shortfall).

    Raises
    ------
    InsufficientDataError
        If fewer than 2 observations are provided.
    FinancialCalculationError
        If no returns fall below the specified threshold or inputs/method are invalid.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_BASIC)
    c = _validate_confidence_level(confidence_level)
    m = _validate_quantile_method(method)
    alpha = 1.0 - c

    if var_threshold is not None:
        if isinstance(var_threshold, (bool, np.bool_)):
            raise FinancialCalculationError(
                f"var_threshold must be a numeric float, got bool ({var_threshold!r}).",
                details={"type": "bool"},
            )
        if not isinstance(var_threshold, (int, float, np.integer, np.floating)):
            raise FinancialCalculationError(
                f"var_threshold must be a numeric float, got {type(var_threshold).__name__}.",
                details={"type": type(var_threshold).__name__},
            )
        v_thresh = float(var_threshold)
        if not np.isfinite(v_thresh):
            raise FinancialCalculationError(
                f"var_threshold must be finite, got {var_threshold!r}.",
                details={"var_threshold": var_threshold},
            )
        threshold_return = -v_thresh
    else:
        # Canonical default: empirical alpha-quantile
        try:
            threshold_return = float(np.quantile(arr, alpha, method=m))
        except (ValueError, TypeError) as exc:
            raise FinancialCalculationError(
                f"Invalid quantile calculation with method={method!r}: {exc}.",
                details={"method": method, "error": str(exc)},
            ) from exc

    # Identify tail observations strictly under the canonical condition R_{p, t} <= threshold_return
    tail_mask = arr <= threshold_return
    tail_returns = arr[tail_mask]

    if len(tail_returns) == 0:
        min_ret = float(np.min(arr))
        raise FinancialCalculationError(
            f"No return observations fell in the tail below threshold {threshold_return:.6f} (min return={min_ret:.6f}).",
            details={"threshold_return": threshold_return, "min_return": min_ret},
        )

    expected_shortfall = -float(np.mean(tail_returns))
    return float(expected_shortfall)


# ---------------------------------------------------------------------------
# Comprehensive Risk Metrics Calculator & Engine
# ---------------------------------------------------------------------------

def calculate_risk_metrics(
    returns: Union[pd.Series, np.ndarray, Sequence[float]],
    *,
    confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    bias: bool = False,
    method: str = "linear",
) -> RiskMetricsResult:
    """Calculate the complete suite of downside risk metrics.

    Computes:
        - Parametric VaR (95% and 99%)
        - Historical VaR (95% and 99%)
        - Cornish-Fisher VaR (95% and 99%)
        - Conditional Value-at-Risk / Expected Shortfall (95% and 99%)
        - Summary distribution statistics (mean, volatility, skewness, excess kurtosis)

    Sign Convention:
        Positive values represent potential losses; negative values represent gains.

    Notes:
        RiskMetricsResult provides fixed canonical top-level fields for the project's
        standard 95% and 99% confidence levels (e.g. parametric_var_95, parametric_var_99).
        When custom confidence levels are supplied in confidence_levels, all requested
        levels are evaluated and recorded in details['by_confidence_level'], while the
        canonical 95% and 99% fields remain populated for standard reporting compatibility.

    Parameters
    ----------
    returns:
        Sequence of historical portfolio returns (requires T >= 4).
    confidence_levels:
        Sequence of confidence levels c in (0, 1) (default (0.95, 0.99)).
    bias:
        Whether to use biased sample skewness and kurtosis (default False).
    method:
        Quantile interpolation method for Historical VaR and CVaR (default 'linear').

    Returns
    -------
    RiskMetricsResult
        Immutable result container with all downside risk metrics and diagnostics.

    Raises
    ------
    InsufficientDataError
        If sample size is strictly less than 4 observations.
    FinancialCalculationError
        If returns contain non-finite, complex, or non-numeric elements, or if
        confidence_levels, bias, or method are invalid.
    """
    arr = _validate_and_normalize_returns(returns, min_observations=MIN_SAMPLE_SIZE_KURTOSIS)
    conf_tuple = _validate_confidence_levels(confidence_levels)
    b = _validate_bias(bias)
    m = _validate_quantile_method(method)
    n_obs = len(arr)

    mean_ret = float(np.mean(arr))
    vol = float(np.std(arr, ddof=1))

    if vol <= _ZERO_VOLATILITY_THRESHOLD:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(arr, bias=b))
        kurt_val = float(stats.kurtosis(arr, bias=b, fisher=True))

    # Evaluate metrics across all requested confidence levels
    by_conf_level: Dict[str, Dict[str, Any]] = {}
    for c in conf_tuple:
        pvar_c = calculate_parametric_var(arr, confidence_level=c)
        hvar_c = calculate_historical_var(arr, confidence_level=c, method=m)
        cf_res_c = calculate_cornish_fisher_expansion(arr, confidence_level=c, bias=b)
        cfvar_c = -(mean_ret + cf_res_c.adjusted_z * vol) if vol > _ZERO_VOLATILITY_THRESHOLD else -mean_ret
        cvar_c = calculate_cvar(arr, confidence_level=c, method=m)

        c_key = _format_confidence_key(c)
        if c_key in by_conf_level:
            raise FinancialCalculationError(
                f"Confidence level key collision detected: {c_key!r} for level {c}. "
                "Distinct confidence levels must have distinct formatted keys.",
                details={"key": c_key, "confidence_level": c},
            )
        by_conf_level[c_key] = {
            "confidence_level": c,
            "parametric_var": float(pvar_c),
            "historical_var": float(hvar_c),
            "cornish_fisher_var": float(cfvar_c),
            "cvar": float(cvar_c),
            "cornish_fisher_expansion": cf_res_c.to_dict(),
        }

    # Canonical 95% and 99% metrics
    if "95%" in by_conf_level:
        pvar_95 = by_conf_level["95%"]["parametric_var"]
        hvar_95 = by_conf_level["95%"]["historical_var"]
        cfvar_95 = by_conf_level["95%"]["cornish_fisher_var"]
        cvar_95 = by_conf_level["95%"]["cvar"]
        cf_res_95_dict = by_conf_level["95%"]["cornish_fisher_expansion"]
    else:
        pvar_95 = calculate_parametric_var(arr, confidence_level=0.95)
        hvar_95 = calculate_historical_var(arr, confidence_level=0.95, method=m)
        cf_res_95 = calculate_cornish_fisher_expansion(arr, confidence_level=0.95, bias=b)
        cfvar_95 = -(mean_ret + cf_res_95.adjusted_z * vol) if vol > _ZERO_VOLATILITY_THRESHOLD else -mean_ret
        cvar_95 = calculate_cvar(arr, confidence_level=0.95, method=m)
        cf_res_95_dict = cf_res_95.to_dict()

    if "99%" in by_conf_level:
        pvar_99 = by_conf_level["99%"]["parametric_var"]
        hvar_99 = by_conf_level["99%"]["historical_var"]
        cfvar_99 = by_conf_level["99%"]["cornish_fisher_var"]
        cvar_99 = by_conf_level["99%"]["cvar"]
        cf_res_99_dict = by_conf_level["99%"]["cornish_fisher_expansion"]
    else:
        pvar_99 = calculate_parametric_var(arr, confidence_level=0.99)
        hvar_99 = calculate_historical_var(arr, confidence_level=0.99, method=m)
        cf_res_99 = calculate_cornish_fisher_expansion(arr, confidence_level=0.99, bias=b)
        cfvar_99 = -(mean_ret + cf_res_99.adjusted_z * vol) if vol > _ZERO_VOLATILITY_THRESHOLD else -mean_ret
        cvar_99 = calculate_cvar(arr, confidence_level=0.99, method=m)
        cf_res_99_dict = cf_res_99.to_dict()

    details: Dict[str, Any] = {
        "quantile_method": m,
        "sample_bias": b,
        "confidence_levels_requested": list(conf_tuple),
        "by_confidence_level": by_conf_level,
        "is_degenerate": vol <= _ZERO_VOLATILITY_THRESHOLD,
        "cornish_fisher_95": cf_res_95_dict,
        "cornish_fisher_99": cf_res_99_dict,
        "sign_convention": "positive_represents_loss_negative_represents_gain",
    }

    return RiskMetricsResult(
        parametric_var_95=float(pvar_95),
        parametric_var_99=float(pvar_99),
        historical_var_95=float(hvar_95),
        historical_var_99=float(hvar_99),
        cornish_fisher_var_95=float(cfvar_95),
        cornish_fisher_var_99=float(cfvar_99),
        cvar_95=float(cvar_95),
        cvar_99=float(cvar_99),
        mean_return=mean_ret,
        volatility=vol,
        skewness=skew_val,
        excess_kurtosis=kurt_val,
        sample_size=n_obs,
        details=details,
    )


class RiskMetricsEngine:
    """Object-oriented engine for downside and tail risk analysis.

    Parameters
    ----------
    bias:
        Whether to calculate biased skewness and kurtosis (default False).
    method:
        Quantile interpolation method (default 'linear').
    """

    def __init__(self, *, bias: bool = False, method: str = "linear"):
        self.bias = _validate_bias(bias)
        self.method = _validate_quantile_method(method)

    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray, Sequence[float]],
        confidence_levels: Sequence[float] = DEFAULT_CONFIDENCE_LEVELS,
    ) -> RiskMetricsResult:
        """Calculate complete risk metrics for the supplied return series."""
        return calculate_risk_metrics(
            returns,
            confidence_levels=confidence_levels,
            bias=self.bias,
            method=self.method,
        )

    def parametric_var(
        self,
        returns: Union[pd.Series, np.ndarray, Sequence[float]],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Parametric VaR."""
        return calculate_parametric_var(returns, confidence_level=confidence_level)

    def historical_var(
        self,
        returns: Union[pd.Series, np.ndarray, Sequence[float]],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Historical VaR."""
        return calculate_historical_var(
            returns, confidence_level=confidence_level, method=self.method
        )

    def cornish_fisher_var(
        self,
        returns: Union[pd.Series, np.ndarray, Sequence[float]],
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Cornish-Fisher VaR."""
        return calculate_cornish_fisher_var(
            returns, confidence_level=confidence_level, bias=self.bias
        )

    def cvar(
        self,
        returns: Union[pd.Series, np.ndarray, Sequence[float]],
        confidence_level: float = 0.95,
        var_threshold: Optional[float] = None,
    ) -> float:
        """Calculate Conditional Value-at-Risk."""
        return calculate_cvar(
            returns,
            confidence_level=confidence_level,
            var_threshold=var_threshold,
            method=self.method,
        )
