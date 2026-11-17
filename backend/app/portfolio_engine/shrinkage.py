"""Quantamental AI - Ledoit-Wolf Shrinkage Covariance Estimator.

Implements Phase 2.2, Step 2: Ledoit-Wolf shrinkage covariance estimation
using scikit-learn's analytically optimal implementation.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §4.1, Estimator 2):

    Ledoit-Wolf shrinkage target — scaled identity matrix:

        Σ_SKLW = (1 − δ̂) · S + δ̂ · μ_var · I

    where:
        S        = empirical covariance matrix (sklearn MLE convention, ddof=0)
        μ_var    = Tr(S) / N   (mean variance, trace-normalized scalar)
        I        = N × N identity matrix
        δ̂ ∈ [0, 1] = analytically optimal shrinkage intensity
                      (Ledoit & Wolf 2004)

    Annualization (FINANCIAL_METHODOLOGY.md §2.2):

        Σ_LW_annual = trading_days_per_year · Σ_LW_daily

Canonical design references:
    - ARCHITECTURE.md §4.2  : Covariance taxonomy; LedoitWolf is listed as the
                              "Default" estimator for the optimization pipeline.
    - FINANCIAL_METHODOLOGY.md §4.1 : Shrinkage formula and annualization policy.
    - DATA_STRATEGY.md §3.2 : Listwise alignment policy (no ffill / bfill).

Important — ddof Convention Distinction vs. Step 1 (Sample Covariance):
    scikit-learn's LedoitWolf estimator uses maximum-likelihood (ML)
    normalization internally (ddof=0, divide by T rather than T−1).  This is
    the standard convention in the shrinkage literature (Ledoit & Wolf 2004).
    Step 1's sample covariance uses the unbiased estimator (ddof=1, divide by
    T−1, pandas .cov() default).

    Consequence: for the same input data, Σ_LW_daily ≠ (1 − δ̂) · S_step1.
    Both are mathematically legitimate; they represent different estimators
    with different bias-variance tradeoffs.  Do NOT force the two to agree.

Shrinkage Intensity API:
    ``calculate_ledoit_wolf_covariance`` returns a plain ``pd.DataFrame``
    (the annualized covariance matrix) so the calling convention stays simple
    and consistent with Step 1.

    The fitted scikit-learn estimator — including the shrinkage intensity
    ``δ̂`` — is exposed through ``fit_ledoit_wolf`` for callers that need
    metadata without changing the primary return contract.

Scope:
    This module implements ONLY Ledoit-Wolf shrinkage (Step 2).
    Constant-Correlation shrinkage and Higham PD projection are separate
    modules (not yet implemented — see PROJECT_PLAN.md Phase 2.2 Steps 3+).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from app.config import settings
from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine._validation import (
    ABSOLUTE_MIN_OBSERVATIONS,
    validate_and_convert_values,
    validate_min_observations,
    validate_returns_dataframe,
    validate_trading_days_per_year,
)

logger = logging.getLogger("quantamental.shrinkage")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_TRADING_DAYS_PER_YEAR: int = settings.TRADING_DAYS_PER_YEAR

# Tolerance for the post-calculation symmetry check: |Σ − Σᵀ|_max < tol
_SYMMETRY_TOL: float = 1e-10

_ESTIMATOR_NAME: str = "Ledoit-Wolf shrinkage"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_observation_count(
    n_obs: int,
    n_assets: int,
    min_observations: Optional[int],
) -> None:
    """Raise InsufficientDataError when T < effective minimum."""
    effective_min: int = (
        min_observations
        if min_observations is not None
        else max(ABSOLUTE_MIN_OBSERVATIONS, n_assets + 1)
    )
    if n_obs < effective_min:
        raise InsufficientDataError(
            f"{_ESTIMATOR_NAME}: insufficient observations: "
            f"need at least {effective_min}, got {n_obs}.",
            details={
                "n_observations": n_obs,
                "min_required": effective_min,
                "n_assets": n_assets,
            },
        )


def _symmetrize_and_validate(
    cov_annual: pd.DataFrame,
) -> pd.DataFrame:
    """Enforce numerical symmetry and validate finiteness of output matrix.

    Parameters
    ----------
    cov_annual:
        Annualized covariance DataFrame (N × N, labelled).

    Returns
    -------
    pd.DataFrame
        Symmetrized, validated DataFrame.

    Raises
    ------
    CovarianceValidationError
        If the matrix contains non-finite values or fails the symmetry check.
    """
    vals = cov_annual.values
    sym_vals = (vals + vals.T) / 2.0
    cov_annual = pd.DataFrame(
        sym_vals,
        index=cov_annual.index,
        columns=cov_annual.columns,
    )

    if not np.all(np.isfinite(cov_annual.values)):
        raise CovarianceValidationError(
            f"{_ESTIMATOR_NAME}: annualized covariance matrix contains non-finite values.",
            details={"shape": list(cov_annual.shape)},
        )

    max_asymmetry = float(np.max(np.abs(cov_annual.values - cov_annual.values.T)))
    if max_asymmetry > _SYMMETRY_TOL:
        raise CovarianceValidationError(
            f"{_ESTIMATOR_NAME}: annualized covariance matrix failed symmetry check "
            f"(max asymmetry={max_asymmetry:.2e}, tol={_SYMMETRY_TOL:.2e}).",
            details={"max_asymmetry": max_asymmetry, "tolerance": _SYMMETRY_TOL},
        )

    return cov_annual


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_ledoit_wolf(
    returns: pd.DataFrame,
    *,
    min_observations: Optional[int] = None,
) -> LedoitWolf:
    """Fit and return the scikit-learn LedoitWolf estimator on ``returns``.

    This is a lower-level accessor that exposes the full fitted estimator,
    including the shrinkage intensity ``estimator.shrinkage_``, the
    daily covariance array ``estimator.covariance_``, and the precision
    matrix ``estimator.precision_``.

    The same input validation as ``calculate_ledoit_wolf_covariance`` is
    applied.  The result is NOT annualized — it is the raw daily estimator.

    Parameters
    ----------
    returns:
        Wide-format DataFrame of daily simple returns.  Rows = dates,
        columns = asset identifiers.  Pandas nullable extension dtypes
        (``Float64``, ``Int64``, etc.) are accepted.
    min_observations:
        Minimum trading-day observations required.  Defaults to
        ``max(2, N + 1)``.  Validated strictly (bool, float, str, zero,
        negative, below-floor all rejected).

    Returns
    -------
    sklearn.covariance.LedoitWolf
        Fitted estimator.  Key attributes:
            - ``shrinkage_``  : float, optimal δ̂ ∈ [0, 1]
            - ``covariance_`` : ndarray (N × N), daily LW covariance
            - ``precision_``  : ndarray (N × N), precision matrix

    Raises
    ------
    FinancialCalculationError / InsufficientDataError / CovarianceValidationError
        Same conditions as ``calculate_ledoit_wolf_covariance``.
    """
    # Validate parameters
    validate_min_observations(min_observations, estimator_name=_ESTIMATOR_NAME)
    returns_df, n_assets, n_obs = validate_returns_dataframe(
        returns, estimator_name=_ESTIMATOR_NAME
    )
    _check_observation_count(n_obs, n_assets, min_observations)
    values_f64 = validate_and_convert_values(returns_df, estimator_name=_ESTIMATOR_NAME)

    try:
        estimator = LedoitWolf(assume_centered=False)
        estimator.fit(values_f64)
    except Exception as exc:
        raise FinancialCalculationError(
            f"{_ESTIMATOR_NAME}: sklearn LedoitWolf.fit() failed: {exc}",
            details={"error": str(exc)},
        ) from exc

    return estimator


def calculate_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    min_observations: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate the annualized Ledoit-Wolf shrinkage covariance matrix.

    Uses scikit-learn's ``LedoitWolf`` estimator, which implements the
    Ledoit-Wolf shrinkage covariance estimator toward a scaled identity target:

        Σ_LW = (1 − δ̂) · S_MLE + δ̂ · μ_var · I

    where S_MLE is sklearn's internal MLE covariance (ddof=0), μ_var =
    Tr(S_MLE) / N, and δ̂ ∈ [0, 1] is the analytically optimal shrinkage
    intensity.

    **Important — ddof Convention:**
    sklearn LedoitWolf uses ddof=0 (maximum-likelihood normalization)
    internally.  This differs from the Step 1 unbiased sample covariance
    (ddof=1, pandas ``.cov()``).  The two estimators therefore produce
    different daily covariance values on the same input data.  This is
    mathematically correct behavior and must NOT be corrected.

    Annualization:

        Σ_LW_annual = trading_days_per_year · Σ_LW_daily

    The result is symmetrized numerically as (Σ + Σᵀ) / 2.

    To access the fitted sklearn estimator and shrinkage intensity, use
    ``fit_ledoit_wolf(returns)`` directly.

    Parameters
    ----------
    returns:
        Wide-format DataFrame of daily simple returns.  Rows = dates,
        columns = asset identifiers.  The caller is responsible for
        listwise alignment (all NaN rows should be dropped beforehand).
        This function does NOT forward-fill or backward-fill missing values.
        Pandas nullable extension dtypes (``Float64``, ``Int64``, etc.) are
        accepted and converted internally to ``float64`` after validation.
        Complex dtypes are explicitly rejected.
    trading_days_per_year:
        Annualization factor (default: ``settings.TRADING_DAYS_PER_YEAR``,
        typically 252 for NSE).  Bool, float, zero, and negative values are
        rejected.
    min_observations:
        Minimum number of trading-day observations required.  Defaults to
        ``max(2, N + 1)``.  Bool, float, string, zero, negative, and
        values below the absolute minimum of 2 are rejected.

    Returns
    -------
    pd.DataFrame
        Symmetric N × N annualized Ledoit-Wolf shrinkage covariance matrix,
        labelled with asset names on both axes in the same column order as
        the input.

    Raises
    ------
    FinancialCalculationError
        If ``returns`` is None, not a DataFrame, contains non-numeric or
        complex-valued columns, NaN, Inf, duplicate column names, or
        duplicate date indices.  Also raised for invalid
        ``trading_days_per_year`` or ``min_observations``, or if the
        sklearn estimator raises an unexpected error.
    InsufficientDataError
        Fewer than 2 assets, or T < ``min_observations``.
    CovarianceValidationError
        If the output matrix contains non-finite entries or fails the
        symmetry check.
    """
    # ------------------------------------------------------------------
    # 1. Validate parameters
    # ------------------------------------------------------------------
    validate_min_observations(min_observations, estimator_name=_ESTIMATOR_NAME)
    validate_trading_days_per_year(trading_days_per_year, estimator_name=_ESTIMATOR_NAME)

    # ------------------------------------------------------------------
    # 2. Validate returns structure and values
    # ------------------------------------------------------------------
    returns_df, n_assets, n_obs = validate_returns_dataframe(
        returns, estimator_name=_ESTIMATOR_NAME
    )
    _check_observation_count(n_obs, n_assets, min_observations)
    values_f64 = validate_and_convert_values(returns_df, estimator_name=_ESTIMATOR_NAME)

    # ------------------------------------------------------------------
    # 3. Fit Ledoit-Wolf estimator (ddof=0 internally)
    # ------------------------------------------------------------------
    logger.debug(
        "Fitting Ledoit-Wolf: n_assets=%d, n_obs=%d, trading_days=%d",
        n_assets,
        n_obs,
        trading_days_per_year,
    )

    try:
        estimator = LedoitWolf(assume_centered=False)
        estimator.fit(values_f64)
    except Exception as exc:
        raise FinancialCalculationError(
            f"{_ESTIMATOR_NAME}: sklearn LedoitWolf.fit() failed: {exc}",
            details={"error": str(exc)},
        ) from exc

    shrinkage: float = float(estimator.shrinkage_)
    logger.debug("Ledoit-Wolf shrinkage intensity: delta=%.6f", shrinkage)

    # ------------------------------------------------------------------
    # 4. Build labelled daily covariance DataFrame
    # ------------------------------------------------------------------
    cov_daily = pd.DataFrame(
        estimator.covariance_,
        index=returns_df.columns,
        columns=returns_df.columns,
    )

    # ------------------------------------------------------------------
    # 5. Annualize
    # ------------------------------------------------------------------
    cov_annual: pd.DataFrame = trading_days_per_year * cov_daily

    # ------------------------------------------------------------------
    # 6. Symmetrize and validate output
    # ------------------------------------------------------------------
    cov_annual = _symmetrize_and_validate(cov_annual)

    logger.info(
        "Ledoit-Wolf covariance computed: shape=%s, delta=%.6f, trading_days=%d",
        cov_annual.shape,
        shrinkage,
        trading_days_per_year,
    )

    return cov_annual
