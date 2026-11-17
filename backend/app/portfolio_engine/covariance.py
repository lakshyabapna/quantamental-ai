"""Quantamental AI - Sample Covariance Matrix Estimation.

Implements Phase 2.2, Step 1: sample covariance estimation from aligned
daily simple returns.

Mathematical Reference (FINANCIAL_METHODOLOGY.md §4.1):
    Daily sample covariance:
        S = 1 / (T - 1) * Σ_t (R_t - R̄)(R_t - R̄)ᵀ

    Annualized covariance (i.i.d. scaling, FINANCIAL_METHODOLOGY.md §4.1):
        Σ_ann = 252 · S

    where T is the number of observations (trading days) and 252 is the
    conventional number of NSE trading days per year.

Canonical design references:
    - ARCHITECTURE.md §4.2  : Covariance taxonomy and estimator catalogue
    - FINANCIAL_METHODOLOGY.md §4.1 : Sample covariance formula and annualization
    - DATA_STRATEGY.md §3.2 : Listwise alignment policy (no ffill / bfill)

Scope:
    This module implements ONLY sample covariance estimation.
    Ledoit-Wolf shrinkage and Higham PD projection are separate modules
    (not yet implemented — see PROJECT_PLAN.md Phase 2.2 Steps 2–3).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    InsufficientDataError,
)

logger = logging.getLogger("quantamental.covariance")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_TRADING_DAYS_PER_YEAR: int = settings.TRADING_DAYS_PER_YEAR

# Minimum number of observations required to compute a sample covariance.
# T >= 2 is the absolute floor (ddof=1 requires at least 2 rows).
# Users are strongly advised to supply T >> N observations for numerical
# stability; see FINANCIAL_METHODOLOGY.md §4.1 and ARCHITECTURE.md §4.2.
_ABSOLUTE_MIN_OBSERVATIONS: int = 2

# Tolerance used when checking output symmetry: |S - Sᵀ|_max < tol
_SYMMETRY_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_sample_covariance(
    returns: pd.DataFrame,
    *,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    min_observations: Optional[int] = None,
) -> pd.DataFrame:
    """Estimate the annualized sample covariance matrix from daily simple returns.

    Uses the unbiased sample covariance estimator (ddof=1):

        S = 1 / (T - 1) · Σ_t (R_t - R̄)(R_t - R̄)ᵀ

    and then annualizes by the configured trading days per year:

        Σ_ann = trading_days_per_year · S

    The result is symmetrized numerically as (Σ_ann + Σ_annᵀ) / 2 to
    eliminate floating-point rounding asymmetry before returning.

    Parameters
    ----------
    returns:
        Wide-format DataFrame of daily simple returns.
        Rows must be dates (DatetimeIndex or date-like Index); columns are
        asset identifiers.  The caller is responsible for listwise alignment
        (all NaN rows should already be dropped).  This function does NOT
        forward-fill or backward-fill any missing values.
        Pandas nullable extension dtypes (``Float64``, ``Int64``, etc.) are
        accepted and are converted internally to ``float64`` after validation.
        Complex dtypes are explicitly rejected.
    trading_days_per_year:
        Annualization factor (default: ``settings.TRADING_DAYS_PER_YEAR``,
        typically 252 for NSE).
    min_observations:
        Minimum number of trading-day observations required.  Defaults to
        ``max(2, N + 1)`` where N is the number of assets, which is the
        weakest threshold for a non-singular sample covariance.  Callers
        may pass a stricter value (e.g. 252 for one full year).

    Returns
    -------
    pd.DataFrame
        Symmetric N × N annualized sample covariance matrix labelled with
        asset names on both axes, in the same column order as the input.

    Raises
    ------
    FinancialCalculationError
        If ``returns`` is None, not a DataFrame, contains non-numeric or
        complex-valued columns, contains NaN or Inf entries, has duplicate
        column names, or has a duplicate date index.
        Also raised if ``trading_days_per_year`` or ``min_observations`` are
        invalid (bool, non-integer, zero, or negative), or if the DataFrame
        cannot be converted to a ``float64`` NumPy array.
    InsufficientDataError
        If fewer than 2 assets are provided, or the number of observations
        is below ``min_observations``.
    CovarianceValidationError
        If the computed output matrix contains non-finite entries or fails
        the symmetry check (indicates a numerical error in the estimator).
    """
    # ------------------------------------------------------------------
    # 1. Basic type and emptiness checks
    # ------------------------------------------------------------------
    if returns is None:
        raise FinancialCalculationError(
            "returns must be a pandas DataFrame, got None.",
            details={"type": "NoneType"},
        )
    if not isinstance(returns, pd.DataFrame):
        raise FinancialCalculationError(
            f"returns must be a pandas DataFrame, got {type(returns).__name__}.",
            details={"type": type(returns).__name__},
        )
    if returns.empty:
        raise InsufficientDataError(
            "returns DataFrame is empty; at least 2 assets and 2 observations are required.",
            details={"shape": list(returns.shape)},
        )

    # ------------------------------------------------------------------
    # 2. Asset (column) dimension checks
    # ------------------------------------------------------------------
    n_assets: int = returns.shape[1]
    if n_assets < 2:
        raise InsufficientDataError(
            f"Sample covariance requires at least 2 assets; got {n_assets}.",
            details={"n_assets": n_assets},
        )

    # Duplicate column names
    if returns.columns.duplicated().any():
        dupes = returns.columns[returns.columns.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "returns DataFrame contains duplicate asset (column) names.",
            details={"duplicate_assets": dupes},
        )

    # ------------------------------------------------------------------
    # 3. Observation (row) dimension checks
    # ------------------------------------------------------------------
    n_obs: int = returns.shape[0]

    # Validate min_observations when explicitly supplied.
    # bool is a subclass of int in Python; a boolean flag is never a valid
    # observation count and must be rejected explicitly.
    if min_observations is not None:
        if isinstance(min_observations, bool):
            raise FinancialCalculationError(
                f"min_observations must be a positive integer, got bool ({min_observations!r}).",
                details={"min_observations": min_observations, "type": "bool"},
            )
        if not isinstance(min_observations, int):
            raise FinancialCalculationError(
                f"min_observations must be a positive integer, "
                f"got {type(min_observations).__name__} ({min_observations!r}).",
                details={
                    "min_observations": min_observations,
                    "type": type(min_observations).__name__,
                },
            )
        if min_observations <= 0:
            raise FinancialCalculationError(
                f"min_observations must be a positive integer, got {min_observations!r}.",
                details={"min_observations": min_observations},
            )
        if min_observations < _ABSOLUTE_MIN_OBSERVATIONS:
            raise FinancialCalculationError(
                f"min_observations ({min_observations}) is below the absolute minimum "
                f"of {_ABSOLUTE_MIN_OBSERVATIONS} required by the ddof=1 estimator.",
                details={
                    "min_observations": min_observations,
                    "absolute_minimum": _ABSOLUTE_MIN_OBSERVATIONS,
                },
            )

    effective_min: int = (
        min_observations
        if min_observations is not None
        else max(_ABSOLUTE_MIN_OBSERVATIONS, n_assets + 1)
    )

    if n_obs < effective_min:
        raise InsufficientDataError(
            f"Insufficient observations: need at least {effective_min}, got {n_obs}.",
            details={
                "n_observations": n_obs,
                "min_required": effective_min,
                "n_assets": n_assets,
            },
        )

    # Duplicate index entries
    if returns.index.duplicated().any():
        dupes = returns.index[returns.index.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            "returns DataFrame contains duplicate index (date) entries.",
            details={"duplicate_dates": [str(d) for d in dupes[:10]]},
        )

    # ------------------------------------------------------------------
    # 4. Value checks — numeric, real-valued, finite, no NaN
    # ------------------------------------------------------------------
    # Check all columns are numeric dtype first (fast path).
    non_numeric_cols = [
        col
        for col in returns.columns
        if not pd.api.types.is_numeric_dtype(returns[col])
    ]
    if non_numeric_cols:
        raise FinancialCalculationError(
            "returns DataFrame contains non-numeric columns.",
            details={"non_numeric_columns": non_numeric_cols},
        )

    # Reject complex-valued columns explicitly.
    # pd.api.types.is_numeric_dtype returns True for complex dtypes
    # (complex64, complex128), but this engine expects real-valued simple
    # returns only.  Complex input is a programming error, not a data-quality
    # issue, and must be rejected rather than silently cast.
    complex_cols = [
        col
        for col in returns.columns
        if pd.api.types.is_complex_dtype(returns[col])
    ]
    if complex_cols:
        raise FinancialCalculationError(
            "returns DataFrame contains complex-valued columns; "
            "this engine requires real-valued simple returns.",
            details={"complex_columns": complex_cols},
        )

    # NaN check — no imputation permitted (DATA_STRATEGY.md §3.2).
    # pandas .isnull() correctly handles nullable extension dtypes (Float64,
    # Int64, etc.) whose underlying NA sentinel is not NumPy NaN.  This check
    # must be performed BEFORE converting to a NumPy array because the
    # conversion itself would propagate NA as NaN, masking the original source.
    if returns.isnull().values.any():
        nan_counts = returns.isnull().sum()
        nan_assets = nan_counts[nan_counts > 0].to_dict()
        raise FinancialCalculationError(
            "returns DataFrame contains NaN values; listwise alignment must be "
            "applied by the caller before calling calculate_sample_covariance. "
            "No forward-fill or backward-fill is performed.",
            details={"nan_counts_by_asset": {str(k): int(v) for k, v in nan_assets.items()}},
        )

    # Convert to a plain NumPy float64 array.
    #
    # Pandas nullable extension dtypes (e.g. Float64, Int64) back their
    # columns with an object-dtype ndarray when `.values` is called, which
    # causes `np.isinf()` and `np.isfinite()` to raise TypeError.  Calling
    # `.to_numpy(dtype=float, na_value=np.nan)` produces a well-typed
    # float64 C-contiguous array regardless of the source dtype, making all
    # subsequent NumPy operations safe and consistent.
    #
    # The conversion is performed AFTER the pandas-aware NaN check above so
    # that NA sentinels in extension types are detected correctly before they
    # are silently converted to NaN by this step.
    try:
        values_f64: np.ndarray = returns.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"Could not convert returns DataFrame to a NumPy float64 array: {exc}. "
            "Ensure all columns contain real-valued numeric data.",
            details={
                "dtypes": {str(col): str(dtype) for col, dtype in returns.dtypes.items()},
                "error": str(exc),
            },
        ) from exc

    # Inf check — performed on the validated float64 array.
    if np.isinf(values_f64).any():
        inf_counts = np.isinf(values_f64).sum(axis=0)
        inf_assets = {
            str(col): int(cnt)
            for col, cnt in zip(returns.columns, inf_counts)
            if cnt > 0
        }
        raise FinancialCalculationError(
            "returns DataFrame contains infinite values.",
            details={"inf_counts_by_asset": inf_assets},
        )

    # Build a plain float64 DataFrame from the validated array, preserving
    # original asset labels and row index.  This is the representation used
    # for all downstream numerical computation (covariance, annualization,
    # symmetrization).  Using a homogeneous float64 backing array avoids any
    # extension-type overhead in pandas arithmetic and guarantees that
    # `.cov()` operates on a well-typed matrix.
    returns_f64: pd.DataFrame = pd.DataFrame(
        values_f64,
        index=returns.index,
        columns=returns.columns,
    )

    # ------------------------------------------------------------------
    # 5. Validate trading_days_per_year
    # ------------------------------------------------------------------
    # bool is a subclass of int in Python; True (==1) and False (==0) must
    # be rejected — a boolean flag is never a valid annualization factor.
    if isinstance(trading_days_per_year, bool):
        raise FinancialCalculationError(
            f"trading_days_per_year must be a positive integer, got bool ({trading_days_per_year!r}).",
            details={"trading_days_per_year": trading_days_per_year, "type": "bool"},
        )
    if not isinstance(trading_days_per_year, int) or trading_days_per_year <= 0:
        raise FinancialCalculationError(
            f"trading_days_per_year must be a positive integer; got {trading_days_per_year!r}.",
            details={"trading_days_per_year": trading_days_per_year},
        )

    # ------------------------------------------------------------------
    # 6. Compute daily sample covariance (ddof=1, pandas default)
    # ------------------------------------------------------------------
    logger.debug(
        "Computing sample covariance: n_assets=%d, n_obs=%d, trading_days=%d",
        n_assets,
        n_obs,
        trading_days_per_year,
    )

    try:
        cov_daily: pd.DataFrame = returns_f64.cov()  # ddof=1 by default
    except Exception as exc:
        raise FinancialCalculationError(
            f"Sample covariance computation failed: {exc}",
            details={"error": str(exc)},
        ) from exc

    # ------------------------------------------------------------------
    # 7. Annualize
    # ------------------------------------------------------------------
    cov_annual: pd.DataFrame = trading_days_per_year * cov_daily

    # ------------------------------------------------------------------
    # 8. Enforce symmetry numerically
    # ------------------------------------------------------------------
    # pandas .cov() is symmetric by construction, but floating-point
    # accumulation can introduce sub-epsilon asymmetry.  Symmetrize to
    # guarantee downstream callers receive a valid symmetric matrix.
    cov_values = cov_annual.values
    cov_sym = (cov_values + cov_values.T) / 2.0
    cov_annual = pd.DataFrame(
        cov_sym,
        index=cov_annual.index,
        columns=cov_annual.columns,
    )

    # ------------------------------------------------------------------
    # 9. Post-calculation validation
    # ------------------------------------------------------------------
    # Check all values are finite.
    if not np.all(np.isfinite(cov_annual.values)):
        raise CovarianceValidationError(
            "Annualized sample covariance matrix contains non-finite values.",
            details={"shape": list(cov_annual.shape)},
        )

    # Check symmetry within tolerance (after explicit symmetrization, this
    # should always pass — kept as a safety net).
    max_asymmetry = float(np.max(np.abs(cov_annual.values - cov_annual.values.T)))
    if max_asymmetry > _SYMMETRY_TOL:
        raise CovarianceValidationError(
            f"Annualized sample covariance matrix failed symmetry check "
            f"(max asymmetry={max_asymmetry:.2e}, tol={_SYMMETRY_TOL:.2e}).",
            details={"max_asymmetry": max_asymmetry, "tolerance": _SYMMETRY_TOL},
        )

    logger.info(
        "Sample covariance computed successfully: shape=%s, trading_days=%d",
        cov_annual.shape,
        trading_days_per_year,
    )

    return cov_annual
