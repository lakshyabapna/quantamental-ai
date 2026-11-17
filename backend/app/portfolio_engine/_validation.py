"""Quantamental AI - Shared Input Validation Helpers for Covariance Estimators.

Internal module — not part of the public API.

Provides reusable validation functions for Phase 2.2 covariance estimators.
These helpers are currently used by the Ledoit-Wolf shrinkage estimator and
are available for subsequent covariance estimators (such as Constant-Correlation
shrinkage). Note that Step 1 (sample covariance) retains its original self-contained
validation implementation in ``covariance.py``.

All functions raise project-specific exceptions defined in
``app.core.exceptions``. See ARCHITECTURE.md §4.2 and
DATA_STRATEGY.md §3.2 for the policies these checks enforce.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
)

# Project-wide absolute minimum observation policy: at least 2 rows.
ABSOLUTE_MIN_OBSERVATIONS: int = 2


def validate_returns_dataframe(
    returns: object,
    *,
    estimator_name: str = "covariance estimator",
) -> Tuple[pd.DataFrame, int, int]:
    """Validate ``returns`` type, shape, columns, and index.

    Performs structural checks only (type, emptiness, asset count, duplicate
    column names, duplicate index entries).  Value checks (NaN, Inf, complex,
    dtype) are separated into ``validate_and_convert_values`` so that callers
    can interleave their own logic between the two stages if needed.

    Parameters
    ----------
    returns:
        Object to validate.  Expected to be a wide-format pandas DataFrame
        of daily simple returns (rows = dates, columns = asset identifiers).
    estimator_name:
        Human-readable estimator name used in error messages.

    Returns
    -------
    (returns, n_assets, n_obs):
        The validated DataFrame, number of asset columns, and number of rows.

    Raises
    ------
    FinancialCalculationError
        Type, emptiness, duplicate-column, or duplicate-index violations.
    InsufficientDataError
        Fewer than 2 assets, or empty DataFrame.
    """
    if returns is None:
        raise FinancialCalculationError(
            f"{estimator_name}: returns must be a pandas DataFrame, got None.",
            details={"type": "NoneType"},
        )
    if not isinstance(returns, pd.DataFrame):
        raise FinancialCalculationError(
            f"{estimator_name}: returns must be a pandas DataFrame, "
            f"got {type(returns).__name__}.",
            details={"type": type(returns).__name__},
        )
    if returns.empty:
        raise InsufficientDataError(
            f"{estimator_name}: returns DataFrame is empty; "
            "at least 2 assets and 2 observations are required.",
            details={"shape": list(returns.shape)},
        )

    n_assets: int = returns.shape[1]
    if n_assets < 2:
        raise InsufficientDataError(
            f"{estimator_name}: requires at least 2 assets; got {n_assets}.",
            details={"n_assets": n_assets},
        )

    if returns.columns.duplicated().any():
        dupes = returns.columns[returns.columns.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains duplicate asset (column) names.",
            details={"duplicate_assets": dupes},
        )

    n_obs: int = returns.shape[0]

    if returns.index.duplicated().any():
        dupes_idx = returns.index[returns.index.duplicated(keep=False)].tolist()
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains duplicate index (date) entries.",
            details={"duplicate_dates": [str(d) for d in dupes_idx[:10]]},
        )

    return returns, n_assets, n_obs


def validate_min_observations(
    min_observations: Optional[int],
    *,
    estimator_name: str = "covariance estimator",
) -> None:
    """Validate the ``min_observations`` parameter when explicitly supplied.

    Rejects ``bool``, ``float``, ``str``, zero, negative, and values below
    ``ABSOLUTE_MIN_OBSERVATIONS``.  ``None`` is accepted (callers apply their
    own default).

    Raises
    ------
    FinancialCalculationError
        Any invalid non-None value.
    """
    if min_observations is None:
        return
    if isinstance(min_observations, bool):
        raise FinancialCalculationError(
            f"{estimator_name}: min_observations must be a positive integer, "
            f"got bool ({min_observations!r}).",
            details={"min_observations": min_observations, "type": "bool"},
        )
    if not isinstance(min_observations, int):
        raise FinancialCalculationError(
            f"{estimator_name}: min_observations must be a positive integer, "
            f"got {type(min_observations).__name__} ({min_observations!r}).",
            details={
                "min_observations": min_observations,
                "type": type(min_observations).__name__,
            },
        )
    if min_observations <= 0:
        raise FinancialCalculationError(
            f"{estimator_name}: min_observations must be a positive integer, "
            f"got {min_observations!r}.",
            details={"min_observations": min_observations},
        )
    if min_observations < ABSOLUTE_MIN_OBSERVATIONS:
        raise FinancialCalculationError(
            f"{estimator_name}: min_observations ({min_observations}) is below "
            f"the absolute minimum of {ABSOLUTE_MIN_OBSERVATIONS}.",
            details={
                "min_observations": min_observations,
                "absolute_minimum": ABSOLUTE_MIN_OBSERVATIONS,
            },
        )


def validate_trading_days_per_year(
    trading_days_per_year: object,
    *,
    estimator_name: str = "covariance estimator",
) -> None:
    """Validate ``trading_days_per_year``.

    Rejects ``bool`` (even though it is an ``int`` subclass in Python),
    non-integers, zero, and negative values.

    Raises
    ------
    FinancialCalculationError
        Any invalid value.
    """
    if isinstance(trading_days_per_year, bool):
        raise FinancialCalculationError(
            f"{estimator_name}: trading_days_per_year must be a positive integer, "
            f"got bool ({trading_days_per_year!r}).",
            details={"trading_days_per_year": trading_days_per_year, "type": "bool"},
        )
    if not isinstance(trading_days_per_year, int) or trading_days_per_year <= 0:
        raise FinancialCalculationError(
            f"{estimator_name}: trading_days_per_year must be a positive integer; "
            f"got {trading_days_per_year!r}.",
            details={"trading_days_per_year": trading_days_per_year},
        )


def validate_and_convert_values(
    returns: pd.DataFrame,
    *,
    estimator_name: str = "covariance estimator",
) -> np.ndarray:
    """Validate column dtypes and return values, then convert to float64.

    Checks:
    1. All columns are numeric (using pandas dtype API).
    2. No complex-valued columns (``is_numeric_dtype`` returns True for complex;
       this engine requires real-valued simple returns only).
    3. No NaN / pd.NA values — pandas ``.isnull()`` is used because it
       correctly detects the ``pd.NA`` sentinel in nullable extension types
       *before* float64 conversion, which would silently promote ``pd.NA`` to
       ``NaN``.
    4. Convert to a homogeneous ``float64`` NumPy array via
       ``.to_numpy(dtype=np.float64, na_value=np.nan)``.
    5. No infinite values in the converted float64 array.

    Parameters
    ----------
    returns:
        DataFrame that has already passed ``validate_returns_dataframe``.
    estimator_name:
        Human-readable estimator name for error messages.

    Returns
    -------
    np.ndarray
        Shape ``(n_obs, n_assets)`` homogeneous ``float64`` array.

    Raises
    ------
    FinancialCalculationError
        Any value violation.
    """
    # 1. Non-numeric dtype check.
    non_numeric_cols = [
        col
        for col in returns.columns
        if not pd.api.types.is_numeric_dtype(returns[col])
    ]
    if non_numeric_cols:
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains non-numeric columns.",
            details={"non_numeric_columns": non_numeric_cols},
        )

    # 2. Complex dtype check.
    complex_cols = [
        col
        for col in returns.columns
        if pd.api.types.is_complex_dtype(returns[col])
    ]
    if complex_cols:
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains complex-valued columns; "
            "this engine requires real-valued simple returns.",
            details={"complex_columns": complex_cols},
        )

    # 3. NaN / pd.NA check — must precede float64 conversion.
    if returns.isnull().values.any():
        nan_counts = returns.isnull().sum()
        nan_assets = nan_counts[nan_counts > 0].to_dict()
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains NaN values; "
            "listwise alignment must be applied by the caller. "
            "No forward-fill or backward-fill is performed.",
            details={"nan_counts_by_asset": {str(k): int(v) for k, v in nan_assets.items()}},
        )

    # 4. Convert to homogeneous float64 array.
    try:
        values_f64: np.ndarray = returns.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise FinancialCalculationError(
            f"{estimator_name}: could not convert returns DataFrame to a NumPy "
            f"float64 array: {exc}. Ensure all columns contain real-valued numeric data.",
            details={
                "dtypes": {str(col): str(dt) for col, dt in returns.dtypes.items()},
                "error": str(exc),
            },
        ) from exc

    # 5. Inf check on the validated float64 array.
    if np.isinf(values_f64).any():
        inf_counts = np.isinf(values_f64).sum(axis=0)
        inf_assets = {
            str(col): int(cnt)
            for col, cnt in zip(returns.columns, inf_counts)
            if cnt > 0
        }
        raise FinancialCalculationError(
            f"{estimator_name}: returns DataFrame contains infinite values.",
            details={"inf_counts_by_asset": inf_assets},
        )

    return values_f64
