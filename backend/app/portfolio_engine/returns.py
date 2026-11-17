"""Quantamental AI - Historical Returns & Expected Returns Engine.

Module: Phase 2.1 — Historical Returns and Expected Returns Module
Conforms to:
- ARCHITECTURE.md (FinancialCore: ReturnsEngine)
- docs/FINANCIAL_METHODOLOGY.md (Sections 2 & 3: Returns & Asset Moments)
- docs/DATA_STRATEGY.md (Section 3: Missing-Data & Alignment Policies)
- docs/PROJECT_PLAN.md (Phase 2, Task 1)

Key Mathematical Principles & Conventions:
1. Daily Simple Returns:
   R_{i, t} = (P_{adj, i, t} / P_{adj, i, t-1}) - 1
   Primary return representation for portfolio aggregation and covariance estimation.
   Simple returns are cross-sectionally linear: R_{p, t} = sum(w_i * R_{i, t}).
   Multi-session returns must NEVER be misclassified as single daily returns.
   Missing observations produce NaN daily returns; listwise-complete alignment
   is applied strictly to the resulting returns matrix, NOT to the price series
   before calculating returns.

2. Daily Logarithmic Returns:
   r_{i, t} = ln(P_{adj, i, t} / P_{adj, i, t-1}) = ln(1 + R_{i, t})
   Calculated strictly for time-series modeling and predictive machine learning.
   NOT cross-sectionally additive (ln sum w_i P_i != sum w_i ln P_i).
   Must NEVER be substituted for simple returns in portfolio optimization.

3. Annualized Arithmetic Expected Return:
   mu_{i, annual} = 252 * mean(daily simple returns)
   Single-period expected return input required by Markowitz Modern Portfolio Theory.
   Multi-asset portfolios evaluate over a common listwise-complete return set.

4. Compound Annual Growth Rate (CAGR):
   CAGR_i = (Ending Value / Beginning Value) ** (252 / N) - 1
          = ( prod_{t=1}^N (1 + R_{i, t}) ) ** (252 / N) - 1
   Multi-period geometric realized growth rate over N return observations.
   Price-Input Mode (is_prices=True):
       CAGR_i = (P_{i, T} / P_{i, 0}) ** (252 / T) - 1
       where T is elapsed trading sessions between evaluated boundaries
       derived from the official NSE exchange calendar.
       Evaluated against data-quality policy (coverage >= 95%, max consecutive missing <= 3).
   Return-Input Mode (is_prices=False):
       CAGR_i = ( prod_{t=1}^N (1 + R_{i, t}) ) ** (252 / N) - 1
       where R_{i, t} is a continuous sequence of valid daily simple returns (no NaNs).
   Must NEVER be substituted for arithmetic expected return in Markowitz optimization.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.config import settings
from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    InvalidPriceError,
    UnalignedObservationsError,
)
from app.core.logger import get_logger
from app.data_engine.calendar import (
    CalendarDateRangeError,
    HolidayCircularNotFoundError,
    InvalidCalendarDateError,
    nse_calendar,
)

logger = get_logger("quantamental.returns")

DEFAULT_TRADING_DAYS_PER_YEAR: int = settings.TRADING_DAYS_PER_YEAR
MIN_CAGR_HISTORY_COVERAGE_RATIO: float = 0.95
MAX_CAGR_CONSECUTIVE_MISSING_DAYS: int = 3


def _normalize_to_date(val: Any) -> Optional[date]:
    """Normalize date, datetime, pd.Timestamp, or ISO string to datetime.date."""
    if isinstance(val, date) and not isinstance(val, (datetime, pd.Timestamp)):
        return val
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    if isinstance(val, str):
        try:
            return date.fromisoformat(val[:10])
        except (ValueError, TypeError):
            return None
    return None


def _validate_trading_days_per_year(trading_days_per_year: Any) -> int:
    """Validate that trading_days_per_year is a strictly positive integer.

    Rejects:
    - Booleans (bool is a subclass of int in Python: isinstance(True, int) is True!)
    - Floats (e.g. 252.0)
    - Strings, None, or other non-integer types
    - Zero or negative integers (<= 0)

    Raises:
        FinancialCalculationError: If parameter is not a positive integer.
    """
    if isinstance(trading_days_per_year, bool) or not isinstance(trading_days_per_year, int):
        raise FinancialCalculationError(
            f"trading_days_per_year must be a positive integer, received {type(trading_days_per_year).__name__} ({trading_days_per_year})",
            details={"trading_days_per_year": str(trading_days_per_year), "type": type(trading_days_per_year).__name__},
        )
    if trading_days_per_year <= 0:
        raise FinancialCalculationError(
            f"trading_days_per_year must be strictly positive (> 0), received {trading_days_per_year}",
            details={"trading_days_per_year": trading_days_per_year},
        )
    return trading_days_per_year


@dataclass(frozen=True)
class ReturnsResult:
    """Immutable container for multi-asset return calculations and summary statistics.

    Attributes:
        simple_returns: Daily simple returns DataFrame (T x K).
        log_returns: Daily logarithmic returns DataFrame (T x K).
        expected_returns: Annualized arithmetic mean simple returns (K,).
        cagr: Compound Annual Growth Rate per asset (K,).
        trading_days_per_year: Annualization factor (typically 252).
        observation_count: Number of return observations T.
        symbols: List of constituent asset symbols.
        start_date: First date of return series.
        end_date: Last date of return series.
    """

    simple_returns: pd.DataFrame
    log_returns: pd.DataFrame
    expected_returns: pd.Series
    cagr: pd.Series
    trading_days_per_year: int
    observation_count: int
    symbols: List[str]
    start_date: date
    end_date: date

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary metrics to dictionary representation."""
        return {
            "symbols": self.symbols,
            "observation_count": self.observation_count,
            "start_date": self.start_date.isoformat() if isinstance(self.start_date, date) else str(self.start_date),
            "end_date": self.end_date.isoformat() if isinstance(self.end_date, date) else str(self.end_date),
            "trading_days_per_year": self.trading_days_per_year,
            "expected_returns": self.expected_returns.to_dict(),
            "cagr": self.cagr.to_dict(),
        }


def _validate_and_prepare_prices(
    prices: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    price_column: str = "adj_close",
    date_column: str = "trading_date",
    symbol_column: str = "symbol",
) -> Tuple[pd.DataFrame, bool]:
    """Validate raw price input and standardize into a sorted chronological wide DataFrame.

    Checks:
    - Non-empty input.
    - Required columns in long-format DataFrames.
    - Strictly positive numeric prices (P > 0).
    - No infinite prices within observations.
    - No duplicate trading dates per symbol.
    - Chronological ordering of dates.
    - At least 2 price observations for return computation.

    CRITICAL NOTE:
    - This function preserves the original chronological date sequence and does NOT drop
      dates with NaNs. Dropping dates prior to computing returns causes multi-session price
      changes to be misclassified as single-session daily returns.

    Args:
        prices: Raw price data in wide format, long format, Series, or dict.
        price_column: Column name for prices in long-format DataFrame.
        date_column: Column name for dates in long-format DataFrame.
        symbol_column: Column name for symbols in long-format DataFrame.

    Returns:
        Tuple of:
        - pd.DataFrame: Standardized wide DataFrame (index=trading_date, columns=symbols).
        - bool: True if the original input was a single Series / 1D structure.

    Raises:
        InsufficientDataError: If input is empty or has < 2 observations.
        InvalidPriceError: If prices are non-positive, Inf, or non-numeric.
        FinancialCalculationError: If duplicate trading dates or missing required columns.
    """
    if prices is None:
        raise InsufficientDataError("Input prices cannot be None")

    is_single_series = False

    # 1. Standardize input types to pd.DataFrame
    if isinstance(prices, dict):
        if not prices:
            raise InsufficientDataError("Input price dictionary is empty")
        try:
            df = pd.DataFrame(prices)
        except Exception as exc:
            raise FinancialCalculationError(
                f"Failed to construct DataFrame from price dictionary: {exc}",
                details={"error": str(exc)},
            ) from exc
    elif isinstance(prices, pd.Series):
        if prices.empty:
            raise InsufficientDataError("Input price Series is empty")
        is_single_series = True
        col_name = str(prices.name) if prices.name is not None else "ASSET"
        df = prices.to_frame(name=col_name)
    elif isinstance(prices, pd.DataFrame):
        if prices.empty:
            raise InsufficientDataError("Input price DataFrame is empty")
        df = prices.copy()
    else:
        raise FinancialCalculationError(
            f"Unsupported price input type: {type(prices).__name__}; expected DataFrame, Series, or Dict",
            details={"received_type": type(prices).__name__},
        )

    # 2. Check for Long-Format vs Wide-Format
    if date_column in df.columns:
        # Long-format DataFrame
        if symbol_column in df.columns:
            if price_column not in df.columns:
                raise FinancialCalculationError(
                    f"Long-format DataFrame missing required price column '{price_column}'",
                    details={"missing_column": price_column, "available_columns": list(df.columns)},
                )
            # Check duplicate (symbol, trading_date)
            dup_mask = df.duplicated(subset=[symbol_column, date_column], keep=False)
            if dup_mask.any():
                dup_count = int(dup_mask.sum())
                raise FinancialCalculationError(
                    f"Found {dup_count} duplicate trading date records across symbols in long-format DataFrame",
                    details={"duplicate_count": dup_count},
                )
            try:
                wide_df = df.pivot(index=date_column, columns=symbol_column, values=price_column)
            except Exception as exc:
                raise FinancialCalculationError(
                    f"Failed to pivot long-format price DataFrame: {exc}",
                    details={"error": str(exc)},
                ) from exc
        else:
            # Single-asset long format with date_column
            if price_column not in df.columns:
                raise FinancialCalculationError(
                    f"Single-asset DataFrame missing required price column '{price_column}'",
                    details={"missing_column": price_column, "available_columns": list(df.columns)},
                )
            if df[date_column].duplicated().any():
                raise FinancialCalculationError("Duplicate trading dates detected in single-asset DataFrame")
            is_single_series = True
            wide_df = df.set_index(date_column)[[price_column]]
            wide_df.columns = ["ASSET"]
    else:
        # Wide-format: trading dates are in the DataFrame index
        wide_df = df.copy()
        if wide_df.index.duplicated().any():
            dup_dates = wide_df.index[wide_df.index.duplicated()].unique().tolist()
            raise FinancialCalculationError(
                f"Duplicate trading dates detected in DataFrame index: {dup_dates}",
                details={"duplicate_dates": [str(d) for d in dup_dates]},
            )

    # 3. Chronological sorting
    if not wide_df.index.is_monotonic_increasing:
        wide_df = wide_df.sort_index()

    # 4. Check minimum required observations
    if len(wide_df) < 2:
        raise InsufficientDataError(
            f"At least 2 price observations are required to compute returns (received {len(wide_df)})",
            details={"observations": len(wide_df), "minimum_required": 2},
        )

    # 5. Numeric & Sanity Validation
    for col in wide_df.columns:
        if not pd.api.types.is_numeric_dtype(wide_df[col]):
            raise InvalidPriceError(
                f"Price column '{col}' contains non-numeric data",
                details={"column": col, "dtype": str(wide_df[col].dtype)},
            )

    # Check for infinite values
    if np.isinf(wide_df.values).any():
        inf_cols = wide_df.columns[np.isinf(wide_df).any()].tolist()
        raise InvalidPriceError(
            f"Found infinite price values in columns: {inf_cols}",
            details={"columns_with_inf": inf_cols},
        )

    # Check for non-positive prices (P <= 0), ignoring NaNs (which represent unobserved sessions)
    non_pos_mask = (wide_df <= 0) & wide_df.notna()
    if non_pos_mask.any().any():
        bad_cols = wide_df.columns[non_pos_mask.any()].tolist()
        raise InvalidPriceError(
            f"Asset prices must be strictly positive (> 0); found non-positive values in columns: {bad_cols}",
            details={"columns_with_non_positive": bad_cols},
        )

    return wide_df, is_single_series


def calculate_simple_returns(
    prices: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    aligned: bool = True,
    price_column: str = "adj_close",
    date_column: str = "trading_date",
    symbol_column: str = "symbol",
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate daily simple returns from adjusted closing prices.

    Mathematical Formula:
        R_{i, t} = (P_{adj, i, t} / P_{adj, i, t-1}) - 1

    Financial Properties & Missing-Observation Safety:
    - Preserves the original chronological trading session sequence.
    - If price at t is missing OR price at t-1 is missing, return at t is strictly NaN.
    - Multi-session returns spanning missing sessions are NEVER misclassified as 1-day returns.
    - When consecutive valid observations resume, daily return calculation automatically resumes.
    - When aligned=True, listwise-complete filtering is applied to the RESULTING return matrix,
      ensuring cross-sectional returns share identical, genuine 1-day observation windows.
    - Simple returns are strictly linear and additive across portfolio weights:
        R_{p, t} = sum_{i=1}^K w_i * R_{i, t}

    Args:
        prices: Adjusted price data in wide format, long format, Series, or dict.
        aligned: If True (default), filters resulting return matrix to listwise-complete rows.
        price_column: Column name for price in long-format data.
        date_column: Column name for date in long-format data.
        symbol_column: Column name for symbol in long-format data.

    Returns:
        pd.DataFrame (or pd.Series if input was 1D) containing daily simple returns.

    Raises:
        InsufficientDataError: If < 2 observations or no valid returns.
        InvalidPriceError: If prices <= 0, Inf, or non-numeric.
        UnalignedObservationsError: If aligned=True and no common return dates exist.
        FinancialCalculationError: If schema validation fails.
    """
    wide_df, is_single_series = _validate_and_prepare_prices(
        prices=prices,
        price_column=price_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )

    # Compute daily simple return: (P_t / P_{t-1}) - 1
    # Missing P_t or missing P_{t-1} produces NaN at row t
    returns = (wide_df / wide_df.shift(1) - 1.0).iloc[1:]

    if aligned:
        clean_ret = returns.dropna(axis=0, how="any")
        if clean_ret.empty:
            if not is_single_series and len(wide_df.columns) > 1:
                raise UnalignedObservationsError(
                    "No common trading dates found with valid returns across all assets for listwise-complete alignment",
                    details={"assets": list(wide_df.columns)},
                )
            else:
                raise InsufficientDataError("No valid consecutive daily return observations available")
        returns = clean_ret

    if is_single_series:
        series_res = returns.iloc[:, 0]
        series_res.name = wide_df.columns[0]
        return series_res
    return returns


def calculate_log_returns(
    prices: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    aligned: bool = True,
    price_column: str = "adj_close",
    date_column: str = "trading_date",
    symbol_column: str = "symbol",
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate daily logarithmic (continuously compounded) returns from adjusted prices.

    Mathematical Formula:
        r_{i, t} = ln(P_{adj, i, t} / P_{adj, i, t-1}) = ln(1 + R_{i, t})

    Financial Properties & Missing-Observation Safety:
    - Time-additive across periods: r_{0, T} = sum_{t=1}^T r_t
    - Suitable for time-series modeling, predictive machine learning, and stationarity tests.
    - Missing previous or current prices strictly produce NaN.
    - When aligned=True, filters resulting log return matrix to listwise-complete dates.
    - WARNING: Log returns are NOT cross-sectionally additive across portfolio assets!
        ln sum(w_i * P_i) != sum(w_i * ln(P_i))
      Must NEVER be used in place of simple returns for portfolio optimization or
      covariance matrix estimation in Modern Portfolio Theory.

    Args:
        prices: Adjusted price data in wide format, long format, Series, or dict.
        aligned: If True (default), filters resulting return matrix to listwise-complete rows.
        price_column: Column name for price in long-format data.
        date_column: Column name for date in long-format data.
        symbol_column: Column name for symbol in long-format data.

    Returns:
        pd.DataFrame (or pd.Series if input was 1D) containing daily log returns.

    Raises:
        InsufficientDataError: If < 2 observations or no valid returns.
        InvalidPriceError: If prices <= 0, Inf, or non-numeric.
        UnalignedObservationsError: If aligned=True and no common return dates exist.
        FinancialCalculationError: If schema validation fails.
    """
    wide_df, is_single_series = _validate_and_prepare_prices(
        prices=prices,
        price_column=price_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )

    # Compute daily log return: ln(P_t / P_{t-1})
    returns = np.log(wide_df / wide_df.shift(1)).iloc[1:]

    if aligned:
        clean_ret = returns.dropna(axis=0, how="any")
        if clean_ret.empty:
            if not is_single_series and len(wide_df.columns) > 1:
                raise UnalignedObservationsError(
                    "No common trading dates found with valid returns across all assets for listwise-complete alignment",
                    details={"assets": list(wide_df.columns)},
                )
            else:
                raise InsufficientDataError("No valid consecutive daily return observations available")
        returns = clean_ret

    if is_single_series:
        series_res = returns.iloc[:, 0]
        series_res.name = wide_df.columns[0]
        return series_res
    return returns


def calculate_expected_returns(
    data: Union[pd.DataFrame, pd.Series],
    is_returns: bool = True,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    aligned: bool = True,
) -> Union[pd.Series, float]:
    """Calculate annualized arithmetic expected returns for Markowitz optimization.

    Mathematical Formula:
        mu_{i, annual} = N_trading * mean(R_{i, daily})
                       = 252 * (1 / T) * sum_{t=1}^T R_{i, t}

    Approved Financial Standard:
    - This is the canonical expected-return input for single-period Markowitz (1952)
      mean-variance portfolio optimization.
    - Assumes independent and identically distributed (i.i.d.) returns over trading days.
    - The standard scalar for Indian equities (NSE) is N = 252 trading days per year.
    - For multi-asset portfolio input, returns are evaluated strictly across a common
      listwise-complete observation set when aligned=True.
    - Multi-period CAGR must NOT be substituted for arithmetic expected returns in Markowitz.

    Args:
        data: Daily simple returns (if is_returns=True) or adjusted prices (if is_returns=False).
        is_returns: If True, data is treated as daily simple returns. If False, simple returns
            are computed from prices first.
        trading_days_per_year: Annualization factor (positive integer, default 252).
        aligned: If True (default), enforces listwise-complete alignment across assets.

    Returns:
        pd.Series of annualized expected returns per asset (or float if single-asset Series).

    Raises:
        InsufficientDataError: If data is empty or contains zero valid return observations.
        FinancialCalculationError: If trading_days_per_year is invalid or returns contain Inf.
    """
    _validate_trading_days_per_year(trading_days_per_year)

    if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
        raise InsufficientDataError("Input data for expected returns calculation cannot be empty")

    if not is_returns:
        returns_df = calculate_simple_returns(data, aligned=aligned)
    else:
        returns_df = data.copy()

    # Numeric & infinite validation on returns
    if isinstance(returns_df, pd.Series):
        if not pd.api.types.is_numeric_dtype(returns_df):
            raise FinancialCalculationError("Returns series contains non-numeric data")
        if np.isinf(returns_df.values).any():
            raise FinancialCalculationError("Found infinite values in returns series")
        clean_ret = returns_df.dropna()
        if clean_ret.empty:
            raise InsufficientDataError("No valid return observations available after dropping NaNs")
        return float(clean_ret.mean() * trading_days_per_year)

    elif isinstance(returns_df, pd.DataFrame):
        for col in returns_df.columns:
            if not pd.api.types.is_numeric_dtype(returns_df[col]):
                raise FinancialCalculationError(f"Returns column '{col}' contains non-numeric data")
        if np.isinf(returns_df.values).any():
            raise FinancialCalculationError("Found infinite values in returns DataFrame")

        # Multi-asset portfolio alignment: enforce common listwise-complete return observations
        if aligned:
            clean_ret = returns_df.dropna(axis=0, how="any")
            if clean_ret.empty:
                raise InsufficientDataError("No common listwise-complete return observations across assets")
        else:
            clean_ret = returns_df

        expected_ret = clean_ret.mean(axis=0) * trading_days_per_year
        return expected_ret

    else:
        raise FinancialCalculationError(
            f"Unsupported input type for expected returns: {type(data).__name__}; expected DataFrame or Series"
        )


def _evaluate_asset_cagr_from_prices(
    price_series: pd.Series,
    col_name: str,
    trading_days_per_year: int,
    min_coverage_ratio: float = MIN_CAGR_HISTORY_COVERAGE_RATIO,
    max_consecutive_missing_days: int = MAX_CAGR_CONSECUTIVE_MISSING_DAYS,
) -> float:
    """Evaluate CAGR for an individual asset price series with strict quality gates.

    Formula:
        CAGR = (P_T / P_0) ** (252 / T) - 1

    Enforces:
    1. Strictly positive, finite boundary prices P_0 and P_T.
    2. Date-based price series require verified official NSE exchange calendar coverage.
       If calendar coverage cannot be established, raises InsufficientDataError.
       Consecutive calendar dates are NEVER assumed to establish trading sessions.
    3. Compliance with data quality gates: coverage >= 95%, max consecutive missing <= 3.
    4. Consecutive integer index fallback is supported strictly for synthetic non-date numeric data.
    """
    if len(price_series) < 2:
        raise InsufficientDataError(
            f"Asset '{col_name}' requires at least 2 price observations to compute CAGR",
            details={"asset": col_name, "observations": len(price_series)},
        )

    t_0 = price_series.index[0]
    t_T = price_series.index[-1]

    p_0 = price_series.iloc[0]
    p_T = price_series.iloc[-1]

    if pd.isna(p_0):
        raise InsufficientDataError(
            f"Starting price for asset '{col_name}' is missing at evaluation boundary {t_0}",
            details={"asset": col_name, "boundary": str(t_0)},
        )
    if pd.isna(p_T):
        raise InsufficientDataError(
            f"Ending price for asset '{col_name}' is missing at evaluation boundary {t_T}",
            details={"asset": col_name, "boundary": str(t_T)},
        )

    if p_0 <= 0 or np.isinf(p_0):
        raise InvalidPriceError(
            f"Starting price for asset '{col_name}' must be strictly positive and finite (got {p_0})",
            details={"asset": col_name, "p_0": p_0},
        )
    if p_T <= 0 or np.isinf(p_T):
        raise InvalidPriceError(
            f"Ending price for asset '{col_name}' must be strictly positive and finite (got {p_T})",
            details={"asset": col_name, "p_T": p_T},
        )

    d_0 = _normalize_to_date(t_0)
    d_T = _normalize_to_date(t_T)

    normalized_indices = [_normalize_to_date(idx) for idx in price_series.index]
    is_date_based = None not in normalized_indices and d_0 is not None and d_T is not None

    if is_date_based:
        try:
            expected_sessions = nse_calendar.get_trading_days(d_0, d_T)
        except (HolidayCircularNotFoundError, CalendarDateRangeError, InvalidCalendarDateError, Exception) as exc:
            raise InsufficientDataError(
                f"Cannot establish verified exchange calendar coverage for price series dates between {d_0} and {d_T}: {exc}",
                details={"asset": col_name, "start_date": str(d_0), "end_date": str(d_T), "error": str(exc)},
            ) from exc

        if d_0 not in expected_sessions:
            raise InsufficientDataError(
                f"Starting date {d_0} for asset '{col_name}' is not an active exchange trading session",
                details={"asset": col_name, "start_date": str(d_0)},
            )
        if d_T not in expected_sessions:
            raise InsufficientDataError(
                f"Ending date {d_T} for asset '{col_name}' is not an active exchange trading session",
                details={"asset": col_name, "end_date": str(d_T)},
            )

        # Validate that normalized dates are unique (e.g. no multiple intraday timestamps on the same calendar date)
        norm_idx_series = pd.Series(normalized_indices)
        if norm_idx_series.duplicated().any():
            dup_dates = norm_idx_series[norm_idx_series.duplicated()].unique().tolist()
            raise FinancialCalculationError(
                f"Duplicate trading dates detected after date normalization for asset '{col_name}': {dup_dates}",
                details={"asset": col_name, "duplicate_dates": [str(d) for d in dup_dates]},
            )

        # Validate that every supplied date belongs to the verified NSE trading-session sequence
        expected_sessions_set = set(expected_sessions)
        invalid_dates = [d for d in normalized_indices if d not in expected_sessions_set]
        if invalid_dates:
            raise InsufficientDataError(
                f"Price series for asset '{col_name}' contains date(s) that are not active exchange trading sessions (e.g. weekend or holiday): {invalid_dates}",
                details={"asset": col_name, "invalid_dates": [str(d) for d in invalid_dates]},
            )

        T = len(expected_sessions) - 1
        if T < 1:
            raise InsufficientDataError(
                f"At least 2 distinct trading sessions required to compute CAGR for asset '{col_name}'",
                details={"asset": col_name},
            )

        date_series = pd.Series(price_series.values, index=normalized_indices)
        session_prices = date_series.reindex(expected_sessions)

        p_start = session_prices.iloc[0]
        p_end = session_prices.iloc[-1]
        if pd.isna(p_start) or p_start <= 0 or np.isinf(p_start):
            raise InsufficientDataError(
                f"Starting price for asset '{col_name}' is missing or invalid on first trading day {expected_sessions[0]}"
            )
        if pd.isna(p_end) or p_end <= 0 or np.isinf(p_end):
            raise InsufficientDataError(
                f"Ending price for asset '{col_name}' is missing or invalid on last trading day {expected_sessions[-1]}"
            )

        n_expected = len(expected_sessions)
        n_valid = int(session_prices.notna().sum())
        coverage = n_valid / n_expected
        if coverage < min_coverage_ratio:
            raise InsufficientDataError(
                f"Asset '{col_name}' trading history coverage ratio {coverage:.2%} is below minimum required {min_coverage_ratio:.0%} for CAGR",
                details={
                    "asset": col_name,
                    "coverage": coverage,
                    "min_coverage_ratio": min_coverage_ratio,
                    "valid_count": n_valid,
                    "expected_count": n_expected,
                },
            )

        max_consec = 0
        curr_consec = 0
        for is_nan in session_prices.isna():
            if is_nan:
                curr_consec += 1
                if curr_consec > max_consec:
                    max_consec = curr_consec
            else:
                curr_consec = 0

        if max_consec > max_consecutive_missing_days:
            raise InsufficientDataError(
                f"Asset '{col_name}' has {max_consec} consecutive missing trading sessions (exceeding limit of {max_consecutive_missing_days})",
                details={
                    "asset": col_name,
                    "consecutive_missing": max_consec,
                    "limit": max_consecutive_missing_days,
                },
            )

        growth = float(p_end / p_start)
        if growth <= 0:
            raise InvalidPriceError(
                f"Compounded growth for asset '{col_name}' is non-positive ({growth})",
                details={"asset": col_name, "growth": growth},
            )
        return float(np.power(growth, trading_days_per_year / T) - 1.0)

    else:
        # Non-date index: strictly consecutive integer index fallback for synthetic data
        if price_series.isna().any():
            raise InsufficientDataError(
                f"Cannot establish elapsed trading sessions for asset '{col_name}': non-date price series contains missing observations",
                details={"asset": col_name},
            )

        if pd.api.types.is_integer_dtype(price_series.index):
            if not price_series.index.is_monotonic_increasing:
                raise FinancialCalculationError(
                    f"Non-date numeric index for asset '{col_name}' must be strictly monotonic increasing",
                    details={"asset": col_name},
                )
            diffs = np.diff(price_series.index)
            if not (diffs == 1).all():
                raise InsufficientDataError(
                    f"Cannot establish elapsed trading sessions for asset '{col_name}': non-consecutive numeric index",
                    details={"asset": col_name, "diffs": diffs.tolist()},
                )
        else:
            raise InsufficientDataError(
                f"Cannot establish elapsed trading sessions for asset '{col_name}': non-date index must be strictly consecutive integers",
                details={"asset": col_name, "index_type": type(price_series.index).__name__},
            )

        T = len(price_series) - 1
        if T < 1:
            raise InsufficientDataError(f"At least 2 observations required for CAGR for asset '{col_name}'")

        growth = float(p_T / p_0)
        if growth <= 0:
            raise InvalidPriceError(f"Compounded growth for asset '{col_name}' is non-positive ({growth})")
        return float(np.power(growth, trading_days_per_year / T) - 1.0)


def _validate_return_index_continuity(index: pd.Index) -> int:
    """Validate that returns index is strictly continuous and without missing sessions.

    Rules:
    1. For date-based indices:
       - Validate sorted, unique dates (no duplicates, strictly monotonic increasing).
       - Query nse_calendar between first and last return date.
       - If calendar coverage cannot be established, raise InsufficientDataError.
       - Require supplied return index to match complete expected session sequence exactly.
       - If any exchange trading session is missing, raise InsufficientDataError.
    2. For non-date numeric indices:
       - Require strictly consecutive integer indices with increments of exactly 1.
       - Reject discontinuous indices with InsufficientDataError.
    3. Any other index type:
       - Reject with InsufficientDataError.

    Returns:
        int: Number of return periods N.

    Raises:
        FinancialCalculationError: If duplicate or unsorted dates/indices.
        InsufficientDataError: If discontinuous, missing sessions, or unverified calendar.
    """
    if len(index) == 0:
        raise InsufficientDataError("Returns index is empty")

    if index.duplicated().any():
        dup_vals = index[index.duplicated()].unique().tolist()
        raise FinancialCalculationError(
            f"Duplicate dates/indices detected in returns: {dup_vals}",
            details={"duplicates": [str(v) for v in dup_vals]},
        )

    normalized_dates = [_normalize_to_date(x) for x in index]
    is_date_based = None not in normalized_dates

    if is_date_based:
        for i in range(len(normalized_dates) - 1):
            if normalized_dates[i] >= normalized_dates[i + 1]:
                raise FinancialCalculationError(
                    f"Returns date index must be strictly increasing and chronological (found {normalized_dates[i]} >= {normalized_dates[i+1]})",
                    details={"prev_date": str(normalized_dates[i]), "curr_date": str(normalized_dates[i+1])},
                )

        d_0 = normalized_dates[0]
        d_T = normalized_dates[-1]

        try:
            expected_sessions = nse_calendar.get_trading_days(d_0, d_T)
        except (HolidayCircularNotFoundError, CalendarDateRangeError, InvalidCalendarDateError, Exception) as exc:
            raise InsufficientDataError(
                f"Cannot establish verified exchange calendar coverage for return dates between {d_0} and {d_T}: {exc}",
                details={"start_date": str(d_0), "end_date": str(d_T), "error": str(exc)},
            ) from exc

        if normalized_dates != expected_sessions:
            missing_sessions = [d for d in expected_sessions if d not in normalized_dates]
            extra_sessions = [d for d in normalized_dates if d not in expected_sessions]
            if missing_sessions:
                raise InsufficientDataError(
                    f"Returns sequence is missing expected trading session(s): {missing_sessions}",
                    details={
                        "missing_sessions": [str(d) for d in missing_sessions],
                        "expected_count": len(expected_sessions),
                        "provided_count": len(normalized_dates),
                    },
                )
            if extra_sessions:
                raise InsufficientDataError(
                    f"Returns sequence contains non-trading session(s): {extra_sessions}",
                    details={"extra_sessions": [str(d) for d in extra_sessions]},
                )

        return len(normalized_dates)

    if pd.api.types.is_integer_dtype(index):
        if not index.is_monotonic_increasing:
            raise FinancialCalculationError(
                "Returns numeric index must be strictly monotonic increasing",
                details={"index": list(index)},
            )
        diffs = np.diff(index)
        if not (diffs == 1).all():
            raise InsufficientDataError(
                "Returns numeric index is discontinuous; increments must be exactly 1",
                details={"diffs": diffs.tolist()},
            )
        return len(index)

    raise InsufficientDataError(
        f"Unsupported or discontinuous returns index type: {type(index).__name__}; expected date-based or consecutive integer index",
        details={"index_type": type(index).__name__},
    )


def calculate_cagr(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    is_prices: bool = True,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    aligned: bool = True,
    min_coverage_ratio: float = MIN_CAGR_HISTORY_COVERAGE_RATIO,
    max_consecutive_missing_days: int = MAX_CAGR_CONSECUTIVE_MISSING_DAYS,
    price_column: str = "adj_close",
    date_column: str = "trading_date",
    symbol_column: str = "symbol",
) -> Union[pd.Series, float]:
    """Calculate the Compound Annual Growth Rate (CAGR).

    Mathematical Formulation:
        Price-Input Mode (is_prices=True):
            CAGR_i = (P_{i, T} / P_{i, 0}) ** (252 / T) - 1
            where:
            - P_{i, 0} is initial valid adjusted price at start of evaluation window.
            - P_{i, T} is final valid adjusted price at end of evaluation window.
            - T represents elapsed trading sessions between evaluated boundaries
              (derived from official NSE exchange calendar), NOT merely count of valid returns.
            - Internal missing observations are strictly evaluated against configured data-quality
              policy (coverage >= 95%, max consecutive missing trading sessions <= 3).
            - Unacceptable discontinuity or insufficient history raises InsufficientDataError.

        Return-Input Mode (is_prices=False):
            CAGR_i = ( prod_{t=1}^N (1 + R_{i, t}) ) ** (252 / N) - 1
            where:
            - R_{i, t} is a continuous sequence of valid daily simple returns (R_{i, t} > -100%).
            - Internal missing observations (NaN) are strictly prohibited and raise InsufficientDataError.
            - N is the number of actual daily return periods.

    Distinction Between CAGR and Markowitz Expected Returns:
    - CAGR measures the realized *geometric* multi-period compounding growth rate.
    - Due to Jensen's inequality and volatility drag (variance drag):
        CAGR approx mu_arithmetic - 0.5 * sigma_annual^2
    - CAGR is strictly path-dependent and non-linear across assets.
    - CAGR is mandatory for multi-period backtest performance reporting, but must
      NEVER be used as the expected-return vector in the Markowitz quadratic optimizer.

    Args:
        data: Adjusted price data (if is_prices=True) or daily simple returns (if is_prices=False).
        is_prices: If True, data is treated as price series. If False, treated as simple returns.
        trading_days_per_year: Annualization factor (positive integer, default 252).
        aligned: If True (default), enforces common evaluation boundaries across all assets.
        min_coverage_ratio: Minimum required coverage ratio (default 0.95).
        max_consecutive_missing_days: Maximum allowable consecutive missing sessions (default 3).
        price_column: Column name for price in long-format data.
        date_column: Column name for date in long-format data.
        symbol_column: Column name for symbol in long-format data.

    Returns:
        pd.Series of CAGR per asset (or float if single-asset Series).

    Raises:
        InsufficientDataError: If < 2 observations, missing boundary prices, internal NaNs in returns,
            unacceptable price discontinuity, or coverage < min_coverage_ratio.
        InvalidPriceError: If return <= -100% or prices non-positive.
        FinancialCalculationError: If trading_days_per_year is invalid or input contains Inf.
    """
    _validate_trading_days_per_year(trading_days_per_year)

    if data is None:
        raise InsufficientDataError("Input data for CAGR calculation cannot be None")

    if is_prices:
        wide_df, is_single_series = _validate_and_prepare_prices(
            prices=data,
            price_column=price_column,
            date_column=date_column,
            symbol_column=symbol_column,
        )

        if aligned:
            # Enforce common evaluation boundaries across all assets
            t_0 = wide_df.index[0]
            t_T = wide_df.index[-1]
            for col in wide_df.columns:
                p_0 = wide_df[col].iloc[0]
                p_T = wide_df[col].iloc[-1]
                if pd.isna(p_0):
                    raise InsufficientDataError(
                        f"Starting price for asset '{col}' is missing at common evaluation boundary {t_0}",
                        details={"asset": col, "boundary": str(t_0)},
                    )
                if pd.isna(p_T):
                    raise InsufficientDataError(
                        f"Ending price for asset '{col}' is missing at common evaluation boundary {t_T}",
                        details={"asset": col, "boundary": str(t_T)},
                    )

            cagr_results = {}
            for col in wide_df.columns:
                cagr_val = _evaluate_asset_cagr_from_prices(
                    price_series=wide_df[col],
                    col_name=col,
                    trading_days_per_year=trading_days_per_year,
                    min_coverage_ratio=min_coverage_ratio,
                    max_consecutive_missing_days=max_consecutive_missing_days,
                )
                cagr_results[col] = cagr_val

            if is_single_series:
                return cagr_results[wide_df.columns[0]]
            return pd.Series(cagr_results, index=wide_df.columns)

        else:
            # Unaligned: each asset evaluates over its own valid boundary window
            cagr_results = {}
            for col in wide_df.columns:
                valid_mask = wide_df[col].notna()
                valid_idx = wide_df.index[valid_mask]
                if len(valid_idx) < 2:
                    raise InsufficientDataError(
                        f"Asset '{col}' has fewer than 2 valid price observations",
                        details={"asset": col},
                    )
                sub_series = wide_df.loc[valid_idx[0]:valid_idx[-1], col]
                cagr_val = _evaluate_asset_cagr_from_prices(
                    price_series=sub_series,
                    col_name=col,
                    trading_days_per_year=trading_days_per_year,
                    min_coverage_ratio=min_coverage_ratio,
                    max_consecutive_missing_days=max_consecutive_missing_days,
                )
                cagr_results[col] = cagr_val

            if is_single_series:
                return cagr_results[wide_df.columns[0]]
            return pd.Series(cagr_results, index=wide_df.columns)

    else:
        # data is daily simple returns
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as exc:
                raise FinancialCalculationError(
                    f"Failed to convert dictionary to returns DataFrame: {exc}",
                    details={"error": str(exc)},
                ) from exc

        if isinstance(data, pd.Series):
            if data.empty:
                raise InsufficientDataError("Returns series is empty for CAGR calculation")
            if not pd.api.types.is_numeric_dtype(data):
                raise FinancialCalculationError("Returns series contains non-numeric data")
            if np.isinf(data.values).any():
                raise FinancialCalculationError("Returns series contains infinite values")
            if data.isna().any():
                raise InsufficientDataError(
                    "Returns sequence contains missing observations; CAGR requires a continuous sequence of valid daily returns",
                    details={"missing_count": int(data.isna().sum())},
                )
            if (data <= -1.0).any():
                raise InvalidPriceError("Return <= -100% detected; total loss cannot be compounded")

            n_periods = _validate_return_index_continuity(data.index)
            if n_periods < 1:
                raise InsufficientDataError("At least 1 valid return observation required for CAGR")
            total_growth = float((1.0 + data).prod())
            if total_growth <= 0:
                raise InvalidPriceError("Total compounded return is non-positive; CAGR undefined")
            exponent = trading_days_per_year / n_periods
            return float(np.power(total_growth, exponent) - 1.0)

        elif isinstance(data, pd.DataFrame):
            if data.empty:
                raise InsufficientDataError("Returns DataFrame is empty for CAGR calculation")
            for col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise FinancialCalculationError(f"Returns column '{col}' contains non-numeric data")
            if np.isinf(data.values).any():
                raise FinancialCalculationError("Returns DataFrame contains infinite values")

            if data.isna().any().any():
                raise InsufficientDataError(
                    "Returns DataFrame contains missing observations; CAGR requires a continuous sequence of valid daily returns across all assets",
                    details={"missing_columns": data.columns[data.isna().any()].tolist()},
                )
            if (data <= -1.0).any().any():
                raise InvalidPriceError("Return <= -100% detected; total loss cannot be compounded")

            n_periods = _validate_return_index_continuity(data.index)
            if n_periods < 1:
                raise InsufficientDataError("At least 1 valid return observation required for CAGR")
            total_growth = (1.0 + data).prod(axis=0)
            if (total_growth <= 0).any():
                bad_cols = data.columns[total_growth <= 0].tolist()
                raise InvalidPriceError(f"Non-positive compounded growth in columns: {bad_cols}")
            exponent = trading_days_per_year / n_periods
            return np.power(total_growth, exponent) - 1.0

        else:
            raise FinancialCalculationError(
                f"Unsupported returns input type for CAGR: {type(data).__name__}; expected Series or DataFrame"
            )


def compute_returns_summary(
    prices: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    aligned: bool = True,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    price_column: str = "adj_close",
    date_column: str = "trading_date",
    symbol_column: str = "symbol",
) -> ReturnsResult:
    """Compute a complete quantitative returns summary for portfolio assets.

    Executes full data validation, aligned returns computation, log returns,
    annualized arithmetic expected returns, and geometric CAGR in a single
    deterministic, audited pass.

    Args:
        prices: Adjusted price data in wide format, long format, Series, or dict.
        aligned: If True (default), enforces listwise-complete alignment across all assets.
        trading_days_per_year: Annualization factor (positive integer, default 252).
        price_column: Column name for price in long-format data.
        date_column: Column name for date in long-format data.
        symbol_column: Column name for symbol in long-format data.

    Returns:
        ReturnsResult containing simple returns, log returns, expected returns,
        CAGR, and observation metadata.

    Raises:
        InsufficientDataError: If < 2 observations or empty input.
        InvalidPriceError: If prices <= 0, Inf, or non-numeric.
        FinancialCalculationError: If schema validation fails or invalid trading_days_per_year.
    """
    _validate_trading_days_per_year(trading_days_per_year)

    # 1. Simple Returns (T x K)
    simple_ret = calculate_simple_returns(
        prices=prices,
        aligned=aligned,
        price_column=price_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )
    if isinstance(simple_ret, pd.Series):
        simple_df = simple_ret.to_frame()
    else:
        simple_df = simple_ret

    # 2. Log Returns (T x K)
    log_ret = calculate_log_returns(
        prices=prices,
        aligned=aligned,
        price_column=price_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )
    if isinstance(log_ret, pd.Series):
        log_df = log_ret.to_frame()
    else:
        log_df = log_ret

    # 3. Expected Returns (K,)
    expected_ret = calculate_expected_returns(
        simple_df,
        is_returns=True,
        trading_days_per_year=trading_days_per_year,
        aligned=aligned,
    )
    if isinstance(expected_ret, float):
        expected_ret_series = pd.Series([expected_ret], index=simple_df.columns)
    else:
        expected_ret_series = expected_ret

    # 4. CAGR (K,)
    cagr_res = calculate_cagr(
        prices,
        is_prices=True,
        trading_days_per_year=trading_days_per_year,
        aligned=aligned,
        price_column=price_column,
        date_column=date_column,
        symbol_column=symbol_column,
    )
    if isinstance(cagr_res, float):
        cagr_series = pd.Series([cagr_res], index=simple_df.columns)
    else:
        cagr_series = cagr_res

    symbols = list(simple_df.columns)
    n_periods = len(simple_df)
    start_dt = simple_df.index[0]
    end_dt = simple_df.index[-1]

    if hasattr(start_dt, "date"):
        start_date_obj = start_dt.date()
    else:
        start_date_obj = start_dt

    if hasattr(end_dt, "date"):
        end_date_obj = end_dt.date()
    else:
        end_date_obj = end_dt

    logger.info(
        "Computed returns summary for %d assets (%d observations) from %s to %s",
        len(symbols),
        n_periods,
        start_date_obj,
        end_date_obj,
    )

    return ReturnsResult(
        simple_returns=simple_df,
        log_returns=log_df,
        expected_returns=expected_ret_series,
        cagr=cagr_series,
        trading_days_per_year=trading_days_per_year,
        observation_count=n_periods,
        symbols=symbols,
        start_date=start_date_obj,
        end_date=end_date_obj,
    )
