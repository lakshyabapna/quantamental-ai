"""Quantamental AI - Feature Engineering & Anti-Leakage Pipeline (Module B).

Constructs strictly causal, lagged technical and statistical features from log returns
for forward realized volatility forecasting:
    - Feature timestamps strictly satisfy tau <= t - 1 for target beginning at t.
    - Uses production ``calculate_log_returns`` from ``app.portfolio_engine.returns``.
    - Prohibits ``bfill`` and unconditional ``ffill``.
    - Explicitly validates missing values, index monotonicity, and minimum history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.config import settings
from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)
from app.portfolio_engine.returns import calculate_log_returns


DEFAULT_HORIZON_DAYS: int = 21
DEFAULT_ROLLING_WINDOWS: Tuple[int, ...] = (5, 10, 21, 63)
DEFAULT_RETURN_LAGS: Tuple[int, ...] = (1, 2, 5, 21)
_MIN_VOL_DENOM: float = 1e-12


@dataclass(frozen=True)
class FeatureDataset:
    """Immutable container for aligned, anti-leakage features and forward volatility target.

    For every index timestamp ``t`` in ``X.index``:
        - ``X.loc[t]`` is computed strictly from observations at ``tau <= t - 1``.
        - ``y.loc[t]`` is the forward ``horizon_days`` annualized realized volatility
          over ``[t, t + horizon_days - 1]``.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_names: Tuple[str, ...]
    horizon_days: int
    trading_days_per_year: int
    target_start_dates: pd.DatetimeIndex
    target_end_dates: pd.DatetimeIndex
    data_snapshot_hash: str
    metadata: Dict[str, Union[int, float, str, bool]] = field(default_factory=dict)


def _validate_price_series(
    prices: Union[pd.Series, pd.DataFrame],
    name: str,
    target_column: Optional[str] = None,
) -> pd.Series:
    """Validate and extract a strictly sorted, finite, positive price Series."""
    if isinstance(prices, pd.DataFrame):
        if target_column is not None:
            if target_column not in prices.columns:
                raise AIEngineException(
                    f"Target column '{target_column}' not found in {name} DataFrame.",
                    details={"available_columns": list(prices.columns)},
                )
            series = prices[target_column].copy()
        elif prices.shape[1] == 1:
            series = prices.iloc[:, 0].copy()
        else:
            raise AIEngineException(
                f"{name} DataFrame has multiple columns; specify 'target_column'.",
                details={"columns": list(prices.columns)},
            )
    elif isinstance(prices, pd.Series):
        series = prices.copy()
    else:
        raise AIEngineException(
            f"{name} must be a pandas Series or DataFrame, got {type(prices).__name__}."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as exc:
            raise AIEngineException(
                f"{name} index must be convertible to DatetimeIndex."
            ) from exc

    if series.index.has_duplicates:
        raise AIEngineException(f"{name} index contains duplicate timestamps.")

    if not series.index.is_monotonic_increasing:
        raise DataLeakageError(
            f"{name} index must be strictly sorted in ascending chronological order."
        )

    if series.isna().any():
        n_missing = int(series.isna().sum())
        raise InsufficientDataError(
            f"{name} contains {n_missing} missing price observation(s). "
            "Unconditional forward-fill and backward-fill are prohibited.",
            details={"series": name, "missing_count": n_missing},
        )

    values = np.asarray(series, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise AIEngineException(
            f"{name} must contain strictly positive, finite prices."
        )

    return series


def _compute_snapshot_hash(
    asset_series: pd.Series,
    benchmark_series: Optional[pd.Series],
    volume_series: Optional[pd.Series],
    horizon_days: int,
) -> str:
    """Compute deterministic SHA-256 hash of input data and horizon configuration."""
    if not isinstance(asset_series.index, pd.DatetimeIndex):
        raise AIEngineException("asset_series index must be a DatetimeIndex.")
    hasher = hashlib.sha256()
    hasher.update(f"horizon={horizon_days};len={len(asset_series)};".encode("utf-8"))
    hasher.update(np.asarray(asset_series.index, dtype="datetime64[ns]").astype(np.int64).tobytes())
    hasher.update(np.asarray(asset_series, dtype=np.float64).tobytes())
    if benchmark_series is not None:
        hasher.update(b"bm:")
        hasher.update(np.asarray(benchmark_series, dtype=np.float64).tobytes())
    if volume_series is not None:
        hasher.update(b"vol:")
        hasher.update(np.asarray(volume_series, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def build_volatility_features(
    asset_prices: Union[pd.Series, pd.DataFrame],
    *,
    target_column: Optional[str] = None,
    benchmark_prices: Optional[Union[pd.Series, pd.DataFrame]] = None,
    benchmark_column: Optional[str] = None,
    volumes: Optional[Union[pd.Series, pd.DataFrame]] = None,
    volume_column: Optional[str] = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    trading_days_per_year: int = settings.TRADING_DAYS_PER_YEAR,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    return_lags: Sequence[int] = DEFAULT_RETURN_LAGS,
) -> FeatureDataset:
    """Build strictly causal lagged features (tau <= t-1) and forward h-day volatility target.

    Parameters
    ----------
    asset_prices:
        Adjusted closing prices for the target asset (Series or DataFrame).
    target_column:
        Column name if ``asset_prices`` is a multi-column DataFrame.
    benchmark_prices:
        Optional benchmark adjusted closing prices (e.g., ``^NSEI``).
    benchmark_column:
        Column name if ``benchmark_prices`` is a multi-column DataFrame.
    volumes:
        Optional daily trading volume series.
    volume_column:
        Column name if ``volumes`` is a multi-column DataFrame.
    horizon_days:
        Forward realized volatility horizon in trading days (default: 21).
    trading_days_per_year:
        Annualization factor (default: 252).
    rolling_windows:
        Rolling volatility lookback windows ending at ``t - 1`` (default: (5, 10, 21, 63)).
    return_lags:
        Lagged log-return offsets >= 1 (default: (1, 2, 5, 21)).

    Returns
    -------
    FeatureDataset
        Clean, aligned feature matrix ``X`` and target Series ``y``.
    """
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
        raise AIEngineException(
            f"horizon_days must be a positive integer, got {horizon_days!r}."
        )
    if trading_days_per_year < 1:
        raise AIEngineException(
            f"trading_days_per_year must be >= 1, got {trading_days_per_year!r}."
        )

    lags = tuple(sorted(set(return_lags)))
    windows = tuple(sorted(set(rolling_windows)))
    if not lags or any(lag < 1 for lag in lags):
        raise DataLeakageError(
            "All return_lags must be >= 1 so features strictly precede target t."
        )
    if not windows or any(w < 2 for w in windows):
        raise AIEngineException("All rolling_windows must be >= 2.")

    price_s = _validate_price_series(asset_prices, "asset_prices", target_column=target_column)

    bm_s: Optional[pd.Series] = None
    if benchmark_prices is not None:
        bm_s = _validate_price_series(
            benchmark_prices, "benchmark_prices", target_column=benchmark_column
        )

    vol_s: Optional[pd.Series] = None
    if volumes is not None:
        if isinstance(volumes, pd.DataFrame):
            col = volume_column or (target_column if target_column in volumes.columns else None)
            if col is None and volumes.shape[1] == 1:
                vol_s = volumes.iloc[:, 0].copy()
            elif col is not None:
                vol_s = volumes[col].copy()
            else:
                raise AIEngineException("Specify volume_column for multi-column volumes.")
        else:
            vol_s = volumes.copy()
        if not isinstance(vol_s.index, pd.DatetimeIndex):
            vol_s.index = pd.to_datetime(vol_s.index)
        if not vol_s.index.is_monotonic_increasing:
            raise DataLeakageError("volumes index must be sorted chronologically.")
        if vol_s.isna().any() or (np.asarray(vol_s, dtype=np.float64) < 0.0).any():
            raise InsufficientDataError(
                "volumes contains missing or negative observations."
            )

    # Align optional benchmark/volumes strictly by intersection (no ffill/bfill)
    if bm_s is not None:
        common_idx = price_s.index.intersection(bm_s.index)
        if len(common_idx) < len(price_s):
            price_s = price_s.loc[common_idx]
        bm_s = bm_s.loc[common_idx]
    if vol_s is not None:
        common_idx = price_s.index.intersection(vol_s.index)
        price_s = price_s.loc[common_idx]
        if bm_s is not None:
            bm_s = bm_s.loc[common_idx]
        vol_s = vol_s.loc[common_idx]

    min_required = max(max(windows), max(lags), 63) + horizon_days + 2
    if len(price_s) < min_required:
        raise InsufficientDataError(
            f"Insufficient history: need at least {min_required} aligned prices, got {len(price_s)}.",
            details={"required": min_required, "actual": len(price_s)},
        )

    # Production log returns from portfolio_engine.returns
    log_ret = calculate_log_returns(price_s, aligned=True)
    if not isinstance(log_ret, pd.Series):
        raise AIEngineException("Expected Series from calculate_log_returns.")

    ann_factor = float(np.sqrt(trading_days_per_year))
    features: Dict[str, pd.Series] = {}

    # 1. Strictly lagged log returns (tau <= t - 1)
    for lag in lags:
        features[f"ret_lag_{lag}"] = log_ret.shift(lag)

    # 2. Rolling annualized realized volatility ending at t - 1
    vol_map: Dict[int, pd.Series] = {}
    for w in windows:
        vol_w = log_ret.rolling(window=w, min_periods=w).std(ddof=1).shift(1) * ann_factor
        vol_map[w] = vol_w
        features[f"vol_{w}"] = vol_w

    # 3. Volatility term-structure ratios (ending at t - 1)
    if 5 in vol_map and 21 in vol_map:
        features["vol_ratio_5_21"] = vol_map[5] / vol_map[21].clip(lower=_MIN_VOL_DENOM)
    if 21 in vol_map and 63 in vol_map:
        features["vol_ratio_21_63"] = vol_map[21] / vol_map[63].clip(lower=_MIN_VOL_DENOM)

    # 4. Higher moments & downside risk proxies ending at t - 1
    features["skew_21"] = log_ret.rolling(window=21, min_periods=21).skew().shift(1)
    features["kurt_21"] = log_ret.rolling(window=21, min_periods=21).kurt().shift(1)

    downside_sq = log_ret.clip(upper=0.0) ** 2
    features["downside_vol_21"] = (
        np.sqrt(downside_sq.rolling(window=21, min_periods=21).mean()).shift(1) * ann_factor
    )

    # Rolling 21d max drawdown strictly on prices up to t - 1
    lagged_price = price_s.shift(1)
    rolling_peak = lagged_price.rolling(window=21, min_periods=21).max()
    drawdown_series = lagged_price / rolling_peak - 1.0
    features["max_dd_21"] = drawdown_series.rolling(window=21, min_periods=21).min()

    # 5. Deterministic calendar metadata known pre-open on day t
    if not isinstance(log_ret.index, pd.DatetimeIndex):
        raise AIEngineException("log_ret index must be a DatetimeIndex.")
    features["day_of_week"] = pd.Series(
        log_ret.index.dayofweek.astype(np.float64),
        index=log_ret.index,
        dtype=float,
    )

    # 6. Optional volume z-score ending at t - 1
    if vol_s is not None:
        log_vol_vals = np.log1p(np.asarray(vol_s, dtype=np.float64)).tolist()
        log_vol = pd.Series(log_vol_vals, index=vol_s.index, dtype=float)
        vol_mean = log_vol.rolling(window=21, min_periods=21).mean()
        vol_std = log_vol.rolling(window=21, min_periods=21).std(ddof=1).clip(lower=_MIN_VOL_DENOM)
        features["vol_z_21"] = ((log_vol - vol_mean) / vol_std).shift(1)

    # 7. Optional benchmark (^NSEI) context ending at t - 1
    if bm_s is not None:
        bm_ret = calculate_log_returns(bm_s, aligned=True)
        if not isinstance(bm_ret, pd.Series):
            raise AIEngineException("Expected Series from benchmark calculate_log_returns.")
        features["bm_ret_lag_1"] = bm_ret.shift(1)
        features["bm_vol_21"] = (
            bm_ret.rolling(window=21, min_periods=21).std(ddof=1).shift(1) * ann_factor
        )
        bm_var_63 = bm_ret.rolling(window=63, min_periods=63).var(ddof=1).clip(lower=_MIN_VOL_DENOM)
        cov_63 = log_ret.rolling(window=63, min_periods=63).cov(bm_ret)
        features["beta_63"] = (cov_63 / bm_var_63).shift(1)

    X_raw = pd.DataFrame(features, index=log_ret.index)

    # Target: forward h-day annualized realized volatility over [t, t + h - 1]
    # rolling(h).mean() at index (t + h - 1) covers [t, t + h - 1]; shift(-(h - 1)) aligns to t.
    fwd_var = log_ret.pow(2).rolling(window=horizon_days, min_periods=horizon_days).mean()
    y_raw = np.sqrt(fwd_var.shift(-(horizon_days - 1))) * ann_factor
    y_raw.name = f"fwd_vol_{horizon_days}d"

    # Explicit alignment: keep only rows where all features and target are finite
    valid_mask = X_raw.notna().all(axis=1) & y_raw.notna()
    X_clean = X_raw.loc[valid_mask].copy()
    y_clean = y_raw.loc[valid_mask].copy()

    if len(X_clean) == 0:
        raise InsufficientDataError(
            "No valid feature-target rows remain after warm-up and horizon alignment."
        )

    # Compute exact target end date for each sample row t: index position pos + horizon_days - 1
    ret_idx = log_ret.index
    pos_map = {ts: i for i, ts in enumerate(ret_idx)}
    start_dates = pd.DatetimeIndex(X_clean.index)
    end_dates = pd.DatetimeIndex(
        [ret_idx[pos_map[pd.Timestamp(ts)] + horizon_days - 1] for ts in start_dates]
    )

    snapshot_hash = _compute_snapshot_hash(price_s, bm_s, vol_s, horizon_days)

    return FeatureDataset(
        X=X_clean,
        y=y_clean,
        feature_names=tuple(str(c) for c in X_clean.columns),
        horizon_days=horizon_days,
        trading_days_per_year=trading_days_per_year,
        target_start_dates=start_dates,
        target_end_dates=end_dates,
        data_snapshot_hash=snapshot_hash,
        metadata={
            "n_samples": int(len(X_clean)),
            "n_features": int(X_clean.shape[1]),
            "horizon_days": horizon_days,
            "has_benchmark": bm_s is not None,
            "has_volumes": vol_s is not None,
        },
    )
