"""Unit & anti-leakage tests for ``app.ai_engine.feature_pipeline``."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
    DEFAULT_HORIZON_DAYS,
    FeatureDataset,
    build_volatility_features,
)
from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)
from app.portfolio_engine.returns import calculate_log_returns


def _make_synthetic_market(
    n_days: int = 300, seed: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate deterministic asset prices, benchmark prices, and volumes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    bm_ret = rng.normal(0.0004, 0.01, size=n_days)
    asset_ret = 1.1 * bm_ret + rng.normal(0.0001, 0.008, size=n_days)

    asset_prices = pd.Series(
        (1000.0 * np.exp(np.cumsum(asset_ret))).tolist(),
        index=dates,
        name="RELIANCE.NS",
        dtype=float,
    )
    bm_prices = pd.Series(
        (18000.0 * np.exp(np.cumsum(bm_ret))).tolist(),
        index=dates,
        name="^NSEI",
        dtype=float,
    )
    volumes = pd.Series(
        rng.integers(1_000_000, 5_000_000, size=n_days).astype(np.float64).tolist(),
        index=dates,
        name="volume",
        dtype=float,
    )
    return asset_prices, bm_prices, volumes


class TestFeaturePipelineAntiLeakage:
    """Critical anti-leakage and temporal alignment tests."""

    def test_future_perturbation_at_t_and_later_leaves_features_bitwise_identical(
        self,
    ) -> None:
        """Perturbing prices/benchmark/volumes at >= t must not change features at t."""
        prices, bm, vols = _make_synthetic_market(n_days=260, seed=101)
        ds_orig = build_volatility_features(
            prices,
            benchmark_prices=bm,
            volumes=vols,
            horizon_days=21,
        )

        # Pick an interior prediction date t
        t_idx = len(ds_orig.X) // 2
        t_date = pd.Timestamp(ds_orig.X.index[t_idx])

        # Inject extreme shock at t_date and all subsequent dates (>= t)
        prices_shock = prices.copy()
        bm_shock = bm.copy()
        vols_shock = vols.copy()
        shock_mask = prices_shock.index >= t_date
        prices_shock.loc[shock_mask] = prices_shock.loc[shock_mask] * 3.5
        bm_shock.loc[shock_mask] = bm_shock.loc[shock_mask] * 0.4
        vols_shock.loc[shock_mask] = vols_shock.loc[shock_mask] * 10.0

        ds_shock = build_volatility_features(
            prices_shock,
            benchmark_prices=bm_shock,
            volumes=vols_shock,
            horizon_days=21,
        )

        # Every feature row at <= t_date must be bitwise identical
        orig_prefix = ds_orig.X.loc[:t_date]
        shock_prefix = ds_shock.X.loc[:t_date]
        assert orig_prefix.index.equals(shock_prefix.index)
        np.testing.assert_array_equal(
            np.asarray(orig_prefix, dtype=np.float64),
            np.asarray(shock_prefix, dtype=np.float64),
        )

    def test_rolling_windows_and_lags_strictly_end_at_t_minus_1(self) -> None:
        """Verify hand-computed lag-1 return and 21d vol at t use returns up to t-1 only."""
        prices, _, _ = _make_synthetic_market(n_days=200, seed=7)
        ds = build_volatility_features(prices, horizon_days=5)

        log_ret = calculate_log_returns(prices, aligned=True)
        assert isinstance(log_ret, pd.Series)

        t_date = pd.Timestamp(ds.X.index[10])
        ret_pos = int(log_ret.index.get_loc(t_date))  # type: ignore[arg-type]

        # ret_lag_1 at t must equal log_ret at t - 1
        expected_lag1 = float(log_ret.iloc[ret_pos - 1])
        actual_lag1 = float(np.asarray(ds.X.loc[t_date, "ret_lag_1"], dtype=np.float64))
        assert actual_lag1 == pytest.approx(expected_lag1, abs=1e-15)

        # vol_21 at t must equal std(log_ret[t-21 : t]) * sqrt(252) (strictly excluding t)
        window_slice = log_ret.iloc[ret_pos - 21 : ret_pos]
        assert len(window_slice) == 21
        expected_vol21 = float(window_slice.std(ddof=1) * np.sqrt(252.0))
        actual_vol21 = float(np.asarray(ds.X.loc[t_date, "vol_21"], dtype=np.float64))
        assert actual_vol21 == pytest.approx(expected_vol21, rel=1e-12, abs=1e-12)

    def test_target_matches_forward_horizon_volatility(self) -> None:
        """Verify y[t] equals sqrt(252 * mean(r[t : t+h]^2))."""
        prices, _, _ = _make_synthetic_market(n_days=200, seed=19)
        horizon = 21
        ds = build_volatility_features(prices, horizon_days=horizon)
        assert ds.horizon_days == DEFAULT_HORIZON_DAYS

        log_ret = calculate_log_returns(prices, aligned=True)
        assert isinstance(log_ret, pd.Series)

        t_date = pd.Timestamp(ds.X.index[5])
        ret_pos = int(log_ret.index.get_loc(t_date))  # type: ignore[arg-type]
        fwd_slice = np.asarray(log_ret.iloc[ret_pos : ret_pos + horizon], dtype=np.float64)
        expected_target = float(np.sqrt(252.0 * np.mean(fwd_slice**2)))
        assert float(ds.y.loc[t_date]) == pytest.approx(expected_target, rel=1e-12)
        assert ds.target_start_dates[5] == t_date
        assert ds.target_end_dates[5] == log_ret.index[ret_pos + horizon - 1]

    def test_missing_values_raise_without_ffill_or_bfill(self) -> None:
        """Missing prices or volumes must raise InsufficientDataError (never filled)."""
        prices, _, vols = _make_synthetic_market(n_days=200, seed=3)
        prices_na = prices.copy()
        prices_na.iloc[50] = np.nan

        with pytest.raises(InsufficientDataError):
            build_volatility_features(prices_na, horizon_days=21)

        vols_na = vols.copy()
        vols_na.iloc[60] = np.nan
        with pytest.raises(InsufficientDataError):
            build_volatility_features(prices, volumes=vols_na, horizon_days=21)

    def test_unsorted_index_rejected_as_data_leakage(self) -> None:
        """Non-monotonic timestamps must raise DataLeakageError."""
        prices, _, _ = _make_synthetic_market(n_days=200, seed=11)
        prices_rev = prices.iloc[::-1]

        with pytest.raises(DataLeakageError):
            build_volatility_features(prices_rev, horizon_days=21)

    def test_configurable_alternative_horizon_and_insufficient_history(self) -> None:
        """Support alternative horizons (e.g. 5d) and reject too-short series."""
        prices, _, _ = _make_synthetic_market(n_days=180, seed=23)
        ds_5 = build_volatility_features(prices, horizon_days=5)
        ds_21 = build_volatility_features(prices, horizon_days=21)

        assert isinstance(ds_5, FeatureDataset)
        assert ds_5.horizon_days == 5
        # 5-day horizon retains 16 more trailing samples than 21-day horizon
        assert len(ds_5.X) == len(ds_21.X) + 16
        assert not ds_5.X.isna().any().any()
        assert not ds_5.y.isna().any()

        with pytest.raises(InsufficientDataError):
            build_volatility_features(prices.iloc[:50], horizon_days=21)

        with pytest.raises(AIEngineException):
            build_volatility_features(prices, horizon_days=0)

    def test_max_dd_21_hand_calculated_and_exact_lookback_boundary(self) -> None:
        """max_dd_21 must use exact 21 observations [t-21..t-1] relative to running peak."""
        n_days = 160
        dates = pd.bdate_range("2022-01-03", periods=n_days)
        rng = np.random.default_rng(77)
        rets = rng.normal(0.0002, 0.01, size=n_days)
        prices = pd.Series(
            (100.0 * np.exp(np.cumsum(rets))).tolist(),
            index=dates,
            name="TEST",
            dtype=float,
        )
        # Put a huge spike at t - 22 (outside 21-obs window) and a known peak/trough inside [t-21..t-1]
        ds_base = build_volatility_features(prices, horizon_days=5)
        t_date = pd.Timestamp(ds_base.X.index[40])
        pos_t = int(prices.index.get_loc(t_date))  # type: ignore[arg-type]

        # Hand-compute max_dd_21 on exact slice prices[pos_t - 21 : pos_t]
        window_21 = np.asarray(prices.iloc[pos_t - 21 : pos_t], dtype=np.float64)
        assert len(window_21) == 21
        running_peak = np.maximum.accumulate(window_21)
        expected_dd = float(np.min(window_21 / running_peak - 1.0))
        actual_dd = float(np.asarray(ds_base.X.loc[t_date, "max_dd_21"], dtype=np.float64))
        assert actual_dd == pytest.approx(expected_dd, rel=1e-12, abs=1e-12)

        # Perturb price at pos_t - 22 (outside 21-obs window) AND at pos_t (future/current)
        prices_boundary = prices.copy()
        prices_boundary.iloc[pos_t - 22] = prices_boundary.iloc[pos_t - 22] * 5.0
        prices_boundary.iloc[pos_t] = prices_boundary.iloc[pos_t] * 2.5
        ds_boundary = build_volatility_features(prices_boundary, horizon_days=5)
        boundary_dd = float(np.asarray(ds_boundary.X.loc[t_date, "max_dd_21"], dtype=np.float64))
        assert boundary_dd == pytest.approx(expected_dd, rel=1e-14, abs=1e-14)

    def test_snapshot_hash_deterministic_and_timestamp_alignment_sensitive(self) -> None:
        """Snapshot hash must be deterministic and sensitive to timestamps, values, and config."""
        prices, bm, vols = _make_synthetic_market(n_days=180, seed=99)
        ds1 = build_volatility_features(prices, benchmark_prices=bm, volumes=vols, horizon_days=21)
        ds2 = build_volatility_features(prices, benchmark_prices=bm, volumes=vols, horizon_days=21)
        assert ds1.data_snapshot_hash == ds2.data_snapshot_hash

        # Same values, shifted timestamps -> different hash
        shifted_dates = pd.bdate_range("2022-01-04", periods=len(prices))
        prices_shift = pd.Series(prices.tolist(), index=shifted_dates, dtype=float)
        bm_shift = pd.Series(bm.tolist(), index=shifted_dates, dtype=float)
        vols_shift = pd.Series(vols.tolist(), index=shifted_dates, dtype=float)
        ds_shift = build_volatility_features(
            prices_shift, benchmark_prices=bm_shift, volumes=vols_shift, horizon_days=21
        )
        assert ds_shift.data_snapshot_hash != ds1.data_snapshot_hash

        # Different horizon / annualization -> different hash
        ds_h5 = build_volatility_features(prices, benchmark_prices=bm, volumes=vols, horizon_days=5)
        assert ds_h5.data_snapshot_hash != ds1.data_snapshot_hash

    def test_input_validation_duplicates_nat_inf_and_integer_sequences(self) -> None:
        """Reject duplicate timestamps, NaT, Inf volumes, and non-integer/bool configs."""
        prices, _, vols = _make_synthetic_market(n_days=160, seed=55)

        # Duplicate timestamp in prices
        dup_idx = prices.index.tolist()
        dup_idx[10] = dup_idx[9]
        prices_dup = pd.Series(prices.tolist(), index=pd.DatetimeIndex(dup_idx), dtype=float)
        with pytest.raises(AIEngineException, match="duplicate"):
            build_volatility_features(prices_dup, horizon_days=5)

        # NaT in prices index
        nat_idx = prices.index.tolist()
        nat_idx[5] = pd.NaT
        prices_nat = pd.Series(prices.tolist(), index=pd.DatetimeIndex(nat_idx), dtype=float)
        with pytest.raises(AIEngineException, match="NaT"):
            build_volatility_features(prices_nat, horizon_days=5)

        # +inf and -inf in volumes
        for bad_val in (np.inf, -np.inf):
            vols_inf = vols.copy()
            vols_inf.iloc[15] = bad_val
            with pytest.raises(AIEngineException, match="non-finite"):
                build_volatility_features(prices, volumes=vols_inf, horizon_days=5)

        # Invalid rolling_windows / return_lags (bool, float, empty)
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, rolling_windows=[True, 21])  # type: ignore[list-item]
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, rolling_windows=[5.5, 21])  # type: ignore[list-item]
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, return_lags=[1.0, 2])  # type: ignore[list-item]
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, return_lags=[])

        # Invalid trading_days_per_year
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, trading_days_per_year=True)  # type: ignore[arg-type]
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, trading_days_per_year=252.0)  # type: ignore[arg-type]
        with pytest.raises(AIEngineException):
            build_volatility_features(prices, trading_days_per_year=0)

    def test_missing_intermediate_benchmark_or_volume_date_rejected_without_stitching(
        self,
    ) -> None:
        """Missing intermediate dates in benchmark or volumes must raise without stitching returns."""
        prices, bm, vols = _make_synthetic_market(n_days=200, seed=88)

        # Drop one intermediate trading session from benchmark_prices
        bm_missing = bm.drop(bm.index[50])
        with pytest.raises(InsufficientDataError, match="benchmark_prices is missing"):
            build_volatility_features(prices, benchmark_prices=bm_missing, horizon_days=21)

        # Drop one intermediate trading session from volumes
        vols_missing = vols.drop(vols.index[75])
        with pytest.raises(InsufficientDataError, match="volumes is missing"):
            build_volatility_features(prices, volumes=vols_missing, horizon_days=21)

        # Drop one intermediate trading session from asset_prices while benchmark is complete
        prices_missing = prices.drop(prices.index[60])
        with pytest.raises(InsufficientDataError, match="asset_prices is missing"):
            build_volatility_features(prices_missing, benchmark_prices=bm, horizon_days=21)
