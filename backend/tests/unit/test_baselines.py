"""Unit and integration tests for Phase 3, Step 3.1: Historical Baseline Models."""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
    BaselineMetrics,
    ForecastFloorPolicy,
    OLSBaseline,
    RidgeBaseline,
    RollingMeanBaseline,
    RollingVolBaseline,
    WalkForwardSplitter,
    aggregate_metrics,
    apply_volatility_floor,
    build_volatility_features,
    evaluate_baselines,
    mae,
    qlike,
    r_squared,
    rmse,
    scale_fold_features,
)
from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)
from app.portfolio_engine.returns import calculate_log_returns


def _make_synthetic_prices(n_days: int = 550, seed: int = 42) -> pd.Series:
    """Create a deterministic synthetic NSE-like price series with regime shifts."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    vols = np.where(np.arange(n_days) % 120 < 60, 0.010, 0.022)
    rets = rng.normal(loc=0.0004, scale=vols, size=n_days)
    prices = 1000.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices.tolist(), index=dates, name="RELIANCE.NS", dtype=float)


# ---------------------------------------------------------------------------
# 1. Rolling volatility baseline tests
# ---------------------------------------------------------------------------


def test_rolling_vol_baseline_matches_hand_calculated_values() -> None:
    """RollingVolBaseline(window=21) must extract exact lagged 21d annualized vol."""
    prices = _make_synthetic_prices(n_days=200, seed=10)
    dataset = build_volatility_features(prices, horizon_days=21)

    baseline = RollingVolBaseline(window=21)
    baseline.fit(dataset.X, dataset.y)
    preds = baseline.predict(dataset.X)

    log_ret = calculate_log_returns(prices, aligned=True)
    assert isinstance(log_ret, pd.Series)
    expected_series = (
        log_ret.rolling(window=21, min_periods=21).std(ddof=1).shift(1)
        * float(np.sqrt(252))
    )
    expected = np.asarray(expected_series.loc[dataset.X.index], dtype=np.float64)

    np.testing.assert_allclose(preds, expected, rtol=1e-12, atol=1e-12)
    assert baseline.model_name == "rolling_vol_21d"


def test_rolling_vol_annualization_uses_sqrt_252() -> None:
    """Verify RollingVolBaseline(63) matches daily std * sqrt(252) on the feature matrix."""
    prices = _make_synthetic_prices(n_days=220, seed=11)
    dataset = build_volatility_features(prices, horizon_days=21)

    baseline_63 = RollingVolBaseline(window=63)
    preds = baseline_63.predict(dataset.X)

    log_ret = calculate_log_returns(prices, aligned=True)
    assert isinstance(log_ret, pd.Series)
    daily_std = log_ret.rolling(window=63, min_periods=63).std(ddof=1).shift(1)
    expected = np.asarray(daily_std.loc[dataset.X.index], dtype=np.float64) * float(
        np.sqrt(252)
    )

    np.testing.assert_allclose(preds, expected, rtol=1e-12, atol=1e-12)
    assert baseline_63.model_name == "rolling_vol_63d"


def test_rolling_vol_does_not_accept_unused_trading_days_param() -> None:
    """RollingVolBaseline must not silently accept an unused trading_days_per_year arg."""
    with pytest.raises(TypeError):
        RollingVolBaseline(window=21, trading_days_per_year=365)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2. Rolling mean baseline tests
# ---------------------------------------------------------------------------


def test_rolling_mean_baseline_uses_only_past_observations() -> None:
    """Perturbing test/future y values must not change RollingMeanBaseline predictions."""
    prices = _make_synthetic_prices(n_days=300, seed=20)
    dataset = build_volatility_features(prices, horizon_days=21)

    splitter = WalkForwardSplitter(
        mode="expanding",
        train_size=120,
        test_size=40,
        horizon_days=21,
    )
    first_fold = next(splitter.split(dataset.X, dataset.y))

    train_idx = first_fold.train_indices.tolist()
    test_idx = first_fold.test_indices.tolist()

    X_train = dataset.X.iloc[train_idx]
    y_train = dataset.y.iloc[train_idx]
    X_test = dataset.X.iloc[test_idx]

    model = RollingMeanBaseline(window=21)
    model.fit(X_train, y_train)
    preds_original = model.predict(X_test)

    expected_mean = float(np.mean(np.asarray(y_train, dtype=np.float64)[-21:]))
    np.testing.assert_allclose(
        preds_original, np.full(len(X_test), expected_mean, dtype=np.float64)
    )

    y_perturbed = dataset.y.copy()
    y_perturbed.iloc[test_idx] = 999.0
    model_perturbed = RollingMeanBaseline(window=21)
    model_perturbed.fit(X_train, y_perturbed.iloc[train_idx])
    preds_perturbed = model_perturbed.predict(X_test)

    np.testing.assert_allclose(preds_original, preds_perturbed, rtol=1e-15, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Linear regression (OLS & Ridge) baseline tests + input validation
# ---------------------------------------------------------------------------


def test_ols_baseline_coefficients_deterministic() -> None:
    """OLSBaseline must produce identical predictions across repeated fits."""
    prices = _make_synthetic_prices(n_days=350, seed=30)
    dataset = build_volatility_features(prices, horizon_days=21)

    splitter = WalkForwardSplitter(
        mode="expanding",
        train_size=150,
        test_size=40,
        horizon_days=21,
    )
    fold = next(splitter.split(dataset.X, dataset.y))
    scaled = scale_fold_features(dataset.X, dataset.y, fold)

    ols_a = OLSBaseline().fit(scaled.X_train, scaled.y_train)
    ols_b = OLSBaseline().fit(scaled.X_train, scaled.y_train)

    preds_a = ols_a.predict(scaled.X_test)
    preds_b = ols_b.predict(scaled.X_test)

    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-14, atol=1e-14)
    assert ols_a.model_name == "ols"


def test_ridge_baseline_regularization_effect() -> None:
    """Higher Ridge alpha must shrink coefficient L2 norm relative to lower alpha."""
    prices = _make_synthetic_prices(n_days=350, seed=31)
    dataset = build_volatility_features(prices, horizon_days=21)

    splitter = WalkForwardSplitter(
        mode="expanding",
        train_size=150,
        test_size=40,
        horizon_days=21,
    )
    fold = next(splitter.split(dataset.X, dataset.y))
    scaled = scale_fold_features(dataset.X, dataset.y, fold)

    ridge_low = RidgeBaseline(alpha=0.01)
    ridge_low.fit(scaled.X_train, scaled.y_train)

    ridge_high = RidgeBaseline(alpha=1000.0)
    ridge_high.fit(scaled.X_train, scaled.y_train)

    assert ridge_low._model is not None
    assert ridge_high._model is not None
    norm_low = float(np.linalg.norm(np.asarray(ridge_low._model.coef_, dtype=np.float64)))
    norm_high = float(np.linalg.norm(np.asarray(ridge_high._model.coef_, dtype=np.float64)))

    assert norm_high < norm_low
    assert ridge_low.model_name == "ridge_alpha_0.01"
    assert ridge_high.model_name == "ridge_alpha_1000"


@pytest.mark.parametrize("model_cls", [OLSBaseline, RidgeBaseline])
def test_linear_baselines_predict_validates_empty_nan_inf(
    model_cls: type[OLSBaseline] | type[RidgeBaseline],
) -> None:
    """OLSBaseline.predict and RidgeBaseline.predict must reject empty, NaN, and Inf X_test."""
    prices = _make_synthetic_prices(n_days=300, seed=32)
    dataset = build_volatility_features(prices, horizon_days=21)
    splitter = WalkForwardSplitter(
        mode="expanding", train_size=120, test_size=30, horizon_days=21
    )
    fold = next(splitter.split(dataset.X, dataset.y))
    scaled = scale_fold_features(dataset.X, dataset.y, fold)

    model = model_cls().fit(scaled.X_train, scaled.y_train)

    # Empty DataFrame
    with pytest.raises(AIEngineException):
        model.predict(scaled.X_test.iloc[:0])

    # NaN in X_test
    X_nan = scaled.X_test.copy()
    X_nan.iloc[0, 0] = np.nan
    with pytest.raises(AIEngineException):
        model.predict(X_nan)

    # Inf in X_test
    X_inf = scaled.X_test.copy()
    X_inf.iloc[0, 0] = np.inf
    with pytest.raises(AIEngineException):
        model.predict(X_inf)


# ---------------------------------------------------------------------------
# 4. Positive volatility floor policy tests
# ---------------------------------------------------------------------------


def test_apply_volatility_floor_preserves_raw_and_counts_floored() -> None:
    """apply_volatility_floor must preserve raw predictions and record exact counts."""
    raw = np.array([-0.02, 0.0, 0.0005, 0.15, 0.25], dtype=np.float64)
    post = apply_volatility_floor(raw, min_volatility=1e-3)

    np.testing.assert_allclose(post.raw_predictions, raw, rtol=1e-15)
    np.testing.assert_allclose(
        post.predictions,
        np.array([1e-3, 1e-3, 1e-3, 0.15, 0.25], dtype=np.float64),
        rtol=1e-15,
    )
    assert post.n_floored == 3
    assert post.floored_pct == pytest.approx(60.0, rel=1e-12)
    assert post.min_volatility == pytest.approx(1e-3, rel=1e-12)


def test_floor_policy_rejects_non_positive_min_volatility() -> None:
    """ForecastFloorPolicy must reject zero, negative, or non-finite floors."""
    with pytest.raises(AIEngineException):
        ForecastFloorPolicy(min_volatility=0.0)
    with pytest.raises(AIEngineException):
        ForecastFloorPolicy(min_volatility=-1e-3)
    with pytest.raises(AIEngineException):
        ForecastFloorPolicy(min_volatility=float("nan"))


# ---------------------------------------------------------------------------
# 5. Evaluation metrics tests (RMSE, MAE, R², normalized QLIKE)
# ---------------------------------------------------------------------------


def test_rmse_mae_r2_against_hand_calculated() -> None:
    """Verify RMSE, MAE, and R² against hand-calculated exact values."""
    y_true = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64)
    y_pred = np.array([0.12, 0.18, 0.33, 0.37], dtype=np.float64)
    expected_rmse = float(np.sqrt(0.00065))
    expected_mae = 0.025
    expected_r2 = 1.0 - 0.0026 / 0.05

    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse, rel=1e-12)
    assert mae(y_true, y_pred) == pytest.approx(expected_mae, rel=1e-12)
    assert r_squared(y_true, y_pred) == pytest.approx(expected_r2, rel=1e-12)


def test_qlike_normalized_matches_hand_calculation() -> None:
    """Verify normalized Patton (2011) QLIKE = mean(v_t/v_p - log(v_t/v_p) - 1)."""
    y_true = np.array([0.20, 0.30], dtype=np.float64)
    y_pred = np.array([0.25, 0.35], dtype=np.float64)

    var_true = y_true ** 2
    var_pred = y_pred ** 2
    ratio = var_true / var_pred
    expected = float(np.mean(ratio - np.log(ratio) - 1.0))

    val = qlike(y_true, y_pred)
    assert val == pytest.approx(expected, rel=1e-12)
    assert val >= 0.0


def test_qlike_equal_variance_is_zero_and_non_negative() -> None:
    """Equal actual and predicted volatility must yield exact 0.0 QLIKE; mismatched > 0."""
    y_true = np.array([0.12, 0.22, 0.35], dtype=np.float64)
    assert qlike(y_true, y_true.copy()) == pytest.approx(0.0, abs=1e-15)

    # Arbitrary perturbed positive forecasts must remain non-negative
    y_pred = np.array([0.15, 0.18, 0.42], dtype=np.float64)
    assert qlike(y_true, y_pred) >= -1e-15


def test_qlike_scale_invariance() -> None:
    """Multiplying both y_true and y_pred by c > 0 must leave QLIKE unchanged."""
    y_true = np.array([0.15, 0.25, 0.40], dtype=np.float64)
    y_pred = np.array([0.18, 0.22, 0.32], dtype=np.float64)

    base_qlike = qlike(y_true, y_pred)
    scaled_qlike = qlike(y_true * 3.5, y_pred * 3.5)
    assert scaled_qlike == pytest.approx(base_qlike, rel=1e-12)


def test_qlike_rejects_non_positive_predictions_without_silent_clipping() -> None:
    """qlike() must raise AIEngineException on zero or negative y_pred or y_true."""
    y_true = np.array([0.15, 0.20], dtype=np.float64)

    with pytest.raises(AIEngineException):
        qlike(y_true, np.array([0.0, 0.20], dtype=np.float64))
    with pytest.raises(AIEngineException):
        qlike(y_true, np.array([-0.01, 0.20], dtype=np.float64))
    with pytest.raises(AIEngineException):
        qlike(np.array([0.0, 0.20], dtype=np.float64), y_true)


def test_qlike_raises_on_nan_inputs() -> None:
    """QLIKE and all metrics must raise AIEngineException on NaN/Inf inputs."""
    y_true = np.array([0.15, np.nan], dtype=np.float64)
    y_pred = np.array([0.15, 0.20], dtype=np.float64)

    with pytest.raises(AIEngineException):
        qlike(y_true, y_pred)
    with pytest.raises(AIEngineException):
        rmse(y_true, y_pred)
    with pytest.raises(AIEngineException):
        mae(y_true, y_pred)
    with pytest.raises(AIEngineException):
        r_squared(y_true, y_pred)


def test_metrics_raise_on_mismatched_lengths() -> None:
    """All metrics must raise AIEngineException when array lengths differ."""
    y_true = np.array([0.15, 0.20, 0.25], dtype=np.float64)
    y_pred = np.array([0.15, 0.20], dtype=np.float64)

    with pytest.raises(AIEngineException):
        rmse(y_true, y_pred)
    with pytest.raises(AIEngineException):
        mae(y_true, y_pred)
    with pytest.raises(AIEngineException):
        r_squared(y_true, y_pred)
    with pytest.raises(AIEngineException):
        qlike(y_true, y_pred)


# ---------------------------------------------------------------------------
# 6. Anti-leakage, walk-forward & configuration invariants
# ---------------------------------------------------------------------------


def test_walk_forward_chronological_ordering() -> None:
    """Fold test dates must advance strictly chronologically without overlap."""
    prices = _make_synthetic_prices(n_days=550, seed=40)
    results = evaluate_baselines(
        prices,
        horizon_days=21,
        train_size=200,
        test_size=50,
        mode="expanding",
    )

    folds = results["rolling_vol_21d"]
    assert len(folds) >= 3
    for i in range(1, len(folds)):
        prev_end = folds[i - 1].test_end_date
        curr_start = folds[i].test_start_date
        assert prev_end is not None and curr_start is not None
        assert pd.Timestamp(curr_start) > pd.Timestamp(prev_end)


def test_purge_gap_at_least_horizon_days() -> None:
    """WalkForwardSplitter must raise DataLeakageError if purge_gap < horizon_days."""
    with pytest.raises(DataLeakageError):
        WalkForwardSplitter(
            mode="expanding",
            train_size=200,
            test_size=50,
            horizon_days=21,
            purge_gap=10,
        )


def test_train_only_preprocessing() -> None:
    """Perturbing test features must not alter StandardScaler mean/scale fitted on train."""
    prices = _make_synthetic_prices(n_days=350, seed=41)
    dataset = build_volatility_features(prices, horizon_days=21)

    splitter = WalkForwardSplitter(
        mode="expanding",
        train_size=150,
        test_size=40,
        horizon_days=21,
    )
    fold = next(splitter.split(dataset.X, dataset.y))

    scaled_orig = scale_fold_features(dataset.X, dataset.y, fold)

    X_perturbed = dataset.X.copy()
    X_perturbed.iloc[fold.test_indices.tolist()] = 1e5
    scaled_pert = scale_fold_features(X_perturbed, dataset.y, fold)

    np.testing.assert_allclose(
        np.asarray(scaled_orig.scaler.mean_, dtype=np.float64),
        np.asarray(scaled_pert.scaler.mean_, dtype=np.float64),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(scaled_orig.scaler.scale_, dtype=np.float64),
        np.asarray(scaled_pert.scaler.scale_, dtype=np.float64),
        rtol=1e-15,
        atol=1e-15,
    )


def test_future_perturbation_invariance() -> None:
    """Perturbing prices after t + horizon_days must not alter features or RollingVol predictions at t."""
    prices = _make_synthetic_prices(n_days=250, seed=42)
    ds_orig = build_volatility_features(prices, horizon_days=21)

    prices_pert = prices.copy()
    prices_pert.iloc[-10:] = prices_pert.iloc[-10:] * 3.0
    ds_pert = build_volatility_features(prices_pert, horizon_days=21)

    safe_idx = ds_orig.X.index[: -31]
    np.testing.assert_allclose(
        np.asarray(ds_orig.X.loc[safe_idx], dtype=np.float64),
        np.asarray(ds_pert.X.loc[safe_idx], dtype=np.float64),
        rtol=1e-14,
        atol=1e-14,
    )

    model = RollingVolBaseline(window=21)
    preds_orig = model.predict(ds_orig.X.loc[safe_idx])
    preds_pert = model.predict(ds_pert.X.loc[safe_idx])
    np.testing.assert_allclose(preds_orig, preds_pert, rtol=1e-14, atol=1e-14)


def test_missing_data_handling() -> None:
    """Missing observations in asset_prices must raise InsufficientDataError."""
    prices = _make_synthetic_prices(n_days=300, seed=43)
    prices.iloc[50] = np.nan

    with pytest.raises(InsufficientDataError):
        evaluate_baselines(prices, train_size=150, test_size=40)


def test_evaluate_baselines_rejects_duplicate_model_names() -> None:
    """Duplicate model_name entries in baselines list must raise AIEngineException."""
    prices = _make_synthetic_prices(n_days=400, seed=44)
    with pytest.raises(AIEngineException, match="duplicate model_name"):
        evaluate_baselines(
            prices,
            train_size=150,
            test_size=40,
            baselines=[RollingVolBaseline(window=21), RollingVolBaseline(window=21)],
        )


def test_evaluate_baselines_does_not_accept_unused_val_size() -> None:
    """evaluate_baselines must not accept val_size since Step 3.1 does no tuning."""
    prices = _make_synthetic_prices(n_days=400, seed=45)
    with pytest.raises(TypeError):
        evaluate_baselines(prices, val_size=20)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 7. End-to-end evaluation & pooled vs macro aggregation integration tests
# ---------------------------------------------------------------------------


def test_evaluate_baselines_integration() -> None:
    """End-to-end evaluate_baselines must evaluate all 5 default models across folds."""
    prices = _make_synthetic_prices(n_days=550, seed=50)
    results = evaluate_baselines(
        prices,
        horizon_days=21,
        train_size=200,
        test_size=50,
        mode="expanding",
    )

    expected_models = {
        "rolling_vol_21d",
        "rolling_vol_63d",
        "rolling_mean_21d",
        "ols",
        "ridge_alpha_1",
    }
    assert set(results.keys()) == expected_models

    for name, fold_list in results.items():
        assert len(fold_list) >= 3, f"Expected >= 3 folds for {name}"
        for m in fold_list:
            assert isinstance(m, BaselineMetrics)
            assert m.horizon_days == 21
            assert m.n_samples == 50
            assert np.isfinite(m.rmse) and m.rmse >= 0.0
            assert np.isfinite(m.mae_val) and m.mae_val >= 0.0
            assert np.isfinite(m.qlike_val) and m.qlike_val >= 0.0
            assert np.isfinite(m.raw_rmse) and m.raw_rmse >= 0.0


def test_aggregate_metrics_pooled_and_macro_distinction() -> None:
    """aggregate_metrics must compute exact pooled OOS metrics distinct from macro fold averages."""
    prices = _make_synthetic_prices(n_days=550, seed=51)
    results = evaluate_baselines(
        prices,
        horizon_days=21,
        train_size=200,
        test_size=50,
        mode="rolling",
    )
    summary = aggregate_metrics(results)

    assert len(summary) == 5
    expected_cols = {
        "model_name",
        "pooled_rmse",
        "pooled_mae",
        "pooled_r2",
        "pooled_qlike",
        "raw_pooled_rmse",
        "raw_pooled_mae",
        "raw_pooled_r2",
        "macro_rmse",
        "macro_mae",
        "macro_r2",
        "macro_qlike",
        "n_floored",
        "floored_pct",
        "n_folds",
        "total_samples",
        "horizon_days",
    }
    assert expected_cols.issubset(set(summary.columns))
    assert summary["pooled_rmse"].is_monotonic_increasing
    assert (summary["horizon_days"] == 21).all()
    assert (summary["pooled_qlike"] >= 0.0).all()

    # Verify pooled metrics match direct concatenation for OLS
    ols_folds = results["ols"]
    y_all = np.concatenate([np.asarray(f.oos_y_true, dtype=np.float64) for f in ols_folds])
    p_all = np.concatenate(
        [np.asarray(f.oos_y_pred_processed, dtype=np.float64) for f in ols_folds]
    )
    ols_row = summary.loc[summary["model_name"] == "ols"].iloc[0]
    assert ols_row["pooled_rmse"] == pytest.approx(rmse(y_all, p_all), rel=1e-12)
    assert ols_row["pooled_r2"] == pytest.approx(r_squared(y_all, p_all), rel=1e-12)
    assert ols_row["pooled_qlike"] == pytest.approx(qlike(y_all, p_all), rel=1e-12)


# ---------------------------------------------------------------------------
# 8. Part A: Numerically stable QLIKE edge-case & overflow tests
# ---------------------------------------------------------------------------


def test_qlike_extreme_small_and_large_positive_inputs() -> None:
    """Log-space QLIKE must handle 1e-200 and 1e+200 without underflow or overflow."""
    y_small_t = np.array([1e-200, 2e-200], dtype=np.float64)
    y_small_p = np.array([1.2e-200, 1.8e-200], dtype=np.float64)
    val_small = qlike(y_small_t, y_small_p)
    assert np.isfinite(val_small) and val_small > 0.0

    y_large_t = np.array([1e200, 2e200], dtype=np.float64)
    y_large_p = np.array([1.2e200, 1.8e200], dtype=np.float64)
    val_large = qlike(y_large_t, y_large_p)
    assert np.isfinite(val_large) and val_large > 0.0
    assert val_small == pytest.approx(val_large, rel=1e-12)


def test_qlike_nearly_equal_inputs_non_negative() -> None:
    """Nearly equal actual and predicted values must remain strictly > 0 and accurate."""
    y_true = np.array([0.20, 0.35], dtype=np.float64)
    y_pred = np.array([0.20 + 1e-12, 0.35 - 1e-12], dtype=np.float64)
    val = qlike(y_true, y_pred)
    assert np.isfinite(val)
    assert val > 0.0
    assert val < 1e-20


def test_qlike_adjacent_float64_preserves_positive_loss() -> None:
    """Adjacent representable float64 values (np.nextafter) must yield a representable > 0 loss."""
    y_true = np.array([0.20], dtype=np.float64)
    y_pred_down = np.array([np.nextafter(0.20, 0.0)], dtype=np.float64)
    y_pred_up = np.array([np.nextafter(0.20, np.inf)], dtype=np.float64)

    loss_down = qlike(y_true, y_pred_down)
    loss_up = qlike(y_true, y_pred_up)
    assert np.isfinite(loss_down) and loss_down > 0.0
    assert np.isfinite(loss_up) and loss_up > 0.0


def test_qlike_taylor_branch_matches_high_precision_reference() -> None:
    """Taylor branch (|log_ratio| < 1e-3) must match Decimal high-precision reference."""
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    for delta in (1e-10, -1e-8, 5e-6, -2e-4, 9e-4):
        yt = 0.25
        yp = yt * float(np.exp(-0.5 * delta))
        actual = qlike(np.array([yt], dtype=np.float64), np.array([yp], dtype=np.float64))

        d_yt = Decimal.from_float(yt)
        d_yp = Decimal.from_float(yp)
        d_x = Decimal(2) * (d_yt.ln() - d_yp.ln())
        ref = float(d_x.exp() - Decimal(1) - d_x)
        assert actual == pytest.approx(ref, rel=1e-13, abs=1e-35)


def test_qlike_scale_invariance_extreme_magnitudes() -> None:
    """Scale invariance must hold across 1e-180 and 1e+180 multipliers."""
    y_true = np.array([0.15, 0.25, 0.40], dtype=np.float64)
    y_pred = np.array([0.18, 0.22, 0.32], dtype=np.float64)
    base = qlike(y_true, y_pred)

    assert qlike(y_true * 1e-180, y_pred * 1e-180) == pytest.approx(base, rel=1e-11)
    assert qlike(y_true * 1e180, y_pred * 1e180) == pytest.approx(base, rel=1e-11)


def test_qlike_explicit_overflow_raises() -> None:
    """Variance ratios that overflow float64 must raise AIEngineException explicitly."""
    y_true = np.array([1e200], dtype=np.float64)
    y_pred = np.array([1e-200], dtype=np.float64)
    with pytest.raises(AIEngineException, match="overflow"):
        qlike(y_true, y_pred)


# ---------------------------------------------------------------------------
# 9. Part B: Adversarial OOS observation identity tests
# ---------------------------------------------------------------------------


def test_aggregate_metrics_adversarial_oos_identity_checks() -> None:
    """aggregate_metrics must reject mismatched dates, lengths, fold_ids, horizons, and duplicates."""
    prices = _make_synthetic_prices(n_days=450, seed=60)
    results = evaluate_baselines(
        prices,
        horizon_days=21,
        train_size=200,
        test_size=50,
        baselines=[RollingVolBaseline(window=21), OLSBaseline()],
    )
    # Valid baseline aggregation succeeds
    summary = aggregate_metrics(results)
    assert len(summary) == 2

    # 1. Identical target values but different OOS dates
    bad_dates = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": [
            replace(
                f,
                oos_dates=tuple(f"{d[:10]}T01:00:00" for d in f.oos_dates),
            )
            for f in results["ols"]
        ],
    }
    with pytest.raises(AIEngineException, match="OOS dates"):
        aggregate_metrics(bad_dates)

    # 2. Incorrect prediction array length
    bad_len = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": [
            replace(results["ols"][0], oos_y_pred_raw=results["ols"][0].oos_y_pred_raw[:-1]),
            *results["ols"][1:],
        ],
    }
    with pytest.raises(AIEngineException, match="incomplete or mismatched"):
        aggregate_metrics(bad_len)

    # 3. Differing fold_id or horizon_days
    bad_fold_id = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": [
            replace(results["ols"][0], fold_id=99),
            *results["ols"][1:],
        ],
    }
    with pytest.raises(AIEngineException, match="fold_id"):
        aggregate_metrics(bad_fold_id)

    bad_horizon = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": [
            replace(f, horizon_days=5) for f in results["ols"]
        ],
    }
    with pytest.raises(AIEngineException, match="horizon_days"):
        aggregate_metrics(bad_horizon)

    # 4. Duplicate OOS dates within a fold
    dup_dates = (results["ols"][0].oos_dates[0],) + results["ols"][0].oos_dates[:-1]
    bad_dup = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": [
            replace(results["ols"][0], oos_dates=dup_dates),
            *results["ols"][1:],
        ],
    }
    with pytest.raises(AIEngineException, match="duplicate OOS timestamps"):
        aggregate_metrics(bad_dup)

    # 5. Differing fold order
    bad_order = {
        "rolling_vol_21d": results["rolling_vol_21d"],
        "ols": list(reversed(results["ols"])),
    }
    with pytest.raises(AIEngineException, match="strictly increasing"):
        aggregate_metrics(bad_order)
