"""Unit, anti-leakage, registry, and integration tests for Stage 3.2 XGBoost models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
    STAGE1_BASELINE_NAMES,
    BaselineMetrics,
    BaselinePredictor,
    Gate32Report,
    ModelRegistry,
    OLSBaseline,
    RidgeBaseline,
    RollingMeanBaseline,
    RollingVolBaseline,
    TunedXGBoostVolatilityRegressor,
    WalkForwardSplitter,
    XGBoostHyperparameters,
    XGBoostVolatilityRegressor,
    aggregate_metrics,
    build_volatility_features,
    evaluate_baselines,
    evaluate_stage2_gate,
)
from app.core.exceptions import (
    AIEngineException,
    ArtifactIntegrityError,
    DataLeakageError,
    InsufficientDataError,
)


def _make_nonlinear_vol_prices(n_days: int = 650, seed: int = 123) -> pd.Series:
    """Deterministic synthetic NSE-like price series with leverage & regime clustering."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = np.zeros(n_days, dtype=np.float64)
    vol = 0.012
    for i in range(1, n_days):
        # Asymmetric leverage + clustering: negative shocks spike future volatility
        shock_term = 0.014 if rets[i - 1] < -0.012 else 0.0
        vol = 0.85 * vol + 0.15 * (0.010 + shock_term)
        rets[i] = rng.normal(0.0003, vol)
    prices = 1000.0 * np.exp(np.cumsum(rets))
    return pd.Series(prices.tolist(), index=dates, name="RELIANCE.NS", dtype=float)


def test_xgboost_fixed_deterministic_and_raw_features() -> None:
    """Fixed XGBoost must be deterministic across repeated fits and preserve feature names."""
    prices = _make_nonlinear_vol_prices(n_days=350, seed=10)
    ds = build_volatility_features(prices, horizon_days=21)
    splitter = WalkForwardSplitter(
        mode="expanding", train_size=150, test_size=40, horizon_days=21
    )
    fold = next(splitter.split(ds.X, ds.y))
    X_tr = ds.X.iloc[fold.train_indices.tolist()]
    y_tr = ds.y.iloc[fold.train_indices.tolist()]
    X_te = ds.X.iloc[fold.test_indices.tolist()]

    m1 = XGBoostVolatilityRegressor().fit(X_tr, y_tr)
    m2 = XGBoostVolatilityRegressor().fit(X_tr, y_tr)

    p1 = m1.predict(X_te)
    p2 = m2.predict(X_te)
    np.testing.assert_allclose(p1, p2, rtol=1e-14, atol=1e-14)
    assert m1.feature_names == tuple(str(c) for c in X_tr.columns)
    assert m1.model_name == "xgboost_fixed"


def test_xgboost_input_validation_and_column_order() -> None:
    """XGBoostVolatilityRegressor must reject unfitted predict, empty, NaN, Inf, and reordered cols."""
    prices = _make_nonlinear_vol_prices(n_days=260, seed=11)
    ds = build_volatility_features(prices, horizon_days=21)
    X_tr = ds.X.iloc[:100]
    y_tr = ds.y.iloc[:100]
    X_te = ds.X.iloc[121:141]

    unfitted = XGBoostVolatilityRegressor()
    with pytest.raises(AIEngineException, match="before predict"):
        unfitted.predict(X_te)

    model = XGBoostVolatilityRegressor().fit(X_tr, y_tr)

    # Empty DataFrame
    with pytest.raises(AIEngineException):
        model.predict(X_te.iloc[:0])

    # NaN in X_test
    X_nan = X_te.copy()
    X_nan.iloc[0, 0] = np.nan
    with pytest.raises(AIEngineException):
        model.predict(X_nan)

    # Reordered columns must be rejected
    reordered_cols = list(reversed(X_te.columns.tolist()))
    with pytest.raises(AIEngineException, match="feature columns"):
        model.predict(X_te[reordered_cols])


def test_xgboost_native_json_registry_round_trip_and_tamper(tmp_path: Path) -> None:
    """Native XGBoost JSON serialization via ModelRegistry must preserve predictions and SHA-256."""
    prices = _make_nonlinear_vol_prices(n_days=280, seed=12)
    ds = build_volatility_features(prices, horizon_days=21)
    X_tr = ds.X.iloc[:110]
    y_tr = ds.y.iloc[:110]
    X_te = ds.X.iloc[131:151]

    model = XGBoostVolatilityRegressor().fit(X_tr, y_tr)
    orig_preds = model.predict(X_te)

    registry = ModelRegistry(artifact_dir=tmp_path)
    manifest = model.save_to_registry(
        registry,
        model_id="xgb_vol_21d",
        version="v3.2.0",
        data_snapshot_hash=ds.data_snapshot_hash,
        train_start_date=str(pd.Timestamp(X_tr.index[0]).date()),
        train_end_date=str(pd.Timestamp(X_tr.index[-1]).date()),
    )
    assert manifest.stage == "STAGE_2_XGBOOST"
    assert manifest.artifact_format == "json"
    assert manifest.dependencies["xgboost"] != "not-installed"
    assert manifest.dependencies["shap"] != "not-installed"

    loaded_model, loaded_manifest = XGBoostVolatilityRegressor.load_from_registry(
        registry, model_id="xgb_vol_21d", version="v3.2.0"
    )
    assert loaded_manifest.artifact_sha256 == manifest.artifact_sha256
    loaded_preds = loaded_model.predict(X_te)
    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-14, atol=1e-14)

    # Tampering with JSON artifact must fail SHA-256 verification
    artifact_file = tmp_path / manifest.artifact_filename
    artifact_file.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        XGBoostVolatilityRegressor.load_from_registry(
            registry, model_id="xgb_vol_21d", version="v3.2.0"
        )


def test_tuned_xgboost_nested_purge_and_test_perturbation_invariance() -> None:
    """Nested tuning must enforce purge_gap >= horizon_days and remain invariant to outer test shocks."""
    with pytest.raises(DataLeakageError):
        TunedXGBoostVolatilityRegressor(horizon_days=21, purge_gap=10)

    prices = _make_nonlinear_vol_prices(n_days=380, seed=13)
    ds = build_volatility_features(prices, horizon_days=21)
    splitter = WalkForwardSplitter(
        mode="expanding", train_size=150, test_size=40, horizon_days=21
    )
    fold = next(splitter.split(ds.X, ds.y))
    tr_idx = fold.train_indices.tolist()
    te_idx = fold.test_indices.tolist()

    small_grid = (
        XGBoostHyperparameters(max_depth=2, learning_rate=0.03, n_estimators=40),
        XGBoostHyperparameters(max_depth=3, learning_rate=0.07, n_estimators=40),
    )

    tuned_orig = TunedXGBoostVolatilityRegressor(
        horizon_days=21, param_grid=small_grid
    ).fit(ds.X.iloc[tr_idx], ds.y.iloc[tr_idx])

    # Perturb outer test data — tuned params and predictions on unperturbed test must be identical
    X_pert = ds.X.copy()
    y_pert = ds.y.copy()
    X_pert.iloc[te_idx] = 999.0
    y_pert.iloc[te_idx] = 999.0

    tuned_pert = TunedXGBoostVolatilityRegressor(
        horizon_days=21, param_grid=small_grid
    ).fit(X_pert.iloc[tr_idx], y_pert.iloc[tr_idx])

    assert tuned_orig.selected_params == tuned_pert.selected_params
    assert tuned_orig.best_inner_val_rmse == pytest.approx(
        tuned_pert.best_inner_val_rmse, rel=1e-15
    )
    np.testing.assert_allclose(
        tuned_orig.predict(ds.X.iloc[te_idx]),
        tuned_pert.predict(ds.X.iloc[te_idx]),
        rtol=1e-14,
        atol=1e-14,
    )

    # Insufficient training data for nested purged split must raise InsufficientDataError
    with pytest.raises(InsufficientDataError):
        TunedXGBoostVolatilityRegressor(horizon_days=21, param_grid=small_grid).fit(
            ds.X.iloc[:50], ds.y.iloc[:50]
        )


def test_stage2_comparison_identical_oos_and_gate_report() -> None:
    """Stage 2 XGBoost + all 5 Phase 3.1 baselines must evaluate on identical OOS dates/targets."""
    prices = _make_nonlinear_vol_prices(n_days=650, seed=14)
    small_grid = (
        XGBoostHyperparameters(max_depth=2, learning_rate=0.05, n_estimators=60),
        XGBoostHyperparameters(max_depth=3, learning_rate=0.05, n_estimators=60),
    )
    models: list[BaselinePredictor] = [
        RollingVolBaseline(window=21),
        RollingVolBaseline(window=63),
        RollingMeanBaseline(window=21),
        OLSBaseline(),
        RidgeBaseline(alpha=1.0),
        XGBoostVolatilityRegressor(
            params=XGBoostHyperparameters(max_depth=3, learning_rate=0.05, n_estimators=60)
        ),
        TunedXGBoostVolatilityRegressor(horizon_days=21, param_grid=small_grid),
    ]
    results = evaluate_baselines(
        prices,
        horizon_days=21,
        train_size=220,
        test_size=50,
        mode="expanding",
        baselines=models,
    )
    summary = aggregate_metrics(results)
    assert len(summary) == 7
    assert "xgboost_fixed" in set(summary["model_name"])
    assert "xgboost_tuned" in set(summary["model_name"])

    gate = evaluate_stage2_gate(results, summary, xgb_model_name="xgboost_tuned")
    assert isinstance(gate, Gate32Report)
    assert gate.baseline_names == STAGE1_BASELINE_NAMES
    assert "xgboost_fixed" not in gate.baseline_names
    assert "xgboost_tuned" not in gate.baseline_names
    assert gate.n_folds >= 3
    assert 0.0 <= gate.fold_win_rate <= 1.0
    assert np.isfinite(gate.xgb_pooled_rmse)
    assert np.isfinite(gate.best_baseline_pooled_rmse)


def test_gate32_baseline_validation_and_guardrail_scenarios() -> None:
    """Gate 3.2 must reject Stage 2 models as baselines and enforce MAE/QLIKE guardrails."""
    base_fold = BaselineMetrics(
        model_name="m",
        rmse=0.10,
        mae_val=0.08,
        r2=0.1,
        qlike_val=0.20,
        n_samples=2,
        horizon_days=21,
        fold_id=0,
        oos_dates=("2023-01-02T00:00:00", "2023-01-03T00:00:00"),
        oos_y_true=(0.20, 0.25),
    )
    fold_results: dict[str, list[BaselineMetrics]] = {
        name: [replace(base_fold, model_name=name)] for name in STAGE1_BASELINE_NAMES
    }
    fold_results["xgboost_fixed"] = [replace(base_fold, model_name="xgboost_fixed", rmse=0.05)]
    fold_results["xgboost_tuned"] = [replace(base_fold, model_name="xgboost_tuned", rmse=0.08)]

    def _make_summary(xgb_rmse: float, xgb_mae: float, xgb_qlike: float) -> pd.DataFrame:
        rows = [
            {
                "model_name": name,
                "pooled_rmse": 0.10,
                "pooled_mae": 0.08,
                "pooled_r2": 0.10,
                "pooled_qlike": 0.20,
                "floored_pct": 0.0,
            }
            for name in STAGE1_BASELINE_NAMES
        ]
        rows.append(
            {
                "model_name": "xgboost_fixed",
                "pooled_rmse": 0.05,
                "pooled_mae": 0.04,
                "pooled_r2": 0.50,
                "pooled_qlike": 0.10,
                "floored_pct": 0.0,
            }
        )
        rows.append(
            {
                "model_name": "xgboost_tuned",
                "pooled_rmse": xgb_rmse,
                "pooled_mae": xgb_mae,
                "pooled_r2": 0.25,
                "pooled_qlike": xgb_qlike,
                "floored_pct": 0.0,
            }
        )
        return pd.DataFrame(rows)

    # Genuine pass: RMSE improved and MAE/QLIKE within guardrails (xgboost_fixed is ignored)
    g_pass = evaluate_stage2_gate(
        fold_results, _make_summary(0.08, 0.075, 0.18), xgb_model_name="xgboost_tuned"
    )
    assert g_pass.best_baseline_pooled_rmse == pytest.approx(0.10)
    assert g_pass.primary_rmse_improved is True
    assert g_pass.mae_guardrail_passed is True
    assert g_pass.qlike_guardrail_passed is True
    assert g_pass.gate_passed is True

    # Primary RMSE improved, but MAE deteriorated > +5% -> overall gate fails
    g_mae_fail = evaluate_stage2_gate(
        fold_results, _make_summary(0.08, 0.095, 0.18), xgb_model_name="xgboost_tuned"
    )
    assert g_mae_fail.primary_rmse_improved is True
    assert g_mae_fail.mae_guardrail_passed is False
    assert g_mae_fail.gate_passed is False

    # Primary RMSE improved, but QLIKE deteriorated > +0.05 -> overall gate fails
    g_qlike_fail = evaluate_stage2_gate(
        fold_results, _make_summary(0.08, 0.075, 0.35), xgb_model_name="xgboost_tuned"
    )
    assert g_qlike_fail.primary_rmse_improved is True
    assert g_qlike_fail.qlike_guardrail_passed is False
    assert g_qlike_fail.gate_passed is False

    # Reject Stage 2 model explicitly passed in baseline_names
    with pytest.raises(AIEngineException, match="Stage 2 model"):
        evaluate_stage2_gate(
            fold_results,
            _make_summary(0.08, 0.075, 0.18),
            baseline_names=["rolling_vol_21d", "xgboost_fixed"],
        )

    # Reject misaligned fold IDs
    bad_folds = dict(fold_results)
    bad_folds["ols"] = [replace(base_fold, model_name="ols", fold_id=7)]
    with pytest.raises(AIEngineException, match="fold_id"):
        evaluate_stage2_gate(bad_folds, _make_summary(0.08, 0.075, 0.18))


def test_gate32_fold_identity_and_summary_df_validation() -> None:
    """Gate 3.2 must reject fold OOS mismatches, duplicate/missing summary rows, and NaN QLIKE."""
    fold_template = BaselineMetrics(
        model_name="m",
        rmse=0.10,
        mae_val=0.08,
        r2=0.1,
        qlike_val=0.20,
        n_samples=2,
        horizon_days=21,
        fold_id=0,
        oos_dates=("2023-01-02T00:00:00", "2023-01-03T00:00:00"),
        oos_y_true=(0.20, 0.25),
    )
    fold_results: dict[str, list[BaselineMetrics]] = {
        name: [replace(fold_template, model_name=name)] for name in STAGE1_BASELINE_NAMES
    }
    fold_results["xgboost_tuned"] = [replace(fold_template, model_name="xgboost_tuned", rmse=0.08)]

    def _summary(xgb_qlike: float = 0.18, base_qlike: float = 0.20) -> pd.DataFrame:
        rows = [
            {
                "model_name": name,
                "pooled_rmse": 0.10,
                "pooled_mae": 0.08,
                "pooled_r2": 0.10,
                "pooled_qlike": base_qlike,
                "floored_pct": 0.0,
            }
            for name in STAGE1_BASELINE_NAMES
        ]
        rows.append(
            {
                "model_name": "xgboost_tuned",
                "pooled_rmse": 0.08,
                "pooled_mae": 0.075,
                "pooled_r2": 0.25,
                "pooled_qlike": xgb_qlike,
                "floored_pct": 0.0,
            }
        )
        return pd.DataFrame(rows)

    # 1. Matching fold_id but different oos_dates
    bad_dates = dict(fold_results)
    bad_dates["ols"] = [
        replace(fold_template, model_name="ols", oos_dates=("2023-01-04T00:00:00", "2023-01-05T00:00:00"))
    ]
    with pytest.raises(AIEngineException, match="oos_dates"):
        evaluate_stage2_gate(bad_dates, _summary())

    # 2. Matching fold_id but different oos_y_true
    bad_targets = dict(fold_results)
    bad_targets["ols"] = [
        replace(fold_template, model_name="ols", oos_y_true=(0.20, 0.30))
    ]
    with pytest.raises(AIEngineException, match="oos_y_true"):
        evaluate_stage2_gate(bad_targets, _summary())

    # 3. Different n_samples or horizon_days
    bad_samples = dict(fold_results)
    bad_samples["ols"] = [replace(fold_template, model_name="ols", n_samples=3)]
    with pytest.raises(AIEngineException, match="n_samples"):
        evaluate_stage2_gate(bad_samples, _summary())

    bad_horizon = dict(fold_results)
    bad_horizon["ols"] = [replace(fold_template, model_name="ols", horizon_days=5)]
    with pytest.raises(AIEngineException, match="horizon_days"):
        evaluate_stage2_gate(bad_horizon, _summary())

    # 4. Duplicate summary rows
    df_dup = pd.concat([_summary(), _summary().iloc[[0]]], ignore_index=True)
    with pytest.raises(AIEngineException, match="duplicate"):
        evaluate_stage2_gate(fold_results, df_dup)

    # 5. Missing summary columns
    with pytest.raises(AIEngineException, match="missing required columns"):
        evaluate_stage2_gate(fold_results, _summary().drop(columns=["pooled_qlike"]))

    # 6. Non-finite QLIKE on XGBoost or comparator fails QLIKE guardrail safely
    g_nan_xgb = evaluate_stage2_gate(fold_results, _summary(xgb_qlike=float("nan")))
    assert g_nan_xgb.qlike_guardrail_passed is False
    assert g_nan_xgb.gate_passed is False

    g_nan_base = evaluate_stage2_gate(fold_results, _summary(base_qlike=float("nan")))
    assert g_nan_base.qlike_guardrail_passed is False
    assert g_nan_base.gate_passed is False


def test_gate32_oos_metadata_completeness_validation() -> None:
    """evaluate_stage2_gate must reject empty, incomplete, or non-finite OOS metadata."""
    valid_fold = BaselineMetrics(
        model_name="m",
        rmse=0.10,
        mae_val=0.08,
        r2=0.1,
        qlike_val=0.20,
        n_samples=2,
        horizon_days=21,
        fold_id=0,
        oos_dates=("2023-01-02T00:00:00", "2023-01-03T00:00:00"),
        oos_y_true=(0.20, 0.25),
    )
    summary = pd.DataFrame(
        [
            {
                "model_name": name,
                "pooled_rmse": 0.10,
                "pooled_mae": 0.08,
                "pooled_r2": 0.10,
                "pooled_qlike": 0.20,
                "floored_pct": 0.0,
            }
            for name in STAGE1_BASELINE_NAMES
        ]
        + [
            {
                "model_name": "xgboost_tuned",
                "pooled_rmse": 0.08,
                "pooled_mae": 0.075,
                "pooled_r2": 0.25,
                "pooled_qlike": 0.18,
                "floored_pct": 0.0,
            }
        ]
    )

    # 1. Both models have empty OOS metadata despite positive n_samples
    empty_folds: dict[str, list[BaselineMetrics]] = {
        name: [replace(valid_fold, model_name=name, oos_dates=(), oos_y_true=())]
        for name in (*STAGE1_BASELINE_NAMES, "xgboost_tuned")
    }
    with pytest.raises(AIEngineException, match="incomplete"):
        evaluate_stage2_gate(empty_folds, summary)

    base_folds: dict[str, list[BaselineMetrics]] = {
        name: [replace(valid_fold, model_name=name)] for name in STAGE1_BASELINE_NAMES
    }
    base_folds["xgboost_tuned"] = [replace(valid_fold, model_name="xgboost_tuned")]

    # 2. Incomplete oos_dates
    inc_dates = dict(base_folds)
    inc_dates["ols"] = [replace(valid_fold, model_name="ols", oos_dates=("2023-01-02T00:00:00",))]
    with pytest.raises(AIEngineException, match="oos_dates"):
        evaluate_stage2_gate(inc_dates, summary)

    # 3. Incomplete oos_y_true
    inc_targets = dict(base_folds)
    inc_targets["xgboost_tuned"] = [replace(valid_fold, model_name="xgboost_tuned", oos_y_true=(0.20,))]
    with pytest.raises(AIEngineException, match="oos_y_true"):
        evaluate_stage2_gate(inc_targets, summary)

    # 4. Non-finite oos_y_true
    nan_targets = {
        name: [replace(valid_fold, model_name=name, oos_y_true=(0.20, float("nan")))]
        for name in (*STAGE1_BASELINE_NAMES, "xgboost_tuned")
    }
    with pytest.raises(AIEngineException, match="non-finite"):
        evaluate_stage2_gate(nan_targets, summary)


def test_hyperparameter_and_tuned_input_validation() -> None:
    """Validate strict integer/finite hyperparameter checks and tuned fit input checks."""
    with pytest.raises(AIEngineException):
        XGBoostHyperparameters(n_estimators=True)  # type: ignore[arg-type]
    with pytest.raises(AIEngineException):
        XGBoostHyperparameters(max_depth=3.0)  # type: ignore[arg-type]
    with pytest.raises(AIEngineException):
        XGBoostHyperparameters(learning_rate=float("nan"))
    with pytest.raises(AIEngineException):
        XGBoostHyperparameters(reg_lambda=float("inf"))

    with pytest.raises(AIEngineException):
        TunedXGBoostVolatilityRegressor(min_inner_train=0)
    with pytest.raises(AIEngineException):
        TunedXGBoostVolatilityRegressor(min_inner_val=True)  # type: ignore[arg-type]
    with pytest.raises(AIEngineException):
        TunedXGBoostVolatilityRegressor(val_fraction=float("nan"))
    with pytest.raises(AIEngineException):
        TunedXGBoostVolatilityRegressor(val_fraction=0.8)

    prices = _make_nonlinear_vol_prices(n_days=280, seed=19)
    ds = build_volatility_features(prices, horizon_days=21)
    tuned = TunedXGBoostVolatilityRegressor(horizon_days=21)

    # Mismatched index order between X and y
    with pytest.raises(DataLeakageError):
        tuned.fit(ds.X.iloc[:100], ds.y.iloc[:100].iloc[::-1])

    # Non-monotonic X index
    with pytest.raises(DataLeakageError):
        tuned.fit(ds.X.iloc[:100].iloc[::-1], ds.y.iloc[:100].iloc[::-1])

    # Non-finite y_train
    y_bad = ds.y.iloc[:100].copy()
    y_bad.iloc[0] = np.nan
    with pytest.raises(AIEngineException):
        tuned.fit(ds.X.iloc[:100], y_bad)


def test_end_to_end_future_price_perturbation_invariance() -> None:
    """Perturbing raw future prices after Fold 0 train target window must not alter Fold 0 tuning."""
    prices = _make_nonlinear_vol_prices(n_days=450, seed=25)
    ds_orig = build_volatility_features(prices, horizon_days=21)

    splitter = WalkForwardSplitter(
        mode="expanding", train_size=160, test_size=40, horizon_days=21
    )
    fold0 = next(splitter.split(ds_orig.X, ds_orig.y))
    tr_idx = fold0.train_indices.tolist()
    last_train_target_end = ds_orig.target_end_dates[tr_idx[-1]]

    # Shock raw prices strictly after Fold 0's last training target end date
    prices_shock = prices.copy()
    shock_mask = prices_shock.index > last_train_target_end
    assert int(shock_mask.sum()) >= 30
    prices_shock.loc[shock_mask] = prices_shock.loc[shock_mask] * 4.0

    ds_shock = build_volatility_features(prices_shock, horizon_days=21)

    # Fold 0 training features and labels rebuilt from scratch must be identical
    np.testing.assert_allclose(
        np.asarray(ds_orig.X.iloc[tr_idx], dtype=np.float64),
        np.asarray(ds_shock.X.iloc[tr_idx], dtype=np.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(ds_orig.y.iloc[tr_idx], dtype=np.float64),
        np.asarray(ds_shock.y.iloc[tr_idx], dtype=np.float64),
        rtol=1e-14,
        atol=1e-14,
    )

    small_grid = (
        XGBoostHyperparameters(max_depth=2, learning_rate=0.03, n_estimators=40),
        XGBoostHyperparameters(max_depth=3, learning_rate=0.07, n_estimators=40),
    )
    m_orig = TunedXGBoostVolatilityRegressor(horizon_days=21, param_grid=small_grid).fit(
        ds_orig.X.iloc[tr_idx], ds_orig.y.iloc[tr_idx]
    )
    m_shock = TunedXGBoostVolatilityRegressor(horizon_days=21, param_grid=small_grid).fit(
        ds_shock.X.iloc[tr_idx], ds_shock.y.iloc[tr_idx]
    )

    assert m_orig.selected_params == m_shock.selected_params
    assert m_orig.best_inner_val_rmse == pytest.approx(m_shock.best_inner_val_rmse, rel=1e-15)
    np.testing.assert_allclose(
        m_orig.predict(ds_orig.X.iloc[tr_idx]),
        m_shock.predict(ds_shock.X.iloc[tr_idx]),
        rtol=1e-14,
        atol=1e-14,
    )
