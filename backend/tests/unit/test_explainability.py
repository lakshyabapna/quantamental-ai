"""Unit tests for Stage 3.2 TreeSHAP explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
    GlobalShapSummary,
    LocalShapExplanation,
    XGBoostHyperparameters,
    XGBoostVolatilityRegressor,
    build_volatility_features,
    compute_global_shap,
    explain_single_prediction,
)
from app.core.exceptions import AIEngineException


def _make_sample_xgb_and_data(
    n_days: int = 260, seed: int = 77
) -> tuple[XGBoostVolatilityRegressor, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    vols = np.where(np.arange(n_days) % 90 < 45, 0.010, 0.022)
    rets = rng.normal(0.0003, vols, size=n_days)
    prices = pd.Series(
        (1000.0 * np.exp(np.cumsum(rets))).tolist(),
        index=dates,
        name="RELIANCE.NS",
        dtype=float,
    )
    ds = build_volatility_features(prices, horizon_days=21)
    X_tr = ds.X.iloc[:100]
    y_tr = ds.y.iloc[:100]
    X_te = ds.X.iloc[121:145]
    model = XGBoostVolatilityRegressor(
        params=XGBoostHyperparameters(n_estimators=50, max_depth=3, learning_rate=0.05)
    ).fit(X_tr, y_tr)
    return model, X_tr, X_te


def test_global_shap_ranking_and_raw_additivity() -> None:
    """TreeSHAP global summary must rank features descending and satisfy raw additivity."""
    model, _, X_te = _make_sample_xgb_and_data()
    summary = compute_global_shap(model, X_te, top_k=5)

    assert isinstance(summary, GlobalShapSummary)
    assert summary.n_samples == len(X_te)
    assert summary.max_additivity_error <= 1e-4
    assert len(summary.top_k_features) == 5

    ranked_vals = list(summary.mean_abs_shap.values())
    assert all(val >= 0.0 for val in ranked_vals)
    assert ranked_vals == sorted(ranked_vals, reverse=True)


def test_local_shap_explanation_matches_raw_prediction() -> None:
    """Local SHAP base_value + sum(shap_values) must match raw prediction before flooring."""
    model, _, X_te = _make_sample_xgb_and_data()
    local = explain_single_prediction(model, X_te, row_index=3, top_k=4)

    assert isinstance(local, LocalShapExplanation)
    assert local.additivity_error <= 1e-4
    assert len(local.top_contributors) == 4
    reconstructed = local.base_value + sum(local.shap_values.values())
    assert reconstructed == pytest.approx(local.raw_prediction, abs=1e-4)
    assert set(local.feature_values.keys()) == set(model.feature_names)


def test_shap_error_handling_for_invalid_inputs() -> None:
    """Explainability functions must raise AIEngineException on invalid row_index or top_k."""
    model, _, X_te = _make_sample_xgb_and_data()
    with pytest.raises(AIEngineException):
        compute_global_shap(model, X_te, top_k=0)
    with pytest.raises(AIEngineException):
        explain_single_prediction(model, X_te, row_index=-1)
    with pytest.raises(AIEngineException):
        explain_single_prediction(model, X_te, row_index=len(X_te))
