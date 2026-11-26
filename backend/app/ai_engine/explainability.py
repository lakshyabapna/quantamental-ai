"""Quantamental AI - TreeSHAP Explainability Engine (Module B, Step 3.2).

Provides global and local SHAP explanations for fitted ``XGBoostVolatilityRegressor``
models using ``shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")``.

Properties:
    - Zero test-set background leakage: ``tree_path_dependent`` uses only the tree
      structure learned on training data.
    - Verifies SHAP additivity strictly against **raw** model predictions (prior to
      any post-processing volatility floor):
          ``abs(base_value + sum(shap_values) - raw_prediction) <= tol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import shap

from app.ai_engine.tree_models import XGBoostVolatilityRegressor
from app.core.exceptions import AIEngineException


_ADDITIVITY_TOL: float = 1e-4


@dataclass(frozen=True)
class GlobalShapSummary:
    """Global feature importance ranked by mean absolute SHAP value."""

    feature_names: Tuple[str, ...]
    mean_abs_shap: Dict[str, float]
    top_k_features: Tuple[Tuple[str, float], ...]
    base_value: float
    n_samples: int
    max_additivity_error: float


@dataclass(frozen=True)
class LocalShapExplanation:
    """Local SHAP attribution for a single volatility forecast observation."""

    timestamp: str
    raw_prediction: float
    base_value: float
    feature_values: Dict[str, float]
    shap_values: Dict[str, float]
    top_contributors: Tuple[Tuple[str, float], ...]
    additivity_error: float


def _build_tree_explainer(model: XGBoostVolatilityRegressor) -> shap.TreeExplainer:
    """Construct a deterministic tree-path-dependent SHAP explainer."""
    return shap.TreeExplainer(
        model.xgb_model,
        feature_perturbation="tree_path_dependent",
    )


def compute_global_shap(
    model: XGBoostVolatilityRegressor,
    X: pd.DataFrame,
    *,
    top_k: int = 10,
    additivity_tol: float = _ADDITIVITY_TOL,
) -> GlobalShapSummary:
    """Compute global TreeSHAP feature importance on ``X`` and verify additivity.

    Parameters
    ----------
    model:
        Fitted ``XGBoostVolatilityRegressor``.
    X:
        Unscaled DataFrame matching the model's exact training feature columns.
    top_k:
        Number of top-ranked features to include in ``top_k_features``.
    additivity_tol:
        Maximum allowed ``|base_value + sum(shap) - raw_prediction|`` error.

    Returns
    -------
    GlobalShapSummary
    """
    if isinstance(top_k, bool) or top_k < 1:
        raise AIEngineException(f"compute_global_shap: top_k must be >= 1, got {top_k!r}.")

    raw_preds = model.predict(X)
    explainer = _build_tree_explainer(model)
    shap_arr = np.asarray(explainer.shap_values(X), dtype=np.float64)
    if shap_arr.ndim != 2 or shap_arr.shape != (len(X), len(model.feature_names)):
        raise AIEngineException(
            f"compute_global_shap: unexpected SHAP shape {shap_arr.shape}."
        )

    base_val = float(np.asarray(explainer.expected_value, dtype=np.float64).ravel()[0])
    reconstructed = base_val + np.sum(shap_arr, axis=1)
    max_err = float(np.max(np.abs(reconstructed - raw_preds)))
    if max_err > additivity_tol:
        raise AIEngineException(
            f"TreeSHAP additivity check failed: max_err={max_err:.3e} > tol={additivity_tol:.3e}.",
            details={"max_additivity_error": max_err, "tolerance": additivity_tol},
        )

    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    names = model.feature_names
    pairs = sorted(
        ((name, float(val)) for name, val in zip(names, mean_abs)),
        key=lambda item: item[1],
        reverse=True,
    )
    mean_abs_dict = {name: val for name, val in pairs}
    top_tuple = tuple(pairs[: min(top_k, len(pairs))])

    return GlobalShapSummary(
        feature_names=names,
        mean_abs_shap=mean_abs_dict,
        top_k_features=top_tuple,
        base_value=base_val,
        n_samples=len(X),
        max_additivity_error=max_err,
    )


def explain_single_prediction(
    model: XGBoostVolatilityRegressor,
    X: pd.DataFrame,
    *,
    row_index: int = 0,
    top_k: int = 5,
    additivity_tol: float = _ADDITIVITY_TOL,
) -> LocalShapExplanation:
    """Compute local TreeSHAP explanation for one row of ``X`` against raw prediction."""
    if isinstance(top_k, bool) or top_k < 1:
        raise AIEngineException(f"explain_single_prediction: top_k must be >= 1, got {top_k!r}.")
    if isinstance(row_index, bool) or not (0 <= row_index < len(X)):
        raise AIEngineException(
            f"explain_single_prediction: row_index {row_index} out of bounds for len(X)={len(X)}."
        )

    row_df = X.iloc[[row_index]]
    raw_pred = float(model.predict(row_df)[0])
    explainer = _build_tree_explainer(model)
    shap_row = np.asarray(explainer.shap_values(row_df), dtype=np.float64).ravel()
    base_val = float(np.asarray(explainer.expected_value, dtype=np.float64).ravel()[0])

    err = float(abs((base_val + float(np.sum(shap_row))) - raw_pred))
    if err > additivity_tol:
        raise AIEngineException(
            f"TreeSHAP local additivity check failed: err={err:.3e} > tol={additivity_tol:.3e}.",
            details={"additivity_error": err, "tolerance": additivity_tol},
        )

    names = model.feature_names
    feat_vals = {
        name: float(row_df.iloc[0][name]) for name in names
    }
    shap_dict = {name: float(val) for name, val in zip(names, shap_row)}
    ranked = tuple(
        sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[
            : min(top_k, len(names))
        ]
    )

    ts = str(row_df.index[0])
    return LocalShapExplanation(
        timestamp=ts,
        raw_prediction=raw_pred,
        base_value=base_val,
        feature_values=feat_vals,
        shap_values=shap_dict,
        top_contributors=ranked,
        additivity_error=err,
    )
