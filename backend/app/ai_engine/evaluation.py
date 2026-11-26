"""Quantamental AI - Walk-Forward Baseline Evaluation Harness (Module B, Step 3.1).

Orchestrates end-to-end walk-forward evaluation of historical baseline models over a
single asset's price series using Step 3.0 anti-leakage infrastructure.

Key design properties:
    - ``evaluate_baselines``: Builds ``FeatureDataset``, runs ``WalkForwardSplitter``
      (with ``val_size=0`` since Step 3.1 baselines perform no validation-based tuning),
      evaluates every model on identical OOS folds, preserves raw predictions, applies
      ``ForecastFloorPolicy`` explicitly, and records flooring counts/percentages.
    - ``RollingVolBaseline`` and ``RollingMeanBaseline`` receive unscaled features/targets;
      ``OLSBaseline`` and ``RidgeBaseline`` receive ``ScaledFoldData`` fitted strictly
      on training rows.
    - ``aggregate_metrics``: Computes both **pooled out-of-sample metrics** (from
      concatenated OOS targets and predictions across all folds) and **macro fold
      averages** (means of per-fold metrics), clearly distinguishing them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from app.ai_engine.baselines import (
    DEFAULT_MIN_ANNUALIZED_VOL,
    BaselinePredictor,
    ForecastFloorPolicy,
    OLSBaseline,
    RidgeBaseline,
    RollingMeanBaseline,
    RollingVolBaseline,
    apply_volatility_floor,
)
from app.ai_engine.feature_pipeline import FeatureDataset, build_volatility_features
from app.ai_engine.metrics import BaselineMetrics, mae, qlike, r_squared, rmse
from app.ai_engine.splitters import SplitMode, WalkForwardSplitter, scale_fold_features
from app.core.exceptions import AIEngineException, InsufficientDataError


def _make_default_baselines() -> List[BaselinePredictor]:
    """Return a fresh list of default baselines (new instances to avoid state sharing)."""
    return [
        RollingVolBaseline(window=21),
        RollingVolBaseline(window=63),
        RollingMeanBaseline(window=21),
        OLSBaseline(),
        RidgeBaseline(alpha=1.0),
    ]


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² or return NaN if test target has zero variance."""
    try:
        return r_squared(y_true, y_pred)
    except AIEngineException:
        return float("nan")


def _safe_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute normalized QLIKE or return NaN if inputs violate strict positivity."""
    try:
        return qlike(y_true, y_pred)
    except AIEngineException:
        return float("nan")


def evaluate_baselines(
    asset_prices: Union[pd.Series, pd.DataFrame],
    *,
    benchmark_prices: Optional[Union[pd.Series, pd.DataFrame]] = None,
    volumes: Optional[Union[pd.Series, pd.DataFrame]] = None,
    horizon_days: int = 21,
    train_size: int = 252,
    test_size: int = 63,
    step_size: Optional[int] = None,
    mode: SplitMode = "expanding",
    min_volatility: float = DEFAULT_MIN_ANNUALIZED_VOL,
    baselines: Optional[List[BaselinePredictor]] = None,
) -> Dict[str, List[BaselineMetrics]]:
    """Run walk-forward evaluation of historical baseline volatility models.

    Parameters
    ----------
    asset_prices:
        Adjusted closing prices for the target asset (Series or single-column DataFrame).
    benchmark_prices:
        Optional benchmark prices aligned with ``asset_prices``.
    volumes:
        Optional trading volume Series aligned with ``asset_prices``.
    horizon_days:
        Forward realized volatility horizon in trading days. Default: 21.
    train_size:
        Number of training samples per fold. Default: 252.
    test_size:
        Number of test samples per fold. Default: 63.
    step_size:
        Advance per fold. Defaults to ``test_size``.
    mode:
        ``"expanding"`` or ``"rolling"`` window. Default: ``"expanding"``.
    min_volatility:
        Explicit positive annualized volatility floor applied in post-processing
        (default: ``1e-3``, i.e. 0.1% annualized volatility). Raw predictions are
        also preserved and evaluated separately.
    baselines:
        List of ``BaselinePredictor`` instances with unique ``model_name``s.

    Returns
    -------
    Dict[str, List[BaselineMetrics]]
        Mapping from ``model_name`` to per-fold ``BaselineMetrics``.
    """
    if baselines is None:
        baselines = _make_default_baselines()

    if not baselines:
        raise AIEngineException("evaluate_baselines: baselines list must not be empty.")

    # Reject duplicate model names explicitly
    seen_names: set[str] = set()
    for b in baselines:
        if b.model_name in seen_names:
            raise AIEngineException(
                f"evaluate_baselines: duplicate model_name '{b.model_name}' in baselines list.",
                details={"duplicate_model_name": b.model_name},
            )
        seen_names.add(b.model_name)

    # Validate floor policy upfront
    floor_policy = ForecastFloorPolicy(min_volatility=min_volatility)

    dataset: FeatureDataset = build_volatility_features(
        asset_prices,
        benchmark_prices=benchmark_prices,
        volumes=volumes,
        horizon_days=horizon_days,
    )

    X: pd.DataFrame = dataset.X
    y: pd.Series = dataset.y

    splitter = WalkForwardSplitter(
        mode=mode,
        train_size=train_size,
        test_size=test_size,
        val_size=0,
        step_size=step_size,
        horizon_days=horizon_days,
    )

    results: Dict[str, List[BaselineMetrics]] = {b.model_name: [] for b in baselines}

    _feature_extractors = (RollingVolBaseline, RollingMeanBaseline)
    _regression_models = (OLSBaseline, RidgeBaseline)

    n_folds = 0
    for fold in splitter.split(X, y):
        n_folds += 1
        train_idx = fold.train_indices.tolist()
        test_idx = fold.test_indices.tolist()

        X_train_raw = X.iloc[train_idx]
        y_train_raw = y.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_test_arr = np.asarray(y.iloc[test_idx], dtype=np.float64)

        scaled = scale_fold_features(X, y, fold)

        for model in baselines:
            if isinstance(model, _feature_extractors):
                model.fit(X_train_raw, y_train_raw)
                raw_preds = model.predict(X_test_raw)
            elif isinstance(model, _regression_models):
                model.fit(scaled.X_train, scaled.y_train)
                raw_preds = model.predict(scaled.X_test)
            else:
                model.fit(X_train_raw, y_train_raw)
                raw_preds = model.predict(X_test_raw)

            post = apply_volatility_floor(
                raw_preds, min_volatility=floor_policy.min_volatility
            )

            fold_metrics = BaselineMetrics(
                model_name=model.model_name,
                rmse=rmse(y_test_arr, post.predictions),
                mae_val=mae(y_test_arr, post.predictions),
                r2=_safe_r2(y_test_arr, post.predictions),
                qlike_val=_safe_qlike(y_test_arr, post.predictions),
                n_samples=len(test_idx),
                horizon_days=horizon_days,
                fold_id=fold.fold_id,
                test_start_date=str(fold.test_start_date.date()),
                test_end_date=str(fold.test_end_date.date()),
                raw_rmse=rmse(y_test_arr, post.raw_predictions),
                raw_mae=mae(y_test_arr, post.raw_predictions),
                raw_r2=_safe_r2(y_test_arr, post.raw_predictions),
                n_floored=post.n_floored,
                floored_pct=post.floored_pct,
                oos_dates=tuple(pd.Timestamp(ts).isoformat() for ts in X.index[test_idx]),
                oos_y_true=tuple(float(v) for v in y_test_arr),
                oos_y_pred_raw=tuple(float(v) for v in post.raw_predictions),
                oos_y_pred_processed=tuple(float(v) for v in post.predictions),
            )
            results[model.model_name].append(fold_metrics)

    if n_folds == 0:
        raise InsufficientDataError(
            "evaluate_baselines: no folds were generated. "
            "Increase the number of samples or reduce train_size/test_size.",
            details={"train_size": train_size, "test_size": test_size, "n_samples": len(X)},
        )

    return results


def aggregate_metrics(
    fold_results: Dict[str, List[BaselineMetrics]],
) -> pd.DataFrame:
    """Aggregate per-fold ``BaselineMetrics`` into pooled OOS and macro fold-averaged metrics.

    Verifies strict OOS observation identity across all models:
        - Complete OOS metadata (``oos_dates``, ``oos_y_true``, ``oos_y_pred_raw``,
          ``oos_y_pred_processed`` each matching ``n_samples``).
        - Consistent ``horizon_days`` and strictly ordered ``fold_id`` sequences.
        - No duplicate or overlapping OOS timestamps across folds.
        - Identical OOS date sequences and target observations across every model.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by ``pooled_rmse`` ascending.
    """
    if not fold_results:
        raise AIEngineException("aggregate_metrics: fold_results is empty.")

    reference_y: Optional[np.ndarray] = None
    reference_dates: Optional[tuple[str, ...]] = None
    reference_fold_ids: Optional[tuple[int, ...]] = None
    reference_horizon: Optional[int] = None
    rows = []

    for model_name, fold_list in fold_results.items():
        if not fold_list:
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' has no fold results."
            )

        horizon_days = fold_list[0].horizon_days
        fold_ids = tuple(f.fold_id for f in fold_list)
        if len(fold_ids) > 1 and any(
            fold_ids[i] <= fold_ids[i - 1] for i in range(1, len(fold_ids))
        ):
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' fold_ids are not strictly increasing: {fold_ids}."
            )

        for f in fold_list:
            if f.n_samples < 1:
                raise AIEngineException(
                    f"aggregate_metrics: model '{model_name}' fold {f.fold_id} has n_samples={f.n_samples}."
                )
            if f.horizon_days != horizon_days:
                raise AIEngineException(
                    f"aggregate_metrics: model '{model_name}' has inconsistent horizon_days "
                    f"({f.horizon_days} vs {horizon_days})."
                )
            if (
                len(f.oos_dates) != f.n_samples
                or len(f.oos_y_true) != f.n_samples
                or len(f.oos_y_pred_raw) != f.n_samples
                or len(f.oos_y_pred_processed) != f.n_samples
            ):
                raise AIEngineException(
                    f"aggregate_metrics: model '{model_name}' fold {f.fold_id} has incomplete or "
                    "mismatched OOS observation metadata/array lengths."
                )

        dates_all = tuple(d for f in fold_list for d in f.oos_dates)
        if len(set(dates_all)) != len(dates_all):
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' contains duplicate OOS timestamps."
            )
        parsed_dates = pd.DatetimeIndex(dates_all)
        if not parsed_dates.is_monotonic_increasing:
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' OOS timestamps are overlapping or out of order."
            )

        y_all = np.concatenate([np.asarray(f.oos_y_true, dtype=np.float64) for f in fold_list])
        raw_all = np.concatenate(
            [np.asarray(f.oos_y_pred_raw, dtype=np.float64) for f in fold_list]
        )
        proc_all = np.concatenate(
            [np.asarray(f.oos_y_pred_processed, dtype=np.float64) for f in fold_list]
        )

        if reference_horizon is None:
            reference_horizon = horizon_days
        elif horizon_days != reference_horizon:
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' horizon_days ({horizon_days}) "
                f"does not match reference ({reference_horizon})."
            )

        if reference_fold_ids is None:
            reference_fold_ids = fold_ids
        elif fold_ids != reference_fold_ids:
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' fold_ids {fold_ids} "
                f"do not match reference {reference_fold_ids}."
            )

        if reference_dates is None:
            reference_dates = dates_all
        elif dates_all != reference_dates:
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' OOS dates do not match "
                "the reference model's OOS dates."
            )

        if reference_y is None:
            reference_y = y_all
        elif len(y_all) != len(reference_y) or not np.allclose(
            y_all, reference_y, rtol=1e-14, atol=1e-14
        ):
            raise AIEngineException(
                f"aggregate_metrics: model '{model_name}' OOS target observations do not match "
                "the reference model's OOS target observations."
            )

        rmse_vals = [f.rmse for f in fold_list]
        mae_vals = [f.mae_val for f in fold_list]
        r2_vals = [f.r2 for f in fold_list]
        qlike_vals = [f.qlike_val for f in fold_list]
        total_samples = sum(f.n_samples for f in fold_list)
        total_floored = sum(f.n_floored for f in fold_list)
        floored_pct = float(100.0 * total_floored / total_samples) if total_samples > 0 else 0.0

        macro_rmse = float(np.nanmean(rmse_vals))
        macro_mae = float(np.nanmean(mae_vals))
        macro_r2 = float(np.nanmean(r2_vals)) if not np.all(np.isnan(r2_vals)) else float("nan")
        macro_qlike = (
            float(np.nanmean(qlike_vals)) if not np.all(np.isnan(qlike_vals)) else float("nan")
        )

        pooled_rmse = rmse(y_all, proc_all)
        pooled_mae = mae(y_all, proc_all)
        pooled_r2 = _safe_r2(y_all, proc_all)
        pooled_qlike = _safe_qlike(y_all, proc_all)

        raw_pooled_rmse = rmse(y_all, raw_all)
        raw_pooled_mae = mae(y_all, raw_all)
        raw_pooled_r2 = _safe_r2(y_all, raw_all)

        rows.append(
            {
                "model_name": model_name,
                "pooled_rmse": pooled_rmse,
                "pooled_mae": pooled_mae,
                "pooled_r2": pooled_r2,
                "pooled_qlike": pooled_qlike,
                "raw_pooled_rmse": raw_pooled_rmse,
                "raw_pooled_mae": raw_pooled_mae,
                "raw_pooled_r2": raw_pooled_r2,
                "macro_rmse": macro_rmse,
                "macro_mae": macro_mae,
                "macro_r2": macro_r2,
                "macro_qlike": macro_qlike,
                "n_floored": total_floored,
                "floored_pct": floored_pct,
                "n_folds": len(fold_list),
                "total_samples": total_samples,
                "horizon_days": horizon_days,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("pooled_rmse", ascending=True).reset_index(drop=True)
    return df
