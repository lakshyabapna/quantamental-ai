"""Quantamental AI - Stage 2 Tree-Based Volatility Models (Module B, Step 3.2).

Implements XGBoost forward realized volatility regressors trained strictly on raw,
unscaled anti-leakage features (preserving DataFrame feature names and ordering):
    - ``XGBoostVolatilityRegressor``: Fixed regularized XGBoost regressor (default).
    - ``TunedXGBoostVolatilityRegressor``: Nested chronological hyperparameter tuning
      strictly inside each outer training fold with mandatory ``purge_gap >= horizon_days``.

Serialization uses native XGBoost JSON via ``ModelRegistry(..., artifact_format="json")``
with SHA-256 manifest integrity verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.ai_engine.baselines import BaselinePredictor
from app.ai_engine.metrics import rmse
from app.ai_engine.registry import ModelManifest, ModelRegistry
from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)


@dataclass(frozen=True)
class XGBoostHyperparameters:
    """Validated hyperparameter configuration for ``XGBoostVolatilityRegressor``."""

    n_estimators: int = 120
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_alpha: float = 0.1
    reg_lambda: float = 2.0
    random_state: int = 42
    n_jobs: int = 1
    tree_method: str = "hist"

    def __post_init__(self) -> None:
        for int_name, int_val in (
            ("n_estimators", self.n_estimators),
            ("max_depth", self.max_depth),
            ("random_state", self.random_state),
            ("n_jobs", self.n_jobs),
        ):
            if isinstance(int_val, bool) or not isinstance(int_val, int):
                raise AIEngineException(
                    f"{int_name} must be an integer (not bool or float), got {int_val!r}."
                )
        if self.n_estimators < 1:
            raise AIEngineException(f"n_estimators must be >= 1, got {self.n_estimators!r}.")
        if not (1 <= self.max_depth <= 10):
            raise AIEngineException(f"max_depth must be in [1, 10], got {self.max_depth!r}.")
        if self.n_jobs < 1:
            raise AIEngineException(f"n_jobs must be >= 1, got {self.n_jobs!r}.")

        for flt_name, flt_val in (
            ("learning_rate", self.learning_rate),
            ("subsample", self.subsample),
            ("colsample_bytree", self.colsample_bytree),
            ("min_child_weight", self.min_child_weight),
            ("reg_alpha", self.reg_alpha),
            ("reg_lambda", self.reg_lambda),
        ):
            if (
                isinstance(flt_val, bool)
                or not isinstance(flt_val, (int, float))
                or not np.isfinite(float(flt_val))
            ):
                raise AIEngineException(
                    f"{flt_name} must be a finite numeric value, got {flt_val!r}."
                )

        if not (0.0 < float(self.learning_rate) <= 1.0):
            raise AIEngineException(
                f"learning_rate must be in (0, 1], got {self.learning_rate!r}."
            )
        if not (0.0 < float(self.subsample) <= 1.0):
            raise AIEngineException(f"subsample must be in (0, 1], got {self.subsample!r}.")
        if not (0.0 < float(self.colsample_bytree) <= 1.0):
            raise AIEngineException(
                f"colsample_bytree must be in (0, 1], got {self.colsample_bytree!r}."
            )
        if (
            float(self.min_child_weight) < 0.0
            or float(self.reg_alpha) < 0.0
            or float(self.reg_lambda) < 0.0
        ):
            raise AIEngineException("Regularization and min_child_weight must be >= 0.")

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable hyperparameter mapping."""
        return asdict(self)


DEFAULT_XGBOOST_PARAM_GRID: Tuple[XGBoostHyperparameters, ...] = (
    XGBoostHyperparameters(max_depth=2, learning_rate=0.03, reg_lambda=1.0),
    XGBoostHyperparameters(max_depth=2, learning_rate=0.03, reg_lambda=5.0),
    XGBoostHyperparameters(max_depth=2, learning_rate=0.07, reg_lambda=1.0),
    XGBoostHyperparameters(max_depth=2, learning_rate=0.07, reg_lambda=5.0),
    XGBoostHyperparameters(max_depth=3, learning_rate=0.03, reg_lambda=1.0),
    XGBoostHyperparameters(max_depth=3, learning_rate=0.03, reg_lambda=5.0),
    XGBoostHyperparameters(max_depth=3, learning_rate=0.07, reg_lambda=1.0),
    XGBoostHyperparameters(max_depth=3, learning_rate=0.07, reg_lambda=5.0),
)


def _validate_dataframe_features(
    X: pd.DataFrame,
    caller: str,
    expected_columns: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, ...]:
    """Validate non-empty 2-D DataFrame with finite float values and optional column match."""
    if not isinstance(X, pd.DataFrame):
        raise AIEngineException(f"{caller}: expected pd.DataFrame, got {type(X).__name__}.")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise AIEngineException(f"{caller}: X must be a non-empty 2-D DataFrame, got {X.shape}.")
    cols = tuple(str(c) for c in X.columns)
    if expected_columns is not None and cols != expected_columns:
        raise AIEngineException(
            f"{caller}: feature columns {cols} do not match expected {expected_columns}."
        )
    arr = np.asarray(X, dtype=np.float64)
    if np.any(~np.isfinite(arr)):
        raise AIEngineException(f"{caller}: X contains NaN or Inf values.")
    return cols


class XGBoostVolatilityRegressor(BaselinePredictor):
    """Fixed regularized XGBoost regressor trained on raw, unscaled DataFrame features.

    Returns raw unconstrained volatility forecasts; apply ``apply_volatility_floor()``
    explicitly for positive variance-ratio evaluation.
    """

    def __init__(
        self,
        params: Optional[XGBoostHyperparameters] = None,
        *,
        name: str = "xgboost_fixed",
    ) -> None:
        self._params = params or XGBoostHyperparameters()
        self._name = name
        self._model: Optional[XGBRegressor] = None
        self._feature_names: Optional[Tuple[str, ...]] = None

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def hyperparameters(self) -> XGBoostHyperparameters:
        return self._params

    @property
    def feature_names(self) -> Tuple[str, ...]:
        if self._feature_names is None:
            raise AIEngineException("XGBoostVolatilityRegressor: model is not fitted.")
        return self._feature_names

    @property
    def xgb_model(self) -> XGBRegressor:
        if self._model is None:
            raise AIEngineException("XGBoostVolatilityRegressor: model is not fitted.")
        return self._model

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "XGBoostVolatilityRegressor":
        """Fit XGBoost on raw DataFrame features preserving column names."""
        cols = _validate_dataframe_features(X_train, "XGBoostVolatilityRegressor.fit")
        y_arr = np.asarray(y_train, dtype=np.float64)
        if y_arr.ndim != 1 or len(y_arr) != len(X_train):
            raise AIEngineException(
                "XGBoostVolatilityRegressor.fit: y_train length must match X_train rows."
            )
        if np.any(~np.isfinite(y_arr)):
            raise AIEngineException(
                "XGBoostVolatilityRegressor.fit: y_train contains NaN or Inf values."
            )

        model = XGBRegressor(
            n_estimators=self._params.n_estimators,
            max_depth=self._params.max_depth,
            learning_rate=self._params.learning_rate,
            subsample=self._params.subsample,
            colsample_bytree=self._params.colsample_bytree,
            min_child_weight=self._params.min_child_weight,
            reg_alpha=self._params.reg_alpha,
            reg_lambda=self._params.reg_lambda,
            random_state=self._params.random_state,
            n_jobs=self._params.n_jobs,
            tree_method=self._params.tree_method,
            objective="reg:squarederror",
            eval_metric="rmse",
        )
        model.fit(X_train, y_arr)
        self._model = model
        self._feature_names = cols
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return raw XGBoost predictions for ``X_test``."""
        if self._model is None or self._feature_names is None:
            raise AIEngineException("XGBoostVolatilityRegressor: call fit() before predict().")
        _validate_dataframe_features(
            X_test, "XGBoostVolatilityRegressor.predict", expected_columns=self._feature_names
        )
        preds = self._model.predict(X_test)
        return np.asarray(preds, dtype=np.float64).ravel()

    def to_json_dict(self) -> Dict[str, Any]:
        """Export fitted model to native XGBoost JSON dict."""
        booster = self.xgb_model.get_booster()
        raw_bytes = bytes(booster.save_raw(raw_format="json"))
        payload = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise AIEngineException("Expected dict from XGBoost native JSON export.")
        return payload

    @classmethod
    def from_json_dict(
        cls,
        payload: Dict[str, Any],
        *,
        params: Optional[XGBoostHyperparameters] = None,
        name: str = "xgboost_fixed",
    ) -> "XGBoostVolatilityRegressor":
        """Reconstruct a fitted ``XGBoostVolatilityRegressor`` from native XGBoost JSON."""
        raw_bytes = bytearray(json.dumps(payload).encode("utf-8"))
        model = XGBRegressor()
        model.load_model(raw_bytes)
        booster_names = model.get_booster().feature_names
        if not booster_names:
            raise AIEngineException(
                "Loaded XGBoost JSON artifact does not contain feature_names metadata."
            )
        instance = cls(params=params, name=name)
        instance._model = model
        instance._feature_names = tuple(str(c) for c in booster_names)
        return instance

    def save_to_registry(
        self,
        registry: ModelRegistry,
        *,
        model_id: str,
        version: str,
        data_snapshot_hash: str,
        train_start_date: str,
        train_end_date: str,
    ) -> ModelManifest:
        """Persist native XGBoost JSON artifact and reproducibility manifest."""
        return registry.save_artifact(
            self.to_json_dict(),
            model_id=model_id,
            version=version,
            stage="STAGE_2_XGBOOST",
            artifact_format="json",
            hyperparameters=self._params.to_dict(),
            random_seed=self._params.random_state,
            data_snapshot_hash=data_snapshot_hash,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
        )

    @classmethod
    def load_from_registry(
        cls,
        registry: ModelRegistry,
        *,
        model_id: str,
        version: str,
        name: str = "xgboost_fixed",
    ) -> Tuple["XGBoostVolatilityRegressor", ModelManifest]:
        """Load and verify native XGBoost JSON artifact from ``ModelRegistry``."""
        loaded, manifest = registry.load_artifact(model_id, version)
        if manifest.artifact_format != "json" or not isinstance(loaded, dict):
            raise AIEngineException(
                f"Expected native XGBoost JSON dict artifact, got {manifest.artifact_format!r}."
            )
        params = XGBoostHyperparameters(**manifest.hyperparameters)
        model = cls.from_json_dict(loaded, params=params, name=name)
        return model, manifest


class TunedXGBoostVolatilityRegressor(BaselinePredictor):
    """XGBoost regressor with nested chronological purged hyperparameter tuning.

    Tuning is performed strictly inside ``X_train`` / ``y_train`` by splitting
    chronologically into ``inner_train``, ``purge_gap >= horizon_days``, and
    ``inner_val``. Outer test data is never accessed. After selecting the best
    configuration on ``inner_val`` RMSE, the winning configuration is refitted
    on the full ``X_train`` / ``y_train``.
    """

    def __init__(
        self,
        *,
        horizon_days: int = 21,
        purge_gap: Optional[int] = None,
        val_fraction: float = 0.25,
        min_inner_train: int = 60,
        min_inner_val: int = 20,
        param_grid: Sequence[XGBoostHyperparameters] = DEFAULT_XGBOOST_PARAM_GRID,
        name: str = "xgboost_tuned",
    ) -> None:
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
            raise AIEngineException(f"horizon_days must be an integer >= 1, got {horizon_days!r}.")
        resolved_purge = horizon_days if purge_gap is None else purge_gap
        if isinstance(resolved_purge, bool) or not isinstance(resolved_purge, int):
            raise AIEngineException(f"purge_gap must be an integer, got {resolved_purge!r}.")
        if resolved_purge < horizon_days:
            raise DataLeakageError(
                f"Nested purge_gap ({resolved_purge}) must be >= horizon_days ({horizon_days})."
            )
        if (
            isinstance(min_inner_train, bool)
            or not isinstance(min_inner_train, int)
            or min_inner_train < 1
        ):
            raise AIEngineException(
                f"min_inner_train must be a positive integer, got {min_inner_train!r}."
            )
        if (
            isinstance(min_inner_val, bool)
            or not isinstance(min_inner_val, int)
            or min_inner_val < 1
        ):
            raise AIEngineException(
                f"min_inner_val must be a positive integer, got {min_inner_val!r}."
            )
        if (
            isinstance(val_fraction, bool)
            or not isinstance(val_fraction, (int, float))
            or not np.isfinite(float(val_fraction))
            or not (0.05 <= float(val_fraction) <= 0.5)
        ):
            raise AIEngineException(
                f"val_fraction must be finite and in [0.05, 0.5], got {val_fraction!r}."
            )
        if not param_grid:
            raise AIEngineException("param_grid must not be empty.")

        self._horizon_days = horizon_days
        self._purge_gap = resolved_purge
        self._val_fraction = val_fraction
        self._min_inner_train = min_inner_train
        self._min_inner_val = min_inner_val
        self._param_grid = tuple(param_grid)
        self._name = name

        self._fitted_model: Optional[XGBoostVolatilityRegressor] = None
        self._selected_params: Optional[XGBoostHyperparameters] = None
        self._best_inner_val_rmse: Optional[float] = None

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def selected_params(self) -> XGBoostHyperparameters:
        if self._selected_params is None:
            raise AIEngineException("TunedXGBoostVolatilityRegressor: model is not fitted.")
        return self._selected_params

    @property
    def best_inner_val_rmse(self) -> float:
        if self._best_inner_val_rmse is None:
            raise AIEngineException("TunedXGBoostVolatilityRegressor: model is not fitted.")
        return self._best_inner_val_rmse

    @property
    def fitted_regressor(self) -> XGBoostVolatilityRegressor:
        if self._fitted_model is None:
            raise AIEngineException("TunedXGBoostVolatilityRegressor: model is not fitted.")
        return self._fitted_model

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "TunedXGBoostVolatilityRegressor":
        """Run nested purged chronological tuning inside ``X_train`` and refit."""
        _validate_dataframe_features(X_train, "TunedXGBoostVolatilityRegressor.fit")
        if not isinstance(y_train, pd.Series):
            raise AIEngineException(
                f"TunedXGBoostVolatilityRegressor.fit: y_train must be a pd.Series, got {type(y_train).__name__}."
            )
        if len(y_train) != len(X_train) or not X_train.index.equals(y_train.index):
            raise DataLeakageError(
                "TunedXGBoostVolatilityRegressor.fit: X_train and y_train indices must match in exact order."
            )
        if isinstance(X_train.index, pd.DatetimeIndex):
            if X_train.index.hasnans:
                raise AIEngineException("TunedXGBoostVolatilityRegressor.fit: X_train index contains NaT.")
            if X_train.index.has_duplicates or not X_train.index.is_monotonic_increasing:
                raise DataLeakageError(
                    "TunedXGBoostVolatilityRegressor.fit: X_train DatetimeIndex must be strictly increasing."
                )
        y_arr = np.asarray(y_train, dtype=np.float64)
        if y_arr.ndim != 1 or np.any(~np.isfinite(y_arr)):
            raise AIEngineException(
                "TunedXGBoostVolatilityRegressor.fit: y_train must contain only finite values."
            )
        n_total = len(X_train)
        n_val = max(self._min_inner_val, int(round(n_total * self._val_fraction)))
        n_inner_train = n_total - self._purge_gap - n_val
        if n_inner_train < self._min_inner_train:
            raise InsufficientDataError(
                f"Insufficient training rows ({n_total}) for nested purged split "
                f"(need >= {self._min_inner_train + self._purge_gap + self._min_inner_val})."
            )

        X_in_tr = X_train.iloc[:n_inner_train]
        y_in_tr = y_train.iloc[:n_inner_train]
        val_start = n_inner_train + self._purge_gap
        X_in_val = X_train.iloc[val_start:]
        y_in_val = y_train.iloc[val_start:]
        y_in_val_arr = np.asarray(y_in_val, dtype=np.float64)

        best_params: Optional[XGBoostHyperparameters] = None
        best_score = float("inf")

        for candidate in self._param_grid:
            est = XGBoostVolatilityRegressor(params=candidate, name=self._name)
            est.fit(X_in_tr, y_in_tr)
            val_preds = est.predict(X_in_val)
            score = rmse(y_in_val_arr, val_preds)
            if score < best_score:
                best_score = score
                best_params = candidate

        if best_params is None:
            raise AIEngineException("TunedXGBoostVolatilityRegressor: hyperparameter search failed.")

        final_model = XGBoostVolatilityRegressor(params=best_params, name=self._name)
        final_model.fit(X_train, y_train)

        self._fitted_model = final_model
        self._selected_params = best_params
        self._best_inner_val_rmse = float(best_score)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return raw predictions from the refitted winning XGBoost model."""
        if self._fitted_model is None:
            raise AIEngineException(
                "TunedXGBoostVolatilityRegressor: call fit() before predict()."
            )
        return self._fitted_model.predict(X_test)
