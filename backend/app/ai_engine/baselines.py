"""Quantamental AI - Historical Baseline Models (Module B, Step 3.1).

Implements strictly causal, non-data-leaking historical baseline models for forward
realized volatility forecasting, together with an explicit forecast post-processing
floor policy.

Baseline hierarchy:
    - ``RollingVolBaseline``: Extracts pre-computed rolling annualized volatility
      directly from the ``FeatureDataset.X`` feature matrix (column ``vol_{window}``),
      which is already annualized by ``build_volatility_features``.
    - ``RollingMeanBaseline``: Trailing mean of past target values seen in training,
      extrapolated as a constant-mean forecast over the test horizon.
    - ``OLSBaseline``: Ordinary Least Squares regression via scikit-learn.
    - ``RidgeBaseline``: Ridge regression with L2 regularization.

Forecast Post-Processing Policy:
    Unconstrained linear models (OLS, Ridge) can produce non-positive volatility
    predictions. Raw predictions are always preserved for diagnostics; an explicit
    ``ForecastFloorPolicy`` (default ``min_volatility = 1e-3``, i.e. 0.1% annualized
    volatility) is applied separately via ``apply_volatility_floor()`` before
    computing variance-ratio metrics such as QLIKE.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from app.core.exceptions import AIEngineException


DEFAULT_MIN_ANNUALIZED_VOL: float = 1e-3


# ---------------------------------------------------------------------------
# Explicit forecast post-processing policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastFloorPolicy:
    """Explicit positive floor policy for annualized volatility forecasts.

    Attributes
    ----------
    min_volatility:
        Minimum allowable annualized volatility in decimal units
        (default: ``1e-3``, i.e. 0.1% annualized volatility). Must be strictly
        positive and finite.
    """

    min_volatility: float = DEFAULT_MIN_ANNUALIZED_VOL

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_volatility, bool)
            or not isinstance(self.min_volatility, (int, float))
            or not np.isfinite(float(self.min_volatility))
            or float(self.min_volatility) <= 0.0
        ):
            raise AIEngineException(
                "ForecastFloorPolicy: min_volatility must be a finite, strictly positive "
                f"float in annualized volatility units, got {self.min_volatility!r}."
            )


@dataclass(frozen=True)
class PostProcessedForecast:
    """Result of applying ``ForecastFloorPolicy`` while preserving raw predictions."""

    raw_predictions: np.ndarray
    predictions: np.ndarray
    n_floored: int
    floored_pct: float
    min_volatility: float


def apply_volatility_floor(
    raw_predictions: np.ndarray,
    min_volatility: float = DEFAULT_MIN_ANNUALIZED_VOL,
) -> PostProcessedForecast:
    """Apply an explicit positive floor to raw annualized volatility predictions.

    Raw predictions are preserved untouched in ``raw_predictions`` for diagnostics.

    Parameters
    ----------
    raw_predictions:
        1-D array of raw model volatility predictions (annualized).
    min_volatility:
        Positive annualized volatility floor (default: ``1e-3``).

    Returns
    -------
    PostProcessedForecast
        Container holding both ``raw_predictions``, floored ``predictions``,
        and flooring count/percentage diagnostics.
    """
    policy = ForecastFloorPolicy(min_volatility=min_volatility)
    raw = np.asarray(raw_predictions, dtype=np.float64)
    if raw.ndim != 1 or len(raw) == 0:
        raise AIEngineException(
            f"apply_volatility_floor: expected non-empty 1-D array, got shape {raw.shape}."
        )
    if np.any(~np.isfinite(raw)):
        raise AIEngineException(
            "apply_volatility_floor: raw_predictions contains NaN or Inf values."
        )

    floored_mask = raw < policy.min_volatility
    n_floored = int(np.sum(floored_mask))
    floored_pct = float(100.0 * n_floored / len(raw))
    processed = np.maximum(raw, policy.min_volatility)

    return PostProcessedForecast(
        raw_predictions=raw,
        predictions=processed,
        n_floored=n_floored,
        floored_pct=floored_pct,
        min_volatility=float(policy.min_volatility),
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaselinePredictor(abc.ABC):
    """Abstract base class for all historical baseline volatility predictors."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Unique identifier string for this model."""

    @abc.abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaselinePredictor":
        """Fit the model on training data."""

    @abc.abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return raw 1-D annualized volatility predictions for ``X_test``."""


# ---------------------------------------------------------------------------
# Rolling volatility baseline
# ---------------------------------------------------------------------------


class RollingVolBaseline(BaselinePredictor):
    """Predict forward volatility using the pre-computed lagged rolling volatility feature.

    Extracts ``vol_{window}`` from ``FeatureDataset.X``, which was computed by
    ``build_volatility_features`` as the ``window``-day rolling annualized volatility
    ending at ``t - 1`` (strictly causal).

    Parameters
    ----------
    window:
        Rolling lookback window in trading days (e.g., 5, 10, 21, 63). Default: 21.
    """

    def __init__(self, window: int = 21) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise AIEngineException(
                f"RollingVolBaseline: window must be a positive integer, got {window!r}."
            )
        self._window = window
        self._col_name = f"vol_{window}"

    @property
    def model_name(self) -> str:
        return f"rolling_vol_{self._window}d"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RollingVolBaseline":
        """No-op: validates that ``vol_{window}`` exists in ``X_train``."""
        if self._col_name not in X_train.columns:
            raise AIEngineException(
                f"RollingVolBaseline: column '{self._col_name}' not found in X_train. "
                f"Available: {list(X_train.columns)}"
            )
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Extract the pre-computed rolling volatility column from ``X_test``."""
        if len(X_test) == 0:
            raise AIEngineException("RollingVolBaseline.predict: X_test is empty.")
        if self._col_name not in X_test.columns:
            raise AIEngineException(
                f"RollingVolBaseline: column '{self._col_name}' not found in X_test. "
                f"Available: {list(X_test.columns)}"
            )
        preds = np.asarray(X_test[self._col_name], dtype=np.float64)
        if np.any(~np.isfinite(preds)):
            raise AIEngineException(
                f"RollingVolBaseline: '{self._col_name}' contains non-finite values."
            )
        return preds


# ---------------------------------------------------------------------------
# Rolling mean baseline
# ---------------------------------------------------------------------------


class RollingMeanBaseline(BaselinePredictor):
    """Predict forward volatility as the trailing rolling mean of past target values.

    Computes the trailing mean over the last ``window`` observations of ``y_train``
    and broadcasts that constant forecast across ``X_test``.
    """

    def __init__(self, window: int = 21) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise AIEngineException(
                f"RollingMeanBaseline: window must be a positive integer, got {window!r}."
            )
        self._window = window
        self._fitted_mean: Optional[float] = None

    @property
    def model_name(self) -> str:
        return f"rolling_mean_{self._window}d"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RollingMeanBaseline":
        """Compute the trailing mean of the last ``window`` training targets."""
        y_arr = np.asarray(y_train, dtype=np.float64)
        if len(y_arr) == 0:
            raise AIEngineException("RollingMeanBaseline.fit: y_train is empty.")
        if np.any(~np.isfinite(y_arr)):
            raise AIEngineException(
                "RollingMeanBaseline.fit: y_train contains non-finite values."
            )
        lookback = y_arr[-self._window:]
        self._fitted_mean = float(np.mean(lookback))
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return the constant trailing-mean prediction for each row in ``X_test``."""
        if self._fitted_mean is None:
            raise AIEngineException("RollingMeanBaseline: call fit() before predict().")
        if len(X_test) == 0:
            raise AIEngineException("RollingMeanBaseline.predict: X_test is empty.")
        return np.full(len(X_test), fill_value=self._fitted_mean, dtype=np.float64)


# ---------------------------------------------------------------------------
# OLS baseline
# ---------------------------------------------------------------------------


def _validate_regression_test_matrix(X_test: pd.DataFrame, caller: str, expected_cols: int) -> np.ndarray:
    """Validate test feature matrix for linear regression baselines."""
    X_arr = np.asarray(X_test, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise AIEngineException(
            f"{caller}.predict: X_test must be a non-empty 2-D matrix, got shape {X_arr.shape}."
        )
    if X_arr.shape[1] != expected_cols:
        raise AIEngineException(
            f"{caller}.predict: expected {expected_cols} features, got {X_arr.shape[1]}."
        )
    if np.any(~np.isfinite(X_arr)):
        raise AIEngineException(f"{caller}.predict: X_test contains NaN or Inf values.")
    return X_arr


class OLSBaseline(BaselinePredictor):
    """Ordinary Least Squares linear regression baseline for volatility forecasting.

    Returns raw unconstrained predictions; apply ``apply_volatility_floor()``
    for positive variance-ratio evaluation.
    """

    def __init__(self) -> None:
        self._model: Optional[LinearRegression] = None

    @property
    def model_name(self) -> str:
        return "ols"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "OLSBaseline":
        """Fit OLS on scaled training data."""
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise AIEngineException("OLSBaseline.fit: X_train must be a non-empty 2-D matrix.")
        if np.any(~np.isfinite(X_arr)):
            raise AIEngineException("OLSBaseline.fit: X_train contains non-finite values.")
        if np.any(~np.isfinite(y_arr)):
            raise AIEngineException("OLSBaseline.fit: y_train contains non-finite values.")
        self._model = LinearRegression()
        self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return raw OLS predictions (unconstrained)."""
        if self._model is None:
            raise AIEngineException("OLSBaseline: call fit() before predict().")
        X_arr = _validate_regression_test_matrix(
            X_test, "OLSBaseline", int(self._model.n_features_in_)
        )
        preds = self._model.predict(X_arr)
        return np.asarray(preds, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Ridge baseline
# ---------------------------------------------------------------------------


class RidgeBaseline(BaselinePredictor):
    """Ridge regression baseline with L2 regularization for volatility forecasting.

    Returns raw unconstrained predictions; apply ``apply_volatility_floor()``
    for positive variance-ratio evaluation.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha < 0.0:
            raise AIEngineException(
                f"RidgeBaseline: alpha must be a non-negative float, got {alpha!r}."
            )
        self._alpha = float(alpha)
        self._model: Optional[Ridge] = None

    @property
    def model_name(self) -> str:
        return f"ridge_alpha_{self._alpha:.4g}"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeBaseline":
        """Fit Ridge on scaled training data."""
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[0] == 0:
            raise AIEngineException("RidgeBaseline.fit: X_train must be a non-empty 2-D matrix.")
        if np.any(~np.isfinite(X_arr)):
            raise AIEngineException("RidgeBaseline.fit: X_train contains non-finite values.")
        if np.any(~np.isfinite(y_arr)):
            raise AIEngineException("RidgeBaseline.fit: y_train contains non-finite values.")
        self._model = Ridge(alpha=self._alpha)
        self._model.fit(X_arr, y_arr)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return raw Ridge predictions (unconstrained)."""
        if self._model is None:
            raise AIEngineException("RidgeBaseline: call fit() before predict().")
        X_arr = _validate_regression_test_matrix(
            X_test, "RidgeBaseline", int(self._model.n_features_in_)
        )
        preds = self._model.predict(X_arr)
        return np.asarray(preds, dtype=np.float64).ravel()
