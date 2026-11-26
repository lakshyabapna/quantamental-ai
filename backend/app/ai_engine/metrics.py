"""Quantamental AI - Baseline Evaluation Metrics (Module B).

Provides strictly validated scalar evaluation metrics for volatility forecasting:
    - RMSE: Root Mean Squared Error (in annualized volatility units).
    - MAE: Mean Absolute Error (in annualized volatility units).
    - R²: Coefficient of Determination (on annualized volatility).
    - QLIKE: Normalized Patton (2011) Gaussian quasi-likelihood loss on variance.

Units & Variance Convention
---------------------------
All functions accept 1-D arrays of **annualized volatility** (``sigma``, decimal units,
e.g., ``0.20`` for 20% annualized volatility). Inside ``qlike()``, volatilities are
converted to **annualized variances** via ``var_true = y_true ** 2`` and
``var_pred = y_pred ** 2``, and evaluated using the normalized Patton (2011) QLIKE loss:

    L(var_true, var_pred) = var_true / var_pred - log(var_true / var_pred) - 1

Properties of normalized QLIKE:
    - Strictly non-negative (``L >= 0``), with ``L == 0`` iff ``var_true == var_pred``.
    - Scale-invariant: multiplying both ``y_true`` and ``y_pred`` by any ``c > 0``
      leaves QLIKE unchanged.
    - Requires strictly positive ``y_true > 0`` and ``y_pred > 0``. Invalid or
      non-positive inputs raise ``AIEngineException`` without silent clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from app.core.exceptions import AIEngineException


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    caller: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that both arrays are 1-D, same length, finite, and free of NaN."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 1:
        raise AIEngineException(
            f"{caller}: y_true must be 1-D, got shape {y_true.shape}.",
            details={"shape": y_true.shape},
        )
    if y_pred.ndim != 1:
        raise AIEngineException(
            f"{caller}: y_pred must be 1-D, got shape {y_pred.shape}.",
            details={"shape": y_pred.shape},
        )
    if len(y_true) != len(y_pred):
        raise AIEngineException(
            f"{caller}: y_true length {len(y_true)} != y_pred length {len(y_pred)}.",
            details={"len_y_true": len(y_true), "len_y_pred": len(y_pred)},
        )
    if len(y_true) == 0:
        raise AIEngineException(f"{caller}: input arrays must not be empty.")
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise AIEngineException(
            f"{caller}: y_true contains NaN or Inf values.",
            details={"n_nan": int(np.isnan(y_true).sum()), "n_inf": int(np.isinf(y_true).sum())},
        )
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise AIEngineException(
            f"{caller}: y_pred contains NaN or Inf values.",
            details={"n_nan": int(np.isnan(y_pred).sum()), "n_inf": int(np.isinf(y_pred).sum())},
        )
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in annualized volatility units.

    Parameters
    ----------
    y_true:
        Observed realized volatility values (1-D array, annualized).
    y_pred:
        Predicted volatility values (1-D array, annualized).

    Returns
    -------
    float
        Non-negative RMSE.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "rmse")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in annualized volatility units.

    Parameters
    ----------
    y_true:
        Observed realized volatility values (1-D array, annualized).
    y_pred:
        Predicted volatility values (1-D array, annualized).

    Returns
    -------
    float
        Non-negative MAE.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "mae")
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R².

    Parameters
    ----------
    y_true:
        Observed realized volatility values (1-D array, annualized).
    y_pred:
        Predicted volatility values (1-D array, annualized).

    Returns
    -------
    float
        R² value (1.0 is perfect; can be negative when worse than sample mean).

    Raises
    ------
    AIEngineException
        If inputs are invalid or ``y_true`` has zero variance.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "r_squared")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        raise AIEngineException(
            "r_squared: y_true has zero variance; R² is undefined.",
            details={"y_true_mean": float(np.mean(y_true))},
        )
    return 1.0 - ss_res / ss_tot


_QLIKE_TAYLOR_THRESHOLD: float = 1e-3


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Patton (2011) Gaussian Quasi-Likelihood (QLIKE) loss.

    Inputs ``y_true`` and ``y_pred`` are **annualized volatilities** (``sigma``).
    Evaluates the normalized Patton (2011) QLIKE loss in log-space without
    directly squaring volatilities:

        log_ratio = 2 * (log(y_true) - log(y_pred))
        QLIKE = mean(expm1(log_ratio) - log_ratio)

    For ``|log_ratio| < 1e-3``, ``expm1(x) - x`` suffers catastrophic cancellation
    (losing all float64 digits for ``|x| <= 1e-8``). In that region we evaluate
    the 6th-order Taylor expansion via Horner's method:

        x^2/2 + x^3/6 + x^4/24 + x^5/120 + x^6/720

    At ``|x| = 1e-3``, the relative truncation error ``|x|^5 / 2520 ≈ 4e-19``
    is strictly below IEEE-754 float64 machine epsilon (``2.22e-16``), while
    preserving strictly positive representable losses even for adjacent float64
    inputs.

    Both ``y_true`` and ``y_pred`` must be strictly positive (``> 0``) and finite.
    Values that would overflow ``float64`` in ``expm1(log_ratio)`` raise
    ``AIEngineException`` explicitly without silent clipping or capping.

    Parameters
    ----------
    y_true:
        Observed annualized realized volatility (1-D array, strictly positive).
    y_pred:
        Predicted annualized volatility (1-D array, strictly positive).

    Returns
    -------
    float
        Non-negative normalized QLIKE loss (0.0 when ``y_true == y_pred``).

    Raises
    ------
    AIEngineException
        If inputs contain NaN/Inf, have mismatched lengths, contain any value ``<= 0``,
        or produce ``float64`` overflow.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred, "qlike")

    if np.any(y_true <= 0.0):
        raise AIEngineException(
            "qlike: y_true must be strictly positive (> 0) for variance-ratio QLIKE.",
            details={"n_non_positive": int((y_true <= 0.0).sum())},
        )
    if np.any(y_pred <= 0.0):
        raise AIEngineException(
            "qlike: y_pred must be strictly positive (> 0); apply an explicit "
            "volatility floor policy before computing QLIKE.",
            details={
                "n_non_positive": int((y_pred <= 0.0).sum()),
                "min_y_pred": float(np.min(y_pred)),
            },
        )

    # Use log1p((y_true - y_pred) / y_pred) near equality so adjacent float64 values
    # do not lose precision when subtracting log(y_true) - log(y_pred).
    near_mask = np.abs(y_true - y_pred) <= 0.5 * np.minimum(y_true, y_pred)
    log_ratio = np.empty_like(y_true, dtype=np.float64)
    if np.any(near_mask):
        yt_n = y_true[near_mask]
        yp_n = y_pred[near_mask]
        log_ratio[near_mask] = 2.0 * np.log1p((yt_n - yp_n) / yp_n)
    far_mask = ~near_mask
    if np.any(far_mask):
        log_ratio[far_mask] = 2.0 * (np.log(y_true[far_mask]) - np.log(y_pred[far_mask]))
    max_log_float64 = float(np.log(np.finfo(np.float64).max))
    if np.any(log_ratio > max_log_float64):
        raise AIEngineException(
            "qlike: variance ratio overflows float64.",
            details={"max_log_ratio": float(np.max(log_ratio)), "limit": max_log_float64},
        )

    small_mask = np.abs(log_ratio) < _QLIKE_TAYLOR_THRESHOLD
    losses = np.empty_like(log_ratio, dtype=np.float64)

    if np.any(small_mask):
        x = log_ratio[small_mask]
        losses[small_mask] = (x * x) * (
            0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x * (1.0 / 720.0))))
        )

    large_mask = ~small_mask
    if np.any(large_mask):
        x_large = log_ratio[large_mask]
        losses[large_mask] = np.expm1(x_large) - x_large

    if not np.all(np.isfinite(losses)):
        raise AIEngineException("qlike: non-finite loss encountered during computation.")
    return float(np.mean(losses))


@dataclass(frozen=True)
class BaselineMetrics:
    """Frozen container for per-fold evaluation metrics of a single baseline model.

    Distinguishes post-processed (floored) forecast metrics from raw unclipped
    model metrics, and optionally retains fold OOS arrays for exact pooled OOS
    aggregation across folds.
    """

    model_name: str
    rmse: float
    mae_val: float
    r2: float
    qlike_val: float
    n_samples: int
    horizon_days: int
    fold_id: int
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    raw_rmse: float = float("nan")
    raw_mae: float = float("nan")
    raw_r2: float = float("nan")
    n_floored: int = 0
    floored_pct: float = 0.0
    oos_dates: Tuple[str, ...] = field(default=(), repr=False)
    oos_y_true: Tuple[float, ...] = field(default=(), repr=False)
    oos_y_pred_raw: Tuple[float, ...] = field(default=(), repr=False)
    oos_y_pred_processed: Tuple[float, ...] = field(default=(), repr=False)
