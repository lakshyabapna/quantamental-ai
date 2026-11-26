"""Quantamental AI - Walk-Forward Chronological Splitters (Module B).

Provides leak-free expanding-window and rolling-window chronological cross-validation
splitters with mandatory horizon-aware purge/embargo gaps, non-overlapping OOS test
windows (``step_size >= test_size``), and train-only feature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)


SplitMode = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class WalkForwardSplit:
    """Single chronological walk-forward split with explicit index boundaries."""

    fold_id: int
    train_indices: np.ndarray
    val_indices: Optional[np.ndarray]
    test_indices: np.ndarray
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    val_start_date: Optional[pd.Timestamp]
    val_end_date: Optional[pd.Timestamp]
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    purge_gap: int
    horizon_days: int = 21


@dataclass(frozen=True)
class ScaledFoldData:
    """Train/val/test arrays scaled strictly using statistics fitted on training data only."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.Series]
    X_test: pd.DataFrame
    y_test: pd.Series
    scaler: StandardScaler


class WalkForwardSplitter:
    """Chronological expanding or rolling walk-forward splitter with purge-gap enforcement.

    For a forward target spanning ``horizon_days`` trading days ``[t, t + horizon_days - 1]``,
    the purge gap between the end of training (or validation) and the start of the next fold
    must be ``>= horizon_days`` so no target window overlaps across splits.

    Furthermore, ``step_size >= test_size`` is enforced so consecutive out-of-sample
    test windows never overlap or duplicate timestamps.
    """

    def __init__(
        self,
        *,
        mode: SplitMode = "expanding",
        train_size: int,
        test_size: int,
        val_size: int = 0,
        step_size: Optional[int] = None,
        horizon_days: int = 21,
        purge_gap: Optional[int] = None,
    ) -> None:
        if mode not in ("expanding", "rolling"):
            raise AIEngineException(f"Unsupported split mode: {mode!r}.")
        if (
            isinstance(train_size, bool)
            or isinstance(test_size, bool)
            or isinstance(val_size, bool)
            or not isinstance(train_size, int)
            or not isinstance(test_size, int)
            or not isinstance(val_size, int)
            or train_size < 1
            or test_size < 1
            or val_size < 0
        ):
            raise AIEngineException(
                "train_size and test_size must be integers >= 1; val_size must be an integer >= 0."
            )
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
            raise AIEngineException("horizon_days must be an integer >= 1.")

        resolved_purge = horizon_days if purge_gap is None else purge_gap
        if isinstance(resolved_purge, bool) or not isinstance(resolved_purge, int):
            raise AIEngineException("purge_gap must be an integer.")
        if resolved_purge < horizon_days:
            raise DataLeakageError(
                f"purge_gap ({resolved_purge}) must be >= horizon_days ({horizon_days}) "
                "to prevent forward-target window overlap between splits.",
                details={"purge_gap": resolved_purge, "horizon_days": horizon_days},
            )

        resolved_step = test_size if step_size is None else step_size
        if isinstance(resolved_step, bool) or not isinstance(resolved_step, int) or resolved_step < 1:
            raise AIEngineException("step_size must be an integer >= 1.")
        if resolved_step < test_size:
            raise AIEngineException(
                f"step_size ({resolved_step}) must be >= test_size ({test_size}) "
                "to prevent overlapping out-of-sample test windows.",
                details={"step_size": resolved_step, "test_size": test_size},
            )

        self.mode: SplitMode = mode
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = resolved_step
        self.horizon_days = horizon_days
        self.purge_gap = resolved_purge

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Iterator[WalkForwardSplit]:
        """Yield strictly chronological ``WalkForwardSplit`` folds."""
        if not isinstance(X.index, pd.DatetimeIndex):
            raise AIEngineException("X must be indexed by a pd.DatetimeIndex.")
        if X.index.hasnans:
            raise AIEngineException("X index contains NaT timestamps.")
        if not X.index.is_monotonic_increasing:
            raise DataLeakageError("X index must be strictly sorted chronologically.")
        if X.index.has_duplicates:
            raise DataLeakageError("X index must not contain duplicate timestamps.")
        if y is not None and not X.index.equals(y.index):
            raise DataLeakageError("X and y indices must match exactly.")

        n_samples = len(X)
        val_block = (self.purge_gap + self.val_size) if self.val_size > 0 else 0
        min_required = self.train_size + val_block + self.purge_gap + self.test_size
        if n_samples < min_required:
            raise InsufficientDataError(
                f"Insufficient samples ({n_samples}) for walk-forward split; "
                f"need at least {min_required}.",
                details={"n_samples": n_samples, "min_required": min_required},
            )

        fold_id = 0
        train_end = self.train_size  # exclusive upper bound

        while True:
            train_start = 0 if self.mode == "expanding" else train_end - self.train_size

            if self.val_size > 0:
                val_start = train_end + self.purge_gap
                val_end = val_start + self.val_size
                test_start = val_end + self.purge_gap
            else:
                val_start = None
                val_end = None
                test_start = train_end + self.purge_gap

            test_end = test_start + self.test_size
            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end, dtype=np.int64)
            val_idx = (
                np.arange(val_start, val_end, dtype=np.int64)
                if val_start is not None and val_end is not None
                else None
            )
            test_idx = np.arange(test_start, test_end, dtype=np.int64)

            yield WalkForwardSplit(
                fold_id=fold_id,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                train_start_date=pd.Timestamp(X.index[train_idx[0]]),
                train_end_date=pd.Timestamp(X.index[train_idx[-1]]),
                val_start_date=(
                    pd.Timestamp(X.index[val_idx[0]]) if val_idx is not None else None
                ),
                val_end_date=(
                    pd.Timestamp(X.index[val_idx[-1]]) if val_idx is not None else None
                ),
                test_start_date=pd.Timestamp(X.index[test_idx[0]]),
                test_end_date=pd.Timestamp(X.index[test_idx[-1]]),
                purge_gap=self.purge_gap,
                horizon_days=self.horizon_days,
            )

            fold_id += 1
            train_end += self.step_size


def _validate_split_indices(
    indices: np.ndarray,
    n_samples: int,
    name: str,
) -> np.ndarray:
    """Validate that split indices are 1-D, non-empty, integer, unique, strictly increasing, and in-bounds."""
    arr = np.asarray(indices)
    if arr.ndim != 1 or len(arr) == 0:
        raise AIEngineException(
            f"scale_fold_features: {name} must be a non-empty 1-D array, got shape {arr.shape}."
        )
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise AIEngineException(
            f"scale_fold_features: {name} must have integer dtype, got {arr.dtype}."
        )
    int_arr = arr.astype(np.int64)
    if np.any(int_arr < 0) or np.any(int_arr >= n_samples):
        raise AIEngineException(
            f"scale_fold_features: {name} contains out-of-bounds index for n_samples={n_samples}."
        )
    if len(int_arr) > 1 and not np.all(np.diff(int_arr) > 0):
        raise DataLeakageError(
            f"scale_fold_features: {name} must be unique and strictly increasing."
        )
    return int_arr


def scale_fold_features(
    X: pd.DataFrame,
    y: pd.Series,
    split: WalkForwardSplit,
    *,
    horizon_days: Optional[int] = None,
) -> ScaledFoldData:
    """Slice a walk-forward fold and fit ``StandardScaler`` strictly on training rows.

    Validates X/y index alignment, finiteness, split index integrity (integer, non-empty,
    unique, strictly increasing, in-bounds), horizon-aware purge gap (``purge_gap >=
    horizon_days``), date metadata consistency, and chronological ordering before fitting.
    """
    if len(X) == 0:
        raise AIEngineException("scale_fold_features: X must not be empty.")
    if not isinstance(X.index, pd.DatetimeIndex):
        raise AIEngineException("scale_fold_features: X must be indexed by a pd.DatetimeIndex.")
    if X.index.hasnans:
        raise AIEngineException("scale_fold_features: X index contains NaT timestamps.")
    if not X.index.is_monotonic_increasing or X.index.has_duplicates:
        raise DataLeakageError(
            "scale_fold_features: X index must be strictly increasing without duplicates."
        )
    if not X.index.equals(y.index):
        raise DataLeakageError("scale_fold_features: X and y indices must match exactly.")

    X_vals = np.asarray(X, dtype=np.float64)
    y_vals = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(X_vals)):
        raise AIEngineException("scale_fold_features: X contains NaN or Inf values.")
    if not np.all(np.isfinite(y_vals)):
        raise AIEngineException("scale_fold_features: y contains NaN or Inf values.")

    if isinstance(split.horizon_days, bool) or not isinstance(split.horizon_days, int) or split.horizon_days < 1:
        raise AIEngineException(
            f"scale_fold_features: split.horizon_days must be an integer >= 1, got {split.horizon_days!r}."
        )
    if horizon_days is not None:
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
            raise AIEngineException(
                f"scale_fold_features: horizon_days must be an integer >= 1, got {horizon_days!r}."
            )
        if horizon_days != split.horizon_days:
            raise DataLeakageError(
                f"scale_fold_features: split.horizon_days ({split.horizon_days}) does not match "
                f"expected horizon_days ({horizon_days})."
            )
    expected_horizon = split.horizon_days if horizon_days is None else horizon_days

    if isinstance(split.purge_gap, bool) or not isinstance(split.purge_gap, int) or split.purge_gap < 1:
        raise AIEngineException(
            f"scale_fold_features: split.purge_gap must be an integer >= 1, got {split.purge_gap!r}."
        )
    if split.purge_gap < expected_horizon:
        raise DataLeakageError(
            f"scale_fold_features: split.purge_gap ({split.purge_gap}) must be >= "
            f"horizon_days ({expected_horizon}) to prevent target window overlap."
        )
    required_gap = max(split.purge_gap, expected_horizon)

    n_samples = len(X)
    train_idx = _validate_split_indices(split.train_indices, n_samples, "train_indices")
    test_idx = _validate_split_indices(split.test_indices, n_samples, "test_indices")

    val_idx: Optional[np.ndarray] = None
    if split.val_indices is not None:
        val_idx = _validate_split_indices(split.val_indices, n_samples, "val_indices")
        if train_idx[-1] >= val_idx[0]:
            raise DataLeakageError("Training indices must strictly precede validation indices.")
        if val_idx[-1] >= test_idx[0]:
            raise DataLeakageError("Validation indices must strictly precede test indices.")
        if int(val_idx[0] - train_idx[-1] - 1) < required_gap:
            raise DataLeakageError(
                "Train-to-validation index gap is smaller than required purge_gap."
            )
        if int(test_idx[0] - val_idx[-1] - 1) < required_gap:
            raise DataLeakageError(
                "Validation-to-test index gap is smaller than required purge_gap."
            )
    else:
        if train_idx[-1] >= test_idx[0]:
            raise DataLeakageError("Training indices must strictly precede test indices.")
        if int(test_idx[0] - train_idx[-1] - 1) < required_gap:
            raise DataLeakageError(
                "Train-to-test index gap is smaller than required purge_gap."
            )

    # Validate split date metadata consistency
    if pd.Timestamp(split.train_start_date) != pd.Timestamp(X.index[train_idx[0]]):
        raise DataLeakageError("split.train_start_date does not match first training timestamp.")
    if pd.Timestamp(split.train_end_date) != pd.Timestamp(X.index[train_idx[-1]]):
        raise DataLeakageError("split.train_end_date does not match last training timestamp.")
    if pd.Timestamp(split.test_start_date) != pd.Timestamp(X.index[test_idx[0]]):
        raise DataLeakageError("split.test_start_date does not match first test timestamp.")
    if pd.Timestamp(split.test_end_date) != pd.Timestamp(X.index[test_idx[-1]]):
        raise DataLeakageError("split.test_end_date does not match last test timestamp.")

    if val_idx is not None:
        if split.val_start_date is None or split.val_end_date is None:
            raise DataLeakageError(
                "split.val_start_date and split.val_end_date must be set when val_indices is provided."
            )
        if pd.Timestamp(split.val_start_date) != pd.Timestamp(X.index[val_idx[0]]):
            raise DataLeakageError("split.val_start_date does not match first validation timestamp.")
        if pd.Timestamp(split.val_end_date) != pd.Timestamp(X.index[val_idx[-1]]):
            raise DataLeakageError("split.val_end_date does not match last validation timestamp.")
    else:
        if split.val_start_date is not None or split.val_end_date is not None:
            raise DataLeakageError(
                "split.val_start_date and split.val_end_date must be None when val_indices is None."
            )

    train_list = train_idx.tolist()
    test_list = test_idx.tolist()
    X_train_raw = X.iloc[train_list]
    y_train = y.iloc[train_list].copy()
    X_test_raw = X.iloc[test_list]
    y_test = y.iloc[test_list].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(X_train_raw, dtype=np.float64))
    X_train = pd.DataFrame(train_scaled, index=X_train_raw.index, columns=X_train_raw.columns)

    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    if val_idx is not None:
        val_list = val_idx.tolist()
        X_val_raw = X.iloc[val_list]
        y_val = y.iloc[val_list].copy()
        val_scaled = scaler.transform(np.asarray(X_val_raw, dtype=np.float64))
        X_val = pd.DataFrame(val_scaled, index=X_val_raw.index, columns=X_val_raw.columns)

    test_scaled = scaler.transform(np.asarray(X_test_raw, dtype=np.float64))
    X_test = pd.DataFrame(test_scaled, index=X_test_raw.index, columns=X_test_raw.columns)

    return ScaledFoldData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )
