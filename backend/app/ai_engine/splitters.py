"""Quantamental AI - Walk-Forward Chronological Splitters (Module B).

Provides leak-free expanding-window and rolling-window chronological cross-validation
splitters with mandatory horizon-aware purge/embargo gaps and train-only feature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Optional, Tuple

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
        if train_size < 1 or test_size < 1 or val_size < 0:
            raise AIEngineException(
                "train_size and test_size must be >= 1; val_size must be >= 0."
            )
        if horizon_days < 1:
            raise AIEngineException("horizon_days must be >= 1.")

        resolved_purge = horizon_days if purge_gap is None else purge_gap
        if resolved_purge < horizon_days:
            raise DataLeakageError(
                f"purge_gap ({resolved_purge}) must be >= horizon_days ({horizon_days}) "
                "to prevent forward-target window overlap between splits.",
                details={"purge_gap": resolved_purge, "horizon_days": horizon_days},
            )

        resolved_step = test_size if step_size is None else step_size
        if resolved_step < 1:
            raise AIEngineException("step_size must be >= 1.")

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
            )

            fold_id += 1
            train_end += self.step_size


def scale_fold_features(
    X: pd.DataFrame,
    y: pd.Series,
    split: WalkForwardSplit,
) -> ScaledFoldData:
    """Slice a walk-forward fold and fit ``StandardScaler`` strictly on training rows.

    Verifies chronological ordering and non-overlap before fitting.
    """
    if split.train_indices[-1] >= split.test_indices[0]:
        raise DataLeakageError("Training indices must strictly precede test indices.")
    if split.val_indices is not None:
        if split.train_indices[-1] >= split.val_indices[0]:
            raise DataLeakageError("Training indices must strictly precede validation indices.")
        if split.val_indices[-1] >= split.test_indices[0]:
            raise DataLeakageError("Validation indices must strictly precede test indices.")

    train_list = split.train_indices.tolist()
    test_list = split.test_indices.tolist()
    X_train_raw = X.iloc[train_list]
    y_train = y.iloc[train_list].copy()
    X_test_raw = X.iloc[test_list]
    y_test = y.iloc[test_list].copy()

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(np.asarray(X_train_raw, dtype=np.float64))
    X_train = pd.DataFrame(train_scaled, index=X_train_raw.index, columns=X_train_raw.columns)

    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    if split.val_indices is not None:
        val_list = split.val_indices.tolist()
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
