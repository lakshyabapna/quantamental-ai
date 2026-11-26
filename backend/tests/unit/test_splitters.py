"""Unit & anti-leakage tests for ``app.ai_engine.splitters``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
    WalkForwardSplit,
    WalkForwardSplitter,
    build_volatility_features,
    scale_fold_features,
)
from app.core.exceptions import (
    AIEngineException,
    DataLeakageError,
    InsufficientDataError,
)


def _build_sample_xy(n_days: int = 320, horizon_days: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rets = rng.normal(0.0003, 0.012, size=n_days)
    prices = pd.Series(
        (100.0 * np.exp(np.cumsum(rets))).tolist(),
        index=dates,
        name="ASSET",
        dtype=float,
    )
    ds = build_volatility_features(prices, horizon_days=horizon_days)
    return ds.X, ds.y


class TestWalkForwardSplitters:
    """Verify chronological walk-forward splits, purge gaps, and train-only scaling."""

    def test_expanding_and_rolling_splits_chronological_and_purged(self) -> None:
        """Verify expanding/rolling boundaries and purge_gap >= horizon_days."""
        horizon = 21
        X, y = _build_sample_xy(n_days=340, horizon_days=horizon)

        exp_splitter = WalkForwardSplitter(
            mode="expanding",
            train_size=120,
            val_size=20,
            test_size=30,
            horizon_days=horizon,
        )
        exp_folds = list(exp_splitter.split(X, y))
        assert len(exp_folds) >= 1

        for fold in exp_folds:
            assert fold.purge_gap >= horizon
            assert fold.train_indices[0] == 0
            assert fold.val_indices is not None
            # Exact index gap >= horizon_days between train/val and val/test
            assert fold.val_indices[0] - fold.train_indices[-1] - 1 >= horizon
            assert fold.test_indices[0] - fold.val_indices[-1] - 1 >= horizon
            assert fold.val_start_date is not None
            assert fold.val_end_date is not None
            assert fold.train_end_date < fold.val_start_date
            assert fold.val_end_date < fold.test_start_date

        rol_splitter = WalkForwardSplitter(
            mode="rolling",
            train_size=120,
            test_size=30,
            horizon_days=horizon,
        )
        rol_folds = list(rol_splitter.split(X, y))
        assert len(rol_folds) >= 2
        for fold in rol_folds:
            assert len(fold.train_indices) == 120
            assert fold.test_indices[0] - fold.train_indices[-1] - 1 >= horizon
        assert rol_folds[1].train_indices[0] > rol_folds[0].train_indices[0]

    def test_purge_gap_less_than_horizon_raises_data_leakage_error(self) -> None:
        """Attempt to set purge_gap < horizon_days must raise DataLeakageError."""
        with pytest.raises(DataLeakageError):
            WalkForwardSplitter(
                mode="expanding",
                train_size=100,
                test_size=20,
                horizon_days=21,
                purge_gap=10,
            )

    def test_scaler_fits_exclusively_on_training_fold(self) -> None:
        """Perturbing validation/test data must not affect scaler stats or scaled X_train."""
        X, y = _build_sample_xy(n_days=320, horizon_days=21)
        splitter = WalkForwardSplitter(
            mode="expanding",
            train_size=120,
            val_size=20,
            test_size=30,
            horizon_days=21,
        )
        split = next(splitter.split(X, y))

        scaled_orig = scale_fold_features(X, y, split)

        # Training fold mean/std must match X_train raw statistics exclusively
        train_raw = np.asarray(X.iloc[split.train_indices.tolist()], dtype=np.float64)
        assert scaled_orig.scaler.mean_ is not None
        assert scaled_orig.scaler.scale_ is not None
        np.testing.assert_allclose(
            np.asarray(scaled_orig.scaler.mean_, dtype=np.float64),
            train_raw.mean(axis=0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(scaled_orig.scaler.scale_, dtype=np.float64),
            train_raw.std(axis=0, ddof=0),
            atol=1e-12,
        )

        # Corrupt test fold with extreme values and verify scaler & X_train are unchanged
        X_perturbed = X.copy()
        X_perturbed.iloc[split.test_indices.tolist()] = 9999.0
        scaled_pert = scale_fold_features(X_perturbed, y, split)

        np.testing.assert_array_equal(scaled_orig.scaler.mean_, scaled_pert.scaler.mean_)
        np.testing.assert_array_equal(scaled_orig.scaler.scale_, scaled_pert.scaler.scale_)
        np.testing.assert_array_equal(
            np.asarray(scaled_orig.X_train, dtype=np.float64),
            np.asarray(scaled_pert.X_train, dtype=np.float64),
        )

    def test_invalid_split_inputs_rejected(self) -> None:
        """Reject insufficient samples, mismatched indices, or invalid mode."""
        X, y = _build_sample_xy(n_days=200, horizon_days=21)

        with pytest.raises(AIEngineException):
            WalkForwardSplitter(mode="random", train_size=50, test_size=20)  # type: ignore[arg-type]

        splitter = WalkForwardSplitter(train_size=500, test_size=50, horizon_days=21)
        with pytest.raises(InsufficientDataError):
            list(splitter.split(X, y))

        with pytest.raises(DataLeakageError):
            list(WalkForwardSplitter(train_size=50, test_size=20, horizon_days=21).split(X.iloc[::-1]))

    def test_step_size_prevents_overlapping_test_windows(self) -> None:
        """step_size < test_size must raise; step_size >= test_size produces disjoint test windows."""
        X, y = _build_sample_xy(n_days=360, horizon_days=21)

        with pytest.raises(AIEngineException, match="step_size"):
            WalkForwardSplitter(
                mode="expanding",
                train_size=100,
                test_size=30,
                step_size=15,
                horizon_days=21,
            )

        for step in (30, 45):
            splitter = WalkForwardSplitter(
                mode="expanding",
                train_size=100,
                test_size=30,
                step_size=step,
                horizon_days=21,
            )
            seen_test_dates: list[pd.Timestamp] = []
            folds = list(splitter.split(X, y))
            assert len(folds) >= 2
            for f in folds:
                fold_dates = [pd.Timestamp(X.index[idx]) for idx in f.test_indices.tolist()]
                assert set(seen_test_dates).isdisjoint(set(fold_dates))
                seen_test_dates.extend(fold_dates)
            assert len(seen_test_dates) == len(set(seen_test_dates))

    def test_scale_fold_features_rejects_invalid_manual_splits(self) -> None:
        """Manually constructed invalid WalkForwardSplit or X/y inputs must be rejected."""
        X, y = _build_sample_xy(n_days=260, horizon_days=21)
        t0 = pd.Timestamp(X.index[0])

        def _make_split(
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            val_idx: np.ndarray | None = None,
            purge_gap: int = 21,
        ) -> WalkForwardSplit:
            return WalkForwardSplit(
                fold_id=0,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                train_start_date=t0,
                train_end_date=t0,
                val_start_date=t0 if val_idx is not None else None,
                val_end_date=t0 if val_idx is not None else None,
                test_start_date=t0,
                test_end_date=t0,
                purge_gap=purge_gap,
            )

        # Mismatched X/y index
        with pytest.raises(DataLeakageError):
            scale_fold_features(
                X,
                y.iloc[1:],
                _make_split(np.arange(0, 50), np.arange(71, 90)),
            )

        # Empty split indices
        with pytest.raises(AIEngineException):
            scale_fold_features(X, y, _make_split(np.array([], dtype=np.int64), np.arange(71, 90)))

        # Non-integer indices (float)
        with pytest.raises(AIEngineException):
            scale_fold_features(X, y, _make_split(np.array([0.0, 1.0]), np.arange(71, 90)))

        # Duplicate / non-increasing indices
        with pytest.raises(DataLeakageError):
            scale_fold_features(
                X, y, _make_split(np.array([0, 2, 2, 3], dtype=np.int64), np.arange(71, 90))
            )

        # Out-of-bounds index
        with pytest.raises(AIEngineException):
            scale_fold_features(
                X, y, _make_split(np.arange(0, 50), np.array([len(X) + 5], dtype=np.int64))
            )

        # Purge gap violation
        with pytest.raises(DataLeakageError):
            scale_fold_features(
                X, y, _make_split(np.arange(0, 50), np.arange(55, 75), purge_gap=21)
            )

        # Non-finite X or y
        X_nan = X.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(AIEngineException):
            scale_fold_features(X_nan, y, _make_split(np.arange(0, 50), np.arange(71, 90)))

        y_inf = y.copy()
        y_inf.iloc[0] = np.inf
        with pytest.raises(AIEngineException):
            scale_fold_features(X, y_inf, _make_split(np.arange(0, 50), np.arange(71, 90)))

    def test_manual_split_enforces_horizon_aware_purge_and_date_metadata(self) -> None:
        """Manual WalkForwardSplit must enforce purge_gap >= horizon_days and exact date metadata."""
        X, y = _build_sample_xy(n_days=260, horizon_days=21)

        def _valid_manual_split(
            train_idx: np.ndarray,
            test_idx: np.ndarray,
            *,
            val_idx: np.ndarray | None = None,
            purge_gap: int = 21,
            horizon_days: int = 21,
        ) -> WalkForwardSplit:
            return WalkForwardSplit(
                fold_id=0,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                train_start_date=pd.Timestamp(X.index[train_idx[0]]),
                train_end_date=pd.Timestamp(X.index[train_idx[-1]]),
                val_start_date=pd.Timestamp(X.index[val_idx[0]]) if val_idx is not None else None,
                val_end_date=pd.Timestamp(X.index[val_idx[-1]]) if val_idx is not None else None,
                test_start_date=pd.Timestamp(X.index[test_idx[0]]),
                test_end_date=pd.Timestamp(X.index[test_idx[-1]]),
                purge_gap=purge_gap,
                horizon_days=horizon_days,
            )

        # Positive purge_gap (10) < horizon_days (21) must fail
        bad_purge = _valid_manual_split(
            np.arange(0, 50), np.arange(71, 90), purge_gap=10, horizon_days=21
        )
        with pytest.raises(DataLeakageError, match="purge_gap"):
            scale_fold_features(X, y, bad_purge)

        # Even if split.horizon_days=10, passing expected horizon_days=21 must fail
        mismatch_h = _valid_manual_split(
            np.arange(0, 50), np.arange(71, 90), purge_gap=10, horizon_days=10
        )
        with pytest.raises(DataLeakageError, match="horizon_days"):
            scale_fold_features(X, y, mismatch_h, horizon_days=21)

        # Actual excluded observations (15) < required purge_gap (21) must fail
        insufficient_gap = _valid_manual_split(
            np.arange(0, 50), np.arange(65, 85), purge_gap=21, horizon_days=21
        )
        with pytest.raises(DataLeakageError, match="index gap"):
            scale_fold_features(X, y, insufficient_gap)

        # Forged train_start_date / test_end_date must fail
        good_split = _valid_manual_split(np.arange(0, 50), np.arange(71, 90))
        forged_train = WalkForwardSplit(
            fold_id=0,
            train_indices=good_split.train_indices,
            val_indices=None,
            test_indices=good_split.test_indices,
            train_start_date=pd.Timestamp(X.index[1]),
            train_end_date=good_split.train_end_date,
            val_start_date=None,
            val_end_date=None,
            test_start_date=good_split.test_start_date,
            test_end_date=good_split.test_end_date,
            purge_gap=21,
            horizon_days=21,
        )
        with pytest.raises(DataLeakageError, match="train_start_date"):
            scale_fold_features(X, y, forged_train)

        # Inconsistent val_start_date when val_indices is None must fail
        forged_val_meta = WalkForwardSplit(
            fold_id=0,
            train_indices=good_split.train_indices,
            val_indices=None,
            test_indices=good_split.test_indices,
            train_start_date=good_split.train_start_date,
            train_end_date=good_split.train_end_date,
            val_start_date=pd.Timestamp(X.index[55]),
            val_end_date=None,
            test_start_date=good_split.test_start_date,
            test_end_date=good_split.test_end_date,
            purge_gap=21,
            horizon_days=21,
        )
        with pytest.raises(DataLeakageError, match="val_start_date"):
            scale_fold_features(X, y, forged_val_meta)

        # Valid manual split with matching dates and purge_gap >= horizon_days succeeds
        scaled = scale_fold_features(X, y, good_split, horizon_days=21)
        assert len(scaled.X_train) == 50
        assert len(scaled.X_test) == 19
