"""Unit & anti-leakage tests for ``app.ai_engine.splitters``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ai_engine import (
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
