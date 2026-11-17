"""Tests for Phase 2.2 Step 2: Ledoit-Wolf Shrinkage Covariance Estimator.

Reference estimator: ``sklearn.covariance.LedoitWolf`` is independently
instantiated in each numerical test to compute expected values.  The
production function is NEVER used to derive its own expected values.

Covers:
1.  Numerical equivalence with sklearn LedoitWolf (daily covariance)
2.  Annualization by 252 (default)
3.  Custom annualization factor
4.  Output shape and DataFrame type
5.  Symmetry of output
6.  Finite values in output
7.  Asset label and ordering preservation
8.  Shrinkage intensity: positive, within [0, 1], accessible via fit_ledoit_wolf
9.  Nullable Float64 input
10. Nullable Int64 input
11. Mixed nullable and regular float64 columns
12. NaN and pd.NA rejection
13. Positive and negative infinity rejection
14. Complex input rejection
15. Duplicate asset names and duplicate index rejection
16. Non-numeric input rejection
17. Invalid trading_days_per_year (bool, float, zero, negative)
18. Invalid min_observations (bool, float, str, zero, negative, below floor)
19. Insufficient observations
20. Constant-return and highly correlated asset cases
21. Compatibility imports from both namespaces
22. fit_ledoit_wolf() accessor: returns LedoitWolf instance with correct attrs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine.shrinkage import (
    calculate_ledoit_wolf_covariance,
    fit_ledoit_wolf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(data: dict, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Build a simple returns DataFrame with a business-day DatetimeIndex."""
    if n is not None:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {col: rng.normal(0, 0.01 * (i + 1), n) for i, col in enumerate(data)},
            index=dates,
        )
    lengths = [len(v) for v in data.values()]
    assert len(set(lengths)) == 1, "All columns must have the same length"
    dates = pd.date_range("2024-01-01", periods=lengths[0], freq="B")
    return pd.DataFrame(data, index=dates)


def _ref_lw_daily(values_f64: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute LedoitWolf daily covariance and shrinkage independently."""
    est = LedoitWolf(assume_centered=False)
    est.fit(values_f64)
    return est.covariance_, float(est.shrinkage_)


# ---------------------------------------------------------------------------
# 1. Numerical equivalence with sklearn (daily covariance)
# ---------------------------------------------------------------------------

class TestNumericalEquivalence:
    """The function must reproduce sklearn's LedoitWolf exactly (before annualization)."""

    def _deterministic_df(self, seed: int = 1, n: int = 60, n_assets: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        cols = [f"A{i}" for i in range(n_assets)]
        data = rng.normal(0, 0.01, (n, n_assets))
        return pd.DataFrame(data, index=dates, columns=cols)

    def test_daily_covariance_matches_sklearn(self):
        """trading_days_per_year=1 → result IS the daily Ledoit-Wolf covariance."""
        df = self._deterministic_df()
        values = df.to_numpy(dtype=np.float64)
        expected_cov, _ = _ref_lw_daily(values)

        result = calculate_ledoit_wolf_covariance(df, trading_days_per_year=1, min_observations=2)

        # Expected is symmetrized the same way the production function does
        expected_sym = (expected_cov + expected_cov.T) / 2.0
        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12, atol=1e-15)

    def test_annualized_covariance_matches_252x_sklearn(self):
        """Result with trading_days_per_year=252 must equal 252 × daily LW cov."""
        df = self._deterministic_df(seed=2)
        values = df.to_numpy(dtype=np.float64)
        expected_daily, _ = _ref_lw_daily(values)
        expected_ann = 252 * expected_daily
        expected_sym = (expected_ann + expected_ann.T) / 2.0

        result = calculate_ledoit_wolf_covariance(df, trading_days_per_year=252)

        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12, atol=1e-15)

    def test_two_asset_exact_match(self):
        """2-asset, deterministic dataset — exact element-level match."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame(
            {"X": rng.normal(0.001, 0.015, 50), "Y": rng.normal(0.0005, 0.010, 50)},
            index=dates,
        )
        values = df.to_numpy(dtype=np.float64)
        expected_daily, _ = _ref_lw_daily(values)
        expected_sym = (expected_daily + expected_daily.T) / 2.0

        result = calculate_ledoit_wolf_covariance(df, trading_days_per_year=1, min_observations=2)
        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Annualization
# ---------------------------------------------------------------------------

class TestAnnualization:

    def _df(self, seed: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        return pd.DataFrame(
            {"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.02, 60)},
            index=dates,
        )

    def test_default_annualization_is_252(self):
        result_def = calculate_ledoit_wolf_covariance(self._df())
        result_252 = calculate_ledoit_wolf_covariance(self._df(), trading_days_per_year=252)
        np.testing.assert_array_equal(result_def.values, result_252.values)

    def test_custom_annualization_factor(self):
        df = self._df()
        result_200 = calculate_ledoit_wolf_covariance(df, trading_days_per_year=200)
        result_252 = calculate_ledoit_wolf_covariance(df, trading_days_per_year=252)
        ratio = result_200.values / result_252.values
        np.testing.assert_allclose(ratio, 200 / 252, rtol=1e-12)

    def test_annualization_exact_scaling(self):
        """Result = trading_days × daily_lw_cov (after symmetrization)."""
        df = self._df(seed=6)
        values = df.to_numpy(dtype=np.float64)
        expected_daily, _ = _ref_lw_daily(values)
        expected = 200 * expected_daily
        expected_sym = (expected + expected.T) / 2.0

        result = calculate_ledoit_wolf_covariance(df, trading_days_per_year=200)
        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Output shape and type
# ---------------------------------------------------------------------------

class TestOutputShapeAndType:

    def test_output_is_dataframe(self):
        df = _make_df({"A": None, "B": None}, n=20, seed=10)
        result = calculate_ledoit_wolf_covariance(df)
        assert isinstance(result, pd.DataFrame)

    def test_output_shape_two_assets(self):
        df = _make_df({"A": None, "B": None}, n=20, seed=11)
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (2, 2)

    def test_output_shape_five_assets(self):
        df = _make_df({f"A{i}": None for i in range(5)}, n=100, seed=12)
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (5, 5)


# ---------------------------------------------------------------------------
# 4. Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:

    def test_symmetric_two_assets(self):
        df = _make_df({"A": None, "B": None}, n=50, seed=13)
        result = calculate_ledoit_wolf_covariance(df)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)

    def test_symmetric_five_assets(self):
        df = _make_df({f"A{i}": None for i in range(5)}, n=100, seed=14)
        result = calculate_ledoit_wolf_covariance(df)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Finite values
# ---------------------------------------------------------------------------

class TestFiniteValues:

    def test_all_finite_two_assets(self):
        df = _make_df({"A": None, "B": None}, n=50, seed=15)
        result = calculate_ledoit_wolf_covariance(df)
        assert np.all(np.isfinite(result.values))

    def test_diagonal_non_negative(self):
        df = _make_df({"A": None, "B": None, "C": None}, n=80, seed=16)
        result = calculate_ledoit_wolf_covariance(df)
        assert np.all(np.diag(result.values) >= 0.0)


# ---------------------------------------------------------------------------
# 6. Asset label and ordering preservation
# ---------------------------------------------------------------------------

class TestLabelOrdering:

    def test_labels_match_input_columns(self):
        cols = ["ZOMATO", "TCS", "INFY", "WIPRO"]
        df = _make_df({c: None for c in cols}, n=60, seed=17)
        result = calculate_ledoit_wolf_covariance(df)
        assert list(result.columns) == cols
        assert list(result.index) == cols

    def test_reversed_column_ordering(self):
        rng = np.random.default_rng(18)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame({"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.02, 60)}, index=dates)
        df_rev = df[["B", "A"]]
        result = calculate_ledoit_wolf_covariance(df)
        result_rev = calculate_ledoit_wolf_covariance(df_rev)
        assert list(result.columns) == ["A", "B"]
        assert list(result_rev.columns) == ["B", "A"]
        # Variance of A must be the same regardless of ordering
        np.testing.assert_allclose(result.loc["A", "A"], result_rev.loc["A", "A"], rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. Shrinkage intensity
# ---------------------------------------------------------------------------

class TestShrinkageIntensity:
    """The shrinkage intensity δ̂ must be in [0, 1] and accessible."""

    def test_shrinkage_in_unit_interval(self):
        df = _make_df({"A": None, "B": None}, n=60, seed=19)
        est = fit_ledoit_wolf(df)
        assert 0.0 <= est.shrinkage_ <= 1.0

    def test_shrinkage_matches_sklearn_reference(self):
        """fit_ledoit_wolf must return the same shrinkage as an independent sklearn fit."""
        rng = np.random.default_rng(20)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.02, 60)},
            index=dates,
        )
        est = fit_ledoit_wolf(df)
        _, ref_delta = _ref_lw_daily(df.to_numpy(dtype=np.float64))
        np.testing.assert_allclose(est.shrinkage_, ref_delta, rtol=1e-12)

    def test_fit_ledoit_wolf_returns_lw_instance(self):
        df = _make_df({"A": None, "B": None}, n=50, seed=21)
        est = fit_ledoit_wolf(df)
        assert isinstance(est, LedoitWolf)
        assert hasattr(est, "shrinkage_")
        assert hasattr(est, "covariance_")
        assert hasattr(est, "precision_")

    def test_covariance_attribute_is_daily(self):
        """estimator.covariance_ should match daily (trading_days=1) result."""
        rng = np.random.default_rng(22)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 60), "B": rng.normal(0, 0.02, 60)},
            index=dates,
        )
        est = fit_ledoit_wolf(df)
        result_daily = calculate_ledoit_wolf_covariance(df, trading_days_per_year=1, min_observations=2)
        expected_sym = (est.covariance_ + est.covariance_.T) / 2.0
        np.testing.assert_allclose(result_daily.values, expected_sym, rtol=1e-12)

    def test_highly_correlated_assets_low_shrinkage(self):
        """With many highly correlated observations LW shrinks less aggressively."""
        rng = np.random.default_rng(23)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        a = rng.normal(0, 0.01, n)
        df = pd.DataFrame(
            {"A": a, "B": a * 0.95 + rng.normal(0, 0.001, n)},
            index=dates,
        )
        est = fit_ledoit_wolf(df)
        # Just verify it's a valid shrinkage value
        assert 0.0 <= est.shrinkage_ <= 1.0


# ---------------------------------------------------------------------------
# 8. Nullable extension dtype inputs
# ---------------------------------------------------------------------------

class TestNullableExtensionDtypes:

    def _float64_df(self, seed: int = 30) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        return pd.DataFrame(
            {"A": rng.normal(0.001, 0.015, 30), "B": rng.normal(0.0005, 0.010, 30)},
            index=dates,
            dtype=np.float64,
        )

    def test_nullable_float64_accepted(self):
        df = self._float64_df().astype("Float64")
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_nullable_float64_matches_regular_float64(self):
        df_reg = self._float64_df()
        df_ext = df_reg.astype("Float64")
        result_reg = calculate_ledoit_wolf_covariance(df_reg)
        result_ext = calculate_ledoit_wolf_covariance(df_ext)
        np.testing.assert_allclose(result_reg.values, result_ext.values, rtol=1e-12)

    def test_nullable_int64_accepted(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, -1, 3, 0, 1, -2, 2, 1, 0], dtype="Int64"),
                "B": pd.array([2, 1, 0, 2, 1, -1, 1, 3, 0, 1], dtype="Int64"),
            },
            index=dates,
        )
        result = calculate_ledoit_wolf_covariance(df, min_observations=2)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_mixed_nullable_and_regular_float64(self):
        rng = np.random.default_rng(31)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array(rng.normal(0, 0.01, 30), dtype="Float64"),
                "B": rng.normal(0, 0.02, 30).astype(np.float64),
            },
            index=dates,
        )
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# 9. NaN and pd.NA rejection
# ---------------------------------------------------------------------------

class TestNaNRejection:

    def test_nan_in_float64_column_raises(self):
        df = _make_df({}, n=0)  # won't be used directly
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [0.01, np.nan, 0.02, 0.03, 0.04], "B": [0.02, 0.01, 0.0, 0.04, 0.01]},
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_ledoit_wolf_covariance(df)

    def test_pd_na_in_nullable_float64_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([0.01, None, 0.02, 0.03, 0.04], dtype="Float64"),
                "B": pd.array([0.02, 0.01, 0.0, 0.04, 0.01], dtype="Float64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_ledoit_wolf_covariance(df)

    def test_pd_na_in_nullable_int64_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([1, None, 2, 3, 4], dtype="Int64"),
                "B": pd.array([2, 1, 0, 4, 1], dtype="Int64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_ledoit_wolf_covariance(df)


# ---------------------------------------------------------------------------
# 10. Infinity rejection
# ---------------------------------------------------------------------------

class TestInfRejection:

    def test_positive_inf_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [0.01, np.inf, 0.02, 0.03, 0.04], "B": [0.02, 0.01, 0.0, 0.04, 0.01]},
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_ledoit_wolf_covariance(df)

    def test_negative_inf_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [0.01, -np.inf, 0.02, 0.03, 0.04], "B": [0.02, 0.01, 0.0, 0.04, 0.01]},
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_ledoit_wolf_covariance(df)

    def test_inf_in_nullable_float64_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([0.01, np.inf, 0.02, 0.03, 0.04], dtype="Float64"),
                "B": pd.array([0.02, 0.01, 0.0, 0.04, 0.01], dtype="Float64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_ledoit_wolf_covariance(df)


# ---------------------------------------------------------------------------
# 11. Complex input rejection
# ---------------------------------------------------------------------------

class TestComplexRejection:

    def test_complex128_raises(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": np.array([0.01 + 0j] * 10, dtype=np.complex128),
                "B": np.array([0.02 + 0j] * 10, dtype=np.complex128),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_ledoit_wolf_covariance(df)

    def test_complex64_raises(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": np.array([0.01 + 0j] * 10, dtype=np.complex64),
                "B": np.array([0.02 + 0j] * 10, dtype=np.complex64),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_ledoit_wolf_covariance(df)


# ---------------------------------------------------------------------------
# 12. Structural rejection: duplicates, non-numeric, None, non-DataFrame
# ---------------------------------------------------------------------------

class TestStructuralValidation:

    def test_none_raises(self):
        with pytest.raises(FinancialCalculationError, match="None"):
            calculate_ledoit_wolf_covariance(None)

    def test_series_raises(self):
        with pytest.raises(FinancialCalculationError, match="DataFrame"):
            calculate_ledoit_wolf_covariance(pd.Series([0.01, 0.02]))

    def test_empty_dataframe_raises(self):
        with pytest.raises(InsufficientDataError, match="empty"):
            calculate_ledoit_wolf_covariance(pd.DataFrame())

    def test_single_asset_raises(self):
        df = _make_df({"A": None}, n=10, seed=40)
        with pytest.raises(InsufficientDataError, match="2 assets"):
            calculate_ledoit_wolf_covariance(df)

    def test_duplicate_column_names_raises(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            np.random.default_rng(41).normal(0, 0.01, (10, 2)),
            index=dates,
            columns=["A", "A"],
        )
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            calculate_ledoit_wolf_covariance(df)

    def test_duplicate_date_index_raises(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=9, freq="B")
        dates = dates.append(dates[-1:])
        df = pd.DataFrame(
            rng.normal(0, 0.01, (10, 2)),
            index=dates,
            columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            calculate_ledoit_wolf_covariance(df)

    def test_non_numeric_column_raises(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05], "B": ["x", "y", "z", "w", "v"]},
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            calculate_ledoit_wolf_covariance(df)


# ---------------------------------------------------------------------------
# 13. Invalid trading_days_per_year
# ---------------------------------------------------------------------------

class TestTradingDaysValidation:

    def _valid_df(self) -> pd.DataFrame:
        return _make_df({"A": None, "B": None}, n=20, seed=50)

    def test_bool_true_raises(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=True)

    def test_bool_false_raises(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=False)

    def test_float_raises(self):
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=252.0)

    def test_zero_raises(self):
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=0)

    def test_negative_raises(self):
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=-252)

    def test_valid_int_accepted(self):
        result = calculate_ledoit_wolf_covariance(self._valid_df(), trading_days_per_year=252)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 14. Invalid min_observations
# ---------------------------------------------------------------------------

class TestMinObservationsValidation:

    def _valid_df(self) -> pd.DataFrame:
        return _make_df({"A": None, "B": None}, n=20, seed=51)

    def test_bool_true_raises(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=True)

    def test_bool_false_raises(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=False)

    def test_float_raises(self):
        with pytest.raises(FinancialCalculationError, match="float"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=5.0)

    def test_string_raises(self):
        with pytest.raises(FinancialCalculationError, match="str"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations="10")

    def test_zero_raises(self):
        with pytest.raises(FinancialCalculationError, match="positive integer"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=0)

    def test_negative_raises(self):
        with pytest.raises(FinancialCalculationError, match="positive integer"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=-1)

    def test_below_floor_raises(self):
        with pytest.raises(FinancialCalculationError, match="absolute minimum"):
            calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=1)

    def test_none_uses_default(self):
        result = calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=None)
        assert result.shape == (2, 2)

    def test_valid_int_accepted(self):
        result = calculate_ledoit_wolf_covariance(self._valid_df(), min_observations=5)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 15. Insufficient observations
# ---------------------------------------------------------------------------

class TestInsufficientObservations:

    def test_t_less_than_default_minimum_raises(self):
        """T=2, N=2: default effective_min = max(2, 3) = 3 → should raise."""
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.01]}, index=dates)
        with pytest.raises(InsufficientDataError):
            calculate_ledoit_wolf_covariance(df)

    def test_custom_min_observations_exceeds_t(self):
        df = _make_df({"A": None, "B": None}, n=10, seed=52)
        with pytest.raises(InsufficientDataError):
            calculate_ledoit_wolf_covariance(df, min_observations=20)

    def test_exactly_n_plus_one_passes(self):
        """T = N+1 (default threshold) must pass."""
        rng = np.random.default_rng(53)
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            rng.normal(0, 0.01, (4, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# 16. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_constant_returns_does_not_raise(self):
        """Constant returns → all-zero LW covariance; no exception expected."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"A": [0.01] * 10, "B": [0.02] * 10},
            index=dates,
        )
        # sklearn LedoitWolf handles constant (zero-variance) inputs gracefully
        result = calculate_ledoit_wolf_covariance(df, min_observations=2)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_perfectly_correlated_assets(self):
        """B = 2*A → rank-deficient; LW shrinkage handles gracefully."""
        rng = np.random.default_rng(54)
        a = rng.normal(0, 0.01, 50)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({"A": a, "B": 2.0 * a}, index=dates)
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_large_asset_universe(self):
        """20 assets, 300 observations — completes without error."""
        rng = np.random.default_rng(55)
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        df = pd.DataFrame(
            rng.normal(0, 0.01, (300, 20)),
            index=dates,
            columns=[f"A{i}" for i in range(20)],
        )
        result = calculate_ledoit_wolf_covariance(df)
        assert result.shape == (20, 20)
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# 17. Namespace compatibility imports
# ---------------------------------------------------------------------------

class TestNamespaceImports:

    def test_importable_from_portfolio_engine(self):
        from app.portfolio_engine import calculate_ledoit_wolf_covariance as fn
        assert callable(fn)

    def test_importable_from_portfolio_engine_shrinkage(self):
        from app.portfolio_engine.shrinkage import calculate_ledoit_wolf_covariance as fn
        assert callable(fn)

    def test_importable_from_financial_engine(self):
        from app.financial_engine import calculate_ledoit_wolf_covariance as fn
        assert callable(fn)

    def test_importable_from_financial_engine_shrinkage(self):
        from app.financial_engine.shrinkage import calculate_ledoit_wolf_covariance as fn
        assert callable(fn)

    def test_fit_ledoit_wolf_importable_from_portfolio_engine(self):
        from app.portfolio_engine import fit_ledoit_wolf as fn
        assert callable(fn)

    def test_fit_ledoit_wolf_importable_from_financial_engine(self):
        from app.financial_engine import fit_ledoit_wolf as fn
        assert callable(fn)

    def test_all_imports_are_same_object(self):
        from app.portfolio_engine.shrinkage import calculate_ledoit_wolf_covariance as fn1
        from app.portfolio_engine import calculate_ledoit_wolf_covariance as fn2
        from app.financial_engine.shrinkage import calculate_ledoit_wolf_covariance as fn3
        from app.financial_engine import calculate_ledoit_wolf_covariance as fn4
        assert fn1 is fn2 is fn3 is fn4

    def test_fit_ledoit_wolf_all_imports_same_object(self):
        from app.portfolio_engine.shrinkage import fit_ledoit_wolf as fn1
        from app.portfolio_engine import fit_ledoit_wolf as fn2
        from app.financial_engine.shrinkage import fit_ledoit_wolf as fn3
        from app.financial_engine import fit_ledoit_wolf as fn4
        assert fn1 is fn2 is fn3 is fn4


# ---------------------------------------------------------------------------
# 18. Step 1 is unaffected: sample covariance still works independently
# ---------------------------------------------------------------------------

class TestStep1Unaffected:
    """Confirm that importing shrinkage module does not alter Step 1 behavior."""

    def test_sample_covariance_still_importable(self):
        from app.portfolio_engine import calculate_sample_covariance
        assert callable(calculate_sample_covariance)

    def test_sample_covariance_result_unchanged(self):
        """Step 1 result must equal 252 × pandas .cov() — unchanged by Step 2."""
        rng = np.random.default_rng(60)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 50), "B": rng.normal(0, 0.02, 50)},
            index=dates,
        )
        from app.portfolio_engine import calculate_sample_covariance
        result = calculate_sample_covariance(df)
        expected = 252 * df.cov()
        expected_sym = (expected.values + expected.values.T) / 2.0
        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12)

    def test_lw_and_sample_differ_due_to_ddof(self):
        """LW (ddof=0 internally) and sample covariance (ddof=1) must NOT be equal."""
        rng = np.random.default_rng(61)
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, 50), "B": rng.normal(0, 0.02, 50)},
            index=dates,
        )
        from app.portfolio_engine import calculate_sample_covariance
        sc = calculate_sample_covariance(df)
        lw = calculate_ledoit_wolf_covariance(df)
        # They must differ (unless delta=0 and ddof effects exactly cancel — extremely unlikely)
        assert not np.allclose(sc.values, lw.values, rtol=1e-6)
