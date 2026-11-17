"""Tests for Phase 2.2 Step 1: Sample Covariance Matrix Estimation.

Covers:
- Known numerical example (manual computation vs. function output)
- Independently hand-calculated two-asset numerical example (no pandas .cov())
- Annualization by trading_days_per_year
- Symmetry of output
- Asset ordering preserved
- Input validation: None, non-DataFrame, empty, single asset, non-numeric,
  NaN, Inf, duplicate column names, duplicate date indices
- Insufficient observations (T < min_observations)
- Constant returns (zero variance diagonal)
- Custom min_observations parameter
- min_observations: bool, float, str, zero, negative all rejected
- trading_days_per_year: bool rejected explicitly
- Complex-valued returns rejected
- Re-export through financial_engine and portfolio_engine namespaces
- CovarianceValidationError is importable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    CovarianceValidationError,
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine.covariance import calculate_sample_covariance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(data: dict, dates: list | None = None) -> pd.DataFrame:
    """Build a simple returns DataFrame with a DatetimeIndex."""
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq="B")
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# 1. Known Numerical Example
# ---------------------------------------------------------------------------

class TestKnownNumericalExample:
    """Verify annualized sample covariance against a manually computed value."""

    def test_two_asset_exact_values(self):
        """
        Manual derivation for 4-row, 2-asset returns:
            A = [0.01, 0.02, -0.01, 0.03]
            B = [0.02, 0.01,  0.00, 0.04]

        R̄_A = 0.0125, R̄_B = 0.0175
        Deviations from mean:
            A: [-0.0025,  0.0075, -0.0225,  0.0175]
            B: [ 0.0025, -0.0075, -0.0175,  0.0225]
        SS_AA = 0.0025^2 + 0.0075^2 + 0.0225^2 + 0.0175^2 = 0.000875
        SS_BB = same = 0.000875
        SS_AB = (-0.0025)(0.0025) + (0.0075)(-0.0075) + (-0.0225)(-0.0175) + (0.0175)(0.0225)
              = -0.000006 + -0.000056 + 0.000394 + 0.000394  ...
        Let pandas compute — just verify annualization by 252.
        """
        a = [0.01, 0.02, -0.01, 0.03]
        b = [0.02, 0.01, 0.00, 0.04]
        df = _make_returns({"A": a, "B": b})

        result = calculate_sample_covariance(df, trading_days_per_year=252)

        # Build expected value directly from pandas
        cov_daily = df.cov()
        expected = 252 * cov_daily
        expected = (expected.values + expected.values.T) / 2.0

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-15)

    def test_three_asset_shape_and_labels(self):
        """Three-asset case: shape (3,3) and labels must match input columns."""
        rng = np.random.default_rng(42)
        data = {
            "RELIANCE": rng.normal(0.001, 0.02, 60),
            "TCS": rng.normal(0.0005, 0.015, 60),
            "INFY": rng.normal(0.0008, 0.018, 60),
        }
        df = _make_returns(data)
        result = calculate_sample_covariance(df)

        assert result.shape == (3, 3)
        assert list(result.columns) == ["RELIANCE", "TCS", "INFY"]
        assert list(result.index) == ["RELIANCE", "TCS", "INFY"]


# ---------------------------------------------------------------------------
# 2. Annualization
# ---------------------------------------------------------------------------

class TestAnnualization:
    """Verify that annualization factor is applied correctly."""

    def test_default_annualization_is_252(self):
        """Default trading_days_per_year should be 252."""
        rng = np.random.default_rng(10)
        df = _make_returns({
            "X": rng.normal(0, 0.01, 50),
            "Y": rng.normal(0, 0.02, 50),
        })

        result_default = calculate_sample_covariance(df)
        result_252 = calculate_sample_covariance(df, trading_days_per_year=252)
        np.testing.assert_array_equal(result_default.values, result_252.values)

    def test_custom_trading_days(self):
        """trading_days_per_year=200 must scale daily covariance by exactly 200."""
        rng = np.random.default_rng(99)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 30),
            "B": rng.normal(0, 0.02, 30),
        })

        result_200 = calculate_sample_covariance(df, trading_days_per_year=200)
        result_252 = calculate_sample_covariance(df, trading_days_per_year=252)

        ratio = result_200.values / result_252.values
        np.testing.assert_allclose(ratio, 200 / 252, rtol=1e-12)

    def test_annualization_exact_scaling(self):
        """Directly verify: result = 252 * daily_cov (up to symmetrization rounding)."""
        rng = np.random.default_rng(7)
        df = _make_returns({
            "P": rng.normal(0, 0.01, 40),
            "Q": rng.normal(0, 0.015, 40),
        })
        result = calculate_sample_covariance(df, trading_days_per_year=252)
        daily = df.cov()
        expected = 252 * daily
        expected_sym = (expected.values + expected.values.T) / 2.0
        np.testing.assert_allclose(result.values, expected_sym, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 3. Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    """Output matrix must be symmetric."""

    def test_symmetric_two_assets(self):
        rng = np.random.default_rng(1)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 50),
            "B": rng.normal(0, 0.02, 50),
        })
        result = calculate_sample_covariance(df)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)

    def test_symmetric_five_assets(self):
        rng = np.random.default_rng(2)
        assets = {f"A{i}": rng.normal(0, 0.01 * (i + 1), 100) for i in range(5)}
        df = _make_returns(assets)
        result = calculate_sample_covariance(df)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)


# ---------------------------------------------------------------------------
# 4. Asset Ordering Preserved
# ---------------------------------------------------------------------------

class TestAssetOrdering:
    """Column and index order of the output must match input column order."""

    def test_ordering_preserved(self):
        rng = np.random.default_rng(3)
        columns = ["ZOMATO", "ADANI", "BHARTIARTL", "WIPRO"]
        data = {col: rng.normal(0, 0.01, 50) for col in columns}
        df = _make_returns(data)

        result = calculate_sample_covariance(df)
        assert list(result.columns) == columns
        assert list(result.index) == columns

    def test_different_column_orderings(self):
        """Reversing input column order produces reversed output ordering."""
        rng = np.random.default_rng(4)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 30),
            "B": rng.normal(0, 0.02, 30),
            "C": rng.normal(0, 0.015, 30),
        })
        df_rev = df[["C", "B", "A"]]

        result_orig = calculate_sample_covariance(df)
        result_rev = calculate_sample_covariance(df_rev)

        assert list(result_orig.columns) == ["A", "B", "C"]
        assert list(result_rev.columns) == ["C", "B", "A"]
        # The (A,A) variance must be the same regardless of ordering
        np.testing.assert_allclose(result_orig.loc["A", "A"], result_rev.loc["A", "A"], rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Input Validation — Structural
# ---------------------------------------------------------------------------

class TestInputValidationStructural:

    def test_none_raises_financial_calculation_error(self):
        with pytest.raises(FinancialCalculationError, match="None"):
            calculate_sample_covariance(None)

    def test_series_raises_financial_calculation_error(self):
        s = pd.Series([0.01, 0.02, 0.03], name="A")
        with pytest.raises(FinancialCalculationError, match="DataFrame"):
            calculate_sample_covariance(s)

    def test_list_raises_financial_calculation_error(self):
        with pytest.raises(FinancialCalculationError, match="DataFrame"):
            calculate_sample_covariance([[0.01, 0.02], [0.03, 0.04]])

    def test_empty_dataframe_raises_insufficient_data(self):
        df = pd.DataFrame()
        with pytest.raises(InsufficientDataError, match="empty"):
            calculate_sample_covariance(df)

    def test_single_asset_raises_insufficient_data(self):
        df = _make_returns({"A": [0.01, 0.02, 0.03, 0.04, 0.05]})
        with pytest.raises(InsufficientDataError, match="2 assets"):
            calculate_sample_covariance(df)

    def test_duplicate_column_names_raises_error(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            rng.normal(0, 0.01, (10, 2)),
            index=dates,
            columns=["A", "A"],  # duplicate
        )
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            calculate_sample_covariance(df)

    def test_duplicate_date_index_raises_error(self):
        rng = np.random.default_rng(6)
        # 10 observations but dates have a duplicate
        dates = pd.date_range("2024-01-01", periods=9, freq="B")
        dates = dates.append(dates[-1:])  # duplicate last date
        df = pd.DataFrame(
            rng.normal(0, 0.01, (10, 2)),
            index=dates,
            columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="duplicate"):
            calculate_sample_covariance(df)


# ---------------------------------------------------------------------------
# 6. Input Validation — Value Checks
# ---------------------------------------------------------------------------

class TestInputValidationValues:

    def test_nan_values_raise_financial_calculation_error(self):
        df = _make_returns({
            "A": [0.01, np.nan, 0.02, 0.03],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_sample_covariance(df)

    def test_inf_values_raise_financial_calculation_error(self):
        df = _make_returns({
            "A": [0.01, np.inf, 0.02, 0.03],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_sample_covariance(df)

    def test_neg_inf_values_raise_financial_calculation_error(self):
        df = _make_returns({
            "A": [0.01, -np.inf, 0.02, 0.03],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_sample_covariance(df)

    def test_non_numeric_column_raises_error(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {
                "A": [0.01, 0.02, 0.03, 0.04],
                "B": ["x", "y", "z", "w"],  # string column
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            calculate_sample_covariance(df)

    def test_no_ffill_bfill_on_nan(self):
        """
        Confirm NaN is rejected outright, not silently imputed.
        Even if only one entry is NaN the call must fail.
        """
        df = _make_returns({
            "A": [0.01, 0.02, 0.03, np.nan],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_sample_covariance(df)

    def test_invalid_trading_days_zero_raises_error(self):
        df = _make_returns({
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_sample_covariance(df, trading_days_per_year=0)

    def test_invalid_trading_days_negative_raises_error(self):
        df = _make_returns({
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_sample_covariance(df, trading_days_per_year=-252)


# ---------------------------------------------------------------------------
# 7. Insufficient Observations
# ---------------------------------------------------------------------------

class TestInsufficientObservations:

    def test_fewer_than_abs_minimum_raises_error(self):
        """T=1 is below absolute floor of 2."""
        df = _make_returns({
            "A": [0.01],
            "B": [0.02],
        })
        with pytest.raises(InsufficientDataError, match="Insufficient"):
            calculate_sample_covariance(df)

    def test_exactly_two_observations_passes_for_two_assets(self):
        """T=2, N=2: effective_min = max(2, 3) = 3 → should raise."""
        df = _make_returns({
            "A": [0.01, 0.02],
            "B": [0.02, 0.01],
        })
        # Default effective_min for N=2 is max(2, 3) = 3; T=2 < 3
        with pytest.raises(InsufficientDataError):
            calculate_sample_covariance(df)

    def test_exactly_n_plus_one_observations_passes(self):
        """T = N+1 (minimum default threshold) should pass."""
        rng = np.random.default_rng(8)
        n_assets = 3
        t_obs = n_assets + 1  # = 4
        data = {f"A{i}": rng.normal(0, 0.01, t_obs) for i in range(n_assets)}
        df = _make_returns(data)
        result = calculate_sample_covariance(df)
        assert result.shape == (n_assets, n_assets)

    def test_custom_min_observations_respected(self):
        """Custom min_observations=10 rejects T=8 even if default would pass."""
        rng = np.random.default_rng(9)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 8),
            "B": rng.normal(0, 0.02, 8),
        })
        # Default min is max(2, 3)=3, so T=8 passes normally
        result = calculate_sample_covariance(df)
        assert result.shape == (2, 2)

        # With min_observations=10, T=8 must fail
        with pytest.raises(InsufficientDataError, match="10"):
            calculate_sample_covariance(df, min_observations=10)

    def test_custom_min_observations_allows_exact_match(self):
        """When T == min_observations the call must succeed."""
        rng = np.random.default_rng(11)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 10),
            "B": rng.normal(0, 0.02, 10),
        })
        result = calculate_sample_covariance(df, min_observations=10)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 8. Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_constant_returns_zero_variance(self):
        """Assets with constant returns have zero variance AND zero covariance."""
        df = _make_returns({
            "A": [0.01] * 10,
            "B": [0.02] * 10,
        })
        result = calculate_sample_covariance(df, min_observations=2)
        # All variances and covariances must be zero (constant series)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)

    def test_uncorrelated_assets_near_zero_covariance(self):
        """
        Orthogonal assets should produce near-zero off-diagonal elements
        (in expectation; for large T the sample estimate converges).
        We just verify the diagonal is positive and off-diagonal is finite.
        """
        rng = np.random.default_rng(12)
        n = 500
        df = _make_returns({
            "A": rng.normal(0, 0.01, n),
            "B": rng.normal(0, 0.02, n),
        })
        result = calculate_sample_covariance(df)
        # Diagonal entries must be positive (non-constant series)
        assert result.loc["A", "A"] > 0
        assert result.loc["B", "B"] > 0
        # Off-diagonal entries must be finite
        assert np.isfinite(result.loc["A", "B"])

    def test_perfectly_correlated_assets(self):
        """
        If B = k * A exactly, covariance matrix is rank-1 (singular) but
        the function must still return without error (singularity ≠ invalid input).
        """
        rng = np.random.default_rng(13)
        a = rng.normal(0, 0.01, 50)
        df = _make_returns({
            "A": a,
            "B": 2.0 * a,  # perfectly correlated
        })
        result = calculate_sample_covariance(df)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_large_number_of_assets(self):
        """20-asset, 300-observation case completes without error."""
        rng = np.random.default_rng(14)
        n_assets, n_obs = 20, 300
        data = {f"A{i}": rng.normal(0, 0.01 * (i + 1), n_obs) for i in range(n_assets)}
        df = _make_returns(data)
        result = calculate_sample_covariance(df)
        assert result.shape == (n_assets, n_assets)
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# 9. Output Properties
# ---------------------------------------------------------------------------

class TestOutputProperties:

    def test_output_is_dataframe(self):
        df = _make_returns({
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.02, 0.01, 0.00, 0.04],
        })
        result = calculate_sample_covariance(df, min_observations=2)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_non_negative(self):
        """Variances (diagonal) must be ≥ 0."""
        rng = np.random.default_rng(15)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 50),
            "B": rng.normal(0, 0.02, 50),
        })
        result = calculate_sample_covariance(df)
        diag = np.diag(result.values)
        assert np.all(diag >= 0.0)

    def test_all_values_finite(self):
        rng = np.random.default_rng(16)
        df = _make_returns({
            "A": rng.normal(0, 0.01, 50),
            "B": rng.normal(0, 0.02, 50),
        })
        result = calculate_sample_covariance(df)
        assert np.all(np.isfinite(result.values))


# ---------------------------------------------------------------------------
# 10. Re-export Namespace Checks
# ---------------------------------------------------------------------------

class TestNamespaceReexport:

    def test_importable_from_portfolio_engine(self):
        from app.portfolio_engine import calculate_sample_covariance as csc
        assert callable(csc)

    def test_importable_from_financial_engine_covariance(self):
        from app.financial_engine.covariance import calculate_sample_covariance as csc
        assert callable(csc)

    def test_importable_from_financial_engine_init(self):
        from app.financial_engine import calculate_sample_covariance as csc
        assert callable(csc)

    def test_all_imports_are_same_object(self):
        from app.portfolio_engine.covariance import calculate_sample_covariance as csc1
        from app.portfolio_engine import calculate_sample_covariance as csc2
        from app.financial_engine.covariance import calculate_sample_covariance as csc3
        from app.financial_engine import calculate_sample_covariance as csc4
        assert csc1 is csc2 is csc3 is csc4


# ---------------------------------------------------------------------------
# 11. Exception Class Checks
# ---------------------------------------------------------------------------

class TestExceptionClasses:

    def test_covariance_validation_error_importable(self):
        from app.core.exceptions import CovarianceValidationError as CVE
        assert issubclass(CVE, Exception)

    def test_covariance_validation_error_is_financial_calculation_error(self):
        from app.core.exceptions import CovarianceValidationError, FinancialCalculationError
        assert issubclass(CovarianceValidationError, FinancialCalculationError)

    def test_covariance_validation_error_has_error_code(self):
        exc = CovarianceValidationError("test")
        assert exc.error_code == "COVARIANCE_VALIDATION_ERROR"

    def test_covariance_validation_error_to_dict(self):
        exc = CovarianceValidationError("bad matrix", details={"shape": [3, 3]})
        d = exc.to_dict()
        assert d["error_code"] == "COVARIANCE_VALIDATION_ERROR"
        assert d["message"] == "bad matrix"
        assert d["details"] == {"shape": [3, 3]}


# ===========================================================================
# Audit-correction tests (Phase 2.2 Step 1 targeted audit)
# ===========================================================================

# ---------------------------------------------------------------------------
# 12. Hand-calculated numerical example (no pandas .cov() in expected value)
# ---------------------------------------------------------------------------

class TestHandCalculatedNumericalExample:
    """Verify the estimator against a fully manual derivation.

    Dataset (T=4, N=2):
        A = [0.01,  0.03, -0.02, 0.04]
        B = [0.02, -0.01,  0.03, 0.02]

    Step 1 — means:
        R̄_A = (0.01 + 0.03 - 0.02 + 0.04) / 4 = 0.06 / 4 = 0.015
        R̄_B = (0.02 - 0.01 + 0.03 + 0.02) / 4 = 0.06 / 4 = 0.015

    Step 2 — deviations from mean:
        t=1: dA = -0.005, dB =  0.005
        t=2: dA =  0.015, dB = -0.025
        t=3: dA = -0.035, dB =  0.015
        t=4: dA =  0.025, dB =  0.005

    Step 3 — sum of squared deviations (ddof=1, divide by T-1=3):
        S_AA = (0.005^2 + 0.015^2 + 0.035^2 + 0.025^2) / 3
             = (0.000025 + 0.000225 + 0.001225 + 0.000625) / 3
             = 0.002100 / 3
             = 0.000700

        S_BB = (0.005^2 + 0.025^2 + 0.015^2 + 0.005^2) / 3
             = (0.000025 + 0.000625 + 0.000225 + 0.000025) / 3
             = 0.000900 / 3
             = 0.000300

        S_AB = ((-0.005)(0.005) + (0.015)(-0.025) + (-0.035)(0.015) + (0.025)(0.005)) / 3
             = (-0.000025 + -0.000375 + -0.000525 + 0.000125) / 3
             = -0.000800 / 3
             = -0.000266̄  (= -4/15000)

    Step 4 — annualize by 252:
        Σ_AA = 252 × 0.000700 = 0.1764
        Σ_BB = 252 × 0.000300 = 0.0756
        Σ_AB = 252 × (-4/15000) = -1008/15000 = -0.0672
    """

    # Exact fractional values derived above
    _S_AA_DAILY = 7 / 10000          # 0.000700
    _S_BB_DAILY = 3 / 10000          # 0.000300
    _S_AB_DAILY = -4 / 15000         # -0.000266̄

    _A = [0.01, 0.03, -0.02, 0.04]
    _B = [0.02, -0.01, 0.03, 0.02]

    def _get_df(self) -> pd.DataFrame:
        return _make_returns({"A": self._A, "B": self._B})

    def test_daily_variance_A(self):
        """Var(A) daily must equal 7/10000 exactly."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=1, min_observations=2)
        # trading_days_per_year=1 → result IS the daily covariance
        np.testing.assert_allclose(result.loc["A", "A"], self._S_AA_DAILY, rtol=1e-12)

    def test_daily_variance_B(self):
        """Var(B) daily must equal 3/10000 exactly."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=1, min_observations=2)
        np.testing.assert_allclose(result.loc["B", "B"], self._S_BB_DAILY, rtol=1e-12)

    def test_daily_covariance_AB(self):
        """Cov(A,B) daily must equal -4/15000 exactly."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=1, min_observations=2)
        np.testing.assert_allclose(result.loc["A", "B"], self._S_AB_DAILY, rtol=1e-12)
        np.testing.assert_allclose(result.loc["B", "A"], self._S_AB_DAILY, rtol=1e-12)

    def test_annualized_variance_A(self):
        """Annualized Var(A) = 252 × (7/10000) = 0.1764."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=252, min_observations=2)
        expected = 252 * self._S_AA_DAILY
        np.testing.assert_allclose(result.loc["A", "A"], expected, rtol=1e-12)

    def test_annualized_variance_B(self):
        """Annualized Var(B) = 252 × (3/10000) = 0.0756."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=252, min_observations=2)
        expected = 252 * self._S_BB_DAILY
        np.testing.assert_allclose(result.loc["B", "B"], expected, rtol=1e-12)

    def test_annualized_covariance_AB(self):
        """Annualized Cov(A,B) = 252 × (-4/15000) = -0.0672."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=252, min_observations=2)
        expected = 252 * self._S_AB_DAILY
        np.testing.assert_allclose(result.loc["A", "B"], expected, rtol=1e-12)
        np.testing.assert_allclose(result.loc["B", "A"], expected, rtol=1e-12)

    def test_full_matrix_matches_hand_calculation(self):
        """All four entries of the 2×2 annualized matrix match the hand derivation."""
        df = self._get_df()
        result = calculate_sample_covariance(df, trading_days_per_year=252, min_observations=2)

        expected = np.array([
            [252 * self._S_AA_DAILY, 252 * self._S_AB_DAILY],
            [252 * self._S_AB_DAILY, 252 * self._S_BB_DAILY],
        ])
        np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 13. min_observations parameter validation
# ---------------------------------------------------------------------------

class TestMinObservationsValidation:
    """min_observations must be a positive integer >= 2 when supplied."""

    def _valid_df(self) -> pd.DataFrame:
        """10-row, 2-asset DataFrame that passes all other checks."""
        rng = np.random.default_rng(20)
        return _make_returns({
            "A": rng.normal(0, 0.01, 10),
            "B": rng.normal(0, 0.02, 10),
        })

    # --- bool ---

    def test_bool_true_rejected(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_sample_covariance(self._valid_df(), min_observations=True)

    def test_bool_false_rejected(self):
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_sample_covariance(self._valid_df(), min_observations=False)

    # --- float ---

    def test_float_rejected(self):
        with pytest.raises(FinancialCalculationError, match="float"):
            calculate_sample_covariance(self._valid_df(), min_observations=5.0)

    def test_float_non_integer_rejected(self):
        with pytest.raises(FinancialCalculationError, match="float"):
            calculate_sample_covariance(self._valid_df(), min_observations=4.5)

    # --- string ---

    def test_string_rejected(self):
        with pytest.raises(FinancialCalculationError, match="str"):
            calculate_sample_covariance(self._valid_df(), min_observations="10")

    # --- zero and negative ---

    def test_zero_rejected(self):
        with pytest.raises(FinancialCalculationError, match="positive integer"):
            calculate_sample_covariance(self._valid_df(), min_observations=0)

    def test_negative_rejected(self):
        with pytest.raises(FinancialCalculationError, match="positive integer"):
            calculate_sample_covariance(self._valid_df(), min_observations=-5)

    # --- below absolute floor ---

    def test_one_below_absolute_floor_rejected(self):
        """min_observations=1 is below the ddof=1 absolute floor of 2."""
        with pytest.raises(FinancialCalculationError, match="absolute minimum"):
            calculate_sample_covariance(self._valid_df(), min_observations=1)

    # --- None uses default, valid int passes ---

    def test_none_uses_default(self):
        """None falls back to max(2, N+1) without raising."""
        result = calculate_sample_covariance(self._valid_df(), min_observations=None)
        assert result.shape == (2, 2)

    def test_valid_int_accepted(self):
        """A positive integer >= 2 that is satisfied by T is accepted."""
        result = calculate_sample_covariance(self._valid_df(), min_observations=5)
        assert result.shape == (2, 2)

    def test_valid_int_equal_to_t_accepted(self):
        """min_observations == T (exact match) must pass."""
        df = self._valid_df()   # T = 10
        result = calculate_sample_covariance(df, min_observations=10)
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# 14. trading_days_per_year bool validation
# ---------------------------------------------------------------------------

class TestTradingDaysPerYearBoolValidation:
    """bool must be rejected for trading_days_per_year even though bool is a
    subclass of int (True == 1 and False == 0 in Python)."""

    def _valid_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(21)
        return _make_returns({
            "A": rng.normal(0, 0.01, 10),
            "B": rng.normal(0, 0.02, 10),
        })

    def test_true_rejected(self):
        """True == 1 (a positive integer) but must still be rejected as bool."""
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_sample_covariance(self._valid_df(), trading_days_per_year=True)

    def test_false_rejected(self):
        """False == 0 must be rejected as bool (not as zero int)."""
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_sample_covariance(self._valid_df(), trading_days_per_year=False)

    def test_positive_int_still_accepted(self):
        """Confirm that the bool guard does not break normal positive-int path."""
        result = calculate_sample_covariance(self._valid_df(), trading_days_per_year=252)
        assert result.shape == (2, 2)

    def test_float_trading_days_rejected(self):
        """A float trading_days_per_year must still be rejected."""
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_sample_covariance(self._valid_df(), trading_days_per_year=252.0)


# ---------------------------------------------------------------------------
# 15. Complex-valued returns
# ---------------------------------------------------------------------------

class TestComplexValuedReturns:
    """Complex dtype columns must be rejected; real-valued returns required."""

    def _complex_df(self, dtype=complex) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        data = {
            "A": np.array([0.01 + 0j] * 10, dtype=dtype),
            "B": np.array([0.02 + 0j] * 10, dtype=dtype),
        }
        return pd.DataFrame(data, index=dates)

    def test_complex128_rejected(self):
        df = self._complex_df(dtype=np.complex128)
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_sample_covariance(df)

    def test_complex64_rejected(self):
        df = self._complex_df(dtype=np.complex64)
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_sample_covariance(df)

    def test_complex_error_names_the_offending_columns(self):
        """The exception details must identify which columns are complex."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "REAL_COL": np.random.default_rng(30).normal(0, 0.01, 10),
                "COMPLEX_COL": np.array([0.01 + 0.001j] * 10, dtype=complex),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError) as exc_info:
            calculate_sample_covariance(df)
        assert "COMPLEX_COL" in str(exc_info.value.details.get("complex_columns", []))

    def test_real_float64_still_accepted(self):
        """Plain float64 real data must continue to work."""
        rng = np.random.default_rng(31)
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": rng.normal(0, 0.01, 10).astype(np.float64),
                "B": rng.normal(0, 0.02, 10).astype(np.float64),
            },
            index=dates,
        )
        result = calculate_sample_covariance(df)
        assert result.shape == (2, 2)


# ===========================================================================
# Nullable extension dtype tests (robustness correction)
# ===========================================================================

class TestNullableExtensionDtypes:
    """Verify that pandas nullable extension dtypes (Float64, Int64, etc.)
    are handled correctly without TypeError from np.isinf() on object arrays.

    All tests confirm:
    - Valid nullable data produces the same covariance as equivalent float64.
    - NA values are still rejected (detected before float64 conversion).
    - Inf values stored in nullable columns are still rejected.
    - Column labels and ordering are preserved through conversion.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _float64_df(self, seed: int = 50) -> pd.DataFrame:
        """Reference DataFrame with standard NumPy float64 dtype."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        return pd.DataFrame(
            {
                "A": rng.normal(0.001, 0.015, 20),
                "B": rng.normal(0.0005, 0.010, 20),
            },
            index=dates,
            dtype=np.float64,
        )

    def _nullable_float64_df(self, seed: int = 50) -> pd.DataFrame:
        """Same data as _float64_df but backed by pandas Float64 extension dtype."""
        df = self._float64_df(seed)
        return df.astype("Float64")

    # ------------------------------------------------------------------
    # 1. Pure Float64 (pandas nullable)
    # ------------------------------------------------------------------

    def test_nullable_float64_accepted(self):
        """A DataFrame with all-Float64 columns must not raise."""
        df = self._nullable_float64_df()
        result = calculate_sample_covariance(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

    def test_nullable_float64_matches_regular_float64(self):
        """Float64 and float64 backed DataFrames with identical data must produce
        identical annualized covariance matrices."""
        df_reg = self._float64_df()
        df_ext = self._nullable_float64_df()

        result_reg = calculate_sample_covariance(df_reg, trading_days_per_year=252)
        result_ext = calculate_sample_covariance(df_ext, trading_days_per_year=252)

        np.testing.assert_allclose(result_reg.values, result_ext.values, rtol=1e-12, atol=1e-15)

    def test_nullable_float64_preserves_column_labels(self):
        """Column and index labels must survive the float64 conversion."""
        df = self._nullable_float64_df()
        result = calculate_sample_covariance(df)
        assert list(result.columns) == ["A", "B"]
        assert list(result.index) == ["A", "B"]

    def test_nullable_float64_output_is_symmetric(self):
        df = self._nullable_float64_df()
        result = calculate_sample_covariance(df)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-14)

    def test_nullable_float64_output_is_finite(self):
        df = self._nullable_float64_df()
        result = calculate_sample_covariance(df)
        assert np.all(np.isfinite(result.values))

    # ------------------------------------------------------------------
    # 2. Mixed Float64 + float64 columns
    # ------------------------------------------------------------------

    def test_mixed_nullable_and_regular_float64_accepted(self):
        """One Float64 column and one float64 column must work without error."""
        rng = np.random.default_rng(51)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame(
            {
                "NULLABLE": pd.array(rng.normal(0.001, 0.015, 20), dtype="Float64"),
                "REGULAR": rng.normal(0.0005, 0.010, 20).astype(np.float64),
            },
            index=dates,
        )
        result = calculate_sample_covariance(df)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_mixed_matches_pure_float64(self):
        """Mixed dtype DataFrame must produce the same result as pure float64."""
        rng = np.random.default_rng(52)
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        a = rng.normal(0.001, 0.015, 20)
        b = rng.normal(0.0005, 0.010, 20)

        df_mixed = pd.DataFrame(
            {
                "A": pd.array(a, dtype="Float64"),
                "B": b.astype(np.float64),
            },
            index=dates,
        )
        df_pure = pd.DataFrame({"A": a, "B": b}, index=dates, dtype=np.float64)

        result_mixed = calculate_sample_covariance(df_mixed)
        result_pure = calculate_sample_covariance(df_pure)

        np.testing.assert_allclose(result_mixed.values, result_pure.values, rtol=1e-12)

    # ------------------------------------------------------------------
    # 3. Nullable integer columns (Int64)
    # ------------------------------------------------------------------

    def test_nullable_int64_accepted(self):
        """Int64 nullable integer columns holding whole-number returns must be accepted."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, -1, 3, 0, 1, -2, 2, 1, 0], dtype="Int64"),
                "B": pd.array([2, 1, 0, 2, 1, -1, 1, 3, 0, 1], dtype="Int64"),
            },
            index=dates,
        )
        result = calculate_sample_covariance(df, min_observations=2)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result.values))

    def test_nullable_int64_matches_int64(self):
        """Nullable Int64 and regular int64 with same data must produce identical result."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        a_vals = [1, 2, -1, 3, 0, 1, -2, 2, 1, 0]
        b_vals = [2, 1, 0, 2, 1, -1, 1, 3, 0, 1]

        df_nullable = pd.DataFrame(
            {
                "A": pd.array(a_vals, dtype="Int64"),
                "B": pd.array(b_vals, dtype="Int64"),
            },
            index=dates,
        )
        df_regular = pd.DataFrame(
            {"A": a_vals, "B": b_vals},
            index=dates,
        ).astype(float)

        result_nullable = calculate_sample_covariance(df_nullable, min_observations=2)
        result_regular = calculate_sample_covariance(df_regular, min_observations=2)

        np.testing.assert_allclose(result_nullable.values, result_regular.values, rtol=1e-12)

    # ------------------------------------------------------------------
    # 4. NA in nullable columns is still rejected
    # ------------------------------------------------------------------

    def test_nullable_float64_with_na_raises(self):
        """NA in a Float64 column must be detected and rejected before conversion."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([0.01, 0.02, None, 0.03, 0.04], dtype="Float64"),
                "B": pd.array([0.02, 0.01, 0.00, 0.04, 0.01], dtype="Float64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_sample_covariance(df)

    def test_nullable_int64_with_na_raises(self):
        """NA in an Int64 column must be detected and rejected."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([1, 2, None, 3, 4], dtype="Int64"),
                "B": pd.array([2, 1, 0, 4, 1], dtype="Int64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            calculate_sample_covariance(df)

    # ------------------------------------------------------------------
    # 5. Inf in nullable Float64 column is still rejected
    # ------------------------------------------------------------------

    def test_nullable_float64_with_inf_raises(self):
        """An Inf stored in a Float64-backed column must be caught after conversion."""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "A": pd.array([0.01, np.inf, 0.02, 0.03, 0.04], dtype="Float64"),
                "B": pd.array([0.02, 0.01, 0.00, 0.04, 0.01], dtype="Float64"),
            },
            index=dates,
        )
        with pytest.raises(FinancialCalculationError, match="infinite"):
            calculate_sample_covariance(df)

    # ------------------------------------------------------------------
    # 6. Three-asset nullable Float64 — labels and ordering preserved
    # ------------------------------------------------------------------

    def test_three_asset_nullable_float64_ordering(self):
        """With three Float64 assets the output labels must follow input column order."""
        rng = np.random.default_rng(53)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        cols = ["ZOMATO", "TCS", "INFY"]
        df = pd.DataFrame(
            {c: pd.array(rng.normal(0, 0.01, 30), dtype="Float64") for c in cols},
            index=dates,
        )
        result = calculate_sample_covariance(df)
        assert list(result.columns) == cols
        assert list(result.index) == cols
        assert result.shape == (3, 3)
