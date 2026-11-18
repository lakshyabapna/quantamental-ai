"""Quantamental AI - Unit Tests for Portfolio Optimization Engine.

Tests Phase 2.3 Step 1: Long-Only Global Minimum Variance (GMV) Portfolio Optimization.
Validates:
    1. Analytical closed-form solution parity for 2-asset portfolios.
    2. Long-only boundary constraints (clamping negative weights to 0.0).
    3. Multi-asset optimization (equal-variance, unequal-variance, 5-asset portfolios).
    4. Exact budget constraint satisfaction (|sum(w) - 1.0| <= 1e-6) and non-negativity (0 <= w_i <= 1).
    5. User-supplied initial weights (pd.Series, np.ndarray, list, label reordering, invalid inputs).
    6. Strict covariance matrix validation (type, shape, duplicate labels, dtypes, NaN, Inf, complex,
       non-positive diagonal variances, asymmetry).
    7. Spectral checks (NonPositiveDefiniteMatrixError, IllConditionedMatrixError, configurable thresholds).
    8. Numerical parameter validation (NaN, Inf, negative, bool for tolerance, condition number, etc.).
    9. Solver non-convergence and fallback policies (RAISE_EXCEPTION, EQUAL_WEIGHT).
    10. OptimizationResult attributes, immutability, to_dict() serialization, and package re-exports.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    IllConditionedMatrixError,
    InsufficientDataError,
    NonPositiveDefiniteMatrixError,
    OptimizationException,
)
from app.financial_engine import (
    OptimizationResult as FinancialOptimizationResult,
    optimize_global_minimum_variance as financial_optimize_gmv,
)
from app.financial_engine.optimizer import (
    DEFAULT_INPUT_SYMMETRY_TOLERANCE,
    DEFAULT_MAX_CONDITION_NUMBER,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_OPTIMIZER_FALLBACK,
    DEFAULT_OPTIMIZER_TOLERANCE,
    OptimizationResult as FinancialOptimizerModuleResult,
    optimize_global_minimum_variance as financial_optimizer_module_gmv,
)
from app.portfolio_engine import (
    OptimizationResult,
    optimize_global_minimum_variance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """Standard 2-asset covariance matrix with known analytical solution."""
    # sigma_1 = 0.20, sigma_2 = 0.30, rho = 0.25 -> cov_12 = 0.015
    return pd.DataFrame(
        [[0.04, 0.015], [0.015, 0.09]],
        index=["INFY", "TCS"],
        columns=["INFY", "TCS"],
    )


@pytest.fixture
def three_asset_cov() -> pd.DataFrame:
    """3-asset covariance matrix with distinct variances."""
    return pd.DataFrame(
        [
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.015],
            [0.005, 0.015, 0.16],
        ],
        index=["RELIANCE", "TCS", "HDFCBANK"],
        columns=["RELIANCE", "TCS", "HDFCBANK"],
    )


@pytest.fixture
def five_asset_cov() -> pd.DataFrame:
    """5-asset well-conditioned positive-definite covariance matrix."""
    rng = np.random.default_rng(42)
    A = rng.normal(0.0, 0.1, size=(5, 5))
    cov_arr = A @ A.T + 0.05 * np.eye(5)
    assets = ["NIFTY_A", "NIFTY_B", "NIFTY_C", "NIFTY_D", "NIFTY_E"]
    return pd.DataFrame(cov_arr, index=assets, columns=assets)


# ---------------------------------------------------------------------------
# 1. Analytical Closed-Form Parity Tests
# ---------------------------------------------------------------------------

class TestAnalyticalGMVParity:
    """Verify numerical optimizer matches analytical 2-asset closed-form solution."""

    def test_two_asset_analytical_solution(self, two_asset_cov: pd.DataFrame) -> None:
        """Verify w1* = (sigma_2^2 - sigma_12) / (sigma_1^2 + sigma_2^2 - 2*sigma_12)."""
        res = optimize_global_minimum_variance(two_asset_cov)

        s1_sq = two_asset_cov.iloc[0, 0]
        s2_sq = two_asset_cov.iloc[1, 1]
        s12 = two_asset_cov.iloc[0, 1]

        denom = s1_sq + s2_sq - 2.0 * s12
        w1_expected = (s2_sq - s12) / denom  # (0.09 - 0.015) / 0.10 = 0.75
        w2_expected = (s1_sq - s12) / denom  # (0.04 - 0.015) / 0.10 = 0.25
        var_expected = (
            w1_expected**2 * s1_sq
            + w2_expected**2 * s2_sq
            + 2.0 * w1_expected * w2_expected * s12
        )  # 0.03375
        vol_expected = np.sqrt(var_expected)

        assert res.success is True
        assert not res.fallback_executed
        np.testing.assert_allclose(res.weights["INFY"], w1_expected, atol=1e-6)
        np.testing.assert_allclose(res.weights["TCS"], w2_expected, atol=1e-6)
        np.testing.assert_allclose(res.variance, var_expected, atol=1e-6)
        np.testing.assert_allclose(res.volatility, vol_expected, atol=1e-6)
        np.testing.assert_allclose(res.objective_value, 0.5 * var_expected, atol=1e-6)

    def test_two_asset_uncorrelated(self) -> None:
        """Verify uncorrelated 2 assets: w1* = s2^2 / (s1^2 + s2^2)."""
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.16]],
            index=["ASSET1", "ASSET2"],
            columns=["ASSET1", "ASSET2"],
        )
        res = optimize_global_minimum_variance(cov)

        # w1 = 0.16 / 0.20 = 0.80, w2 = 0.04 / 0.20 = 0.20
        np.testing.assert_allclose(res.weights["ASSET1"], 0.80, atol=1e-6)
        np.testing.assert_allclose(res.weights["ASSET2"], 0.20, atol=1e-6)
        expected_var = 0.80**2 * 0.04 + 0.20**2 * 0.16  # 0.0256 + 0.0064 = 0.032
        np.testing.assert_allclose(res.variance, expected_var, atol=1e-6)

    def test_identical_assets_equal_weights(self) -> None:
        """Verify identical assets yield exact equal weighting."""
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.01], [0.01, 0.04, 0.01], [0.01, 0.01, 0.04]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        res = optimize_global_minimum_variance(cov)
        assert res.success is True
        for asset in ["A", "B", "C"]:
            np.testing.assert_allclose(res.weights[asset], 1.0 / 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Long-Only Boundary & Clamping Tests
# ---------------------------------------------------------------------------

class TestLongOnlyConstraints:
    """Verify long-only constraint clamps unconstrained negative weights to 0.0."""

    def test_unconstrained_negative_weight_clamped_to_zero(self) -> None:
        """When unconstrained w2* < 0, long-only solution must clamp w2 = 0.0, w1 = 1.0."""
        # s1^2 = 0.01, s2^2 = 0.25, s12 = 0.03
        # Denom = 0.01 + 0.25 - 0.06 = 0.20
        # Unconstrained w1* = (0.25 - 0.03) / 0.20 = 1.10
        # Unconstrained w2* = (0.01 - 0.03) / 0.20 = -0.10
        cov = pd.DataFrame(
            [[0.01, 0.03], [0.03, 0.25]],
            index=["LOW_VOL", "HIGH_VOL"],
            columns=["LOW_VOL", "HIGH_VOL"],
        )
        res = optimize_global_minimum_variance(cov)

        assert res.success is True
        np.testing.assert_allclose(res.weights["LOW_VOL"], 1.0, atol=1e-6)
        np.testing.assert_allclose(res.weights["HIGH_VOL"], 0.0, atol=1e-6)
        np.testing.assert_allclose(res.variance, 0.01, atol=1e-6)

    def test_weights_sum_to_one_strictly(self, five_asset_cov: pd.DataFrame) -> None:
        """Verify sum(w) = 1.0 within 1e-6 tolerance across multi-asset universes."""
        res = optimize_global_minimum_variance(five_asset_cov)
        assert res.success is True
        assert abs(float(res.weights.sum()) - 1.0) <= 1e-6
        assert (res.weights >= 0.0).all()
        assert (res.weights <= 1.0).all()

    def test_three_asset_dominated_asset_zero_weight(self) -> None:
        """An asset with high variance and high correlation to lower variance assets gets 0 weight."""
        cov = pd.DataFrame(
            [
                [0.01, 0.005, 0.05],
                [0.005, 0.02, 0.06],
                [0.05, 0.06, 0.50],
            ],
            index=["A1", "A2", "DOMINATED"],
            columns=["A1", "A2", "DOMINATED"],
        )
        res = optimize_global_minimum_variance(cov)
        assert res.success is True
        np.testing.assert_allclose(res.weights["DOMINATED"], 0.0, atol=1e-5)
        assert abs(float(res.weights.sum()) - 1.0) <= 1e-6


# ---------------------------------------------------------------------------
# 3. User-Provided Initial Weights (w0) Tests
# ---------------------------------------------------------------------------

class TestInitialWeights:
    """Verify validation and handling of custom initial weights."""

    def test_initial_weights_series_correct(self, two_asset_cov: pd.DataFrame) -> None:
        """Pass pd.Series initial weights matching covariance columns."""
        w0 = pd.Series([0.6, 0.4], index=["INFY", "TCS"])
        res = optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)
        assert res.success is True
        np.testing.assert_allclose(res.weights["INFY"], 0.75, atol=1e-6)

    def test_initial_weights_series_reordered(self, two_asset_cov: pd.DataFrame) -> None:
        """Pass pd.Series with reversed label order; should reorder and converge correctly."""
        w0 = pd.Series([0.4, 0.6], index=["TCS", "INFY"])
        res = optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)
        assert res.success is True
        np.testing.assert_allclose(res.weights["INFY"], 0.75, atol=1e-6)

    def test_initial_weights_numpy_array(self, two_asset_cov: pd.DataFrame) -> None:
        """Pass np.ndarray initial weights."""
        w0 = np.array([0.5, 0.5])
        res = optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)
        assert res.success is True
        np.testing.assert_allclose(res.weights["INFY"], 0.75, atol=1e-6)

    def test_initial_weights_list(self, two_asset_cov: pd.DataFrame) -> None:
        """Pass list of floats as initial weights."""
        res = optimize_global_minimum_variance(two_asset_cov, initial_weights=[0.3, 0.7])
        assert res.success is True
        np.testing.assert_allclose(res.weights["INFY"], 0.75, atol=1e-6)

    def test_initial_weights_length_mismatch(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights with wrong length."""
        with pytest.raises(FinancialCalculationError, match="initial_weights length"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[0.3, 0.3, 0.4])

    def test_initial_weights_sum_violation(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights not summing to 1.0."""
        with pytest.raises(FinancialCalculationError, match="must sum to 1.0"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[0.5, 0.6])

    def test_initial_weights_negative_entry(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights with negative entries."""
        with pytest.raises(FinancialCalculationError, match="long-only constraint"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[-0.2, 1.2])

    def test_initial_weights_greater_than_one(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights with entry > 1.0."""
        with pytest.raises(FinancialCalculationError, match="long-only constraint"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[1.1, -0.1])

    def test_initial_weights_nan_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights containing NaN."""
        with pytest.raises(FinancialCalculationError, match="NaN or non-finite"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[np.nan, 0.5])

    def test_initial_weights_infinite_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject initial weights containing Inf."""
        with pytest.raises(FinancialCalculationError, match="NaN or non-finite"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[np.inf, -np.inf])

    def test_initial_weights_series_label_mismatch(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject pd.Series with different asset names."""
        w0 = pd.Series([0.5, 0.5], index=["WRONG1", "WRONG2"])
        with pytest.raises(FinancialCalculationError, match="do not match covariance asset labels"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)

    def test_initial_weights_series_duplicate_labels(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject pd.Series with duplicate index labels."""
        w0 = pd.Series([0.5, 0.5], index=["INFY", "INFY"])
        with pytest.raises(FinancialCalculationError, match="duplicate index labels"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)

    def test_initial_weights_boolean_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject scalar boolean passed as initial_weights."""
        with pytest.raises(FinancialCalculationError, match="got bool"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=True)
        with pytest.raises(FinancialCalculationError, match="got bool"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=np.bool_(False))

    def test_initial_weights_sequence_boolean_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject Python list or tuple containing boolean values."""
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[True, False])
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=(False, True))

    def test_initial_weights_numpy_boolean_array_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject NumPy boolean array."""
        w0 = np.array([True, False], dtype=bool)
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0)

    def test_initial_weights_pandas_boolean_series_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject pandas Series with boolean or nullable boolean dtype."""
        w0_bool = pd.Series([True, False], index=["INFY", "TCS"])
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0_bool)

        w0_nullable = pd.Series([True, False], index=["INFY", "TCS"], dtype="boolean")
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0_nullable)

    def test_initial_weights_object_container_with_boolean_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject object array, sequence, or Series containing boolean elements."""
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=[True, 0.5])

        w0_obj_arr = np.array([True, 0.0], dtype=object)
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0_obj_arr)

        w0_obj_series = pd.Series([True, 0.0], index=["INFY", "TCS"], dtype=object)
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_global_minimum_variance(two_asset_cov, initial_weights=w0_obj_series)

    def test_initial_weights_integer_preserved(self, two_asset_cov: pd.DataFrame) -> None:
        """Accept legitimate numeric integer weights (e.g. [1, 0])."""
        # Python integer list
        res_list = optimize_global_minimum_variance(two_asset_cov, initial_weights=[1, 0])
        assert res_list.success is True
        np.testing.assert_allclose(res_list.weights["INFY"], 0.75, atol=1e-6)

        # Python integer tuple
        res_tuple = optimize_global_minimum_variance(two_asset_cov, initial_weights=(0, 1))
        assert res_tuple.success is True
        np.testing.assert_allclose(res_tuple.weights["INFY"], 0.75, atol=1e-6)

        # NumPy integer array
        res_np = optimize_global_minimum_variance(two_asset_cov, initial_weights=np.array([1, 0]))
        assert res_np.success is True
        np.testing.assert_allclose(res_np.weights["INFY"], 0.75, atol=1e-6)

        # pandas integer Series
        res_series = optimize_global_minimum_variance(
            two_asset_cov,
            initial_weights=pd.Series([1, 0], index=["INFY", "TCS"]),
        )
        assert res_series.success is True
        np.testing.assert_allclose(res_series.weights["INFY"], 0.75, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Input Covariance Validation Tests
# ---------------------------------------------------------------------------

class TestCovarianceValidation:
    """Verify strict rejection of invalid covariance DataFrames."""

    def test_none_covariance_rejected(self) -> None:
        """Reject None covariance."""
        with pytest.raises(FinancialCalculationError, match="got None"):
            optimize_global_minimum_variance(None)

    def test_non_dataframe_covariance_rejected(self) -> None:
        """Reject numpy array or dict."""
        with pytest.raises(FinancialCalculationError, match="must be a pandas DataFrame"):
            optimize_global_minimum_variance(np.eye(2))

    def test_empty_dataframe_rejected(self) -> None:
        """Reject empty DataFrame."""
        with pytest.raises(InsufficientDataError, match="empty"):
            optimize_global_minimum_variance(pd.DataFrame())

    def test_single_asset_dataframe_rejected(self) -> None:
        """Reject 1x1 covariance matrix."""
        df = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="at least 2 assets"):
            optimize_global_minimum_variance(df)

    def test_non_square_dataframe_rejected(self) -> None:
        """Reject rectangular DataFrame."""
        df = pd.DataFrame([[0.04, 0.01, 0.02], [0.01, 0.09, 0.03]], index=["A", "B"], columns=["A", "B", "C"])
        with pytest.raises(FinancialCalculationError, match="must be square"):
            optimize_global_minimum_variance(df)

    def test_duplicate_column_labels_rejected(self) -> None:
        """Reject DataFrame with duplicate column labels."""
        df = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "A"])
        with pytest.raises(FinancialCalculationError, match="duplicate column labels"):
            optimize_global_minimum_variance(df)

    def test_duplicate_index_labels_rejected(self) -> None:
        """Reject DataFrame with duplicate index labels."""
        df = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "A"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="duplicate index labels"):
            optimize_global_minimum_variance(df)

    def test_mismatched_index_and_columns_rejected(self) -> None:
        """Reject DataFrame where index and columns don't match."""
        df = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["B", "A"])
        with pytest.raises(FinancialCalculationError, match="must be identical and in the exact same order"):
            optimize_global_minimum_variance(df)

    def test_non_numeric_columns_rejected(self) -> None:
        """Reject string or object columns."""
        df = pd.DataFrame({"A": ["0.04", "0.01"], "B": ["0.01", "0.09"]}, index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            optimize_global_minimum_variance(df)

    def test_complex_columns_rejected(self) -> None:
        """Reject complex valued covariance."""
        df = pd.DataFrame(
            [[0.04 + 0j, 0.01 + 0j], [0.01 + 0j, 0.09 + 0j]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="complex"):
            optimize_global_minimum_variance(df)

    def test_nan_values_rejected(self) -> None:
        """Reject covariance containing NaN."""
        df = pd.DataFrame([[0.04, np.nan], [np.nan, 0.09]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="NaN or missing"):
            optimize_global_minimum_variance(df)

    def test_infinite_values_rejected(self) -> None:
        """Reject covariance containing Inf."""
        df = pd.DataFrame([[0.04, np.inf], [np.inf, 0.09]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="infinite"):
            optimize_global_minimum_variance(df)

    def test_zero_or_negative_diagonal_variance_rejected(self) -> None:
        """Reject non-positive diagonal elements."""
        df_zero = pd.DataFrame([[0.0, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            optimize_global_minimum_variance(df_zero)

        df_neg = pd.DataFrame([[-0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            optimize_global_minimum_variance(df_neg)

    def test_materially_asymmetric_matrix_rejected(self) -> None:
        """Reject matrix with asymmetry exceeding tolerance."""
        df = pd.DataFrame([[0.04, 0.02], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="materially asymmetric"):
            optimize_global_minimum_variance(df, input_symmetry_tolerance=1e-8)

    def test_mildly_asymmetric_matrix_accepted(self) -> None:
        """Accept matrix with tiny numerical asymmetry (< 1e-8)."""
        df = pd.DataFrame(
            [[0.04, 0.015 + 1e-10], [0.015, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        res = optimize_global_minimum_variance(df, input_symmetry_tolerance=1e-8)
        assert res.success is True


# ---------------------------------------------------------------------------
# 5. Spectral & Conditioning Tests
# ---------------------------------------------------------------------------

class TestSpectralChecks:
    """Verify positive definiteness and condition number enforcement."""

    def test_non_positive_definite_matrix_rejected(self) -> None:
        """Reject covariance with negative or zero eigenvalue."""
        # Highly correlated matrix with singular/negative eigenvalue:
        # C = [[1, 1.01], [1.01, 1]] -> eigenvalues: 2.01, -0.01
        cov = pd.DataFrame(
            [[0.04, 0.061], [0.061, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        with pytest.raises(NonPositiveDefiniteMatrixError, match="not strictly positive definite"):
            optimize_global_minimum_variance(cov)

    def test_singular_matrix_rejected(self) -> None:
        """Reject singular covariance matrix (det = 0, rank deficient)."""
        # Exactly collinear: rho = 1.0 -> sigma_12 = sqrt(0.04 * 0.09) = 0.06
        cov = pd.DataFrame(
            [[0.04, 0.06], [0.06, 0.09]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        with pytest.raises((NonPositiveDefiniteMatrixError, IllConditionedMatrixError)):
            optimize_global_minimum_variance(cov)

    def test_ill_conditioned_matrix_rejected(self) -> None:
        """Reject matrix with condition number exceeding max_condition_number."""
        # Matrix with eigenvalues [1e-7, 1.0] -> cond = 1e7 > default 1e6
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.diag([1.0, 1e-7])
        cov_arr = V @ D @ V.T
        cov = pd.DataFrame(cov_arr, index=["A", "B"], columns=["A", "B"])

        with pytest.raises(IllConditionedMatrixError, match="condition number.*exceeds maximum"):
            optimize_global_minimum_variance(cov, max_condition_number=1e6)

    def test_configurable_max_condition_number(self) -> None:
        """Allow ill-conditioned matrix when max_condition_number is explicitly relaxed."""
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.diag([1.0, 1e-7])
        cov_arr = V @ D @ V.T
        cov = pd.DataFrame(cov_arr, index=["A", "B"], columns=["A", "B"])

        # Relax condition number threshold to 1e8
        res = optimize_global_minimum_variance(cov, max_condition_number=1e8)
        assert res.success is True
        assert res.metadata["condition_number"] > 1e6


# ---------------------------------------------------------------------------
# 6. Parameter Validation Tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Verify strict numerical parameter validation."""

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, -1.0, 0.0, "string", True, False])
    def test_invalid_max_condition_number(self, two_asset_cov: pd.DataFrame, bad_val: object) -> None:
        """Reject invalid max_condition_number."""
        with pytest.raises(FinancialCalculationError, match="max_condition_number"):
            optimize_global_minimum_variance(two_asset_cov, max_condition_number=bad_val)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, -1.0, 0.0, "abc", True, False])
    def test_invalid_input_symmetry_tolerance(self, two_asset_cov: pd.DataFrame, bad_val: object) -> None:
        """Reject invalid input_symmetry_tolerance."""
        with pytest.raises(FinancialCalculationError, match="input_symmetry_tolerance"):
            optimize_global_minimum_variance(two_asset_cov, input_symmetry_tolerance=bad_val)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf, -1.0, 0.0, "fast", True, False])
    def test_invalid_tolerance(self, two_asset_cov: pd.DataFrame, bad_val: object) -> None:
        """Reject invalid tolerance."""
        with pytest.raises(FinancialCalculationError, match="tolerance"):
            optimize_global_minimum_variance(two_asset_cov, tolerance=bad_val)

    @pytest.mark.parametrize("bad_val", [-10, 0, 10.5, "100", True, False])
    def test_invalid_max_iterations(self, two_asset_cov: pd.DataFrame, bad_val: object) -> None:
        """Reject invalid max_iterations."""
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            optimize_global_minimum_variance(two_asset_cov, max_iterations=bad_val)

    def test_invalid_fallback_policy(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject unsupported fallback policy."""
        with pytest.raises(FinancialCalculationError, match="Unsupported fallback policy"):
            optimize_global_minimum_variance(two_asset_cov, fallback="INVALID_POLICY")


# ---------------------------------------------------------------------------
# 7. Fallback Policy Tests
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    """Verify optimizer fallback behavior on non-convergence."""

    def test_fallback_raise_exception(self, two_asset_cov: pd.DataFrame) -> None:
        """When max_iterations=1 fails to converge, raise OptimizationException."""
        # Extremely tight tolerance and max_iterations=1 to force non-convergence
        with pytest.raises(OptimizationException, match="SLSQP GMV optimization failed"):
            optimize_global_minimum_variance(
                two_asset_cov,
                max_iterations=1,
                tolerance=1e-20,
                fallback="RAISE_EXCEPTION",
            )

    def test_fallback_equal_weight(self, two_asset_cov: pd.DataFrame) -> None:
        """When max_iterations=1 fails to converge and fallback=EQUAL_WEIGHT, return equal weights."""
        res = optimize_global_minimum_variance(
            two_asset_cov,
            max_iterations=1,
            tolerance=1e-20,
            fallback="EQUAL_WEIGHT",
        )
        assert res.success is False
        assert res.fallback_executed is True
        np.testing.assert_allclose(res.weights["INFY"], 0.5, atol=1e-6)
        np.testing.assert_allclose(res.weights["TCS"], 0.5, atol=1e-6)
        assert "Fallback to EQUAL_WEIGHT" in res.message


# ---------------------------------------------------------------------------
# 8. OptimizationResult Dataclass & Serialization Tests
# ---------------------------------------------------------------------------

class TestOptimizationResultStructure:
    """Verify OptimizationResult dataclass attributes, immutability, and serialization."""

    def test_result_attributes(self, two_asset_cov: pd.DataFrame) -> None:
        """Verify all result attributes are correctly populated."""
        res = optimize_global_minimum_variance(two_asset_cov)

        assert isinstance(res.weights, pd.Series)
        assert list(res.weights.index) == ["INFY", "TCS"]
        assert isinstance(res.variance, float)
        assert isinstance(res.volatility, float)
        assert isinstance(res.objective_value, float)
        assert isinstance(res.success, bool)
        assert isinstance(res.status, int)
        assert isinstance(res.message, str)
        assert isinstance(res.iterations, int)
        assert isinstance(res.metadata, dict)
        assert res.fallback_executed is False

        # Metadata contents
        assert "condition_number" in res.metadata
        assert "min_eigenvalue" in res.metadata
        assert "max_eigenvalue" in res.metadata
        assert res.metadata["n_assets"] == 2
        assert res.metadata["method"] == "SLSQP"

    def test_result_immutability(self, two_asset_cov: pd.DataFrame) -> None:
        """OptimizationResult must be frozen."""
        res = optimize_global_minimum_variance(two_asset_cov)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.variance = 999.0  # type: ignore

    def test_to_dict_serialization(self, two_asset_cov: pd.DataFrame) -> None:
        """Verify to_dict() returns valid serializable dictionary."""
        res = optimize_global_minimum_variance(two_asset_cov)
        d = res.to_dict()

        assert isinstance(d, dict)
        assert d["weights"]["INFY"] == pytest.approx(0.75, abs=1e-6)
        assert d["weights"]["TCS"] == pytest.approx(0.25, abs=1e-6)
        assert isinstance(d["variance"], float)
        assert isinstance(d["volatility"], float)
        assert d["success"] is True
        assert d["fallback_executed"] is False


# ---------------------------------------------------------------------------
# 9. Package & Namespace Re-Export Tests
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package re-exports from both portfolio_engine and financial_engine."""

    def test_portfolio_engine_exports(self) -> None:
        """Verify portfolio_engine exports OptimizationResult and optimize_global_minimum_variance."""
        import app.portfolio_engine as pe
        assert hasattr(pe, "OptimizationResult")
        assert hasattr(pe, "optimize_global_minimum_variance")
        assert pe.OptimizationResult is OptimizationResult
        assert pe.optimize_global_minimum_variance is optimize_global_minimum_variance

    def test_financial_engine_exports(self) -> None:
        """Verify financial_engine exports match portfolio_engine."""
        import app.financial_engine as fe
        assert hasattr(fe, "OptimizationResult")
        assert hasattr(fe, "optimize_global_minimum_variance")
        assert fe.OptimizationResult is OptimizationResult
        assert fe.optimize_global_minimum_variance is optimize_global_minimum_variance

    def test_financial_engine_optimizer_module(self) -> None:
        """Verify financial_engine.optimizer re-export module."""
        import app.financial_engine.optimizer as feo
        assert hasattr(feo, "OptimizationResult")
        assert hasattr(feo, "optimize_global_minimum_variance")
        assert feo.OptimizationResult is OptimizationResult
        assert feo.optimize_global_minimum_variance is optimize_global_minimum_variance
        assert feo.DEFAULT_INPUT_SYMMETRY_TOLERANCE == DEFAULT_INPUT_SYMMETRY_TOLERANCE
        assert feo.DEFAULT_MAX_CONDITION_NUMBER == DEFAULT_MAX_CONDITION_NUMBER
