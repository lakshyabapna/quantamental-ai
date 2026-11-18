"""Quantamental AI - Unit Tests for Maximum Sharpe Ratio (Tangency) Optimization.

Tests Phase 2.3 Step 2: Maximum Sharpe Ratio / Tangency Portfolio Optimizer.
Validates:
    1. Analytical closed-form parity for 2-asset correlated and uncorrelated portfolios.
    2. Analytical gradient verification against independent numerical finite-difference approximation.
    3. Long-only boundary constraints (clamping negative unconstrained weights to 0.0).
    4. Exact budget constraint satisfaction (|sum(w) - 1.0| <= 1e-6) and non-negativity (0 <= w_i <= 1).
    5. User-supplied initial weights (pd.Series, np.ndarray, list, label reordering, invalid inputs).
    6. Strict input validation for expected_returns (type, length, duplicate labels, label alignment,
       non-numeric, complex, NaN, Inf, booleans).
    7. Strict input validation for risk_free_rate (type, finiteness, strict boolean rejection).
    8. Strict covariance matrix validation (shape, labels, dtypes, positive diagonal, PD, conditioning).
    9. Parameter validation (max_condition_number, input_symmetry_tolerance, tolerance, max_iterations, fallback).
    10. Solver non-convergence and fallback policies (RAISE_EXCEPTION, EQUAL_WEIGHT).
    11. TangencyOptimizationResult dataclass attributes, immutability, to_dict() serialization, and package re-exports.
"""

from __future__ import annotations

import dataclasses
import functools
import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from app.core.exceptions import (
    FinancialCalculationError,
    IllConditionedMatrixError,
    InsufficientDataError,
    NonPositiveDefiniteMatrixError,
    OptimizationException,
)
from app.financial_engine import (
    TangencyOptimizationResult as FinancialTangencyOptimizationResult,
    optimize_maximum_sharpe_ratio as financial_optimize_maximum_sharpe_ratio,
    optimize_tangency_portfolio as financial_optimize_tangency_portfolio,
)
from app.financial_engine.tangency import (
    TangencyOptimizationResult as FinancialTangencyModuleResult,
    optimize_maximum_sharpe_ratio as financial_module_optimize_maximum_sharpe_ratio,
    optimize_tangency_portfolio as financial_module_optimize_tangency_portfolio,
)
from app.portfolio_engine import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
    optimize_tangency_portfolio,
)
from app.portfolio_engine.tangency import (
    _negative_sharpe_gradient,
    _negative_sharpe_objective,
)


# ---------------------------------------------------------------------------
# Test-local reference: independent negative Sharpe objective for finite
# differences.  Computed directly from the mathematical definition so that
# finite-difference comparisons are independent of the production objective
# helper.  NEVER import or call _negative_sharpe_objective below this line
# inside the gradient tests; use _ref_neg_sharpe_fd() instead.
# ---------------------------------------------------------------------------

def _ref_neg_sharpe_fd(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> float:
    """Test-local reference: f(w) = -((w @ mu - rf) / sqrt(w @ cov @ w)).

    Implemented from the mathematical definition independently of the
    production _negative_sharpe_objective helper so that finite-difference
    gradient comparisons are not compromised by a shared implementation error.
    """
    portfolio_variance = float(w @ cov @ w)
    portfolio_volatility = np.sqrt(portfolio_variance)
    excess_return = float(w @ mu) - rf
    return -(excess_return / portfolio_volatility)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """Standard 2-asset covariance matrix with known analytical solution."""
    return pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["ASSET1", "ASSET2"],
        columns=["ASSET1", "ASSET2"],
    )


@pytest.fixture
def two_asset_returns() -> pd.Series:
    """Standard 2-asset expected returns series."""
    return pd.Series([0.12, 0.15], index=["ASSET1", "ASSET2"])


@pytest.fixture
def three_asset_cov() -> pd.DataFrame:
    """3-asset positive-definite covariance matrix."""
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
def three_asset_returns() -> pd.Series:
    """3-asset expected returns series."""
    return pd.Series([0.14, 0.16, 0.18], index=["RELIANCE", "TCS", "HDFCBANK"])


# ---------------------------------------------------------------------------
# 1. Analytical Parity & Closed-Form Tests
# ---------------------------------------------------------------------------

class TestAnalyticalTangencyParity:
    """Verify numerical optimizer matches analytical 2-asset closed-form solution."""

    def test_two_asset_analytical_solution(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Verify 2-asset tangency portfolio matches w* = inv(Sigma) @ (mu - rf) normalized."""
        rf = 0.05
        res = optimize_maximum_sharpe_ratio(
            expected_returns=two_asset_returns,
            covariance=two_asset_cov,
            risk_free_rate=rf,
        )

        # Analytical unconstrained weights:
        # Sigma = [[0.04, 0.01], [0.01, 0.09]], det = 0.0035
        # mu - rf = [0.07, 0.10]
        # inv(Sigma) @ [0.07, 0.10] proportional to [53, 33]
        # w1* = 53 / 86 = 0.61627907, w2* = 33 / 86 = 0.38372093
        w1_expected = 53.0 / 86.0
        w2_expected = 33.0 / 86.0
        exp_ret_expected = w1_expected * 0.12 + w2_expected * 0.15
        var_expected = (
            w1_expected**2 * 0.04
            + w2_expected**2 * 0.09
            + 2.0 * w1_expected * w2_expected * 0.01
        )
        vol_expected = np.sqrt(var_expected)
        sharpe_expected = (exp_ret_expected - rf) / vol_expected

        assert res.success is True
        assert not res.fallback_executed
        np.testing.assert_allclose(res.weights["ASSET1"], w1_expected, atol=1e-5)
        np.testing.assert_allclose(res.weights["ASSET2"], w2_expected, atol=1e-5)
        np.testing.assert_allclose(res.expected_return, exp_ret_expected, atol=1e-5)
        np.testing.assert_allclose(res.variance, var_expected, atol=1e-5)
        np.testing.assert_allclose(res.volatility, vol_expected, atol=1e-5)
        np.testing.assert_allclose(res.sharpe_ratio, sharpe_expected, atol=1e-5)
        assert res.risk_free_rate == rf

    def test_two_asset_uncorrelated(self) -> None:
        """Verify uncorrelated 2 assets: w_i proportional to (mu_i - rf) / sigma_i^2."""
        cov = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.16]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        mu = pd.Series([0.10, 0.14], index=["A", "B"])
        rf = 0.06

        # e1 = 0.04, s1^2 = 0.04 -> w1_unnorm = 1.0
        # e2 = 0.08, s2^2 = 0.16 -> w2_unnorm = 0.5
        # w1* = 1.0 / 1.5 = 2/3, w2* = 0.5 / 1.5 = 1/3
        res = optimize_maximum_sharpe_ratio(mu, cov, risk_free_rate=rf)

        assert res.success is True
        np.testing.assert_allclose(res.weights["A"], 2.0 / 3.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["B"], 1.0 / 3.0, atol=1e-5)

    def test_optimize_tangency_portfolio_alias(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Confirm optimize_tangency_portfolio alias produces identical results."""
        res_alias = optimize_tangency_portfolio(two_asset_returns, two_asset_cov, risk_free_rate=0.05)
        res_direct = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=0.05)

        assert res_alias.success is True
        np.testing.assert_allclose(res_alias.weights.values, res_direct.weights.values)
        assert res_alias.sharpe_ratio == res_direct.sharpe_ratio


# ---------------------------------------------------------------------------
# 2. Analytical Gradient Verification Tests
# ---------------------------------------------------------------------------

class TestAnalyticalGradient:
    """Verify actual production analytical gradient against numerical central finite differences.

    Finite differences are computed using the test-local _ref_neg_sharpe_fd() helper,
    which re-implements the negative Sharpe objective independently from the mathematical
    definition.  This ensures the comparison is not compromised by a shared implementation
    error between the production objective and gradient helpers.
    """

    @pytest.mark.parametrize("w_test", [
        np.array([0.5, 0.5]),
        np.array([0.7, 0.3]),
        np.array([0.2, 0.8]),
    ])
    def test_production_gradient_finite_difference_2asset_correlated(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        w_test: np.ndarray,
    ) -> None:
        """Compare production _negative_sharpe_gradient against independent central finite differences (2-asset correlated)."""
        cov_arr = two_asset_cov.to_numpy()
        mu_arr = two_asset_returns.to_numpy()
        rf = 0.05

        exact = _negative_sharpe_gradient(w_test, mu_arr, cov_arr, rf)

        # Central differences computed via the independent test-local reference (eps = 1e-7)
        eps = 1e-7
        numerical = np.zeros_like(w_test)
        for i in range(len(w_test)):
            w_plus = w_test.copy()
            w_plus[i] += eps
            w_minus = w_test.copy()
            w_minus[i] -= eps
            numerical[i] = (
                _ref_neg_sharpe_fd(w_plus, mu_arr, cov_arr, rf)
                - _ref_neg_sharpe_fd(w_minus, mu_arr, cov_arr, rf)
            ) / (2.0 * eps)

        max_abs_err = float(np.max(np.abs(exact - numerical)))
        assert max_abs_err < 1e-7, (
            f"Production gradient deviates from independent finite-difference reference: "
            f"max |err| = {max_abs_err:.4e} (tolerance = 1e-7)"
        )
        np.testing.assert_allclose(exact, numerical, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("w_test", [
        np.array([0.5, 0.5]),
        np.array([0.8, 0.2]),
        np.array([0.3, 0.7]),
    ])
    def test_production_gradient_finite_difference_2asset_uncorrelated(
        self,
        w_test: np.ndarray,
    ) -> None:
        """Compare production _negative_sharpe_gradient against independent central finite differences (2-asset uncorrelated)."""
        cov_arr = np.array([[0.04, 0.0], [0.0, 0.16]])
        mu_arr = np.array([0.10, 0.14])
        rf = 0.06

        exact = _negative_sharpe_gradient(w_test, mu_arr, cov_arr, rf)

        eps = 1e-7
        numerical = np.zeros_like(w_test)
        for i in range(len(w_test)):
            w_plus = w_test.copy()
            w_plus[i] += eps
            w_minus = w_test.copy()
            w_minus[i] -= eps
            numerical[i] = (
                _ref_neg_sharpe_fd(w_plus, mu_arr, cov_arr, rf)
                - _ref_neg_sharpe_fd(w_minus, mu_arr, cov_arr, rf)
            ) / (2.0 * eps)

        max_abs_err = float(np.max(np.abs(exact - numerical)))
        assert max_abs_err < 1e-7, (
            f"Production gradient deviates from independent finite-difference reference: "
            f"max |err| = {max_abs_err:.4e} (tolerance = 1e-7)"
        )
        np.testing.assert_allclose(exact, numerical, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("w_test", [
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        np.array([0.6, 0.25, 0.15]),
        np.array([0.1, 0.2, 0.7]),
    ])
    def test_production_gradient_finite_difference_3asset(
        self,
        three_asset_cov: pd.DataFrame,
        three_asset_returns: pd.Series,
        w_test: np.ndarray,
    ) -> None:
        """Compare production _negative_sharpe_gradient against independent central finite differences (3-asset)."""
        cov_arr = three_asset_cov.to_numpy()
        mu_arr = three_asset_returns.to_numpy()
        rf = 0.065

        exact = _negative_sharpe_gradient(w_test, mu_arr, cov_arr, rf)

        eps = 1e-7
        numerical = np.zeros_like(w_test)
        for i in range(len(w_test)):
            w_plus = w_test.copy()
            w_plus[i] += eps
            w_minus = w_test.copy()
            w_minus[i] -= eps
            numerical[i] = (
                _ref_neg_sharpe_fd(w_plus, mu_arr, cov_arr, rf)
                - _ref_neg_sharpe_fd(w_minus, mu_arr, cov_arr, rf)
            ) / (2.0 * eps)

        max_abs_err = float(np.max(np.abs(exact - numerical)))
        assert max_abs_err < 1e-7, (
            f"Production gradient deviates from independent finite-difference reference: "
            f"max |err| = {max_abs_err:.4e} (tolerance = 1e-7)"
        )
        np.testing.assert_allclose(exact, numerical, atol=1e-5, rtol=1e-4)

    def test_gradient_passed_to_slsqp(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that optimize_maximum_sharpe_ratio supplies the production analytical gradient to SciPy."""
        import app.portfolio_engine.tangency as tang_module

        intercepted = {}
        original_minimize = scipy.optimize.minimize

        def spy_minimize(*args, **kwargs):
            intercepted["jac"] = kwargs.get("jac")
            intercepted["fun"] = kwargs.get("fun")
            return original_minimize(*args, **kwargs)

        monkeypatch.setattr(tang_module.scipy.optimize, "minimize", spy_minimize)

        rf = 0.05
        res = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=rf)

        # 1. Verify a callable analytical Jacobian is supplied
        passed_jac = intercepted.get("jac")
        assert callable(passed_jac), "Optimizer did not supply a callable Jacobian to SciPy minimize."

        # 2. Verify it wraps/is the production gradient helper
        assert isinstance(passed_jac, functools.partial), "Passed Jacobian is not a partial wrapper."
        assert passed_jac.func is _negative_sharpe_gradient, "Passed Jacobian does not wrap _negative_sharpe_gradient."

        # 3. Verify Jacobian output matches the production gradient directly
        test_w = np.array([0.5, 0.5])
        cov_arr = two_asset_cov.to_numpy()
        mu_arr = two_asset_returns.to_numpy()

        jac_output = passed_jac(test_w)
        direct_output = _negative_sharpe_gradient(test_w, mu_arr, cov_arr, rf)
        np.testing.assert_allclose(jac_output, direct_output)

        # 4. Verify Jacobian output also matches the independent finite-difference reference
        eps = 1e-7
        fd_numerical = np.zeros_like(test_w)
        for i in range(len(test_w)):
            w_plus = test_w.copy()
            w_plus[i] += eps
            w_minus = test_w.copy()
            w_minus[i] -= eps
            fd_numerical[i] = (
                _ref_neg_sharpe_fd(w_plus, mu_arr, cov_arr, rf)
                - _ref_neg_sharpe_fd(w_minus, mu_arr, cov_arr, rf)
            ) / (2.0 * eps)

        max_jac_fd_err = float(np.max(np.abs(jac_output - fd_numerical)))
        assert max_jac_fd_err < 1e-7, (
            f"Production Jacobian deviates from independent finite-difference reference: "
            f"max |err| = {max_jac_fd_err:.4e}"
        )
        np.testing.assert_allclose(jac_output, fd_numerical, atol=1e-5, rtol=1e-4)
        assert res.success is True


# ---------------------------------------------------------------------------
# 3. Long-Only Boundary & Clamping Tests
# ---------------------------------------------------------------------------

class TestLongOnlyConstraints:
    """Verify long-only constraint clamps unconstrained negative weights to 0.0."""

    def test_unconstrained_negative_weight_clamped_to_zero(self) -> None:
        """When unconstrained tangency w2* < 0, long-only solution must clamp w2 = 0.0, w1 = 1.0."""
        # Asset 1: low risk, good return -> Sharpe = (0.12 - 0.05) / 0.10 = 0.70
        # Asset 2: high risk, poor return -> Sharpe = (0.06 - 0.05) / 0.30 = 0.033
        # High correlation forces short position on Asset 2 if unconstrained
        cov = pd.DataFrame(
            [[0.01, 0.02], [0.02, 0.09]],
            index=["GOOD", "BAD"],
            columns=["GOOD", "BAD"],
        )
        mu = pd.Series([0.12, 0.06], index=["GOOD", "BAD"])
        res = optimize_maximum_sharpe_ratio(mu, cov, risk_free_rate=0.05)

        assert res.success is True
        np.testing.assert_allclose(res.weights["GOOD"], 1.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["BAD"], 0.0, atol=1e-5)
        np.testing.assert_allclose(res.variance, 0.01, atol=1e-5)
        np.testing.assert_allclose(res.volatility, 0.10, atol=1e-5)

    def test_weights_sum_to_one_strictly(
        self,
        three_asset_cov: pd.DataFrame,
        three_asset_returns: pd.Series,
    ) -> None:
        """Verify sum(w) = 1.0 within 1e-6 tolerance across 3 assets."""
        res = optimize_maximum_sharpe_ratio(three_asset_returns, three_asset_cov, risk_free_rate=0.06)
        assert res.success is True
        assert abs(float(res.weights.sum()) - 1.0) <= 1e-6
        assert (res.weights >= 0.0).all()
        assert (res.weights <= 1.0).all()


# ---------------------------------------------------------------------------
# 4. Expected Returns Validation Tests
# ---------------------------------------------------------------------------

class TestExpectedReturnsValidation:
    """Verify strict validation of expected_returns input Series."""

    def test_none_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject None expected_returns."""
        with pytest.raises(FinancialCalculationError, match="must be a pandas Series, got None"):
            optimize_maximum_sharpe_ratio(None, two_asset_cov)

    def test_non_series_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject list or numpy array for expected_returns."""
        with pytest.raises(FinancialCalculationError, match="must be a pandas Series"):
            optimize_maximum_sharpe_ratio([0.12, 0.15], two_asset_cov)

    def test_empty_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject empty expected_returns Series."""
        with pytest.raises(InsufficientDataError, match="empty"):
            optimize_maximum_sharpe_ratio(pd.Series(dtype=float), two_asset_cov)

    def test_single_asset_expected_returns_rejected(self) -> None:
        """Reject 1-asset expected_returns."""
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="at least 2 assets"):
            optimize_maximum_sharpe_ratio(pd.Series([0.10], index=["A"]), cov)

    def test_duplicate_labels_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject expected_returns with duplicate asset labels."""
        mu = pd.Series([0.12, 0.15], index=["ASSET1", "ASSET1"])
        with pytest.raises(FinancialCalculationError, match="duplicate asset labels"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_label_mismatch_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject expected_returns with labels that don't match covariance."""
        mu = pd.Series([0.12, 0.15], index=["WRONG1", "WRONG2"])
        with pytest.raises(FinancialCalculationError, match="do not match covariance asset labels"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_label_reordering_aligned_properly(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Verify expected_returns in reverse order is properly aligned to covariance."""
        mu_reversed = pd.Series([0.15, 0.12], index=["ASSET2", "ASSET1"])
        res = optimize_maximum_sharpe_ratio(mu_reversed, two_asset_cov, risk_free_rate=0.05)
        assert res.success is True
        np.testing.assert_allclose(res.weights["ASSET1"], 53.0 / 86.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["ASSET2"], 33.0 / 86.0, atol=1e-5)

    def test_non_numeric_dtype_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject string-dtype expected_returns."""
        mu = pd.Series(["0.12", "0.15"], index=["ASSET1", "ASSET2"])
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_complex_dtype_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject complex expected_returns."""
        mu = pd.Series([0.12 + 0j, 0.15 + 0j], index=["ASSET1", "ASSET2"])
        with pytest.raises(FinancialCalculationError, match="complex"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_boolean_dtype_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject boolean expected_returns Series."""
        mu = pd.Series([True, False], index=["ASSET1", "ASSET2"])
        with pytest.raises(FinancialCalculationError, match="boolean"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_object_with_boolean_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject object Series containing boolean element."""
        mu = pd.Series([True, 0.15], index=["ASSET1", "ASSET2"], dtype=object)
        with pytest.raises(FinancialCalculationError, match="boolean"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_nan_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject expected_returns containing NaN."""
        mu = pd.Series([0.12, np.nan], index=["ASSET1", "ASSET2"])
        with pytest.raises(FinancialCalculationError, match="NaN or missing"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)

    def test_inf_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        """Reject expected_returns containing Inf."""
        mu = pd.Series([0.12, np.inf], index=["ASSET1", "ASSET2"])
        with pytest.raises(FinancialCalculationError, match="infinite"):
            optimize_maximum_sharpe_ratio(mu, two_asset_cov)


# ---------------------------------------------------------------------------
# 5. Risk-Free Rate Validation Tests
# ---------------------------------------------------------------------------

class TestRiskFreeRateValidation:
    """Verify strict validation of risk_free_rate."""

    @pytest.mark.parametrize("bad_rf", [True, False, np.bool_(True), np.bool_(False)])
    def test_boolean_rf_rejected(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        bad_rf: object,
    ) -> None:
        """Strictly reject boolean risk_free_rate."""
        with pytest.raises(FinancialCalculationError, match="got bool"):
            optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=bad_rf)

    @pytest.mark.parametrize("bad_rf", [np.nan, np.inf, -np.inf, "0.05", None, [0.05]])
    def test_invalid_rf_rejected(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        bad_rf: object,
    ) -> None:
        """Reject non-numeric or non-finite risk_free_rate."""
        with pytest.raises(FinancialCalculationError, match="risk_free_rate"):
            optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=bad_rf)

    def test_integer_rf_accepted(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Accept integer zero or non-zero risk-free rate."""
        res = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=0)
        assert res.success is True
        assert res.risk_free_rate == 0.0

    @pytest.mark.parametrize("valid_scalar", [
        np.float32(0.05),
        np.float64(0.065),
        np.int32(0),
        np.int64(0),
        np.int32(1),
        np.float32(0.0),
    ])
    def test_numpy_real_scalars_accepted(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        valid_scalar: object,
    ) -> None:
        """Accept legitimate NumPy real numeric scalar types."""
        res = optimize_maximum_sharpe_ratio(
            two_asset_returns,
            two_asset_cov,
            risk_free_rate=valid_scalar,
        )
        assert res.success is True
        assert isinstance(res.risk_free_rate, float)
        assert res.risk_free_rate == float(valid_scalar)

    @pytest.mark.parametrize("complex_rf", [
        0.05 + 0.01j,
        complex(0.05, 0.0),
        np.complex64(0.05 + 0.01j),
        np.complex128(0.05 + 0.01j),
    ])
    def test_complex_rf_rejected(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
        complex_rf: object,
    ) -> None:
        """Strictly reject Python and NumPy complex scalar types."""
        with pytest.raises(FinancialCalculationError, match="real numeric value, got complex"):
            optimize_maximum_sharpe_ratio(
                two_asset_returns,
                two_asset_cov,
                risk_free_rate=complex_rf,
            )


# ---------------------------------------------------------------------------
# 6. User-Provided Initial Weights (w0) Tests
# ---------------------------------------------------------------------------

class TestInitialWeights:
    """Verify initial weights validation in tangency optimizer."""

    def test_valid_custom_initial_weights(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Accept custom initial weights and converge properly."""
        w0 = [0.8, 0.2]
        res = optimize_maximum_sharpe_ratio(
            two_asset_returns,
            two_asset_cov,
            risk_free_rate=0.05,
            initial_weights=w0,
        )
        assert res.success is True
        np.testing.assert_allclose(res.weights["ASSET1"], 53.0 / 86.0, atol=5e-5)

    def test_initial_weights_boolean_rejected(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Reject boolean elements in initial_weights."""
        with pytest.raises(FinancialCalculationError, match="contains boolean values"):
            optimize_maximum_sharpe_ratio(
                two_asset_returns,
                two_asset_cov,
                initial_weights=[True, False],
            )

    def test_initial_weights_sum_violation_rejected(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Reject initial_weights that do not sum to 1.0."""
        with pytest.raises(FinancialCalculationError, match="must sum to 1.0"):
            optimize_maximum_sharpe_ratio(
                two_asset_returns,
                two_asset_cov,
                initial_weights=[0.5, 0.8],
            )


# ---------------------------------------------------------------------------
# 7. Covariance Matrix Validation & Spectral Tests
# ---------------------------------------------------------------------------

class TestCovarianceValidation:
    """Verify that tangency optimizer enforces full covariance validation."""

    def test_asymmetric_covariance_rejected(
        self,
        two_asset_returns: pd.Series,
    ) -> None:
        """Reject materially asymmetric covariance."""
        cov = pd.DataFrame(
            [[0.04, 0.02], [0.01, 0.09]],
            index=["ASSET1", "ASSET2"],
            columns=["ASSET1", "ASSET2"],
        )
        with pytest.raises(FinancialCalculationError, match="materially asymmetric"):
            optimize_maximum_sharpe_ratio(two_asset_returns, cov)

    def test_non_positive_definite_covariance_rejected(
        self,
        two_asset_returns: pd.Series,
    ) -> None:
        """Reject non-positive definite covariance."""
        cov = pd.DataFrame(
            [[0.04, 0.061], [0.061, 0.09]],
            index=["ASSET1", "ASSET2"],
            columns=["ASSET1", "ASSET2"],
        )
        with pytest.raises(NonPositiveDefiniteMatrixError, match="not strictly positive definite"):
            optimize_maximum_sharpe_ratio(two_asset_returns, cov)

    def test_ill_conditioned_covariance_rejected(
        self,
        two_asset_returns: pd.Series,
    ) -> None:
        """Reject ill-conditioned covariance exceeding condition number threshold."""
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.diag([1.0, 1e-7])
        cov_arr = V @ D @ V.T
        cov = pd.DataFrame(cov_arr, index=["ASSET1", "ASSET2"], columns=["ASSET1", "ASSET2"])

        with pytest.raises(IllConditionedMatrixError, match="condition number.*exceeds maximum"):
            optimize_maximum_sharpe_ratio(two_asset_returns, cov, max_condition_number=1e6)


# ---------------------------------------------------------------------------
# 8. Fallback Policy Tests
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    """Verify optimizer fallback behavior on non-convergence."""

    def test_fallback_raise_exception(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """When max_iterations=1 fails to converge, raise OptimizationException."""
        with pytest.raises(OptimizationException, match="SLSQP Tangency optimization failed"):
            optimize_maximum_sharpe_ratio(
                two_asset_returns,
                two_asset_cov,
                max_iterations=1,
                tolerance=1e-20,
                fallback="RAISE_EXCEPTION",
            )

    def test_fallback_equal_weight(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """When max_iterations=1 fails to converge and fallback=EQUAL_WEIGHT, return equal weights."""
        res = optimize_maximum_sharpe_ratio(
            two_asset_returns,
            two_asset_cov,
            max_iterations=1,
            tolerance=1e-20,
            fallback="EQUAL_WEIGHT",
        )
        assert res.success is False
        assert res.fallback_executed is True
        np.testing.assert_allclose(res.weights["ASSET1"], 0.5, atol=1e-6)
        np.testing.assert_allclose(res.weights["ASSET2"], 0.5, atol=1e-6)
        assert "Fallback to EQUAL_WEIGHT" in res.message


# ---------------------------------------------------------------------------
# 9. TangencyOptimizationResult Structure & Serialization Tests
# ---------------------------------------------------------------------------

class TestTangencyOptimizationResultStructure:
    """Verify TangencyOptimizationResult dataclass attributes, immutability, and serialization."""

    def test_result_attributes(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Verify all result attributes are correctly populated."""
        res = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=0.05)

        assert isinstance(res.weights, pd.Series)
        assert list(res.weights.index) == ["ASSET1", "ASSET2"]
        assert isinstance(res.expected_return, float)
        assert isinstance(res.variance, float)
        assert isinstance(res.volatility, float)
        assert isinstance(res.risk_free_rate, float)
        assert isinstance(res.sharpe_ratio, float)
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

    def test_result_immutability(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """TangencyOptimizationResult must be frozen."""
        res = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.sharpe_ratio = 999.0  # type: ignore

    def test_to_dict_serialization(
        self,
        two_asset_cov: pd.DataFrame,
        two_asset_returns: pd.Series,
    ) -> None:
        """Verify to_dict() returns valid serializable dictionary."""
        res = optimize_maximum_sharpe_ratio(two_asset_returns, two_asset_cov, risk_free_rate=0.05)
        d = res.to_dict()

        assert isinstance(d, dict)
        assert "weights" in d
        assert d["weights"]["ASSET1"] == pytest.approx(53.0 / 86.0, abs=1e-5)
        assert d["weights"]["ASSET2"] == pytest.approx(33.0 / 86.0, abs=1e-5)
        assert isinstance(d["expected_return"], float)
        assert isinstance(d["variance"], float)
        assert isinstance(d["volatility"], float)
        assert d["risk_free_rate"] == 0.05
        assert isinstance(d["sharpe_ratio"], float)
        assert d["success"] is True
        assert d["fallback_executed"] is False


# ---------------------------------------------------------------------------
# 10. Package & Namespace Re-Export Tests
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package re-exports from both portfolio_engine and financial_engine."""

    def test_portfolio_engine_exports(self) -> None:
        """Verify portfolio_engine exports TangencyOptimizationResult and functions."""
        import app.portfolio_engine as pe
        assert hasattr(pe, "TangencyOptimizationResult")
        assert hasattr(pe, "optimize_maximum_sharpe_ratio")
        assert hasattr(pe, "optimize_tangency_portfolio")
        assert pe.TangencyOptimizationResult is TangencyOptimizationResult
        assert pe.optimize_maximum_sharpe_ratio is optimize_maximum_sharpe_ratio
        assert pe.optimize_tangency_portfolio is optimize_tangency_portfolio

    def test_financial_engine_exports(self) -> None:
        """Verify financial_engine exports match portfolio_engine."""
        import app.financial_engine as fe
        assert hasattr(fe, "TangencyOptimizationResult")
        assert hasattr(fe, "optimize_maximum_sharpe_ratio")
        assert hasattr(fe, "optimize_tangency_portfolio")
        assert fe.TangencyOptimizationResult is TangencyOptimizationResult
        assert fe.optimize_maximum_sharpe_ratio is optimize_maximum_sharpe_ratio
        assert fe.optimize_tangency_portfolio is optimize_tangency_portfolio

    def test_financial_engine_tangency_module(self) -> None:
        """Verify financial_engine.tangency re-export module."""
        import app.financial_engine.tangency as fet
        assert hasattr(fet, "TangencyOptimizationResult")
        assert hasattr(fet, "optimize_maximum_sharpe_ratio")
        assert hasattr(fet, "optimize_tangency_portfolio")
        assert fet.TangencyOptimizationResult is TangencyOptimizationResult
        assert fet.optimize_maximum_sharpe_ratio is optimize_maximum_sharpe_ratio
        assert fet.optimize_tangency_portfolio is optimize_tangency_portfolio
