"""Quantamental AI - PyPortfolioOpt Shadow Validator Test Suite.

Phase 2.4, Step 1: Comprehensive independent unit tests verifying shadow validation
for Global Minimum Variance (GMV), Maximum Sharpe Ratio (Tangency), and Efficient Frontier.

Testing Scope:
    1. Valid GMV comparison (DataFrame, NumPy, precomputed, auto-run).
    2. Valid Tangency comparison (metrics, weights, Sharpe ratio parity).
    3. Valid Efficient Frontier comparison (target returns, branch separation).
    4. Weight ordering and asset label alignment.
    5. Invalid dimensions, non-square matrices, mismatched labels.
    6. Non-finite, complex, and boolean inputs.
    7. Solver failures and reference fallback handling.
    8. Equivalent optimum detection (collinear assets with flat optima).
    9. Numerical tolerance boundaries.
    10. Immutability and JSON serialization.
    11. Cross-namespace exports and identities.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    OptimizationException,
)
from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    FrontierPortfolio,
    compute_efficient_frontier,
)
from app.portfolio_engine.optimizer import (
    OptimizationResult,
    optimize_global_minimum_variance,
)
from app.portfolio_engine.shadow_validator import (
    DEFAULT_VALIDATION_TOLERANCES,
    FrontierPointValidation,
    FrontierValidationResult,
    GMVValidationResult,
    PortfolioMetrics,
    PyPortfolioOptShadowValidator,
    ShadowValidator,
    TangencyValidationResult,
    ValidationDifferences,
    ValidationStatus,
    ValidationTolerances,
    validate_efficient_frontier,
    validate_gmv,
    validate_tangency,
)
from app.portfolio_engine.tangency import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
)
import app.portfolio_engine as pe
import app.financial_engine as fe
import app.financial_engine.shadow_validator as fesv


# ---------------------------------------------------------------------------
# Test Fixtures (Independently Constructed Reference Data)
# ---------------------------------------------------------------------------

@pytest.fixture
def independent_3asset_data() -> Dict[str, Any]:
    """3-asset annualized expected returns and covariance matrix."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    mu = pd.Series([0.15, 0.20, 0.12], index=tickers)
    cov = pd.DataFrame(
        [
            [0.04, 0.01, 0.015],
            [0.01, 0.05, 0.02],
            [0.015, 0.02, 0.03],
        ],
        index=tickers,
        columns=tickers,
    )
    return {"tickers": tickers, "mu": mu, "cov": cov, "rf": 0.04}


@pytest.fixture
def independent_4asset_data() -> Dict[str, Any]:
    """4-asset annualized expected returns and well-conditioned covariance matrix."""
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]
    mu = pd.Series([0.10, 0.14, 0.18, 0.08], index=tickers)
    # Strictly positive definite covariance
    base_cov = np.array(
        [
            [0.035, 0.008, 0.012, 0.005],
            [0.008, 0.045, 0.015, 0.009],
            [0.012, 0.015, 0.060, 0.011],
            [0.005, 0.009, 0.011, 0.025],
        ]
    )
    cov = pd.DataFrame(base_cov, index=tickers, columns=tickers)
    return {"tickers": tickers, "mu": mu, "cov": cov, "rf": 0.03}


# ---------------------------------------------------------------------------
# 1. Valid GMV Comparison
# ---------------------------------------------------------------------------

class TestValidGMVComparison:
    """Verify GMV validation against PyPortfolioOpt shadow solver."""

    def test_gmv_validation_with_dataframe_covariance(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """GMV validation with pandas DataFrame passes metric and weight parity."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = validate_gmv(covariance=cov, expected_returns=mu)

        assert isinstance(res, GMVValidationResult)
        assert res.is_valid is True
        assert res.status in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)
        assert res.metric_valid is True
        assert res.reference_metrics is not None
        assert res.shadow_metrics is not None
        assert res.differences is not None

        # Verify variance and volatility parity within 1e-4
        assert res.differences.abs_diff_variance < 1e-4
        assert res.differences.abs_diff_volatility < 1e-4
        assert res.differences.abs_diff_return is not None
        assert res.differences.abs_diff_return < 1e-3

        # Weights sum to 1.0
        assert res.shadow_metrics.weights.sum() == pytest.approx(1.0, abs=1e-5)
        assert res.reference_metrics.weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_gmv_validation_with_numpy_array(
        self, independent_4asset_data: Dict[str, Any]
    ) -> None:
        """GMV validation accepts 2D NumPy array covariance."""
        cov_np = independent_4asset_data["cov"].values
        res = validate_gmv(covariance=cov_np)

        assert res.is_valid is True
        assert res.metric_valid is True
        assert res.differences.abs_diff_variance < 1e-4

    def test_gmv_validation_with_precomputed_reference_result(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Passing precomputed OptimizationResult avoids re-running production optimizer."""
        cov = independent_3asset_data["cov"]
        ref_opt = optimize_global_minimum_variance(covariance=cov)

        res = validate_gmv(covariance=cov, reference_result=ref_opt)
        assert res.is_valid is True
        assert res.reference_metrics.variance == pytest.approx(ref_opt.variance)
        assert res.reference_metrics.volatility == pytest.approx(ref_opt.volatility)

    def test_object_oriented_shadow_validator_gmv(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """PyPortfolioOptShadowValidator class instance performs GMV validation."""
        validator = PyPortfolioOptShadowValidator()
        res = validator.validate_gmv(independent_3asset_data["cov"])
        assert res.is_valid is True
        assert res.metric_valid is True


# ---------------------------------------------------------------------------
# 2. Valid Tangency Comparison
# ---------------------------------------------------------------------------

class TestValidTangencyComparison:
    """Verify Maximum Sharpe Ratio (Tangency) validation against PyPortfolioOpt."""

    def test_tangency_validation_metric_and_sharpe_parity(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Tangency validation verifies return, volatility, Sharpe ratio, and weights."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        rf = independent_3asset_data["rf"]

        res = validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=rf)

        assert isinstance(res, TangencyValidationResult)
        assert res.is_valid is True
        assert res.status in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)
        assert res.metric_valid is True
        assert res.risk_free_rate == pytest.approx(rf)

        # Parity checks
        assert res.differences.abs_diff_sharpe < 1e-3
        assert res.differences.abs_diff_return < 1e-3
        assert res.differences.abs_diff_volatility < 1e-3
        assert res.differences.max_weight_diff < 1e-3

    def test_tangency_validation_with_precomputed_result(
        self, independent_4asset_data: Dict[str, Any]
    ) -> None:
        """Tangency validation with precomputed TangencyOptimizationResult."""
        cov = independent_4asset_data["cov"]
        mu = independent_4asset_data["mu"]
        rf = independent_4asset_data["rf"]

        ref_tang = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        res = validate_tangency(
            expected_returns=mu,
            covariance=cov,
            risk_free_rate=rf,
            reference_result=ref_tang,
        )

        assert res.is_valid is True
        assert res.reference_metrics.sharpe_ratio == pytest.approx(ref_tang.sharpe_ratio)

    def test_object_oriented_shadow_validator_tangency(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """PyPortfolioOptShadowValidator class instance performs Tangency validation."""
        validator = PyPortfolioOptShadowValidator()
        res = validator.validate_tangency(
            expected_returns=independent_3asset_data["mu"],
            covariance=independent_3asset_data["cov"],
            risk_free_rate=independent_3asset_data["rf"],
        )
        assert res.is_valid is True


# ---------------------------------------------------------------------------
# 3. Valid Efficient Frontier Comparison
# ---------------------------------------------------------------------------

class TestValidEfficientFrontierComparison:
    """Verify Efficient Frontier target-return portfolio validation."""

    def test_efficient_frontier_validation_selected_targets(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Selected target returns on the efficient branch match PyPortfolioOpt."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        target_returns = [0.15, 0.16, 0.17, 0.18]
        res = validate_efficient_frontier(
            expected_returns=mu,
            covariance=cov,
            target_returns=target_returns,
        )

        assert isinstance(res, FrontierValidationResult)
        assert res.is_valid is True
        assert res.n_points_validated == len(target_returns)
        assert res.n_points_passed == len(target_returns)

        for pt in res.validated_points:
            assert pt.is_valid is True
            assert pt.differences.abs_diff_volatility < 1e-4

    def test_efficient_frontier_inefficient_branch_handling(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Targets strictly below GMV are explicitly warned/skipped because PyPfOpt solves ret >= target."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        # GMV return is ~0.147. A target of 0.12 is on the lower inefficient branch.
        res = validate_efficient_frontier(
            expected_returns=mu,
            covariance=cov,
            target_returns=[0.12, 0.16],
        )

        # 0.12 should be skipped with a warning; 0.16 should be validated and pass
        assert res.is_valid is True
        assert res.n_points_validated == 1
        assert res.validated_points[0].target_return == 0.16
        assert any("strictly below GMV" in w for w in res.warnings)

    def test_object_oriented_shadow_validator_frontier(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """PyPortfolioOptShadowValidator class instance performs Frontier validation."""
        validator = PyPortfolioOptShadowValidator()
        res = validator.validate_efficient_frontier(
            expected_returns=independent_3asset_data["mu"],
            covariance=independent_3asset_data["cov"],
            target_returns=[0.15, 0.17],
        )
        assert res.is_valid is True


# ---------------------------------------------------------------------------
# 4. Weight Ordering & Asset Label Alignment
# ---------------------------------------------------------------------------

class TestWeightOrderingAndAlignment:
    """Verify robust asset label alignment and rejection of missing/mismatched labels."""

    def test_differently_ordered_expected_returns_series_aligned_correctly(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Expected returns indexed in reverse order from covariance are aligned without error."""
        cov = independent_3asset_data["cov"]  # ["AAPL", "MSFT", "GOOG"]
        shuffled_mu = independent_3asset_data["mu"][["GOOG", "AAPL", "MSFT"]]

        res = validate_tangency(
            expected_returns=shuffled_mu,
            covariance=cov,
            risk_free_rate=0.04,
        )
        assert res.is_valid is True
        # Verify alignment: shadow weights index matches covariance index
        assert list(res.shadow_metrics.weights.index) == list(cov.index)
        assert list(res.reference_metrics.weights.index) == list(cov.index)

    def test_reference_result_with_differently_ordered_weights_aligned(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Reference result with permuted weight labels is aligned before computing differences."""
        cov = independent_3asset_data["cov"]
        ref_opt = optimize_global_minimum_variance(covariance=cov)

        # Permute reference weights
        shuffled_weights = ref_opt.weights[["MSFT", "GOOG", "AAPL"]]
        shuffled_ref_opt = OptimizationResult(
            weights=shuffled_weights,
            variance=ref_opt.variance,
            volatility=ref_opt.volatility,
            objective_value=ref_opt.objective_value,
            success=ref_opt.success,
            status=ref_opt.status,
            message=ref_opt.message,
            iterations=ref_opt.iterations,
            metadata=ref_opt.metadata,
        )

        res = validate_gmv(covariance=cov, reference_result=shuffled_ref_opt)
        assert res.is_valid is True
        assert list(res.differences.weight_differences.index) == list(cov.index)


# ---------------------------------------------------------------------------
# 5. Invalid Dimensions and Asset Labels
# ---------------------------------------------------------------------------

class TestInvalidDimensionsAndLabels:
    """Verify rejection of non-square matrices, mismatched dimensions, and missing labels."""

    def test_non_square_covariance_rejected(self) -> None:
        """Non-square covariance matrix raises FinancialCalculationError."""
        non_square = pd.DataFrame(np.ones((3, 2)))
        with pytest.raises(FinancialCalculationError, match="square"):
            validate_gmv(covariance=non_square)

    def test_fewer_than_two_assets_rejected(self) -> None:
        """Single-asset covariance raises InsufficientDataError."""
        single_cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="at least 2 assets"):
            validate_gmv(covariance=single_cov)

    def test_covariance_index_column_mismatch_rejected(self) -> None:
        """Covariance with different index and column labels raises FinancialCalculationError."""
        mismatched_cov = pd.DataFrame(
            np.eye(2) * 0.04,
            index=["A", "B"],
            columns=["X", "Y"],
        )
        with pytest.raises(FinancialCalculationError, match="index and columns must match"):
            validate_gmv(covariance=mismatched_cov)

    def test_expected_returns_missing_asset_label_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Expected returns missing an asset present in covariance raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        incomplete_mu = pd.Series([0.15, 0.20], index=["AAPL", "MSFT"])
        with pytest.raises(FinancialCalculationError, match="length"):
            validate_tangency(expected_returns=incomplete_mu, covariance=cov, risk_free_rate=0.04)

    def test_boolean_scalar_risk_free_rate_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Boolean risk_free_rate raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError, match="bool"):
            validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=True)

    def test_reference_result_risk_free_rate_mismatch_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Supplied risk_free_rate differing from reference_result raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        ref_tang = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=0.04)

        with pytest.raises(FinancialCalculationError, match="does not match"):
            validate_tangency(
                expected_returns=mu,
                covariance=cov,
                risk_free_rate=0.06,  # mismatch with 0.04
                reference_result=ref_tang,
            )


# ---------------------------------------------------------------------------
# 6. Non-Finite Inputs
# ---------------------------------------------------------------------------

class TestNonFiniteInputs:
    """Verify rejection of NaN, Inf, and non-numeric entries."""

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_in_covariance_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_val: float
    ) -> None:
        """NaN or Inf in covariance matrix raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"].copy()
        cov.iloc[0, 1] = bad_val
        with pytest.raises(FinancialCalculationError, match="non-finite"):
            validate_gmv(covariance=cov)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_in_expected_returns_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_val: float
    ) -> None:
        """NaN or Inf in expected returns raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"].copy()
        mu.iloc[0] = bad_val
        with pytest.raises(FinancialCalculationError, match="non-finite"):
            validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=0.04)

    def test_complex_number_in_covariance_rejected(self) -> None:
        """Complex numbers in covariance matrix raise FinancialCalculationError."""
        complex_cov = pd.DataFrame([[0.04 + 1j, 0.01], [0.01, 0.04]])
        with pytest.raises(FinancialCalculationError, match="complex"):
            validate_gmv(covariance=complex_cov)


# ---------------------------------------------------------------------------
# 7. Solver Failures and Reference Fallback Handling
# ---------------------------------------------------------------------------

class TestSolverFailuresAndFallbacks:
    """Verify explicit handling of solver non-convergence and fallback policies."""

    def test_reference_result_with_success_false_fails_validation(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Reference result with success=False results in is_valid=False and status=FAILED."""
        cov = independent_3asset_data["cov"]
        ref_opt = optimize_global_minimum_variance(covariance=cov)

        unconverged_ref = OptimizationResult(
            weights=ref_opt.weights,
            variance=ref_opt.variance,
            volatility=ref_opt.volatility,
            objective_value=ref_opt.objective_value,
            success=False,
            status=99,
            message="Simulated solver iteration limit exceeded",
            iterations=1000,
            metadata=ref_opt.metadata,
            fallback_executed=False,
        )

        res = validate_gmv(covariance=cov, reference_result=unconverged_ref)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("did not converge" in e for e in res.errors)

    def test_reference_result_with_fallback_executed_fails_validation(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Reference result with fallback_executed=True is not falsely certified as optimal."""
        cov = independent_3asset_data["cov"]
        ref_opt = optimize_global_minimum_variance(covariance=cov)

        fallback_ref = OptimizationResult(
            weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=cov.index),
            variance=0.05,
            volatility=0.22,
            objective_value=0.025,
            success=True,
            status=0,
            message="Fallback executed",
            iterations=0,
            metadata={},
            fallback_executed=True,
        )

        res = validate_gmv(covariance=cov, reference_result=fallback_ref)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("used fallback" in e for e in res.errors)

    def test_shadow_solver_failure_when_all_returns_below_risk_free_rate(self) -> None:
        """When all expected returns < rf, PyPortfolioOpt max_sharpe raises ValueError; caught cleanly."""
        tickers = ["A", "B"]
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=tickers, columns=tickers)
        mu = pd.Series([0.02, 0.03], index=tickers)
        rf = 0.06  # rf > all mu

        res = validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("PyPortfolioOpt shadow Tangency optimization failed" in e for e in res.errors)


# ---------------------------------------------------------------------------
# 8. Equivalent Optimum Detection (Collinear Assets)
# ---------------------------------------------------------------------------

class TestEquivalentOptimumAndToleranceBoundaries:
    """Verify handling when multiple numerically equivalent optimal allocations exist."""

    def test_collinear_assets_detected_as_equivalent_optimum(self) -> None:
        """Two perfectly correlated assets permit multiple weight combinations with identical variance.

        When ASSET_1 and ASSET_2 are perfectly correlated (cov[0,1]==cov[0,0]==cov[1,1]), any
        split (a, s-a, c) with fixed total s=w1+w2 yields the same portfolio variance.
        CVXPY (PyPortfolioOpt) may split differently from production SLSQP.
        The test verifies that metric parity is detected despite different individual allocations.
        """
        tickers = ["ASSET_1", "ASSET_2", "ASSET_3"]
        # ASSET_1 and ASSET_2 are perfectly correlated: same variance, same covariance
        cov = pd.DataFrame(
            [
                [0.04, 0.04, 0.01],
                [0.04, 0.04, 0.01],
                [0.01, 0.01, 0.05],
            ],
            index=tickers,
            columns=tickers,
        )

        # True GMV weights have s = w1+w2 = 4/7, w3 = 3/7
        # Both (4/7, 0, 3/7) and (2/7, 2/7, 3/7) are equally optimal GMV allocations
        ref_weights_a = pd.Series([4 / 7, 0.0, 3 / 7], index=tickers)
        ref_weights_b = pd.Series([2 / 7, 2 / 7, 3 / 7], index=tickers)
        cov_np = cov.values

        var_a = float(ref_weights_a.values @ cov_np @ ref_weights_a.values)
        var_b = float(ref_weights_b.values @ cov_np @ ref_weights_b.values)
        # Confirm both allocations have same variance
        assert abs(var_a - var_b) < 1e-12, f"Variances differ: {var_a} vs {var_b}"

        # Hand the solver-A allocation as reference
        ref_opt = OptimizationResult(
            weights=ref_weights_a,
            variance=var_a,
            volatility=np.sqrt(var_a),
            objective_value=0.5 * var_a,
            success=True,
            status=0,
            message="Optimal",
            iterations=10,
            metadata={},
        )

        # Validation: PyPortfolioOpt will return allocation close to (2/7, 2/7, 3/7)
        # Metric parity (variance match) should be verified despite weight divergence
        tol = ValidationTolerances(weight_tol=1e-4, variance_tol=1e-4, volatility_tol=1e-4)
        res = validate_gmv(covariance=cov, reference_result=ref_opt, tolerances=tol)

        # Both allocations are truly optimal — metric parity must hold
        assert res.metric_valid is True
        assert res.differences.abs_diff_variance < 1e-10
        # Weights differ (different optimal split), so weight_valid may be False
        if not res.weight_valid:
            assert res.status == ValidationStatus.EQUIVALENT_OPTIMUM.value
            assert res.is_valid is True
            assert any("equivalent" in w.lower() for w in res.warnings)

    def test_custom_tolerances_respected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Custom tight tolerances are strictly evaluated."""
        cov = independent_3asset_data["cov"]
        # Impossibly tight tolerance (1e-15) causes validation to fail
        ultra_tight_tol = ValidationTolerances(variance_tol=1e-15, volatility_tol=1e-15, relative_tol=1e-15)
        res = validate_gmv(covariance=cov, tolerances=ultra_tight_tol)

        assert res.status == ValidationStatus.FAILED.value
        assert res.is_valid is False


# ---------------------------------------------------------------------------
# 9. Result Structure, Immutability & Serialization
# ---------------------------------------------------------------------------

class TestResultStructureAndSerialization:
    """Verify frozen immutability and JSON serializability of all validation result dataclasses."""

    def test_gmv_validation_result_immutability(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """GMVValidationResult attributes cannot be assigned after initialization."""
        cov = independent_3asset_data["cov"]
        res = validate_gmv(covariance=cov)

        with pytest.raises(FrozenInstanceError):
            res.is_valid = False  # type: ignore

        with pytest.raises(FrozenInstanceError):
            res.status = "MUTATED"  # type: ignore

    def test_tangency_validation_result_immutability(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """TangencyValidationResult attributes cannot be assigned after initialization."""
        res = validate_tangency(
            expected_returns=independent_3asset_data["mu"],
            covariance=independent_3asset_data["cov"],
            risk_free_rate=0.04,
        )

        with pytest.raises(FrozenInstanceError):
            res.risk_free_rate = 0.99  # type: ignore

    def test_portfolio_metrics_defensive_copy(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Mutating returned weights series from PortfolioMetrics does not alter internal state."""
        cov = independent_3asset_data["cov"]
        res = validate_gmv(covariance=cov)
        assert res.reference_metrics is not None

        w_copy = res.reference_metrics.weights
        w_copy.iloc[0] = 999.0
        assert res.reference_metrics.weights.iloc[0] != 999.0

    def test_all_validation_results_to_dict_json_serializable(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """to_dict() outputs for GMV, Tangency, and Frontier are 100% JSON-serializable."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        rf = independent_3asset_data["rf"]

        gmv_res = validate_gmv(covariance=cov, expected_returns=mu)
        tang_res = validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        ef_res = validate_efficient_frontier(expected_returns=mu, covariance=cov, target_returns=[0.16])

        # Test JSON dumping
        json_gmv = json.dumps(gmv_res.to_dict())
        assert json_gmv is not None
        parsed_gmv = json.loads(json_gmv)
        assert parsed_gmv["is_valid"] is True

        json_tang = json.dumps(tang_res.to_dict())
        assert json_tang is not None
        parsed_tang = json.loads(json_tang)
        assert parsed_tang["risk_free_rate"] == 0.04

        json_ef = json.dumps(ef_res.to_dict())
        assert json_ef is not None
        parsed_ef = json.loads(json_ef)
        assert len(parsed_ef["validated_points"]) == 1


# ---------------------------------------------------------------------------
# 10. Namespace Exports and Identities
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package exports and cross-namespace identities."""

    SHADOW_SYMBOLS = [
        "DEFAULT_VALIDATION_TOLERANCES",
        "FrontierPointValidation",
        "FrontierValidationResult",
        "GMVValidationResult",
        "PortfolioMetrics",
        "PyPortfolioOptShadowValidator",
        "ShadowValidator",
        "TangencyValidationResult",
        "ValidationDifferences",
        "ValidationStatus",
        "ValidationTolerances",
        "validate_efficient_frontier",
        "validate_gmv",
        "validate_tangency",
    ]

    def test_portfolio_engine_exports(self) -> None:
        """portfolio_engine exports all shadow validator symbols in __all__."""
        for sym in self.SHADOW_SYMBOLS:
            assert hasattr(pe, sym), f"portfolio_engine missing export {sym}"
            assert sym in pe.__all__, f"{sym} not in portfolio_engine.__all__"

    def test_financial_engine_exports(self) -> None:
        """financial_engine re-exports all shadow validator symbols in __all__."""
        for sym in self.SHADOW_SYMBOLS:
            assert hasattr(fe, sym), f"financial_engine missing export {sym}"
            assert sym in fe.__all__, f"{sym} not in financial_engine.__all__"

    def test_financial_engine_shadow_validator_module_exports(self) -> None:
        """financial_engine.shadow_validator re-exports all symbols in __all__."""
        for sym in self.SHADOW_SYMBOLS:
            assert hasattr(fesv, sym), f"financial_engine.shadow_validator missing {sym}"
            assert sym in fesv.__all__, f"{sym} not in fesv.__all__"

    def test_cross_namespace_identities(self) -> None:
        """Symbols across portfolio_engine and financial_engine point to identical objects."""
        for sym in self.SHADOW_SYMBOLS:
            pe_obj = getattr(pe, sym)
            fe_obj = getattr(fe, sym)
            fesv_obj = getattr(fesv, sym)
            assert pe_obj is fe_obj, f"Identity mismatch between pe and fe for {sym}"
            assert pe_obj is fesv_obj, f"Identity mismatch between pe and fesv for {sym}"


# ---------------------------------------------------------------------------
# 11. Feasibility Gating and Covariance Regression Tests (Audit Findings)
# ---------------------------------------------------------------------------

class TestFeasibilityGatingAndAuditCorrections:
    """Regression tests for audit findings 1-5: weight feasibility, sum gates, EF GMV reference, and covariance validation."""

    def test_shadow_weights_sum_above_tolerance_fails_validation(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 1 & 2: Shadow weights summing beyond weight_sum_tol fails validation (is_valid=False, status=FAILED)."""
        cov = independent_3asset_data["cov"]
        # Return shadow weights that sum to 1.05 (well beyond 1e-5 tolerance)
        monkeypatch.setattr(
            EfficientFrontier,
            "min_volatility",
            lambda self: {"AAPL": 0.50, "MSFT": 0.35, "GOOG": 0.20},
        )
        res = validate_gmv(covariance=cov)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert res.weight_valid is False
        assert any("weight_sum_tol" in err for err in res.errors)

    def test_negative_shadow_weight_fails_validation(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 1: Shadow portfolio with negative weight violates long-only bounds and fails validation."""
        cov = independent_3asset_data["cov"]
        monkeypatch.setattr(
            EfficientFrontier,
            "min_volatility",
            lambda self: {"AAPL": -0.05, "MSFT": 0.55, "GOOG": 0.50},
        )
        res = validate_gmv(covariance=cov)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert res.weight_valid is False
        assert any("bounds [0, 1]" in err for err in res.errors)

    def test_shadow_weight_above_one_fails_validation(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 1: Shadow portfolio with weight > 1.0 fails validation."""
        cov = independent_3asset_data["cov"]
        monkeypatch.setattr(
            EfficientFrontier,
            "min_volatility",
            lambda self: {"AAPL": 1.10, "MSFT": -0.05, "GOOG": -0.05},
        )
        res = validate_gmv(covariance=cov)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert res.weight_valid is False

    def test_reference_weights_sum_incorrectly_raises(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 4: Supplied reference result with weights summing to 0.9 raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        bad_ref = OptimizationResult(
            weights=pd.Series([0.30, 0.30, 0.30], index=cov.index),
            variance=0.03,
            volatility=0.173,
            objective_value=0.015,
            success=True,
            status=0,
            message="Optimal",
            iterations=5,
            metadata={},
            fallback_executed=False,
        )
        with pytest.raises(FinancialCalculationError, match="deviates from 1.0"):
            validate_gmv(covariance=cov, reference_result=bad_ref)

    def test_reference_weights_contain_nan_raises(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 4: Supplied reference result containing NaN weight raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        bad_ref = OptimizationResult(
            weights=pd.Series([np.nan, 0.50, 0.50], index=cov.index),
            variance=0.03,
            volatility=0.173,
            objective_value=0.015,
            success=True,
            status=0,
            message="Optimal",
            iterations=5,
            metadata={},
            fallback_executed=False,
        )
        with pytest.raises(FinancialCalculationError, match="NaN"):
            validate_gmv(covariance=cov, reference_result=bad_ref)

    def test_reference_weights_missing_or_extra_asset_raises(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 4: Supplied reference result with extra or missing asset label raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        bad_ref = OptimizationResult(
            weights=pd.Series([0.50, 0.50, 0.0], index=["AAPL", "MSFT", "EXTRA_TICKER"]),
            variance=0.03,
            volatility=0.173,
            objective_value=0.015,
            success=True,
            status=0,
            message="Optimal",
            iterations=5,
            metadata={},
            fallback_executed=False,
        )
        with pytest.raises(FinancialCalculationError, match="asset labels do not match"):
            validate_gmv(covariance=cov, reference_result=bad_ref)

    def test_ef_validation_with_gmv_portfolio_none_fails(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 3: EF validation when reference gmv_portfolio is None returns is_valid=False and status=FAILED."""
        import unittest.mock
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        mock_ef = unittest.mock.MagicMock(spec=EfficientFrontierResult)
        mock_ef.success = True
        mock_ef.gmv_portfolio = None
        mock_ef.message = "No GMV reference"

        res = validate_efficient_frontier(expected_returns=mu, covariance=cov, reference_result=mock_ef)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("no GMV portfolio" in err for err in res.errors)

    def test_ef_validation_with_failed_gmv_fails(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 3: EF validation when reference GMV failed returns is_valid=False and status=FAILED."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        ref_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov)

        unconverged_gmv = OptimizationResult(
            weights=ref_ef.gmv_portfolio.weights,
            variance=ref_ef.gmv_portfolio.variance,
            volatility=ref_ef.gmv_portfolio.volatility,
            objective_value=ref_ef.gmv_portfolio.objective_value,
            success=False,
            status=99,
            message="Solver did not converge",
            iterations=1000,
            metadata={},
            fallback_executed=False,
        )
        ef_failed_gmv = EfficientFrontierResult(
            portfolios=ref_ef.portfolios,
            efficient_portfolios=ref_ef.efficient_portfolios,
            inefficient_portfolios=ref_ef.inefficient_portfolios,
            minimum_feasible_return=ref_ef.minimum_feasible_return,
            maximum_feasible_return=ref_ef.maximum_feasible_return,
            gmv_portfolio=unconverged_gmv,
            number_of_points=ref_ef.number_of_points,
            success=True,
            metadata=ref_ef.metadata,
        )
        res = validate_efficient_frontier(expected_returns=mu, covariance=cov, reference_result=ef_failed_gmv)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("did not converge" in err for err in res.errors)

    def test_ef_validation_with_fallback_gmv_fails(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 3: EF validation when reference GMV used fallback returns is_valid=False and status=FAILED."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        ref_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov)

        fallback_gmv = OptimizationResult(
            weights=pd.Series([1 / 3, 1 / 3, 1 / 3], index=cov.index),
            variance=0.03,
            volatility=0.173,
            objective_value=0.015,
            success=True,
            status=0,
            message="Fallback executed",
            iterations=0,
            metadata={},
            fallback_executed=True,
        )
        ef_fallback_gmv = EfficientFrontierResult(
            portfolios=ref_ef.portfolios,
            efficient_portfolios=ref_ef.efficient_portfolios,
            inefficient_portfolios=ref_ef.inefficient_portfolios,
            minimum_feasible_return=ref_ef.minimum_feasible_return,
            maximum_feasible_return=ref_ef.maximum_feasible_return,
            gmv_portfolio=fallback_gmv,
            number_of_points=ref_ef.number_of_points,
            success=True,
            metadata=ref_ef.metadata,
        )
        res = validate_efficient_frontier(expected_returns=mu, covariance=cov, reference_result=ef_fallback_gmv)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("used fallback" in err for err in res.errors)

    def test_asymmetric_covariance_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 5: Asymmetric covariance matrix raises FinancialCalculationError."""
        cov_df = independent_3asset_data["cov"].copy()
        cov_df.iloc[0, 1] += 0.05
        with pytest.raises(FinancialCalculationError, match="asymmetric"):
            validate_gmv(covariance=cov_df)

        cov_np = cov_df.values
        with pytest.raises(FinancialCalculationError, match="asymmetric"):
            validate_gmv(covariance=cov_np)

    def test_negative_diagonal_covariance_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Finding 5: Covariance matrix with negative diagonal element raises FinancialCalculationError."""
        cov_df = independent_3asset_data["cov"].copy()
        cov_df.iloc[0, 0] = -0.01
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            validate_gmv(covariance=cov_df)

        cov_np = cov_df.values
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            validate_gmv(covariance=cov_np)

    @pytest.mark.parametrize(
        "bad_weights",
        [
            {"AAPL": 0.60, "MSFT": 0.35, "GOOG": 0.20},  # Sum = 1.15
            {"AAPL": -0.10, "MSFT": 0.60, "GOOG": 0.50}, # Negative weight
            {"AAPL": 1.20, "MSFT": -0.10, "GOOG": -0.10},# Weight > 1.0
            {"AAPL": np.nan, "MSFT": 0.50, "GOOG": 0.50},# NaN weight
        ],
    )
    def test_no_infeasible_portfolio_returns_passed_or_equivalent_optimum(
        self,
        independent_3asset_data: Dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        bad_weights: Dict[str, float],
    ) -> None:
        """Finding 1: Infeasible shadow weights can NEVER produce PASSED or EQUIVALENT_OPTIMUM."""
        cov = independent_3asset_data["cov"]
        monkeypatch.setattr(EfficientFrontier, "min_volatility", lambda self: bad_weights)

        res = validate_gmv(covariance=cov)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert res.status not in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)

    def test_internally_generated_gmv_reference_weights_validated(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 1 (Final Audit): Internally generated GMV reference weights are validated via _validate_reference_weights."""
        cov = independent_3asset_data["cov"]
        bad_generated_ref = OptimizationResult(
            weights=pd.Series([0.30, 0.30, 0.30], index=cov.index),
            variance=0.03,
            volatility=0.173,
            objective_value=0.015,
            success=True,
            status=0,
            message="Optimal",
            iterations=5,
            metadata={},
            fallback_executed=False,
        )
        monkeypatch.setattr(
            "app.portfolio_engine.shadow_validator.optimize_global_minimum_variance",
            lambda **kwargs: bad_generated_ref,
        )
        with pytest.raises(FinancialCalculationError, match="deviates from 1.0"):
            validate_gmv(covariance=cov)

    def test_internally_generated_tangency_reference_weights_validated(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Finding 1 (Final Audit): Internally generated Tangency reference weights are validated via _validate_reference_weights."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        bad_generated_tang = TangencyOptimizationResult(
            weights=pd.Series([0.30, 0.30, 0.30], index=cov.index),
            expected_return=0.15,
            variance=0.03,
            volatility=0.173,
            sharpe_ratio=0.635,
            risk_free_rate=0.04,
            success=True,
            status=0,
            message="Optimal",
            iterations=5,
            metadata={},
            fallback_executed=False,
        )
        monkeypatch.setattr(
            "app.portfolio_engine.shadow_validator.optimize_maximum_sharpe_ratio",
            lambda **kwargs: bad_generated_tang,
        )
        with pytest.raises(FinancialCalculationError, match="deviates from 1.0"):
            validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=0.04)
