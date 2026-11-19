"""Quantamental AI - Unit Tests for Efficient Frontier Optimization Engine.

Tests Phase 2.3 Step 3: Efficient Frontier computation via constrained
minimum-variance optimization.

Validates:
    1. Mathematical correctness: closed-form 2-asset examples, independent
       numpy calculations, weight/return/variance constraints.
    2. Frontier structure: ascending order, efficient/inefficient classification,
       GMV consistency, feasible boundary coverage.
    3. Edge cases: identical returns, explicit target grids, duplicate targets,
       out-of-range targets, empty arrays, single-asset rejection.
    4. Input validation: type, dtype, NaN, Inf, boolean, complex, label alignment.
    5. Solver failure and fallback policies.
    6. Result dataclass structure, immutability, and serialization.
    7. Package namespace re-exports.
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
    EfficientFrontierResult as FinancialEfficientFrontierResult,
    FrontierPortfolio as FinancialFrontierPortfolio,
    compute_efficient_frontier as financial_compute_efficient_frontier,
)
from app.financial_engine.efficient_frontier import (
    EfficientFrontierResult as FinancialModuleEfficientFrontierResult,
    FrontierPortfolio as FinancialModuleFrontierPortfolio,
    compute_efficient_frontier as financial_module_compute_efficient_frontier,
)
from app.portfolio_engine import (
    EfficientFrontierResult,
    FrontierPortfolio,
    compute_efficient_frontier,
)
from app.portfolio_engine.efficient_frontier import (
    DEFAULT_FRONTIER_POINTS,
    _FEASIBILITY_TOLERANCE,
    _TARGET_RETURN_TOLERANCE,
    _WEIGHT_BOUND_TOLERANCE,
    _WEIGHT_SUM_TOLERANCE,
    _feasible_initial_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """Standard 2-asset covariance matrix."""
    return pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )


@pytest.fixture
def two_asset_mu() -> pd.Series:
    """Standard 2-asset expected returns."""
    return pd.Series([0.12, 0.18], index=["A", "B"])


@pytest.fixture
def three_asset_cov() -> pd.DataFrame:
    """3-asset positive-definite covariance matrix."""
    return pd.DataFrame(
        [[0.04, 0.01, 0.005],
         [0.01, 0.09, 0.015],
         [0.005, 0.015, 0.16]],
        index=["X", "Y", "Z"],
        columns=["X", "Y", "Z"],
    )


@pytest.fixture
def three_asset_mu() -> pd.Series:
    """3-asset expected returns."""
    return pd.Series([0.10, 0.15, 0.20], index=["X", "Y", "Z"])


# ---------------------------------------------------------------------------
# 1. Mathematical Correctness Tests
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:
    """Verify frontier portfolios satisfy all constraints and match analytical results."""

    def test_budget_constraint_all_points(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Every frontier portfolio must satisfy sum(w) = 1 within tolerance."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            weight_sum = float(p.weights.sum())
            assert abs(weight_sum - 1.0) <= _WEIGHT_SUM_TOLERANCE, (
                f"Budget constraint violated at target_return={p.target_return:.6f}: "
                f"sum(w)={weight_sum:.8f}"
            )

    def test_long_only_constraint_all_points(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Every frontier portfolio must satisfy 0 <= w_i <= 1."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            assert (p.weights >= -_WEIGHT_BOUND_TOLERANCE).all(), (
                f"Long-only constraint violated at target={p.target_return:.6f}: "
                f"min(w)={p.weights.min():.8f}"
            )
            assert (p.weights <= 1.0 + _WEIGHT_BOUND_TOLERANCE).all()

    def test_target_return_achieved(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Every successful portfolio must achieve expected return within tolerance."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                assert abs(p.expected_return - p.target_return) <= _TARGET_RETURN_TOLERANCE, (
                    f"Target return not achieved: target={p.target_return:.8f}, "
                    f"achieved={p.expected_return:.8f}, "
                    f"err={abs(p.expected_return - p.target_return):.2e}"
                )

    def test_variance_independent_verification(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Independently verify portfolio variance = w.T @ Sigma @ w for each point."""
        cov_arr = two_asset_cov.to_numpy()
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                w = p.weights.to_numpy()
                expected_var = float(w @ cov_arr @ w)
                assert abs(p.variance - expected_var) <= 1e-10, (
                    f"Variance mismatch at target={p.target_return:.6f}: "
                    f"reported={p.variance:.8e}, independent={expected_var:.8e}"
                )

    def test_volatility_equals_sqrt_variance(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Verify volatility = sqrt(variance) for every frontier portfolio."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                expected_vol = float(np.sqrt(p.variance))
                assert abs(p.volatility - expected_vol) <= 1e-10, (
                    f"Volatility != sqrt(variance) at target={p.target_return:.6f}"
                )

    def test_expected_return_independent_verification(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Independently verify portfolio expected return = mu.T @ w for each point."""
        mu_arr = two_asset_mu.to_numpy()
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                w = p.weights.reindex(["A", "B"]).to_numpy()
                independent_ret = float(mu_arr @ w)
                assert abs(p.expected_return - independent_ret) <= 1e-10

    def test_two_asset_boundary_minimum_return(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """At min return boundary, full allocation must go to the low-return asset."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)
        min_pt = result.portfolios[0]
        assert abs(min_pt.target_return - 0.12) <= _FEASIBILITY_TOLERANCE
        assert abs(min_pt.weights["A"] - 1.0) <= 1e-5
        assert abs(min_pt.weights["B"] - 0.0) <= 1e-5
        assert abs(min_pt.volatility - 0.20) <= 1e-5  # sigma_A = sqrt(0.04) = 0.2

    def test_two_asset_boundary_maximum_return(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """At max return boundary, full allocation must go to the high-return asset."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)
        max_pt = result.portfolios[-1]
        assert abs(max_pt.target_return - 0.18) <= _FEASIBILITY_TOLERANCE
        assert abs(max_pt.weights["B"] - 1.0) <= 1e-5
        assert abs(max_pt.weights["A"] - 0.0) <= 1e-5
        assert abs(max_pt.volatility - 0.30) <= 1e-5  # sigma_B = sqrt(0.09) = 0.3

    def test_frontier_variance_convexity(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Variance curve must be convex (U-shaped): decreasing then increasing."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=30)
        variances = [p.variance for p in result.portfolios if p.success]
        # Find the minimum index (GMV)
        min_idx = int(np.argmin(variances))
        # Variance before GMV must be non-increasing; after GMV non-decreasing
        # Allow small numerical tolerance for floating-point equality
        tol = 1e-8
        for i in range(1, min_idx):
            assert variances[i] <= variances[i - 1] + tol, (
                f"Variance increased before GMV at index {i}: "
                f"{variances[i-1]:.8e} -> {variances[i]:.8e}"
            )
        for i in range(min_idx + 1, len(variances)):
            assert variances[i] >= variances[i - 1] - tol, (
                f"Variance decreased after GMV at index {i}: "
                f"{variances[i-1]:.8e} -> {variances[i]:.8e}"
            )

    def test_three_asset_variance_finite_positive(
        self, three_asset_cov: pd.DataFrame, three_asset_mu: pd.Series
    ) -> None:
        """Every 3-asset frontier portfolio must have finite positive variance."""
        result = compute_efficient_frontier(three_asset_mu, three_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                assert np.isfinite(p.variance)
                assert p.variance >= 0.0
                assert np.isfinite(p.volatility)
                assert p.volatility >= 0.0

    def test_three_asset_weights_sum_to_one(
        self, three_asset_cov: pd.DataFrame, three_asset_mu: pd.Series
    ) -> None:
        """Every 3-asset frontier portfolio must have weights summing to 1."""
        result = compute_efficient_frontier(three_asset_mu, three_asset_cov, n_points=15)
        for p in result.portfolios:
            assert abs(p.weights.sum() - 1.0) <= _WEIGHT_SUM_TOLERANCE


# ---------------------------------------------------------------------------
# 2. Frontier Structure Tests
# ---------------------------------------------------------------------------

class TestFrontierStructure:
    """Verify frontier structure, ordering, and classification."""

    def test_ascending_target_return_order(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Frontier portfolios must be in ascending target_return order."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        targets = [p.target_return for p in result.portfolios]
        for i in range(1, len(targets)):
            assert targets[i] >= targets[i - 1], (
                f"Portfolios not in ascending order at index {i}: "
                f"{targets[i-1]:.8f} -> {targets[i]:.8f}"
            )

    def test_explicit_target_returns_sorted(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Caller-supplied unsorted target_returns must be sorted ascending in result."""
        targets = [0.18, 0.12, 0.15, 0.14]
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, target_returns=targets)
        returned_targets = [p.target_return for p in result.portfolios]
        assert returned_targets == sorted(returned_targets)

    def test_default_number_of_points(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Default n_points is 50 and that many portfolios are computed."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov)
        assert result.number_of_points == DEFAULT_FRONTIER_POINTS
        assert len(result.portfolios) == DEFAULT_FRONTIER_POINTS

    def test_custom_number_of_points(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Custom n_points is respected."""
        for n in [5, 10, 25, 100]:
            result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=n)
            assert result.number_of_points == n
            assert len(result.portfolios) == n

    def test_feasible_range_reported(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """minimum and maximum feasible returns must equal min/max(mu)."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov)
        assert abs(result.minimum_feasible_return - 0.12) <= 1e-10
        assert abs(result.maximum_feasible_return - 0.18) <= 1e-10

    def test_all_portfolios_in_efficient_or_inefficient(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Every portfolio must appear in exactly one of efficient or inefficient lists."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        all_targets = {p.target_return for p in result.portfolios}
        eff_targets = {p.target_return for p in result.efficient_portfolios}
        ineff_targets = {p.target_return for p in result.inefficient_portfolios}
        assert eff_targets.isdisjoint(ineff_targets), "Efficient and inefficient sets overlap"
        assert eff_targets | ineff_targets == all_targets, "Partition is incomplete"

    def test_efficient_portfolios_at_or_above_gmv_return(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """All efficient portfolios must have target_return >= GMV expected return."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        gmv_ret = float(
            two_asset_mu.reindex(["A", "B"]).to_numpy()
            @ result.gmv_portfolio.weights.reindex(["A", "B"]).to_numpy()
        )
        for p in result.efficient_portfolios:
            assert p.target_return >= gmv_ret - _FEASIBILITY_TOLERANCE

    def test_inefficient_portfolios_below_gmv_return(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """All inefficient portfolios must have target_return < GMV expected return."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        gmv_ret = float(
            two_asset_mu.reindex(["A", "B"]).to_numpy()
            @ result.gmv_portfolio.weights.reindex(["A", "B"]).to_numpy()
        )
        for p in result.inefficient_portfolios:
            assert p.target_return < gmv_ret + _FEASIBILITY_TOLERANCE

    def test_gmv_portfolio_consistency(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """GMV portfolio in result must match direct call to optimize_global_minimum_variance."""
        from app.portfolio_engine.optimizer import optimize_global_minimum_variance
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        gmv_direct = optimize_global_minimum_variance(two_asset_cov)
        np.testing.assert_allclose(
            result.gmv_portfolio.variance, gmv_direct.variance, atol=1e-8
        )
        np.testing.assert_allclose(
            result.gmv_portfolio.volatility, gmv_direct.volatility, atol=1e-8
        )
        np.testing.assert_allclose(
            result.gmv_portfolio.weights.reindex(["A", "B"]).to_numpy(),
            gmv_direct.weights.reindex(["A", "B"]).to_numpy(),
            atol=1e-5,
        )

    def test_frontier_grid_covers_boundaries(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """The linspace grid must include the min and max feasible returns."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)
        targets = [p.target_return for p in result.portfolios]
        assert abs(targets[0] - 0.12) <= _FEASIBILITY_TOLERANCE
        assert abs(targets[-1] - 0.18) <= _FEASIBILITY_TOLERANCE

    def test_explicit_target_returns_accepted(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Explicit target_returns overrides n_points and covers expected values."""
        targets = [0.12, 0.14, 0.16, 0.18]
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, target_returns=targets)
        assert result.number_of_points == 4
        returned = [p.target_return for p in result.portfolios]
        for t in targets:
            assert any(abs(r - t) <= 1e-9 for r in returned), f"Target {t} not in result"

    def test_all_portfolios_successful(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """All points of a well-conditioned problem must succeed."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        assert result.success is True
        for p in result.portfolios:
            assert p.success is True

    def test_frontier_result_success_flag_reflects_all_points(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """EfficientFrontierResult.success = True iff all individual points succeeded."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)
        all_success = all(p.success for p in result.portfolios)
        assert result.success is all_success


# ---------------------------------------------------------------------------
# 3. GMV and Tangency Consistency Tests
# ---------------------------------------------------------------------------

class TestGMVAndTangencyConsistency:
    """Verify consistency between frontier, GMV and tangency portfolio results."""

    def test_gmv_variance_minimum_on_frontier(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """The GMV portfolio must have variance <= all frontier portfolios."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=30)
        gmv_var = result.gmv_portfolio.variance
        for p in result.portfolios:
            if p.success:
                assert p.variance >= gmv_var - 1e-7, (
                    f"Frontier portfolio at target={p.target_return:.6f} has "
                    f"variance {p.variance:.8e} < GMV variance {gmv_var:.8e}"
                )

    def test_frontier_at_gmv_return_matches_gmv_variance(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """A frontier point targeting the GMV return must achieve near-GMV variance."""
        from app.portfolio_engine.optimizer import optimize_global_minimum_variance
        gmv = optimize_global_minimum_variance(two_asset_cov)
        gmv_ret = float(two_asset_mu.reindex(["A", "B"]).to_numpy() @ gmv.weights.reindex(["A", "B"]).to_numpy())

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[gmv_ret]
        )
        p = result.portfolios[0]
        assert p.success is True
        # Variance at the GMV return must be very close to the GMV variance
        np.testing.assert_allclose(p.variance, gmv.variance, atol=1e-6)

    def test_tangency_on_efficient_frontier(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Tangency portfolio return must be >= GMV return and lie on efficient frontier."""
        from app.portfolio_engine.tangency import optimize_maximum_sharpe_ratio
        tang = optimize_maximum_sharpe_ratio(two_asset_mu, two_asset_cov, risk_free_rate=0.05)
        tang_ret = float(tang.expected_return)

        # Solve frontier at the tangency return
        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[tang_ret]
        )
        p = result.portfolios[0]
        assert p.success is True
        assert p.is_efficient is True
        # Variance from frontier must match tangency portfolio variance (same MV problem)
        np.testing.assert_allclose(p.variance, tang.variance, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Identical Expected Returns Edge Cases
# ---------------------------------------------------------------------------

class TestIdenticalExpectedReturns:
    """Verify robust handling of identical or near-identical expected returns."""

    def test_all_assets_identical_returns(self) -> None:
        """When all assets have identical expected returns, frontier has one point."""
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["A", "B"], columns=["A", "B"],
        )
        mu = pd.Series([0.12, 0.12], index=["A", "B"])
        result = compute_efficient_frontier(mu, cov, n_points=10)
        # mu_min == mu_max, so the grid collapses to 1 point
        assert result.number_of_points == 1
        assert result.minimum_feasible_return == pytest.approx(0.12)
        assert result.maximum_feasible_return == pytest.approx(0.12)
        assert result.portfolios[0].success is True

    def test_two_assets_different_returns_three_explicit_targets(self) -> None:
        """3-asset universe with two assets at same return handles explicit targets."""
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.005],
             [0.01, 0.09, 0.015],
             [0.005, 0.015, 0.16]],
            index=["X", "Y", "Z"], columns=["X", "Y", "Z"],
        )
        # X and Z have same return
        mu = pd.Series([0.12, 0.18, 0.12], index=["X", "Y", "Z"])
        result = compute_efficient_frontier(mu, cov, n_points=5)
        assert result.number_of_points == 5
        assert result.minimum_feasible_return == pytest.approx(0.12)
        assert result.maximum_feasible_return == pytest.approx(0.18)
        for p in result.portfolios:
            assert p.success is True


# ---------------------------------------------------------------------------
# 5. Explicit Target Returns Tests
# ---------------------------------------------------------------------------

class TestExplicitTargetReturns:
    """Verify behavior when explicit target_returns are supplied."""

    def test_explicit_single_target(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Explicit single target return produces exactly one portfolio."""
        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[0.15]
        )
        assert result.number_of_points == 1
        p = result.portfolios[0]
        assert p.success is True
        assert abs(p.expected_return - 0.15) <= _TARGET_RETURN_TOLERANCE

    def test_explicit_targets_sorted_on_input(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reverse-ordered explicit targets are sorted ascending in result."""
        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[0.18, 0.15, 0.12]
        )
        targets = [p.target_return for p in result.portfolios]
        assert targets == sorted(targets)
        assert abs(targets[0] - 0.12) <= 1e-10
        assert abs(targets[-1] - 0.18) <= 1e-10

    def test_numpy_array_target_returns(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """numpy array for target_returns is accepted."""
        targets = np.linspace(0.12, 0.18, 7)
        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=targets
        )
        assert result.number_of_points == 7

    def test_pandas_series_target_returns(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """pd.Series for target_returns is accepted."""
        targets = pd.Series([0.12, 0.14, 0.16, 0.18])
        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=targets
        )
        assert result.number_of_points == 4


# ---------------------------------------------------------------------------
# 6. Expected Returns Validation Tests
# ---------------------------------------------------------------------------

class TestExpectedReturnsValidation:
    """Verify strict validation of expected_returns input."""

    def test_none_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        with pytest.raises(FinancialCalculationError, match="must be a pandas Series, got None"):
            compute_efficient_frontier(None, two_asset_cov)

    def test_non_series_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        with pytest.raises(FinancialCalculationError, match="must be a pandas Series"):
            compute_efficient_frontier([0.12, 0.18], two_asset_cov)

    def test_numpy_array_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        with pytest.raises(FinancialCalculationError, match="must be a pandas Series"):
            compute_efficient_frontier(np.array([0.12, 0.18]), two_asset_cov)

    def test_empty_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        with pytest.raises(InsufficientDataError, match="empty"):
            compute_efficient_frontier(pd.Series(dtype=float), two_asset_cov)

    def test_single_asset_expected_returns_rejected(self) -> None:
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="at least 2 assets"):
            compute_efficient_frontier(pd.Series([0.12], index=["A"]), cov)

    def test_duplicate_labels_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([0.12, 0.18], index=["A", "A"])
        with pytest.raises(FinancialCalculationError, match="duplicate asset labels"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_label_mismatch_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([0.12, 0.18], index=["WRONG1", "WRONG2"])
        with pytest.raises(FinancialCalculationError, match="do not match covariance"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_non_numeric_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series(["0.12", "0.18"], index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_complex_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([0.12 + 0j, 0.18 + 0j], index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="complex"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_boolean_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([True, False], index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_object_series_with_boolean_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([True, 0.18], index=["A", "B"], dtype=object)
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_nan_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([0.12, np.nan], index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="NaN or missing"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_inf_expected_returns_rejected(self, two_asset_cov: pd.DataFrame) -> None:
        mu = pd.Series([0.12, np.inf], index=["A", "B"])
        with pytest.raises(FinancialCalculationError, match="infinite"):
            compute_efficient_frontier(mu, two_asset_cov)

    def test_label_reordering_handled(self, two_asset_cov: pd.DataFrame) -> None:
        """Expected returns in different order from covariance must be aligned correctly."""
        mu_reversed = pd.Series([0.18, 0.12], index=["B", "A"])
        result = compute_efficient_frontier(mu_reversed, two_asset_cov, n_points=5)
        assert result.minimum_feasible_return == pytest.approx(0.12)
        assert result.maximum_feasible_return == pytest.approx(0.18)


# ---------------------------------------------------------------------------
# 7. Target Returns Validation Tests
# ---------------------------------------------------------------------------

class TestTargetReturnsValidation:
    """Verify validation of explicit target_returns argument."""

    def test_out_of_range_target_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="outside the feasible range"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.05]
            )

    def test_target_above_max_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="outside the feasible range"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.25]
            )

    def test_empty_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="empty"):
            compute_efficient_frontier(two_asset_mu, two_asset_cov, target_returns=[])

    def test_nan_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="NaN or infinite"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.15, np.nan]
            )

    def test_inf_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="NaN or infinite"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.15, np.inf]
            )

    def test_negative_inf_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="NaN or infinite"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.15, -np.inf]
            )

    def test_scalar_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """A bare scalar is not a valid target_returns sequence."""
        with pytest.raises(FinancialCalculationError):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=0.15
            )

    def test_dict_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns={"ret": 0.15}
            )

    def test_boolean_sequence_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject Python boolean list [True, False]."""
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[True, False]
            )

    def test_numpy_boolean_array_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject NumPy boolean array."""
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=np.array([True, False])
            )

    def test_pandas_boolean_series_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject Pandas boolean Series."""
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=pd.Series([True, False])
            )

    def test_object_sequence_with_boolean_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject object sequences/arrays containing boolean values."""
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[True, 0.15]
            )
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=pd.Series([True, 0.15], dtype=object)
            )
        with pytest.raises(FinancialCalculationError, match="boolean"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=np.array([True, 0.15], dtype=object)
            )

    def test_complex_target_returns_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject complex-valued target returns."""
        with pytest.raises(FinancialCalculationError, match="complex"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.15 + 0.01j]
            )
        with pytest.raises(FinancialCalculationError, match="complex"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=np.array([0.15 + 0.01j])
            )

    def test_numeric_looking_string_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject numeric-looking strings rather than silently converting."""
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=["0.14", "0.16"]
            )
        with pytest.raises(FinancialCalculationError, match="non-numeric|string"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=np.array(["0.14", "0.16"])
            )
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=pd.Series(["0.14", "0.16"])
            )

    def test_exact_duplicate_targets_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject exact duplicate target returns."""
        with pytest.raises(FinancialCalculationError, match="duplicate or near-duplicate"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.14, 0.14]
            )

    def test_near_duplicate_targets_within_tolerance_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Reject near-duplicate target returns within duplicate tolerance (1e-8)."""
        with pytest.raises(FinancialCalculationError, match="duplicate or near-duplicate"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, target_returns=[0.14, 0.14 + 1e-9]
            )

    def test_targets_outside_duplicate_tolerance_accepted(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Accept distinct target returns separated by more than duplicate tolerance."""
        res = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[0.14, 0.14 + 1e-5]
        )
        assert res.number_of_points == 2

    def test_legitimate_numeric_targets_accepted(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Accept legitimate int, float, and NumPy numeric scalar types."""
        res_float = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=[0.13, 0.16]
        )
        assert res_float.success is True

        res_np = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, target_returns=np.array([np.float64(0.13), np.float64(0.16)])
        )
        assert res_np.success is True


# ---------------------------------------------------------------------------
# 8. n_points Validation Tests
# ---------------------------------------------------------------------------

class TestNPointsValidation:
    """Verify validation of n_points argument."""

    @pytest.mark.parametrize("bad_n", [0, -1, -100])
    def test_non_positive_n_points_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, bad_n: int
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="strictly positive integer"):
            compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=bad_n)

    @pytest.mark.parametrize("bad_n", [True, False])
    def test_boolean_n_points_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, bad_n: object
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="bool"):
            compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=bad_n)

    @pytest.mark.parametrize("bad_n", [1.5, 10.0, "10"])
    def test_non_integer_n_points_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, bad_n: object
    ) -> None:
        with pytest.raises(FinancialCalculationError):
            compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=bad_n)

    def test_n_points_one_accepted(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """n_points=1 must produce a single portfolio at the midpoint of the range."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=1)
        assert result.number_of_points == 1


# ---------------------------------------------------------------------------
# 9. Covariance Validation Tests
# ---------------------------------------------------------------------------

class TestCovarianceValidation:
    """Verify that the frontier enforces full covariance validation."""

    def test_non_dataframe_covariance_rejected(
        self, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="must be a pandas DataFrame"):
            compute_efficient_frontier(two_asset_mu, [[0.04, 0.01], [0.01, 0.09]])

    def test_asymmetric_covariance_rejected(
        self, two_asset_mu: pd.Series
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.02], [0.01, 0.09]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="asymmetric"):
            compute_efficient_frontier(two_asset_mu, cov)

    def test_non_positive_definite_covariance_rejected(
        self, two_asset_mu: pd.Series
    ) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.062], [0.062, 0.09]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(NonPositiveDefiniteMatrixError):
            compute_efficient_frontier(two_asset_mu, cov)

    def test_ill_conditioned_covariance_rejected(
        self, two_asset_mu: pd.Series
    ) -> None:
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        D = np.diag([1.0, 1e-7])
        cov_arr = V @ D @ V.T
        cov = pd.DataFrame(cov_arr, index=["A", "B"], columns=["A", "B"])
        with pytest.raises(IllConditionedMatrixError):
            compute_efficient_frontier(
                pd.Series([0.12, 0.18], index=["A", "B"]),
                cov,
                max_condition_number=1e6,
            )

    def test_nan_covariance_rejected(self, two_asset_mu: pd.Series) -> None:
        cov = pd.DataFrame(
            [[0.04, np.nan], [np.nan, 0.09]],
            index=["A", "B"], columns=["A", "B"],
        )
        with pytest.raises(FinancialCalculationError, match="NaN or missing"):
            compute_efficient_frontier(two_asset_mu, cov)


# ---------------------------------------------------------------------------
# 10. Solver Configuration Validation Tests
# ---------------------------------------------------------------------------

class TestSolverConfigValidation:
    """Verify solver parameter validation."""

    def test_invalid_fallback_policy_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="Unsupported fallback policy"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, fallback="INVALID"
            )

    def test_zero_tolerance_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="tolerance"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, tolerance=0.0
            )

    def test_negative_tolerance_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="tolerance"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, tolerance=-1e-9
            )

    def test_zero_max_iterations_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="max_iterations"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, max_iterations=0
            )

    def test_bool_max_condition_number_rejected(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        with pytest.raises(FinancialCalculationError, match="max_condition_number"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, max_condition_number=True
            )


# ---------------------------------------------------------------------------
# 11. Fallback Policy Tests
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    """Verify optimizer fallback behavior on solver failure."""

    @pytest.fixture
    def hard_cov(self) -> pd.DataFrame:
        """5-asset covariance matrix; more constraints = harder for 1-iteration convergence."""
        # Build a PD covariance from random correlation structure
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        cov_arr = A @ A.T + np.eye(5) * 0.5
        assets = ["P", "Q", "R", "S", "T"]
        return pd.DataFrame(cov_arr, index=assets, columns=assets)

    @pytest.fixture
    def hard_mu(self) -> pd.Series:
        assets = ["P", "Q", "R", "S", "T"]
        return pd.Series([0.08, 0.10, 0.12, 0.14, 0.16], index=assets)

    def test_fallback_raise_exception_on_solver_failure(
        self, hard_cov: pd.DataFrame, hard_mu: pd.Series
    ) -> None:
        """When SLSQP fails and fallback=RAISE_EXCEPTION, raise OptimizationException."""
        with pytest.raises(OptimizationException, match="Efficient Frontier optimization failed"):
            compute_efficient_frontier(
                hard_mu,
                hard_cov,
                n_points=20,
                max_iterations=1,
                tolerance=1e-20,
                fallback="RAISE_EXCEPTION",
            )

    def test_fallback_equal_weight_on_solver_failure(
        self, hard_cov: pd.DataFrame, hard_mu: pd.Series
    ) -> None:
        """When SLSQP fails and fallback=EQUAL_WEIGHT, return equal-weight portfolio with fallback_executed=True."""
        result = compute_efficient_frontier(
            hard_mu,
            hard_cov,
            n_points=20,
            max_iterations=1,
            tolerance=1e-20,
            fallback="EQUAL_WEIGHT",
        )
        # At least some points should have failed
        assert result.success is False
        assert result.fallback_executed is True
        assert result.metadata["frontier_fallback_executed"] is True
        failed = [p for p in result.portfolios if not p.success]
        assert len(failed) > 0
        # Failed portfolios must carry the fallback message and fallback_executed=True
        for p in failed:
            assert p.fallback_executed is True
            assert "Fallback to EQUAL_WEIGHT" in p.solver_message

    def test_gmv_fallback_discoverable_in_result(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GMV reference optimizer executes a fallback, it must be discoverable in result."""
        import app.portfolio_engine.efficient_frontier as ef_mod

        orig_gmv = ef_mod.optimize_global_minimum_variance

        def mock_gmv(*args, **kwargs):
            real_res = orig_gmv(*args, **kwargs)
            return dataclasses.replace(real_res, success=False, fallback_executed=True)

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv)

        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        assert result.fallback_executed is True
        assert result.success is False
        assert result.classification_valid is False
        assert result.is_classification_valid is False
        assert result.metadata["gmv_fallback_executed"] is True
        assert result.metadata["classification_valid"] is False
        assert result.metadata["gmv_expected_return"] is None
        assert "classification_error" in result.metadata
        assert result.gmv_portfolio.fallback_executed is True
        assert result.efficient_portfolios == ()
        assert result.inefficient_portfolios == ()


# ---------------------------------------------------------------------------
# 12. Result Dataclass Structure & Serialization Tests
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Verify dataclass attributes, immutability, and to_dict() serialization."""

    def test_efficient_frontier_result_attributes(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)

        assert isinstance(result.portfolios, tuple)
        assert isinstance(result.efficient_portfolios, tuple)
        assert isinstance(result.inefficient_portfolios, tuple)
        assert isinstance(result.minimum_feasible_return, float)
        assert isinstance(result.maximum_feasible_return, float)
        assert isinstance(result.number_of_points, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.classification_valid, bool)
        assert isinstance(result.is_classification_valid, bool)
        assert isinstance(result.metadata, dict)

        # Metadata keys
        assert "condition_number" in result.metadata
        assert "min_eigenvalue" in result.metadata
        assert "gmv_expected_return" in result.metadata
        assert "n_assets" in result.metadata
        assert result.metadata["n_assets"] == 2

    def test_efficient_frontier_result_immutability(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.success = False  # type: ignore

    def test_frontier_portfolio_attributes(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        p = result.portfolios[0]

        assert isinstance(p.target_return, float)
        assert isinstance(p.expected_return, float)
        assert isinstance(p.variance, float)
        assert isinstance(p.volatility, float)
        assert isinstance(p.weights, pd.Series)
        assert isinstance(p.is_efficient, bool)
        assert isinstance(p.classification_valid, bool)
        assert isinstance(p.is_classification_valid, bool)
        assert isinstance(p.success, bool)
        assert isinstance(p.solver_status, int)
        assert isinstance(p.solver_message, str)
        assert isinstance(p.iterations, int)
        # Weights are indexed by asset labels
        assert list(p.weights.index) == ["A", "B"]

    def test_frontier_portfolio_immutability(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        p = result.portfolios[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.variance = 999.0  # type: ignore

    def test_to_dict_efficient_frontier_result(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "portfolios" in d
        assert "efficient_portfolios" in d
        assert "inefficient_portfolios" in d
        assert "minimum_feasible_return" in d
        assert "maximum_feasible_return" in d
        assert "gmv_portfolio" in d
        assert "number_of_points" in d
        assert "success" in d
        assert "classification_valid" in d
        assert "metadata" in d
        assert isinstance(d["portfolios"], list)
        assert len(d["portfolios"]) == 5

    def test_to_dict_frontier_portfolio(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        p_dict = result.portfolios[0].to_dict()

        assert "target_return" in p_dict
        assert "expected_return" in p_dict
        assert "variance" in p_dict
        assert "volatility" in p_dict
        assert "weights" in p_dict
        assert "is_efficient" in p_dict
        assert "classification_valid" in p_dict
        assert "success" in p_dict
        assert isinstance(p_dict["weights"], dict)
        assert "A" in p_dict["weights"]
        assert "B" in p_dict["weights"]

    def test_to_dict_values_serializable(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """All values in to_dict() must be JSON-serializable native Python types."""
        import json
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        # This will raise if anything is not serializable (numpy scalars, etc.)
        json.dumps(result.to_dict())

    def test_weights_defensive_copy_mutation_does_not_affect_portfolio(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Mutating a returned weights Series must not alter the stored portfolio weights."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        p = result.portfolios[0]
        w1 = p.weights
        orig_val = float(w1["A"])
        # Attempt to mutate returned Series
        try:
            w1.loc["A"] = 999.0
        except Exception:
            pass
        # Second access must return fresh copy with original value
        w2 = p.weights
        assert w2["A"] == pytest.approx(orig_val)
        assert w2["A"] != 999.0

    def test_weights_input_series_mutation_does_not_affect_portfolio(self) -> None:
        """Mutating an input Series after FrontierPortfolio construction must not affect it."""
        s = pd.Series([0.6, 0.4], index=["A", "B"])
        p = FrontierPortfolio(
            target_return=0.14,
            expected_return=0.14,
            variance=0.03,
            volatility=np.sqrt(0.03),
            weights=s,
            is_efficient=True,
            success=True,
            solver_status=0,
            solver_message="ok",
            iterations=5,
            fallback_executed=False,
        )
        s.loc["A"] = 999.0
        assert p.weights["A"] == pytest.approx(0.6)
        assert p.weights["A"] != 999.0

    def test_metadata_defensive_copy_mutation_does_not_affect_result(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Mutating a returned metadata dictionary must not alter stored result metadata."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        m = result.metadata
        m["new_injected_key"] = "malicious"
        assert "new_injected_key" not in result.metadata

    def test_gmv_portfolio_defensive_copy_mutation_does_not_affect_result(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Mutating GMV weights or metadata from result.gmv_portfolio must not alter stored data."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        gmv_w = result.gmv_portfolio.weights
        orig_gmv_val = float(gmv_w["A"])
        try:
            gmv_w.loc["A"] = 999.0
        except Exception:
            pass
        assert result.gmv_portfolio.weights["A"] == pytest.approx(orig_gmv_val)

        gmv_m = result.gmv_portfolio.metadata
        gmv_m["hacked_gmv_key"] = "bad"
        assert "hacked_gmv_key" not in result.gmv_portfolio.metadata

    def test_gmv_portfolio_type_hints_and_instance(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Verify type annotations resolve correctly and gmv_portfolio is an actual OptimizationResult."""
        import typing
        from app.portfolio_engine.optimizer import OptimizationResult

        hints = typing.get_type_hints(EfficientFrontierResult)
        assert hints["gmv_portfolio"] is OptimizationResult, "gmv_portfolio type hint failed to resolve"

        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        assert isinstance(result.gmv_portfolio, OptimizationResult)
        assert isinstance(result.gmv_portfolio.weights, pd.Series)
        assert isinstance(result.gmv_portfolio.variance, float)
        assert isinstance(result.gmv_portfolio.volatility, float)
        assert "variance" in result.gmv_portfolio.to_dict()

    def test_fallback_executed_field_present(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Verify fallback_executed boolean field on both FrontierPortfolio and EfficientFrontierResult."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=5)
        assert hasattr(result, "fallback_executed")
        assert result.fallback_executed is False
        for p in result.portfolios:
            assert hasattr(p, "fallback_executed")
            assert p.fallback_executed is False

    def test_cleaned_portfolio_constraint_violation_triggers_fallback(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If post-cleaning revalidation fails, the solver must treat it as optimization failure."""
        import app.portfolio_engine.efficient_frontier as ef_mod

        # Monkeypatch np.clip inside efficient_frontier to introduce a violation after solver converges
        def bad_clip(a, a_min, a_max):
            return np.zeros_like(a)

        monkeypatch.setattr(ef_mod.np, "clip", bad_clip)

        # Under RAISE_EXCEPTION, should raise OptimizationException
        with pytest.raises(OptimizationException, match="Efficient Frontier optimization failed"):
            compute_efficient_frontier(
                two_asset_mu, two_asset_cov, n_points=5, fallback="RAISE_EXCEPTION"
            )

        # Under EQUAL_WEIGHT, should return fallback portfolio with success=False, fallback_executed=True
        fallback_res = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )
        assert fallback_res.success is False
        assert fallback_res.fallback_executed is True
        for p in fallback_res.portfolios:
            assert p.success is False
            assert p.fallback_executed is True


# ---------------------------------------------------------------------------
# 13. Package Namespace Re-Export Tests
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package re-exports from both portfolio_engine and financial_engine."""

    def test_portfolio_engine_exports(self) -> None:
        import app.portfolio_engine as pe
        assert hasattr(pe, "EfficientFrontierResult")
        assert hasattr(pe, "FrontierPortfolio")
        assert hasattr(pe, "compute_efficient_frontier")
        assert pe.EfficientFrontierResult is EfficientFrontierResult
        assert pe.FrontierPortfolio is FrontierPortfolio
        assert pe.compute_efficient_frontier is compute_efficient_frontier

    def test_financial_engine_exports(self) -> None:
        import app.financial_engine as fe
        assert hasattr(fe, "EfficientFrontierResult")
        assert hasattr(fe, "FrontierPortfolio")
        assert hasattr(fe, "compute_efficient_frontier")
        assert fe.EfficientFrontierResult is EfficientFrontierResult
        assert fe.FrontierPortfolio is FrontierPortfolio
        assert fe.compute_efficient_frontier is compute_efficient_frontier

    def test_financial_engine_module_exports(self) -> None:
        import app.financial_engine.efficient_frontier as fee
        assert hasattr(fee, "EfficientFrontierResult")
        assert hasattr(fee, "FrontierPortfolio")
        assert hasattr(fee, "compute_efficient_frontier")
        assert fee.EfficientFrontierResult is EfficientFrontierResult
        assert fee.FrontierPortfolio is FrontierPortfolio
        assert fee.compute_efficient_frontier is compute_efficient_frontier

    def test_cross_namespace_identity(self) -> None:
        """Financial engine exports must be the same objects as portfolio engine."""
        assert FinancialEfficientFrontierResult is EfficientFrontierResult
        assert FinancialFrontierPortfolio is FrontierPortfolio
        assert financial_compute_efficient_frontier is compute_efficient_frontier
        assert FinancialModuleEfficientFrontierResult is EfficientFrontierResult
        assert FinancialModuleFrontierPortfolio is FrontierPortfolio
        assert financial_module_compute_efficient_frontier is compute_efficient_frontier


# ---------------------------------------------------------------------------
# 14. Warm-Start Initial Weights Helper Tests
# ---------------------------------------------------------------------------

class TestFeasibleInitialWeights:
    """Verify the _feasible_initial_weights helper produces valid initial vectors."""

    def test_basic_convex_combination(self) -> None:
        """Warm-start weights must satisfy budget and target return constraint."""
        mu = np.array([0.10, 0.20])
        w0 = _feasible_initial_weights(
            target_return=0.15,
            mu=mu,
            mu_min=0.10,
            mu_max=0.20,
            n_assets=2,
        )
        assert abs(float(np.sum(w0)) - 1.0) <= 1e-10
        assert abs(float(w0 @ mu) - 0.15) <= 1e-10
        assert np.all(w0 >= 0.0)

    def test_min_boundary_target(self) -> None:
        """At mu_min target, all weight goes to min-return asset."""
        mu = np.array([0.10, 0.20])
        w0 = _feasible_initial_weights(
            target_return=0.10,
            mu=mu,
            mu_min=0.10,
            mu_max=0.20,
            n_assets=2,
        )
        assert abs(float(np.sum(w0)) - 1.0) <= 1e-10
        assert w0[0] == pytest.approx(1.0)
        assert w0[1] == pytest.approx(0.0)

    def test_max_boundary_target(self) -> None:
        """At mu_max target, all weight goes to max-return asset."""
        mu = np.array([0.10, 0.20])
        w0 = _feasible_initial_weights(
            target_return=0.20,
            mu=mu,
            mu_min=0.10,
            mu_max=0.20,
            n_assets=2,
        )
        assert abs(float(np.sum(w0)) - 1.0) <= 1e-10
        assert w0[0] == pytest.approx(0.0)
        assert w0[1] == pytest.approx(1.0)

    def test_identical_returns_equal_weights(self) -> None:
        """When all returns are identical, equal weight is returned."""
        mu = np.array([0.12, 0.12])
        w0 = _feasible_initial_weights(
            target_return=0.12,
            mu=mu,
            mu_min=0.12,
            mu_max=0.12,
            n_assets=2,
        )
        assert abs(float(np.sum(w0)) - 1.0) <= 1e-10
        np.testing.assert_allclose(w0, [0.5, 0.5])

    def test_prev_weights_reused_when_feasible(self) -> None:
        """Previous weights are reused if they already satisfy the target return."""
        mu = np.array([0.12, 0.18])
        w_prev = np.array([0.5, 0.5])  # achieves return 0.15
        w0 = _feasible_initial_weights(
            target_return=0.15,
            mu=mu,
            mu_min=0.12,
            mu_max=0.18,
            n_assets=2,
            prev_weights=w_prev,
        )
        # Should reuse prev_weights since it satisfies the target within tolerance
        np.testing.assert_allclose(w0, w_prev)


# ---------------------------------------------------------------------------
# 15. Numerical Accuracy Tests
# ---------------------------------------------------------------------------

class TestNumericalAccuracy:
    """Independent numerical verification of frontier results."""

    def test_two_asset_min_variance_analytical(self) -> None:
        """For 2-asset uncorrelated portfolio, minimum variance at target return has analytical solution.

        For uncorrelated assets with variances sigma1^2, sigma2^2 and returns mu1, mu2:
        At target return mu_p = w1*mu1 + w2*mu2 with w1 + w2 = 1:
            w1 = (mu_p - mu2) / (mu1 - mu2)
            w2 = 1 - w1
        Portfolio variance = w1^2 * sigma1^2 + w2^2 * sigma2^2
        """
        sigma1_sq = 0.04  # Asset A
        sigma2_sq = 0.09  # Asset B
        mu1 = 0.12
        mu2 = 0.18
        cov = pd.DataFrame(
            [[sigma1_sq, 0.0], [0.0, sigma2_sq]],
            index=["A", "B"], columns=["A", "B"],
        )
        mu = pd.Series([mu1, mu2], index=["A", "B"])

        # Test several interior target returns
        targets = [0.13, 0.14, 0.15, 0.16, 0.17]
        result = compute_efficient_frontier(mu, cov, target_returns=targets)

        for p in result.portfolios:
            if not p.success:
                continue
            tr = p.target_return
            # Analytical solution (uncorrelated, so this is also the global minimum
            # variance for the given target on the full frontier)
            w1_analytical = (tr - mu2) / (mu1 - mu2)
            w2_analytical = 1.0 - w1_analytical
            var_analytical = w1_analytical**2 * sigma1_sq + w2_analytical**2 * sigma2_sq

            np.testing.assert_allclose(p.variance, var_analytical, atol=1e-8, rtol=1e-6)

    def test_three_asset_weights_sum_budget(
        self, three_asset_cov: pd.DataFrame, three_asset_mu: pd.Series
    ) -> None:
        """All 3-asset frontier points satisfy exact budget constraint."""
        result = compute_efficient_frontier(three_asset_mu, three_asset_cov, n_points=20)
        for p in result.portfolios:
            if p.success:
                np.testing.assert_allclose(p.weights.sum(), 1.0, atol=_WEIGHT_SUM_TOLERANCE)

    def test_frontier_variance_less_than_max_asset_variance(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Interior frontier portfolios must have variance <= max single-asset variance."""
        max_single_asset_var = max(
            float(two_asset_cov.loc["A", "A"]),
            float(two_asset_cov.loc["B", "B"]),
        )
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=20)
        # Exclude boundary points (which are single-asset and thus equal the max)
        interior = result.portfolios[1:-1]
        for p in interior:
            if p.success:
                assert p.variance <= max_single_asset_var + 1e-8

    def test_minimum_variance_portfolio_on_frontier(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """The lowest-variance point on the frontier must equal the GMV portfolio variance."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=50)
        min_frontier_var = min(p.variance for p in result.portfolios if p.success)
        gmv_var = result.gmv_portfolio.variance
        # GMV variance must be very close to the minimum frontier variance
        # (they may differ slightly since GMV is solved separately and the grid may not
        # hit it exactly)
        assert min_frontier_var >= gmv_var - 1e-5


# ---------------------------------------------------------------------------
# 16. GMV Fallback & Efficient Frontier Classification Correctness Tests
# ---------------------------------------------------------------------------

class TestGMVFallbackAndFrontierClassification:
    """Verify that GMV fallback behavior never corrupts efficient frontier classification."""

    def test_gmv_optimizer_normal_success_classification(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """When GMV succeeds normally, classification is valid and partitions all portfolios."""
        result = compute_efficient_frontier(two_asset_mu, two_asset_cov, n_points=10)

        assert result.success is True
        assert result.fallback_executed is False
        assert result.classification_valid is True
        assert result.is_classification_valid is True
        assert result.metadata["classification_valid"] is True
        assert result.metadata["gmv_fallback_executed"] is False
        assert result.metadata["frontier_fallback_executed"] is False
        assert isinstance(result.metadata["gmv_expected_return"], float)
        assert len(result.portfolios) == 10
        assert len(result.efficient_portfolios) + len(result.inefficient_portfolios) == 10

        gmv_ret = result.metadata["gmv_expected_return"]
        for p in result.portfolios:
            assert p.classification_valid is True
            assert p.is_classification_valid is True
            assert isinstance(p.is_efficient, bool)
            if p.target_return >= gmv_ret:
                assert p.is_efficient is True
            else:
                assert p.is_efficient is False

    def test_gmv_optimizer_equal_weight_fallback_semantics(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GMV returns EQUAL_WEIGHT fallback, overall success must be False and classification disabled."""
        import app.portfolio_engine.efficient_frontier as ef_mod
        from app.portfolio_engine.optimizer import OptimizationResult

        orig_gmv = ef_mod.optimize_global_minimum_variance

        def mock_gmv_fallback(*args, **kwargs):
            real_res = orig_gmv(*args, **kwargs)
            n = len(real_res.weights)
            eq_w = pd.Series(1.0 / n, index=real_res.weights.index, name="weight")
            var = float(eq_w.to_numpy() @ two_asset_cov.to_numpy() @ eq_w.to_numpy())
            return OptimizationResult(
                weights=eq_w,
                variance=var,
                volatility=float(np.sqrt(var)),
                objective_value=var,
                success=False,
                status=4,
                message="Fallback to EQUAL_WEIGHT: iteration limit reached",
                iterations=1000,
                fallback_executed=True,
                metadata={"fallback_policy": "EQUAL_WEIGHT"},
            )

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv_fallback)

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )

        # Semantics: overall success MUST be False because GMV reference optimization failed
        assert result.success is False
        assert result.fallback_executed is True
        assert result.classification_valid is False
        assert result.is_classification_valid is False
        assert result.metadata["classification_valid"] is False
        assert result.metadata["gmv_fallback_executed"] is True
        assert result.metadata["gmv_success"] is False
        assert result.metadata["gmv_expected_return"] is None
        assert "classification_error" in result.metadata
        assert "GMV reference optimization failed or executed fallback" in result.metadata["classification_error"]

        # Efficient and inefficient subsets must be empty to avoid false classification claims
        assert result.efficient_portfolios == ()
        assert result.inefficient_portfolios == ()

    def test_frontier_points_succeed_while_gmv_fails(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GMV fails, individual frontier points still succeed and remain accessible."""
        import app.portfolio_engine.efficient_frontier as ef_mod
        from app.portfolio_engine.optimizer import OptimizationResult

        orig_gmv = ef_mod.optimize_global_minimum_variance

        def mock_gmv_fail(*args, **kwargs):
            real = orig_gmv(*args, **kwargs)
            return dataclasses.replace(
                real,
                success=False,
                fallback_executed=True,
                message="Fallback to EQUAL_WEIGHT: singular matrix",
            )

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv_fail)

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )

        assert len(result.portfolios) == 5
        assert result.number_of_points == 5

        # All individual frontier points succeeded in their constrained QP
        for p in result.portfolios:
            assert p.success is True
            assert p.fallback_executed is False
            # Classification is unverified (None), not falsely True or False
            assert p.is_efficient is None
            assert p.classification_valid is False
            assert p.is_classification_valid is False
            # Weights are valid long-only, fully-invested
            assert abs(float(p.weights.sum()) - 1.0) <= _WEIGHT_SUM_TOLERANCE
            assert (p.weights >= 0.0).all()
            assert p.variance > 0.0

        # But overall result.success is False due to GMV failure
        assert result.success is False
        assert result.fallback_executed is True
        assert result.classification_valid is False

    def test_no_incorrect_classification_based_on_fallback_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prevent false classification: equal weights must not be used as GMV threshold."""
        import app.portfolio_engine.efficient_frontier as ef_mod
        from app.portfolio_engine.optimizer import OptimizationResult

        # 2-asset universe where equal weights (return 0.15) differs greatly from GMV (return ~0.106)
        # Asset A: low return 0.10, very low variance 0.01
        # Asset B: high return 0.20, high variance 0.16
        cov = pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.16]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        mu = pd.Series([0.10, 0.20], index=["A", "B"])

        # Equal weight expected return: 0.5 * 0.10 + 0.5 * 0.20 = 0.15
        # True GMV weights: w_A = 0.16 / 0.17 = 0.9412, w_B = 0.01 / 0.17 = 0.0588
        # True GMV return: ~0.1059
        #
        # A target return of 0.13 is mathematically EFFICIENT (0.13 >= 0.1059).
        # But if equal-weight fallback (return 0.15) were falsely used as GMV reference,
        # 0.13 would be classified as INEFFICIENT (0.13 < 0.15)!

        def mock_gmv_fallback(*args, **kwargs):
            eq_w = pd.Series([0.5, 0.5], index=["A", "B"], name="weight")
            var = float(eq_w.to_numpy() @ cov.to_numpy() @ eq_w.to_numpy())
            return OptimizationResult(
                weights=eq_w,
                variance=var,
                volatility=float(np.sqrt(var)),
                objective_value=var,
                success=False,
                status=4,
                message="Fallback to EQUAL_WEIGHT",
                iterations=1000,
                fallback_executed=True,
                metadata={"fallback_policy": "EQUAL_WEIGHT"},
            )

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv_fallback)

        result = compute_efficient_frontier(
            mu, cov, target_returns=[0.13], fallback="EQUAL_WEIGHT"
        )

        assert len(result.portfolios) == 1
        p = result.portfolios[0]

        # Solver succeeded for target return 0.13
        assert p.success is True
        np.testing.assert_allclose(float(p.expected_return), 0.13, atol=1e-5)

        # But it must NOT be falsely classified based on equal-weight return 0.15
        assert p.is_efficient is None
        assert p.classification_valid is False
        assert result.classification_valid is False
        assert result.efficient_portfolios == ()
        assert result.inefficient_portfolios == ()
        assert result.success is False

    def test_both_frontier_and_gmv_fallback(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both GMV and frontier points fail, flags reflect both failures."""
        import app.portfolio_engine.efficient_frontier as ef_mod
        from app.portfolio_engine.optimizer import OptimizationResult

        def mock_gmv_fail(*args, **kwargs):
            eq_w = pd.Series([0.5, 0.5], index=["A", "B"], name="weight")
            return OptimizationResult(
                weights=eq_w,
                variance=0.05,
                volatility=np.sqrt(0.05),
                objective_value=0.05,
                success=False,
                status=4,
                message="Fallback to EQUAL_WEIGHT",
                iterations=1000,
                fallback_executed=True,
                metadata={"fallback_policy": "EQUAL_WEIGHT"},
            )

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv_fail)

        # Force solver failure for target returns by monkeypatching scipy.optimize.minimize
        orig_minimize = ef_mod.scipy.optimize.minimize

        def mock_minimize(*args, **kwargs):
            mock_res = orig_minimize(*args, **kwargs)
            mock_res.success = False
            mock_res.message = "Simulated solver failure"
            return mock_res

        monkeypatch.setattr(ef_mod.scipy.optimize, "minimize", mock_minimize)

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )

        assert result.success is False
        assert result.fallback_executed is True
        assert result.classification_valid is False
        assert result.metadata["gmv_fallback_executed"] is True
        assert result.metadata["frontier_fallback_executed"] is True
        for p in result.portfolios:
            assert p.fallback_executed is True
            assert p.success is False
            assert p.is_efficient is None
            assert p.classification_valid is False

    def test_to_dict_preserves_classification_valid_and_none_efficiency(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """to_dict() correctly serializes classification_valid and None is_efficient."""
        import json
        import app.portfolio_engine.efficient_frontier as ef_mod
        from app.portfolio_engine.optimizer import OptimizationResult

        def mock_gmv_fail(*args, **kwargs):
            eq_w = pd.Series([0.5, 0.5], index=["A", "B"], name="weight")
            return OptimizationResult(
                weights=eq_w,
                variance=0.05,
                volatility=np.sqrt(0.05),
                objective_value=0.05,
                success=False,
                status=4,
                message="Fallback to EQUAL_WEIGHT",
                iterations=1000,
                fallback_executed=True,
                metadata={"fallback_policy": "EQUAL_WEIGHT"},
            )

        monkeypatch.setattr(ef_mod, "optimize_global_minimum_variance", mock_gmv_fail)

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["fallback_executed"] is True
        assert d["classification_valid"] is False
        assert d["efficient_portfolios"] == []
        assert d["inefficient_portfolios"] == []
        assert d["metadata"]["classification_valid"] is False
        assert d["metadata"]["gmv_expected_return"] is None

        for p_dict in d["portfolios"]:
            assert p_dict["is_efficient"] is None
            assert p_dict["classification_valid"] is False

        # Must serialize cleanly to JSON
        json_str = json.dumps(d)
        assert json_str is not None

    def test_gmv_succeeds_single_frontier_point_fallback_classification(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GMV succeeds but one frontier point falls back, global classification remains valid,
        the fallback point has is_efficient=None and is in neither subset, and successful points are classified.
        """
        import app.portfolio_engine.efficient_frontier as ef_mod

        # We'll compute 5 points. Choose the 3rd point's target return to simulate failure.
        # Targets will be linspace(0.12, 0.18, 5) -> [0.12, 0.135, 0.15, 0.165, 0.18]
        target_to_fail = 0.15

        orig_minimize = ef_mod.scipy.optimize.minimize

        def selective_minimize(*args, **kwargs):
            # GMV optimization has only 1 constraint (budget); frontier points have 2 (budget + return)
            constraints = kwargs.get("constraints", [])
            if len(constraints) == 2:
                # Check target return from the second constraint
                fun = constraints[1].get("fun")
                if fun is not None:
                    # Target return constraint is: mu @ w - target_return == 0
                    # Evaluate at zero to get -target_return
                    w_dummy = np.zeros(len(two_asset_mu))
                    val = fun(w_dummy)
                    # val == -target_return
                    if abs(-val - target_to_fail) < 1e-4:
                        mock_res = orig_minimize(*args, **kwargs)
                        mock_res.success = False
                        mock_res.status = 5
                        mock_res.message = "Simulated failure for specific target return"
                        return mock_res
            return orig_minimize(*args, **kwargs)

        monkeypatch.setattr(ef_mod.scipy.optimize, "minimize", selective_minimize)

        result = compute_efficient_frontier(
            two_asset_mu, two_asset_cov, n_points=5, fallback="EQUAL_WEIGHT"
        )

        # 1. Global classification reference remains valid because GMV succeeded
        assert result.classification_valid is True
        assert result.is_classification_valid is True
        assert result.metadata["classification_valid"] is True
        assert result.metadata["gmv_fallback_executed"] is False
        assert isinstance(result.metadata["gmv_expected_return"], float)

        # 2. Overall result reflects the frontier point fallback
        assert result.success is False
        assert result.fallback_executed is True
        assert result.metadata["frontier_fallback_executed"] is True

        # 3. All 5 points remain accessible in result.portfolios
        assert len(result.portfolios) == 5
        assert result.number_of_points == 5

        # Find the fallback point and successful points
        fallback_points = [p for p in result.portfolios if p.fallback_executed]
        successful_points = [p for p in result.portfolios if not p.fallback_executed]

        assert len(fallback_points) == 1
        assert len(successful_points) == 4

        fb_p = fallback_points[0]
        assert abs(fb_p.target_return - target_to_fail) < 1e-4
        assert fb_p.success is False
        assert fb_p.fallback_executed is True
        assert fb_p.solver_status == 5
        assert "Simulated failure for specific target return" in fb_p.solver_message
        # is_efficient is strictly None on the fallback point
        assert fb_p.is_efficient is None
        # GMV-reference classification_valid is True on the point
        assert fb_p.classification_valid is True
        assert fb_p.is_classification_valid is True

        # 4. Fallback point is excluded from BOTH classification subsets
        assert fb_p not in result.efficient_portfolios
        assert fb_p not in result.inefficient_portfolios

        # 5. Other successfully optimized points remain properly classified
        for sp in successful_points:
            assert sp.success is True
            assert sp.fallback_executed is False
            assert isinstance(sp.is_efficient, bool)
            assert sp.classification_valid is True
            if sp.is_efficient:
                assert sp in result.efficient_portfolios
            else:
                assert sp in result.inefficient_portfolios

        # Total points in efficient + inefficient equals only the successfully optimized points
        assert len(result.efficient_portfolios) + len(result.inefficient_portfolios) == 4
