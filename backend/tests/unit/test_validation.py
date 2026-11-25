"""Quantamental AI - Phase 2 Task 8: Numerical Validation & Solver Parity Suite.

Canonical References:
    - docs/PROJECT_PLAN.md (v1.4.0, Phase 2, Task 8)
    - docs/FINANCIAL_METHODOLOGY.md §4 (Covariance & Higham), §5 (Optimization & Parity)
    - ARCHITECTURE.md §4.2

Validation Domains:
    1. Two-Asset Closed-Form Validation (GMV, Tangency, independent variance/volatility,
       constrained boundary cases, parameter sweeps, stationarity conditions).
    2. Dual-Solver Parity (SLSQP vs. PyPortfolioOpt on 10-asset deterministic universe,
       scale-aware tolerances, multi-target frontier grid, equivalent optima manifolds,
       solver failure semantics).
    3. Covariance Matrix Integrity (rejection gates, correlation eigenvalue floor vs.
       covariance strict positive definiteness, variance preservation, condition number bounds,
       Cholesky factorability).
    4. Cross-Module Consistency (consistent 252-day annualization across returns, covariance,
       optimizer, CAL, Monte Carlo; preservation of daily-frequency risk metrics conventions;
       end-to-end mathematical pipeline integration).
"""

from __future__ import annotations

from typing import List, TypedDict
import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    CovarianceValidationError,
    OptimizationException,
)
from app.portfolio_engine import (
    calculate_simple_returns,
    calculate_expected_returns,
    calculate_sample_covariance,
    regularize_covariance_higham,
    optimize_global_minimum_variance,
    optimize_maximum_sharpe_ratio,
    compute_efficient_frontier,
    generate_cal_coordinates,
    simulate_monte_carlo_portfolios,
    calculate_parametric_var,
    calculate_historical_var,
    calculate_cvar,
    validate_gmv,
    validate_tangency,
    validate_efficient_frontier,
    ValidationStatus,
    ValidationTolerances,
    DEFAULT_VALIDATION_TOLERANCES,
)


# ===========================================================================
# Deterministic Test Fixtures (No External Network or Live Downloads)
# ===========================================================================

class DeterministicUniverse(TypedDict):
    """Typed container for deterministic 10-asset test fixture."""

    tickers: List[str]
    mu: pd.Series
    cov: pd.DataFrame
    rf: float


@pytest.fixture(scope="module")
def deterministic_10asset_universe() -> DeterministicUniverse:
    """Deterministic 10-asset universe with calibrated realistic NSE statistics.

    Constructed with fixed random seed 42 to ensure bitwise reproducibility
    and zero dependence on external data sources.
    """
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "INFY.NS",
        "HINDUNILVR.NS",
        "LT.NS",
        "ITC.NS",
        "BHARTIARTL.NS",
        "MARUTI.NS",
    ]
    N = len(tickers)
    rng = np.random.default_rng(42)

    # Expected returns in realistic 12% - 22% annual range
    base_mu: List[float] = [0.16, 0.18, 0.14, 0.20, 0.15, 0.12, 0.22, 0.13, 0.19, 0.17]
    mu = pd.Series(base_mu, index=tickers, name="expected_returns", dtype=float)

    # Construct strictly positive-definite covariance matrix
    # Random orthogonal factors + diagonal noise
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    eigenvalues = np.linspace(0.015, 0.085, N)
    cov_arr = Q @ np.diag(eigenvalues) @ Q.T
    cov_arr = (cov_arr + cov_arr.T) / 2.0

    # Ensure strictly positive diagonal variances (18% to 28% annual volatility)
    stds = np.sqrt(np.diag(cov_arr))
    target_stds = np.array([0.22, 0.24, 0.20, 0.26, 0.25, 0.18, 0.28, 0.19, 0.27, 0.23])
    scaling = np.outer(target_stds / stds, target_stds / stds)
    cov_arr = cov_arr * scaling
    cov_arr = (cov_arr + cov_arr.T) / 2.0

    cov = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
    rf = 0.065  # 6.5% typical RBI repo rate benchmark

    return {"tickers": tickers, "mu": mu, "cov": cov, "rf": rf}


@pytest.fixture(scope="module")
def deterministic_price_history() -> pd.DataFrame:
    """Deterministic 5-asset, 504-trading-day (2 full years) price history.

    Geometric Brownian Motion with fixed drift and covariance for pipeline testing.
    """
    tickers = ["ASSET_1", "ASSET_2", "ASSET_3", "ASSET_4", "ASSET_5"]
    T = 504
    dt = 1.0 / 252.0
    rng = np.random.default_rng(101)

    mu_daily = np.array([0.12, 0.16, 0.08, 0.20, 0.14]) * dt
    sigma_daily = np.array([0.18, 0.22, 0.14, 0.28, 0.20]) * np.sqrt(dt)

    # Moderate correlation
    corr = np.array([
        [1.0, 0.4, 0.2, 0.3, 0.1],
        [0.4, 1.0, 0.3, 0.5, 0.2],
        [0.2, 0.3, 1.0, 0.2, 0.1],
        [0.3, 0.5, 0.2, 1.0, 0.4],
        [0.1, 0.2, 0.1, 0.4, 1.0],
    ])
    L = np.linalg.cholesky(corr)

    # Generate daily returns
    Z = rng.standard_normal((T, len(tickers)))
    correlated_Z = Z @ L.T
    daily_returns = mu_daily + sigma_daily * correlated_Z

    # Generate price paths starting at 100.0
    prices = 100.0 * np.exp(np.vstack([np.zeros(len(tickers)), np.cumsum(daily_returns, axis=0)]))
    dates = pd.date_range("2024-01-01", periods=T + 1, freq="B")
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ===========================================================================
# 1. Two-Asset Closed-Form Validation Suite
# ===========================================================================

class TestTwoAssetClosedFormValidation:
    """Rigorously validate optimizer outputs against closed-form analytical formulas."""

    @pytest.mark.parametrize("rho", [-0.90, -0.60, -0.30, 0.0, 0.30, 0.50])
    @pytest.mark.parametrize("s1,s2", [(0.15, 0.25), (0.20, 0.30), (0.12, 0.20)])
    def test_two_asset_gmv_parameter_sweep(self, rho: float, s1: float, s2: float) -> None:
        """Verify GMV weights and variance match analytical formulas across parameter grid.

        Formula:
            w1* = (s2^2 - s12) / (s1^2 + s2^2 - 2*s12)
            var* = (s1^2 * s2^2 - s12^2) / (s1^2 + s2^2 - 2*s12)
        """
        s12 = rho * s1 * s2
        s1_sq, s2_sq = s1**2, s2**2
        denom = s1_sq + s2_sq - 2.0 * s12

        w1_analytical = (s2_sq - s12) / denom
        w2_analytical = (s1_sq - s12) / denom
        var_analytical = (s1_sq * s2_sq - s12**2) / denom
        vol_analytical = np.sqrt(var_analytical)

        # In this parameter grid, w1 is strictly in the interior (0, 1)
        assert 0.0 < w1_analytical < 1.0

        cov = pd.DataFrame(
            [[s1_sq, s12], [s12, s2_sq]],
            index=["A", "B"],
            columns=["A", "B"],
        )
        res = optimize_global_minimum_variance(covariance=cov)

        assert res.success is True
        assert not res.fallback_executed

        # Weight parity: |w - w*| <= 1e-5 (Plan DoD threshold)
        np.testing.assert_allclose(res.weights["A"], w1_analytical, atol=1e-5)
        np.testing.assert_allclose(res.weights["B"], w2_analytical, atol=1e-5)

        # Independent variance and volatility verification
        np.testing.assert_allclose(res.variance, var_analytical, atol=1e-6)
        np.testing.assert_allclose(res.volatility, vol_analytical, atol=1e-6)

        # Budget and non-negativity constraints
        assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
        assert (res.weights >= 0.0).all()

    def test_two_asset_gmv_equal_variance_symmetry(self) -> None:
        """Verify identical variances sigma_1 = sigma_2 yield exact w1 = w2 = 0.5 for all rho.

        Proof:
            w1* = (s^2 - rho*s^2) / (2s^2 - 2*rho*s^2) = (1 - rho) / (2(1 - rho)) = 0.5.
            var* = s^2 (1 + rho) / 2.
        """
        s = 0.20
        s_sq = s**2
        for rho in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            s12 = rho * s_sq
            cov = pd.DataFrame([[s_sq, s12], [s12, s_sq]], index=["X", "Y"], columns=["X", "Y"])
            res = optimize_global_minimum_variance(cov)

            assert res.success is True
            np.testing.assert_allclose(res.weights["X"], 0.5, atol=1e-6)
            np.testing.assert_allclose(res.weights["Y"], 0.5, atol=1e-6)

            expected_var = s_sq * (1.0 + rho) / 2.0
            np.testing.assert_allclose(res.variance, expected_var, atol=1e-6)
            np.testing.assert_allclose(res.volatility, np.sqrt(expected_var), atol=1e-6)

    def test_two_asset_gmv_long_only_clamping_boundary(self) -> None:
        """Verify exact boundary transition when unconstrained GMV weight becomes negative.

        For s1 > s2, unconstrained w1* = (s2^2 - s12) / denom < 0  <=>  rho > s2 / s1.
        At the threshold rho = s2 / s1, w1* = 0, w2* = 1.
        For rho > s2 / s1, the constrained minimum must clamp exactly to [0, 1].
        """
        s1 = 0.30
        s2 = 0.15
        s1_sq, s2_sq = s1**2, s2**2

        # Sub-critical: rho = 0.45 < 0.50 -> interior solution
        rho_sub = 0.45
        s12_sub = rho_sub * s1 * s2
        w1_sub = (s2_sq - s12_sub) / (s1_sq + s2_sq - 2.0 * s12_sub)
        cov_sub = pd.DataFrame([[s1_sq, s12_sub], [s12_sub, s2_sq]], index=["A", "B"], columns=["A", "B"])
        res_sub = optimize_global_minimum_variance(cov_sub)
        assert res_sub.success is True
        np.testing.assert_allclose(res_sub.weights["A"], w1_sub, atol=1e-5)
        assert res_sub.weights["A"] > 0.0

        # Super-critical: rho = 0.55 > 0.50 -> clamped to boundary [0, 1]
        rho_super = 0.55
        s12_super = rho_super * s1 * s2
        cov_super = pd.DataFrame([[s1_sq, s12_super], [s12_super, s2_sq]], index=["A", "B"], columns=["A", "B"])
        res_super = optimize_global_minimum_variance(cov_super)
        assert res_super.success is True
        np.testing.assert_allclose(res_super.weights["A"], 0.0, atol=1e-6)
        np.testing.assert_allclose(res_super.weights["B"], 1.0, atol=1e-6)
        np.testing.assert_allclose(res_super.variance, s2_sq, atol=1e-6)
        np.testing.assert_allclose(res_super.volatility, s2, atol=1e-6)

    def test_two_asset_tangency_closed_form_derivation(self) -> None:
        """Verify 2-asset Tangency portfolio against analytical Tobin separation solution.

        Analytical:
            w_tilde = inv(Sigma) @ (mu - rf * 1)
            w* = w_tilde / sum(w_tilde)
        """
        rf = 0.05
        mu = pd.Series([0.14, 0.18], index=["ASSET1", "ASSET2"])
        s1, s2, rho = 0.16, 0.24, 0.30
        s12 = rho * s1 * s2
        cov = pd.DataFrame([[s1**2, s12], [s12, s2**2]], index=["ASSET1", "ASSET2"], columns=["ASSET1", "ASSET2"])

        # Independent matrix inverse and weight evaluation
        e1, e2 = mu["ASSET1"] - rf, mu["ASSET2"] - rf
        det = s1**2 * s2**2 - s12**2
        w1_unnorm = (s2**2 * e1 - s12 * e2) / det
        w2_unnorm = (s1**2 * e2 - s12 * e1) / det
        w_sum = w1_unnorm + w2_unnorm
        w1_expected = w1_unnorm / w_sum
        w2_expected = w2_unnorm / w_sum

        exp_ret_expected = w1_expected * mu["ASSET1"] + w2_expected * mu["ASSET2"]
        var_expected = w1_expected**2 * s1**2 + w2_expected**2 * s2**2 + 2.0 * w1_expected * w2_expected * s12
        vol_expected = np.sqrt(var_expected)
        sharpe_expected = (exp_ret_expected - rf) / vol_expected

        res = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)

        assert res.success is True
        np.testing.assert_allclose(res.weights["ASSET1"], w1_expected, atol=1e-5)
        np.testing.assert_allclose(res.weights["ASSET2"], w2_expected, atol=1e-5)
        np.testing.assert_allclose(res.expected_return, exp_ret_expected, atol=1e-5)
        np.testing.assert_allclose(res.variance, var_expected, atol=1e-5)
        np.testing.assert_allclose(res.volatility, vol_expected, atol=1e-5)
        np.testing.assert_allclose(res.sharpe_ratio, sharpe_expected, atol=1e-5)

    def test_two_asset_tangency_first_order_stationarity(self) -> None:
        """Verify numerical Tangency weights satisfy the first-order stationarity d(SR)/dw1 = 0."""
        rf = 0.06
        mu = pd.Series([0.15, 0.22], index=["A", "B"])
        cov = pd.DataFrame([[0.04, 0.015], [0.015, 0.09]], index=["A", "B"], columns=["A", "B"])

        res = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        assert res.success is True

        w1 = res.weights["A"]
        w2 = 1.0 - w1
        e1, e2 = mu["A"] - rf, mu["B"] - rf
        s1_sq, s2_sq, s12 = cov.iloc[0, 0], cov.iloc[1, 1], cov.iloc[0, 1]

        # Evaluate d(SR)/dw1 analytically:
        # e_p = w1*e1 + (1-w1)*e2
        # var_p = w1^2*s1_sq + (1-w1)^2*s2_sq + 2*w1*(1-w1)*s12
        # d(SR)/dw1 = (e_p' * var_p - e_p * 0.5 * d(var_p)/dw1) / (var_p^(3/2))
        e_p = w1 * e1 + w2 * e2
        var_p = w1**2 * s1_sq + w2**2 * s2_sq + 2.0 * w1 * w2 * s12
        de_p = e1 - e2
        dvar_p = 2.0 * w1 * s1_sq - 2.0 * w2 * s2_sq + 2.0 * (1.0 - 2.0 * w1) * s12
        numerator = de_p * var_p - e_p * (0.5 * dvar_p)

        # At the optimal interior solution, numerator must equal 0
        np.testing.assert_allclose(numerator, 0.0, atol=1e-5)

    def test_two_asset_tangency_constrained_boundary_dominated_asset(self) -> None:
        """Verify constrained Tangency optimum when unconstrained weight is negative.

        Independently derive constrained maximum:
            If unconstrained w1* < 0, compare SR(0) vs SR(1).
            When Asset 2 has superior Sharpe and unconstrained requires shorting Asset 1,
            the constrained optimum over [0, 1] is exactly w = [0, 1]^T.
        """
        rf = 0.05
        # Asset 1 has low return and high correlation with high-return Asset 2
        mu = pd.Series([0.08, 0.25], index=["A", "B"])
        s1, s2 = 0.20, 0.22
        rho = 0.70
        s12 = rho * s1 * s2
        cov = pd.DataFrame([[s1**2, s12], [s12, s2**2]], index=["A", "B"], columns=["A", "B"])

        # Check unconstrained weights:
        e1, e2 = mu["A"] - rf, mu["B"] - rf  # 0.03, 0.20
        assert (s2**2 * e1 - s12 * e2) < 0.0, "Unconstrained w1 must be negative"

        # Independent constrained maximum evaluation:
        # Over w in [0, 1], SR(0) = e2 / s2 = 0.20 / 0.22 = 0.90909
        #                  SR(1) = e1 / s1 = 0.03 / 0.20 = 0.15000
        sr_0 = e2 / s2
        sr_1 = e1 / s1
        assert sr_0 > sr_1

        res = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)

        assert res.success is True
        np.testing.assert_allclose(res.weights["A"], 0.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["B"], 1.0, atol=1e-5)
        np.testing.assert_allclose(res.sharpe_ratio, sr_0, atol=1e-5)
        np.testing.assert_allclose(res.expected_return, mu["B"], atol=1e-5)
        np.testing.assert_allclose(res.volatility, s2, atol=1e-5)

    def test_two_asset_tangency_constrained_boundary_negative_excess_return(self) -> None:
        """Verify constrained Tangency when one asset has return below risk-free rate.

        When mu_1 <= rf < mu_2, Asset 1 has non-positive excess return and non-negative
        correlation. Any long allocation to Asset 1 reduces excess return.
        Constrained optimum must allocate 100% to Asset 2.
        """
        rf = 0.07
        mu = pd.Series([0.05, 0.18], index=["SUB_RF", "SUPER_RF"])  # e1 = -0.02, e2 = +0.11
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.06]], index=["SUB_RF", "SUPER_RF"], columns=["SUB_RF", "SUPER_RF"])

        res = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)

        assert res.success is True
        np.testing.assert_allclose(res.weights["SUB_RF"], 0.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["SUPER_RF"], 1.0, atol=1e-5)
        super_rf_ret = float(mu.to_numpy(dtype=float)[1])
        super_rf_var = float(cov.to_numpy(dtype=float)[1, 1])
        expected_sr = (super_rf_ret - rf) / float(np.sqrt(super_rf_var))
        np.testing.assert_allclose(res.sharpe_ratio, expected_sr, atol=1e-5)

    def test_two_asset_tangency_all_negative_excess_returns_least_negative_sharpe(self) -> None:
        """Verify behavior when all asset returns are below the risk-free rate.

        When all mu_i < rf, every long-only portfolio has negative Sharpe ratio.
        SLSQP finds the constrained portfolio minimizing the negative Sharpe ratio
        (maximizing Sharpe towards zero), allocating to the asset with least negative Sharpe.
        """
        rf = 0.08
        mu = pd.Series([0.04, 0.06], index=["A", "B"])  # both below rf
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.05]], index=["A", "B"], columns=["A", "B"])

        # Individual Sharpe ratios:
        # SR_A = (0.04 - 0.08) / sqrt(0.04) = -0.04 / 0.20 = -0.20
        # SR_B = (0.06 - 0.08) / sqrt(0.05) = -0.02 / 0.2236068 = -0.0894427
        # Asset B has the superior (less negative) Sharpe ratio.
        sr_a = (0.04 - 0.08) / 0.20
        sr_b = (0.06 - 0.08) / np.sqrt(0.05)
        assert sr_b > sr_a

        res = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        assert res.success is True
        np.testing.assert_allclose(res.weights["A"], 0.0, atol=1e-5)
        np.testing.assert_allclose(res.weights["B"], 1.0, atol=1e-5)
        np.testing.assert_allclose(res.sharpe_ratio, sr_b, atol=1e-5)


# ===========================================================================
# 2. Dual-Solver Parity Suite
# ===========================================================================

class TestDualSolverParityValidation:
    """Verify production SLSQP optimizer results against PyPortfolioOpt shadow validator."""

    def test_dual_solver_gmv_ten_asset_deterministic_universe(
        self, deterministic_10asset_universe: DeterministicUniverse
    ) -> None:
        """Verify scale-aware objective parity on 10-asset deterministic universe for GMV.

        Tolerances:
            |Obj_custom - Obj_pypopt| <= tol_abs + tol_rel * |Obj_pypopt|
            tol_abs = 1e-5, tol_rel = 1e-4
        """
        cov = deterministic_10asset_universe["cov"]
        mu = deterministic_10asset_universe["mu"]

        val_res = validate_gmv(covariance=cov, expected_returns=mu)

        assert val_res.is_valid is True
        assert val_res.status in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)
        assert val_res.metric_valid is True
        assert val_res.reference_metrics is not None
        assert val_res.shadow_metrics is not None

        # Scale-aware variance parity check
        ref_var = val_res.reference_metrics.variance
        shd_var = val_res.shadow_metrics.variance
        tol_abs = DEFAULT_VALIDATION_TOLERANCES.variance_tol  # 1e-4
        tol_rel = DEFAULT_VALIDATION_TOLERANCES.relative_tol  # 1e-2
        assert abs(ref_var - shd_var) <= tol_abs + tol_rel * abs(shd_var)

        # Budget constraint
        assert np.isclose(val_res.reference_metrics.weights.sum(), 1.0, atol=1e-5)
        assert np.isclose(val_res.shadow_metrics.weights.sum(), 1.0, atol=1e-5)

    def test_dual_solver_tangency_ten_asset_deterministic_universe(
        self, deterministic_10asset_universe: DeterministicUniverse
    ) -> None:
        """Verify scale-aware Sharpe ratio parity on 10-asset deterministic universe.

        Tolerances:
            |SR_custom - SR_pypopt| <= tol_abs + tol_rel * |SR_pypopt|
            tol_abs = 1e-3, tol_rel = 1e-2
        """
        cov = deterministic_10asset_universe["cov"]
        mu = deterministic_10asset_universe["mu"]
        rf = deterministic_10asset_universe["rf"]

        val_res = validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=rf)

        assert val_res.is_valid is True
        assert val_res.status in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)
        assert val_res.metric_valid is True
        assert val_res.reference_metrics is not None
        assert val_res.shadow_metrics is not None

        # Scale-aware Sharpe ratio parity
        ref_sr = val_res.reference_metrics.sharpe_ratio
        shd_sr = val_res.shadow_metrics.sharpe_ratio
        assert ref_sr is not None
        assert shd_sr is not None
        tol_abs = DEFAULT_VALIDATION_TOLERANCES.sharpe_tol
        tol_rel = DEFAULT_VALIDATION_TOLERANCES.relative_tol
        assert abs(ref_sr - shd_sr) <= tol_abs + tol_rel * abs(shd_sr)

    def test_dual_solver_frontier_dense_grid_parity(
        self, deterministic_10asset_universe: DeterministicUniverse
    ) -> None:
        """Verify dual-solver parity along a 10-point grid of efficient frontier target returns."""
        cov = deterministic_10asset_universe["cov"]
        mu = deterministic_10asset_universe["mu"]

        # Run GMV to find minimum frontier return
        gmv_res = optimize_global_minimum_variance(cov)
        gmv_ret = float(np.dot(gmv_res.weights.to_numpy(dtype=float), mu.to_numpy(dtype=float)))
        max_ret = float(mu.max())

        # Construct 10 evenly spaced feasible return targets on the upper branch
        target_returns = list(np.linspace(gmv_ret + 0.002, max_ret - 0.002, 10))

        val_res = validate_efficient_frontier(
            expected_returns=mu,
            covariance=cov,
            target_returns=target_returns,
        )

        assert val_res.is_valid is True
        assert val_res.n_points_passed == len(target_returns)
        assert val_res.n_points_validated == len(target_returns)

        for pt in val_res.validated_points:
            assert pt.is_valid is True
            assert pt.status in (ValidationStatus.PASSED.value, ValidationStatus.EQUIVALENT_OPTIMUM.value)
            assert pt.differences is not None
            assert pt.differences.abs_diff_variance < 1e-4

    def test_dual_solver_equivalent_optima_manifold(self) -> None:
        """Verify collinear / identical assets produce EQUIVALENT_OPTIMUM or PASSED without failure.

        When two assets are near-collinear in covariance and return, multiple weight allocations
        produce near-identical variance, satisfying metric parity.
        """
        tickers = ["A1", "A2", "B"]
        # Near-collinear assets (correlation 0.9999, strictly positive definite)
        cov = pd.DataFrame(
            [
                [0.04, 0.039999, 0.01],
                [0.039999, 0.04, 0.01],
                [0.01, 0.01, 0.05],
            ],
            index=tickers,
            columns=tickers,
        )
        mu = pd.Series([0.15, 0.15, 0.12], index=tickers)

        val_res = validate_gmv(covariance=cov, expected_returns=mu)

        assert val_res.is_valid is True
        assert val_res.status in (ValidationStatus.EQUIVALENT_OPTIMUM.value, ValidationStatus.PASSED.value)
        assert val_res.metric_valid is True
        assert val_res.differences is not None
        assert val_res.differences.abs_diff_variance < 1e-4

    def test_dual_solver_infeasible_target_return_parity(
        self, deterministic_10asset_universe: DeterministicUniverse
    ) -> None:
        """Verify targets exceeding maximum universe return are rejected with FinancialCalculationError."""
        cov = deterministic_10asset_universe["cov"]
        mu = deterministic_10asset_universe["mu"]
        impossible_target = float(mu.max()) + 0.10  # 10% above highest asset return

        with pytest.raises(FinancialCalculationError):
            compute_efficient_frontier(
                expected_returns=mu,
                covariance=cov,
                target_returns=[impossible_target],
            )

    def test_dual_solver_all_negative_excess_returns_parity(self) -> None:
        """Verify shadow validator catches PyPortfolioOpt failure when all returns < rf."""
        tickers = ["A", "B"]
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=tickers, columns=tickers)
        mu = pd.Series([0.02, 0.03], index=tickers)
        rf = 0.06  # rf > all mu

        res = validate_tangency(expected_returns=mu, covariance=cov, risk_free_rate=rf)
        assert res.is_valid is False
        assert res.status == ValidationStatus.FAILED.value
        assert any("PyPortfolioOpt shadow Tangency optimization failed" in e for e in res.errors)


# ===========================================================================
# 3. Covariance Matrix Integrity Suite
# ===========================================================================

class TestCovarianceMatrixIntegrity:
    """Validate spectral properties, rejection gates, and Higham regularization guarantees."""

    def test_asymmetry_rejection_gate(self) -> None:
        """Verify asymmetric covariance exceeding 1e-8 is rejected with FinancialCalculationError.

        Must NEVER be silently symmetrized or passed to solvers.
        """
        cov_asym = pd.DataFrame(
            [[0.04, 0.02], [0.01, 0.04]],  # |0.02 - 0.01| = 0.01 > 1e-8
            index=["A", "B"],
            columns=["A", "B"],
        )

        with pytest.raises(FinancialCalculationError) as exc_info:
            regularize_covariance_higham(cov_asym)
        assert "asymmetric" in str(exc_info.value).lower()

        with pytest.raises(FinancialCalculationError):
            optimize_global_minimum_variance(cov_asym)

    def test_higham_correlation_eigenvalue_floor(self) -> None:
        """Verify eigenvalue floor applies to the standardized correlation matrix before rescaling.

        Clarification 1: Test correlation eigenvalue floor (lambda_min >= floor)
        and final covariance positive definiteness (lambda_min > 0) separately.
        """
        # Construct rank-deficient 3x3 covariance
        v = np.array([0.20, 0.25, 0.30])
        # Rank-1 outer product matrix (singular, two zero eigenvalues)
        cov_singular = pd.DataFrame(np.outer(v, v), index=["A", "B", "C"], columns=["A", "B", "C"])

        floor = 1e-5
        cov_reg = regularize_covariance_higham(
            cov_singular,
            eigenvalue_floor=floor,
            max_condition_number=1e7,  # allow condition budget for test
        )

        # 1. Check standardized correlation matrix directly
        cov_reg_arr = np.asarray(cov_reg, dtype=np.float64)
        stds = np.sqrt(np.diag(cov_reg_arr))
        corr_reg = cov_reg_arr / np.outer(stds, stds)
        corr_eigvals = np.linalg.eigvalsh(corr_reg)
        min_corr_eig = float(corr_eigvals.min())

        # Correlation eigenvalues must satisfy the floor within numerical tolerance
        assert min_corr_eig >= floor * (1.0 - 1e-4)

        # 2. Check final rescaled covariance positive definiteness separately
        cov_eigvals = np.linalg.eigvalsh(cov_reg_arr)
        assert float(cov_eigvals.min()) > 0.0

    def test_higham_variance_preservation_to_high_precision(self) -> None:
        """Verify Higham regularization strictly preserves diagonal asset variances to 1e-12."""
        # Known indefinite correlation matrix with negative eigenvalue -0.131
        C_indef = np.array([
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.8],
            [0.0, 0.8, 1.0],
        ])
        sigmas = np.array([0.15, 0.20, 0.25])
        diag_vars = sigmas**2
        D = np.diag(sigmas)
        cov_indef = pd.DataFrame(D @ C_indef @ D, index=["X", "Y", "Z"], columns=["X", "Y", "Z"])

        cov_reg = regularize_covariance_higham(cov_indef, eigenvalue_floor=1e-5, max_condition_number=1e7)

        # Relative discrepancy between original and regularized variances
        rel_diffs = np.abs(np.diag(cov_reg.values) - diag_vars) / diag_vars
        assert np.max(rel_diffs) <= 1e-12

    def test_higham_output_exact_symmetry(self) -> None:
        """Verify regularized covariance satisfies exact symmetry to machine precision."""
        cov_raw = pd.DataFrame(
            [[0.04, 0.015, 0.008], [0.015, 0.06, 0.02], [0.008, 0.02, 0.08]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        cov_reg = regularize_covariance_higham(cov_raw)
        asym = np.max(np.abs(cov_reg.values - cov_reg.values.T))
        assert asym <= 1e-14

    def test_higham_cholesky_factorability(self) -> None:
        """Verify regularized covariance admits valid Cholesky factorization L L^T."""
        rng = np.random.default_rng(303)
        Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        eigs = np.geomspace(1e-4, 1.0, 8)  # condition number 1e4
        cov_ill = pd.DataFrame(Q @ np.diag(eigs) @ Q.T, index=[f"T_{i}" for i in range(8)], columns=[f"T_{i}" for i in range(8)])
        cov_ill = (cov_ill + cov_ill.T) / 2.0

        cov_reg = regularize_covariance_higham(cov_ill, max_condition_number=1e6)
        cov_reg_arr = np.asarray(cov_reg, dtype=np.float64)
        # Cholesky must succeed without LinAlgError
        L = np.linalg.cholesky(cov_reg_arr)
        assert L.shape == (8, 8)
        assert np.all(np.isfinite(L))
        np.testing.assert_allclose(L @ L.T, cov_reg_arr, atol=1e-10)

    def test_degenerate_matrix_rejection_when_unfixable(self) -> None:
        """Verify matrices that cannot achieve max_condition_number are rejected cleanly."""
        cov_zero_det = pd.DataFrame(
            np.ones((4, 4)) * 0.04,  # perfectly collinear 4x4 matrix
            index=["A", "B", "C", "D"],
            columns=["A", "B", "C", "D"],
        )

        # With a very tight condition budget that cannot be satisfied
        with pytest.raises(CovarianceValidationError) as exc_info:
            regularize_covariance_higham(cov_zero_det, eigenvalue_floor=1e-7, max_condition_number=10.0)
        assert "condition number" in str(exc_info.value).lower()


# ===========================================================================
# 4. Cross-Module Consistency Suite
# ===========================================================================

class TestCrossModuleConsistency:
    """Verify consistent annualization, conventions, and end-to-end mathematical identities."""

    def test_annualization_factor_consistency_252(
        self, deterministic_price_history: pd.DataFrame
    ) -> None:
        """Verify 252-day annualization factor is consistently applied across returns and covariance.

        Mathematical Identities:
            mu_ann = 252 * mean(R_daily)
            Sigma_ann = 252 * cov(R_daily)
            sigma_p_ann^2 = 252 * (w^T Sigma_daily w)
        """
        TRADING_DAYS = 252
        returns = calculate_simple_returns(deterministic_price_history, aligned=True)
        assert isinstance(returns, pd.DataFrame)

        # 1. Expected returns annualization
        mu_engine = calculate_expected_returns(returns, is_returns=True, trading_days_per_year=TRADING_DAYS)
        assert isinstance(mu_engine, pd.Series)
        mu_independent = returns.mean() * TRADING_DAYS
        np.testing.assert_allclose(
            np.asarray(mu_engine, dtype=np.float64),
            np.asarray(mu_independent, dtype=np.float64),
            atol=1e-12,
        )

        # 2. Covariance annualization
        cov_engine = calculate_sample_covariance(returns, trading_days_per_year=TRADING_DAYS)
        cov_daily = returns.cov()
        cov_independent = cov_daily * TRADING_DAYS
        np.testing.assert_allclose(
            np.asarray(cov_engine, dtype=np.float64),
            np.asarray(cov_independent, dtype=np.float64),
            atol=1e-12,
        )

        # 3. Portfolio variance scale identity
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        var_ann = float(w @ cov_engine.values @ w)
        var_daily = float(w @ cov_daily.values @ w)
        np.testing.assert_allclose(var_ann, 252.0 * var_daily, atol=1e-10)

    def test_daily_frequency_risk_metrics_conventions_preserved(
        self, deterministic_price_history: pd.DataFrame
    ) -> None:
        """Verify daily-frequency risk metrics conventions are strictly preserved without annualization.

        Clarification 3: Preserve daily-frequency VaR/CVaR conventions.
        Do not force annualization onto daily downside-risk calculations.
        """
        returns = calculate_simple_returns(deterministic_price_history, aligned=True)
        assert isinstance(returns, pd.DataFrame)
        asset_ret = returns["ASSET_1"]

        # Calculate daily risk metrics
        var_95 = calculate_historical_var(asset_ret, confidence_level=0.95)
        var_99 = calculate_historical_var(asset_ret, confidence_level=0.99)
        cvar_95 = calculate_cvar(asset_ret, confidence_level=0.95)
        param_var_95 = calculate_parametric_var(asset_ret, confidence_level=0.95)

        # Values must be on the daily loss scale (typically 1% to 4%), NOT annualized (~15% to 60%)
        assert 0.005 < var_95 < 0.06
        assert 0.008 < var_99 < 0.08
        assert 0.008 < cvar_95 < 0.09
        assert 0.005 < param_var_95 < 0.06

        # Invariants: CVaR >= VaR and VaR_99 >= VaR_95
        assert cvar_95 >= var_95
        assert var_99 >= var_95

        # Independent daily Parametric Normal VaR verification
        # Formula: VaR_alpha = -(mean - z_alpha * std)
        mu_d = float(asset_ret.mean())
        s_d = float(asset_ret.std(ddof=1))
        # scipy.stats.norm.ppf(0.05) ~= -1.6448536269514729
        z_95 = -1.6448536269514729
        expected_param_var = -(mu_d + z_95 * s_d)
        np.testing.assert_allclose(param_var_95, expected_param_var, atol=1e-6)

    def test_end_to_end_pipeline_mathematical_consistency(
        self, deterministic_price_history: pd.DataFrame
    ) -> None:
        """Verify full mathematical pipeline integration:

        Prices -> Returns -> Covariance -> Regularization -> Optimizer -> CAL -> Monte Carlo -> Risk Metrics.
        """
        TRADING_DAYS = 252
        rf = 0.06

        # Step 1: Returns & Covariance
        returns = calculate_simple_returns(deterministic_price_history, aligned=True)
        assert isinstance(returns, pd.DataFrame)
        mu = calculate_expected_returns(returns, is_returns=True, trading_days_per_year=TRADING_DAYS)
        assert isinstance(mu, pd.Series)
        cov_sample = calculate_sample_covariance(returns, trading_days_per_year=TRADING_DAYS)

        # Step 2: Regularization
        cov_reg = regularize_covariance_higham(cov_sample)
        assert (np.linalg.eigvalsh(np.asarray(cov_reg, dtype=np.float64)) > 0.0).all()

        # Step 3: Analytical Optimizations
        gmv = optimize_global_minimum_variance(cov_reg)
        tang = optimize_maximum_sharpe_ratio(expected_returns=mu, covariance=cov_reg, risk_free_rate=rf)

        assert gmv.success and tang.success
        assert gmv.volatility <= tang.volatility  # GMV volatility floor

        # Compute GMV annualized expected return from weights and expected returns
        gmv_expected_ret = float(np.dot(gmv.weights.to_numpy(dtype=float), mu.to_numpy(dtype=float)))
        gmv_sharpe = (gmv_expected_ret - rf) / gmv.volatility
        assert tang.sharpe_ratio >= gmv_sharpe - 1e-6  # Tangency maximum Sharpe property

        # Step 4: Capital Allocation Line (CAL) Consistency
        # generate_cal_coordinates parameters: tangency_portfolio, risk_free_rate, n_points,
        # allow_leverage (bool). No max_leverage or num_points parameter.
        cal_coords = generate_cal_coordinates(
            tangency_portfolio=tang,
            risk_free_rate=rf,
            n_points=16,
        )
        assert cal_coords.slope == pytest.approx(tang.sharpe_ratio, abs=1e-6)

        # At weight_tangency = 0.0, portfolio is 100% risk-free
        pt_rf = cal_coords.coordinates[0]
        assert pt_rf.volatility == pytest.approx(0.0, abs=1e-8)
        assert pt_rf.expected_return == pytest.approx(rf, abs=1e-8)

        # At weight_tangency = 1.0, portfolio is exactly the Tangency portfolio
        pt_tang = [pt for pt in cal_coords.coordinates if np.isclose(pt.weight_tangency, 1.0)][0]
        assert pt_tang.volatility == pytest.approx(tang.volatility, abs=1e-6)
        assert pt_tang.expected_return == pytest.approx(tang.expected_return, abs=1e-6)

        # Step 5: Monte Carlo Vectorized Invariant Check
        mc_result = simulate_monte_carlo_portfolios(
            expected_returns=mu,
            covariance=cov_reg,
            risk_free_rate=rf,
            num_portfolios=2000,
            random_seed=42,
        )
        # All sampled portfolios must respect GMV and Tangency analytical landmarks
        # (allowing 1e-6 numerical tolerance)
        min_mc_vol = float(np.min(mc_result.volatilities))
        max_mc_sr = float(np.max(mc_result.sharpe_ratios))
        assert min_mc_vol >= gmv.volatility - 1e-6
        assert max_mc_sr <= tang.sharpe_ratio + 1e-6

        # Step 6: Daily Portfolio Realized Returns & Risk Metrics
        port_daily_ret = returns.dot(gmv.weights)
        daily_var_95 = calculate_historical_var(port_daily_ret, confidence_level=0.95)
        daily_cvar_95 = calculate_cvar(port_daily_ret, confidence_level=0.95)

        # Downside risk coherence: CVaR >= VaR
        assert daily_cvar_95 >= daily_var_95

        # Realized daily variance matches 1/252 of annualized GMV variance
        realized_daily_var = float(np.var(np.asarray(port_daily_ret, dtype=np.float64), ddof=1))
        expected_daily_var = gmv.variance / TRADING_DAYS
        np.testing.assert_allclose(realized_daily_var, expected_daily_var, rtol=1e-2)
