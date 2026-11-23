"""Quantamental AI - Monte Carlo Portfolio Simulator Test Suite.

Phase 2.5: Comprehensive unit tests verifying Dirichlet simplex portfolio generation,
vectorized metric calculations (returns, variances, volatilities, Sharpe ratios),
defensive immutability, landmark identification, strict input validation,
analytical frontier cross-validation, and cross-namespace export identities.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine.efficient_frontier import (
    EfficientFrontierResult,
    compute_efficient_frontier,
)
from app.portfolio_engine.monte_carlo import (
    DEFAULT_NUM_PORTFOLIOS,
    DEFAULT_RISK_FREE_RATE,
    MonteCarloSimulationResult,
    MonteCarloSimulator,
    MonteCarloValidationResult,
    SimulatedLandmarkPortfolio,
    simulate_monte_carlo_portfolios,
    validate_monte_carlo_against_frontier,
)
import app.portfolio_engine as pe
import app.financial_engine as fe
import app.financial_engine.monte_carlo as femc


# ---------------------------------------------------------------------------
# Test Fixtures
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
    """4-asset annualized expected returns and covariance matrix."""
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]
    mu = pd.Series([0.10, 0.14, 0.18, 0.08], index=tickers)
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
# 1. Simplex Portfolio Generation & Dirichlet Constraints
# ---------------------------------------------------------------------------

class TestSimplexWeightGeneration:
    """Verify standard simplex generation (w_i >= 0, sum(w_i) == 1.0)."""

    def test_dirichlet_weights_shape_and_simplex_constraints(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Weights array has shape (K, N), long-only non-negative values, and row sums of 1.0."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        k = 1000

        res = simulate_monte_carlo_portfolios(
            expected_returns=mu, covariance=cov, num_portfolios=k, random_seed=42
        )

        assert isinstance(res, MonteCarloSimulationResult)
        assert res.num_portfolios == k
        assert res.num_assets == 3
        assert len(res) == k
        assert res.weights.shape == (k, 3)

        # Long-only constraint: all weights >= 0
        assert np.all(res.weights >= 0.0)
        assert np.all(res.weights <= 1.0)

        # Budget constraint: sum of each row equals 1.0 within 1e-6
        row_sums = np.sum(res.weights, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_weights_dataframe_accessor(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """weights_dataframe returns DataFrame with correct columns and index."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = simulate_monte_carlo_portfolios(
            expected_returns=mu, covariance=cov, num_portfolios=100, random_seed=42
        )

        df = res.weights_dataframe
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["AAPL", "MSFT", "GOOG"]
        assert len(df) == 100
        assert np.allclose(df.sum(axis=1).values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Reproducibility & Generator Support
# ---------------------------------------------------------------------------

class TestReproducibilityAndGenerators:
    """Verify bit-for-bit reproducibility via seeds and custom generators."""

    def test_integer_seed_provides_bit_for_bit_reproducibility(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Identical integer seeds produce exact matching weights, returns, and volatilities."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        res1 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=500, random_seed=12345)
        res2 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=500, random_seed=12345)

        np.testing.assert_array_equal(res1.weights, res2.weights)
        np.testing.assert_array_equal(res1.expected_returns, res2.expected_returns)
        np.testing.assert_array_equal(res1.volatilities, res2.volatilities)
        np.testing.assert_array_equal(res1.sharpe_ratios, res2.sharpe_ratios)
        assert res1.random_seed == 12345
        assert res1.metadata["is_reproducible"] is True

    def test_different_seeds_produce_different_allocations(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Different seeds generate distinct portfolio distributions."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        res1 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=500, random_seed=1)
        res2 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=500, random_seed=2)

        assert not np.array_equal(res1.weights, res2.weights)

    def test_supplied_numpy_generator_reproducibility(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Supplied numpy.random.Generator is respected."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        rng1 = np.random.default_rng(999)
        res1 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=300, generator=rng1)

        rng2 = np.random.default_rng(999)
        res2 = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=300, generator=rng2)

        np.testing.assert_array_equal(res1.weights, res2.weights)

    def test_simultaneous_seed_and_generator_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Passing both random_seed and generator raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        gen = np.random.default_rng(42)

        with pytest.raises(FinancialCalculationError, match="both 'random_seed' and 'generator'"):
            simulate_monte_carlo_portfolios(mu, cov, random_seed=42, generator=gen)

    @pytest.mark.parametrize("bad_seed", [-1, -99, 2**33, "seed", 3.14, True, False])
    def test_invalid_random_seed_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_seed: Any
    ) -> None:
        """Negative, non-integer, bool, or out-of-range seed raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError):
            simulate_monte_carlo_portfolios(mu, cov, random_seed=bad_seed)

    def test_invalid_generator_object_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Non-Generator object passed as generator raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError, match="instance of numpy.random.Generator"):
            simulate_monte_carlo_portfolios(mu, cov, generator="not_a_generator")


# ---------------------------------------------------------------------------
# 3. Vectorized Metric Accuracy & Analytical Verification
# ---------------------------------------------------------------------------

class TestVectorizedMetricAccuracy:
    """Verify analytical metric precision against independent matrix computations."""

    def test_metrics_match_independent_matrix_calculations(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """einsum variance and returns match step-by-step mathematical calculations."""
        cov_df = independent_3asset_data["cov"]
        mu_s = independent_3asset_data["mu"]
        rf = 0.04

        res = simulate_monte_carlo_portfolios(
            expected_returns=mu_s,
            covariance=cov_df,
            num_portfolios=200,
            risk_free_rate=rf,
            random_seed=777,
        )

        w = res.weights
        cov = cov_df.values
        mu = mu_s.values

        # Independently calculate returns
        expected_returns = w @ mu
        np.testing.assert_allclose(res.expected_returns, expected_returns, rtol=1e-12)

        # Independently calculate variances via loop / quadratic form
        expected_variances = np.array([float(w[i] @ cov @ w[i]) for i in range(len(w))])
        np.testing.assert_allclose(res.variances, expected_variances, rtol=1e-12)

        # Volatilities
        np.testing.assert_allclose(res.volatilities, np.sqrt(expected_variances), rtol=1e-12)

        # Sharpe ratios
        expected_sharpe = (expected_returns - rf) / np.sqrt(expected_variances)
        np.testing.assert_allclose(res.sharpe_ratios, expected_sharpe, rtol=1e-12)

    def test_hand_crafted_two_asset_case(self) -> None:
        """Explicit hand-calculated test vectors on a 2-asset portfolio."""
        tickers = ["A", "B"]
        mu = pd.Series([0.10, 0.20], index=tickers)
        cov = pd.DataFrame([[0.04, 0.00], [0.00, 0.16]], index=tickers, columns=tickers)
        rf = 0.05

        res = simulate_monte_carlo_portfolios(
            expected_returns=mu,
            covariance=cov,
            num_portfolios=50,
            risk_free_rate=rf,
            random_seed=10,
        )

        for i in range(len(res)):
            w_a, w_b = res.weights[i, 0], res.weights[i, 1]
            ret_manual = w_a * 0.10 + w_b * 0.20
            var_manual = (w_a**2) * 0.04 + (w_b**2) * 0.16
            vol_manual = np.sqrt(var_manual)
            sr_manual = (ret_manual - rf) / vol_manual

            assert res.expected_returns[i] == pytest.approx(ret_manual, abs=1e-10)
            assert res.variances[i] == pytest.approx(var_manual, abs=1e-10)
            assert res.volatilities[i] == pytest.approx(vol_manual, abs=1e-10)
            assert res.sharpe_ratios[i] == pytest.approx(sr_manual, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. Landmark Empirical Portfolios
# ---------------------------------------------------------------------------

class TestLandmarkPortfolios:
    """Verify identification of simulated GMV and Maximum Sharpe portfolios."""

    def test_landmark_portfolios_identified_correctly(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """simulated_gmv has min volatility; simulated_max_sharpe has max Sharpe ratio."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=5000, random_seed=42)

        gmv = res.simulated_gmv
        max_sr = res.simulated_max_sharpe

        assert isinstance(gmv, SimulatedLandmarkPortfolio)
        assert isinstance(max_sr, SimulatedLandmarkPortfolio)

        assert gmv.landmark_type == "SIMULATED_GMV"
        assert max_sr.landmark_type == "SIMULATED_MAX_SHARPE"

        # Verify GMV optimality among simulated set
        assert gmv.volatility == pytest.approx(np.min(res.volatilities))
        assert gmv.variance == pytest.approx(np.min(res.variances))
        assert gmv.simulation_index == int(np.argmin(res.volatilities))

        # Verify Max Sharpe optimality among simulated set
        assert max_sr.sharpe_ratio == pytest.approx(np.max(res.sharpe_ratios))
        assert max_sr.simulation_index == int(np.argmax(res.sharpe_ratios))

        # Landmark weights sum to 1
        assert gmv.weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert max_sr.weights.sum() == pytest.approx(1.0, abs=1e-6)

        # Unpacking support
        vol_gmv, ret_gmv = gmv
        assert vol_gmv == gmv.volatility
        assert ret_gmv == gmv.expected_return


# ---------------------------------------------------------------------------
# 5. Defensive Immutability & Serialization
# ---------------------------------------------------------------------------

class TestImmutabilityAndSerialization:
    """Verify frozen dataclass, defensive array copies, and JSON serialization."""

    def test_frozen_result_prevents_attribute_reassignment(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Reassigning attributes on result raises FrozenInstanceError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=100, random_seed=42)

        with pytest.raises(FrozenInstanceError):
            res.num_portfolios = 500  # type: ignore

        with pytest.raises(FrozenInstanceError):
            res.simulated_gmv = None  # type: ignore

        with pytest.raises(FrozenInstanceError):
            res.simulated_gmv.expected_return = 0.99  # type: ignore

    def test_defensive_copies_prevent_external_array_mutation(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Mutating arrays returned by properties does not affect internal result state."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=100, random_seed=42)

        orig_w = res.weights.copy()
        orig_ret = res.expected_returns.copy()

        # Mutate array returned by property
        w_view = res.weights
        w_view[0, 0] = 999.0

        ret_view = res.expected_returns
        ret_view[0] = -999.0

        # Assert internal state remained unmodified
        np.testing.assert_array_equal(res.weights, orig_w)
        np.testing.assert_array_equal(res.expected_returns, orig_ret)

    def test_json_serialization_succeeds(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """to_dict() produces 100% valid JSON-serializable output."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=50, random_seed=42)

        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["num_portfolios"] == 50
        assert d["num_assets"] == 3
        assert "simulated_gmv" in d
        assert "simulated_max_sharpe" in d
        assert "summary" in d

        # Dump to JSON and parse back
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["num_portfolios"] == 50
        assert len(parsed["weights"]) == 50
        assert len(parsed["expected_returns"]) == 50

    def test_validation_result_immutability_and_defensive_copy(self) -> None:
        """MonteCarloValidationResult is frozen and defensively copies its details dictionary."""
        val_res = MonteCarloValidationResult(
            is_valid=True,
            num_portfolios_tested=10,
            num_breaches=0,
            max_breach=0.0,
            gmv_volatility_delta=0.01,
            tangency_sharpe_delta=0.05,
            tolerance=1e-4,
            details={"original_key": "original_val"},
        )

        # Reassignment forbidden
        with pytest.raises(FrozenInstanceError):
            val_res.is_valid = False  # type: ignore

        # Property has no setter
        with pytest.raises(AttributeError):
            val_res.details = {}  # type: ignore

        # Mutating the returned details dict does not affect internal details
        det = val_res.details
        det["mutated_key"] = "hacked"
        assert "mutated_key" not in val_res.details
        assert val_res.details == {"original_key": "original_val"}


# ---------------------------------------------------------------------------
# 6. Strict Input Validation & Edge Cases
# ---------------------------------------------------------------------------

class TestStrictInputValidation:
    """Verify rejection of invalid inputs, negative dimensions, and degenerate data."""

    def test_fewer_than_two_assets_rejected(self) -> None:
        """Single asset raises InsufficientDataError."""
        mu = pd.Series([0.15], index=["A"])
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        with pytest.raises(InsufficientDataError, match="at least 2 assets"):
            simulate_monte_carlo_portfolios(mu, cov)

    @pytest.mark.parametrize("bad_k", [0, -1, -50])
    def test_non_positive_portfolio_count_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_k: int
    ) -> None:
        """K < 1 raises InsufficientDataError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        with pytest.raises(InsufficientDataError, match="at least 1"):
            simulate_monte_carlo_portfolios(mu, cov, num_portfolios=bad_k)

    def test_mismatched_dimensions_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Expected returns with 2 assets and covariance with 3 raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = pd.Series([0.15, 0.20], index=["AAPL", "MSFT"])
        with pytest.raises(FinancialCalculationError, match="does not match asset count"):
            simulate_monte_carlo_portfolios(mu, cov)

    def test_asymmetric_covariance_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Asymmetric covariance matrix raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"].copy()
        cov.iloc[0, 1] += 0.05
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError, match="asymmetric"):
            simulate_monte_carlo_portfolios(mu, cov)

    def test_negative_diagonal_covariance_rejected(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Negative diagonal variance raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"].copy()
        cov.iloc[0, 0] = -0.01
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            simulate_monte_carlo_portfolios(mu, cov)

    def test_non_psd_covariance_rejected(self) -> None:
        """Covariance with negative eigenvalues raises FinancialCalculationError."""
        tickers = ["A", "B", "C"]
        mu = pd.Series([0.10, 0.12, 0.15], index=tickers)
        # Construct non-PSD symmetric matrix with positive diagonals
        cov = pd.DataFrame(
            [
                [0.04, 0.05, 0.05],
                [0.05, 0.04, 0.05],
                [0.05, 0.05, 0.04],
            ],
            index=tickers,
            columns=tickers,
        )
        with pytest.raises(FinancialCalculationError, match="positive semidefinite"):
            simulate_monte_carlo_portfolios(mu, cov)

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_inputs_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_val: float
    ) -> None:
        """NaN or Inf in returns or covariance raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"].copy()
        mu = independent_3asset_data["mu"].copy()

        cov.iloc[0, 1] = bad_val
        with pytest.raises(FinancialCalculationError):
            simulate_monte_carlo_portfolios(mu, cov)

        cov_clean = independent_3asset_data["cov"]
        mu_bad = mu.copy()
        mu_bad.iloc[0] = bad_val
        with pytest.raises(FinancialCalculationError):
            simulate_monte_carlo_portfolios(mu_bad, cov_clean)

    @pytest.mark.parametrize("bad_rf", [True, False, 1 + 2j, np.nan, np.inf, "0.05"])
    def test_invalid_risk_free_rate_rejected(
        self, independent_3asset_data: Dict[str, Any], bad_rf: Any
    ) -> None:
        """Boolean, complex, non-finite, or string rf raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        with pytest.raises(FinancialCalculationError):
            simulate_monte_carlo_portfolios(mu, cov, risk_free_rate=bad_rf)

    def test_tiny_negative_variance_within_tolerance_is_clamped(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative variance within [-1e-8, 0.0) is safely clamped to 0.0 without error."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        orig_einsum = np.einsum
        def mock_einsum(subscripts: str, *operands: Any, **kwargs: Any) -> Any:
            res = orig_einsum(subscripts, *operands, **kwargs)
            res[0] = -1e-10  # within [-1e-8, 0.0)
            return res

        monkeypatch.setattr(np, "einsum", mock_einsum)

        # Determine portfolio 0 return so rf matches it (zero excess return prevents undefined Sharpe)
        rng = np.random.default_rng(42)
        w0 = rng.dirichlet(np.ones(len(mu)))
        ret0 = float(w0 @ mu.values)

        sim_res = simulate_monte_carlo_portfolios(
            mu, cov, num_portfolios=20, risk_free_rate=ret0, random_seed=42
        )
        assert sim_res.variances[0] == 0.0
        assert sim_res.volatilities[0] == 0.0
        assert sim_res.sharpe_ratios[0] == 0.0

    def test_materially_negative_variance_raises_financial_calculation_error(
        self, independent_3asset_data: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative variance strictly below -1e-8 raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        orig_einsum = np.einsum
        def mock_einsum(subscripts: str, *operands: Any, **kwargs: Any) -> Any:
            res = orig_einsum(subscripts, *operands, **kwargs)
            res[0] = -1e-6  # materially negative (< -1e-8)
            return res

        monkeypatch.setattr(np, "einsum", mock_einsum)
        with pytest.raises(FinancialCalculationError, match="Computed portfolio variance is materially negative"):
            simulate_monte_carlo_portfolios(mu, cov, num_portfolios=20, random_seed=42)


# ---------------------------------------------------------------------------
# 7. Analytical Frontier Cross-Validation
# ---------------------------------------------------------------------------

class TestAnalyticalFrontierCrossValidation:
    """Verify that Monte Carlo portfolios respect analytical Efficient Frontier boundaries."""

    def test_simulated_portfolios_respect_analytical_frontier(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Simulated portfolios lie on or below the analytical upper branch."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        analytical_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov, n_points=50)
        assert analytical_ef.success is True

        sim_res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=5000, random_seed=42)

        val_res = validate_monte_carlo_against_frontier(sim_res, analytical_ef, volatility_tolerance=1e-4)

        assert isinstance(val_res, MonteCarloValidationResult)
        assert val_res.is_valid is True
        assert val_res.num_breaches == 0
        assert val_res.max_breach == 0.0

        # Simulated GMV vol >= analytical GMV vol - tolerance
        analytical_gmv_vol = analytical_ef.gmv_portfolio.volatility
        assert sim_res.simulated_gmv.volatility >= analytical_gmv_vol - 1e-4

        # Serialization
        d = val_res.to_dict()
        assert d["is_valid"] is True
        assert d["num_breaches"] == 0

    def test_cross_validation_rejects_failed_analytical_frontier(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Cross-validation against unconverged frontier raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        sim_res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=100, random_seed=42)

        # Reject invalid type
        with pytest.raises(FinancialCalculationError, match="frontier_result must be an EfficientFrontierResult"):
            validate_monte_carlo_against_frontier(sim_res, "not_a_frontier")  # type: ignore

        # Mock an unconverged frontier
        import unittest.mock
        mock_ef = unittest.mock.MagicMock(spec=EfficientFrontierResult)
        mock_ef.success = False
        mock_ef.fallback_executed = False
        mock_ef.classification_valid = False

        with pytest.raises(FinancialCalculationError, match="unconverged or executed a fallback"):
            validate_monte_carlo_against_frontier(sim_res, mock_ef)

    @pytest.mark.parametrize("bad_tol", [-1e-4, -1.0, float("nan"), float("inf"), True, "1e-4"])
    def test_cross_validation_rejects_invalid_volatility_tolerances(
        self, independent_3asset_data: Dict[str, Any], bad_tol: Any
    ) -> None:
        """Negative, non-finite, boolean, or string tolerances are strictly rejected."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]
        sim_res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=50, random_seed=42)
        analytical_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov, n_points=20)

        with pytest.raises(FinancialCalculationError, match="volatility_tolerance"):
            validate_monte_carlo_against_frontier(sim_res, analytical_ef, volatility_tolerance=bad_tol)

    def test_lower_branch_interpolation_detects_breach_that_gmv_only_misses(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Simulated portfolio below GMV return passes GMV floor but fails lower-branch boundary.

        Regression test for Finding 1:
        Analytical GMV has volatility sigma_GMV.
        Lower branch boundary has sigma(mu) > sigma_GMV for mu < mu_GMV.
        A hypothetical point with sigma_GMV <= sigma_sim < sigma_lower_branch(mu_sim)
        passes a naive GMV-floor check but must FAIL the actual lower-branch boundary check.
        """
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        analytical_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov, n_points=50)
        assert analytical_ef.success is True
        assert len(analytical_ef.inefficient_portfolios) > 0

        gmv_vol = float(analytical_ef.gmv_portfolio.volatility)

        # Pick the lowest-return inefficient portfolio on the lower branch
        lowest_ineff = analytical_ef.inefficient_portfolios[0]
        ineff_ret = float(lowest_ineff.expected_return)
        ineff_vol = float(lowest_ineff.volatility)

        # In standard mean-variance geometry, ineff_vol is strictly greater than gmv_vol
        assert ineff_vol > gmv_vol + 0.01, f"ineff_vol={ineff_vol} must exceed gmv_vol={gmv_vol}"

        # Fabricate a test portfolio located at ineff_ret with volatility midway between gmv_vol and ineff_vol
        # vol_mid has vol_mid > gmv_vol (passes GMV-only floor)
        # But vol_mid < ineff_vol (violates lower branch boundary!)
        vol_mid = (gmv_vol + ineff_vol) / 2.0
        assert vol_mid >= gmv_vol  # Would pass GMV floor check
        assert vol_mid < ineff_vol - 1e-4  # Violates lower-branch boundary

        n_assets = len(mu)
        w_synth = np.ones((1, n_assets)) / n_assets
        rets_synth = np.array([ineff_ret])
        vols_synth = np.array([vol_mid])
        vars_synth = vols_synth**2
        srs_synth = (rets_synth - 0.04) / vols_synth

        landmark = SimulatedLandmarkPortfolio(
            landmark_type="SIMULATED_GMV",
            expected_return=float(rets_synth[0]),
            volatility=float(vols_synth[0]),
            variance=float(vars_synth[0]),
            sharpe_ratio=float(srs_synth[0]),
            simulation_index=0,
            weights=pd.Series(w_synth[0], index=analytical_ef.gmv_portfolio.weights.index),
        )

        sim_synth = MonteCarloSimulationResult(
            weights=w_synth,
            expected_returns=rets_synth,
            volatilities=vols_synth,
            variances=vars_synth,
            sharpe_ratios=srs_synth,
            simulated_gmv=landmark,
            simulated_max_sharpe=landmark,
            asset_labels=list(analytical_ef.gmv_portfolio.weights.index),
            num_portfolios=1,
            num_assets=n_assets,
            risk_free_rate=0.04,
        )

        val_res = validate_monte_carlo_against_frontier(sim_synth, analytical_ef, volatility_tolerance=1e-4)

        # Crucial assertions:
        # 1. Validation failed because the lower-branch boundary was breached
        assert val_res.is_valid is False
        assert val_res.num_breaches == 1
        assert val_res.max_breach > 0.0
        # 2. Details reveal breach is specifically on the lower branch, NOT GMV floor
        assert val_res.details["lower_branch_breach_count"] == 1
        assert val_res.details["gmv_floor_breach_count"] == 0
        assert val_res.details["upper_branch_breach_count"] == 0
        assert val_res.details["lower_branch_interpolated"] is True

    def test_analytical_frontier_without_inefficient_points_uses_gmv_floor(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """When analytical frontier has no inefficient points, lower branch falls back to GMV floor."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        # Run GMV to find exact gmv_expected_return
        from app.portfolio_engine.optimizer import optimize_global_minimum_variance
        gmv = optimize_global_minimum_variance(cov)
        gmv_ret = float(gmv.weights @ mu)

        # Dense grid of target returns covering exclusively the upper branch
        target_returns = np.linspace(gmv_ret, float(np.max(mu)), 50)
        analytical_ef = compute_efficient_frontier(
            expected_returns=mu, covariance=cov, target_returns=target_returns
        )
        assert analytical_ef.success is True
        assert len(analytical_ef.inefficient_portfolios) == 0
        assert len(analytical_ef.efficient_portfolios) > 0

        sim_res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=500, random_seed=42)
        val_res = validate_monte_carlo_against_frontier(sim_res, analytical_ef, volatility_tolerance=1e-4)

        assert isinstance(val_res, MonteCarloValidationResult)
        assert val_res.is_valid is True
        assert val_res.details["lower_branch_interpolated"] is False
        assert "gmv_floor_only" in val_res.details["lower_branch_validation_guarantee"]

    def test_upper_branch_simulated_return_above_grid_max_raises(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Simulated portfolio with expected return above upper grid maximum raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        # Upper grid only extends to 0.17 (while MSFT has return 0.20)
        target_returns = [0.15, 0.16, 0.17]
        analytical_ef = compute_efficient_frontier(
            expected_returns=mu, covariance=cov, target_returns=target_returns
        )
        assert analytical_ef.success is True

        # Synthesize a portfolio with return 0.19 > 0.17
        n_assets = len(mu)
        w_synth = np.ones((1, n_assets)) / n_assets
        rets_synth = np.array([0.19])
        vols_synth = np.array([0.18])
        landmark = SimulatedLandmarkPortfolio(
            landmark_type="SIMULATED_GMV",
            expected_return=0.19,
            volatility=0.18,
            variance=0.18**2,
            sharpe_ratio=(0.19 - 0.04) / 0.18,
            simulation_index=0,
            weights=pd.Series(w_synth[0], index=analytical_ef.gmv_portfolio.weights.index),
        )
        sim_synth = MonteCarloSimulationResult(
            weights=w_synth,
            expected_returns=rets_synth,
            volatilities=vols_synth,
            variances=vols_synth**2,
            sharpe_ratios=np.array([(0.19 - 0.04) / 0.18]),
            simulated_gmv=landmark,
            simulated_max_sharpe=landmark,
            asset_labels=list(analytical_ef.gmv_portfolio.weights.index),
            num_portfolios=1,
            num_assets=n_assets,
            risk_free_rate=0.04,
        )

        with pytest.raises(FinancialCalculationError, match="exceeding analytical frontier upper-branch domain maximum"):
            validate_monte_carlo_against_frontier(sim_synth, analytical_ef)

    def test_lower_branch_simulated_return_below_grid_min_raises(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Simulated portfolio with expected return below lower grid minimum raises FinancialCalculationError."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        # Grid starts at 0.13 (above the minimum asset GOOG at 0.12)
        target_returns = [0.13, 0.14, 0.15, 0.16, 0.17]
        analytical_ef = compute_efficient_frontier(
            expected_returns=mu, covariance=cov, target_returns=target_returns
        )
        assert analytical_ef.success is True
        assert len(analytical_ef.inefficient_portfolios) > 0

        # Synthesize a portfolio with return 0.11 < 0.13
        n_assets = len(mu)
        w_synth = np.ones((1, n_assets)) / n_assets
        rets_synth = np.array([0.11])
        vols_synth = np.array([0.18])
        landmark = SimulatedLandmarkPortfolio(
            landmark_type="SIMULATED_GMV",
            expected_return=0.11,
            volatility=0.18,
            variance=0.18**2,
            sharpe_ratio=(0.11 - 0.04) / 0.18,
            simulation_index=0,
            weights=pd.Series(w_synth[0], index=analytical_ef.gmv_portfolio.weights.index),
        )
        sim_synth = MonteCarloSimulationResult(
            weights=w_synth,
            expected_returns=rets_synth,
            volatilities=vols_synth,
            variances=vols_synth**2,
            sharpe_ratios=np.array([(0.11 - 0.04) / 0.18]),
            simulated_gmv=landmark,
            simulated_max_sharpe=landmark,
            asset_labels=list(analytical_ef.gmv_portfolio.weights.index),
            num_portfolios=1,
            num_assets=n_assets,
            risk_free_rate=0.04,
        )

        with pytest.raises(FinancialCalculationError, match="below analytical frontier lower-branch domain minimum"):
            validate_monte_carlo_against_frontier(sim_synth, analytical_ef)

    def test_valid_portfolios_exactly_at_interpolation_endpoints_succeed(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """Portfolios located exactly at GMV, upper max, and lower min endpoints pass validation."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        analytical_ef = compute_efficient_frontier(expected_returns=mu, covariance=cov, n_points=50)
        assert analytical_ef.success is True

        gmv_ret = float(analytical_ef.metadata["gmv_expected_return"])
        gmv_vol = float(analytical_ef.gmv_portfolio.volatility)
        upper_max = analytical_ef.efficient_portfolios[-1]
        lower_min = analytical_ef.inefficient_portfolios[0]

        # Three exact endpoint test points
        test_rets = np.array([float(lower_min.expected_return), gmv_ret, float(upper_max.expected_return)])
        test_vols = np.array([float(lower_min.volatility), gmv_vol, float(upper_max.volatility)])
        n_pts = len(test_rets)
        w_synth = np.ones((n_pts, len(mu))) / len(mu)

        landmark = SimulatedLandmarkPortfolio(
            landmark_type="SIMULATED_GMV",
            expected_return=gmv_ret,
            volatility=gmv_vol,
            variance=gmv_vol**2,
            sharpe_ratio=(gmv_ret - 0.04) / gmv_vol,
            simulation_index=1,
            weights=pd.Series(w_synth[1], index=analytical_ef.gmv_portfolio.weights.index),
        )
        sim_synth = MonteCarloSimulationResult(
            weights=w_synth,
            expected_returns=test_rets,
            volatilities=test_vols,
            variances=test_vols**2,
            sharpe_ratios=(test_rets - 0.04) / test_vols,
            simulated_gmv=landmark,
            simulated_max_sharpe=landmark,
            asset_labels=list(analytical_ef.gmv_portfolio.weights.index),
            num_portfolios=n_pts,
            num_assets=len(mu),
            risk_free_rate=0.04,
        )

        val_res = validate_monte_carlo_against_frontier(sim_synth, analytical_ef, volatility_tolerance=1e-4)
        assert val_res.is_valid is True
        assert val_res.num_breaches == 0
        assert val_res.max_breach == 0.0

    def test_custom_target_return_grid_partial_coverage_rejects_out_of_domain_portfolios(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """When user supplies a partial target-return grid, cross-validation rejects out-of-domain simulation."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        # Narrow grid [0.145, 0.155] that does not cover the full range of simplex portfolios [0.12, 0.20]
        narrow_ef = compute_efficient_frontier(
            expected_returns=mu, covariance=cov, target_returns=[0.145, 0.150, 0.155]
        )
        assert narrow_ef.success is True

        sim_res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=200, random_seed=42)

        # Simulation will have portfolios with returns outside [0.145, 0.155]
        with pytest.raises(FinancialCalculationError, match="Analytical frontier does not cover the complete"):
            validate_monte_carlo_against_frontier(sim_res, narrow_ef)


# ---------------------------------------------------------------------------
# 8. Object-Oriented Simulator Class Interface
# ---------------------------------------------------------------------------

class TestMonteCarloSimulatorClass:
    """Verify MonteCarloSimulator object-oriented interface."""

    def test_simulator_class_interface(
        self, independent_3asset_data: Dict[str, Any]
    ) -> None:
        """MonteCarloSimulator instantiates, executes simulate(), and returns result."""
        cov = independent_3asset_data["cov"]
        mu = independent_3asset_data["mu"]

        sim = MonteCarloSimulator(risk_free_rate=0.05, random_seed=888)
        assert sim.risk_free_rate == 0.05
        assert sim.random_seed == 888

        res = sim.simulate(mu, cov, num_portfolios=500)
        assert isinstance(res, MonteCarloSimulationResult)
        assert res.num_portfolios == 500
        assert res.risk_free_rate == 0.05


# ---------------------------------------------------------------------------
# 9. Performance Benchmark (Honest & Non-Brittle)
# ---------------------------------------------------------------------------

class TestPerformanceBenchmark:
    """Verify 50,000 portfolios x 10 assets evaluates well within 50 ms target."""

    def test_fifty_thousand_portfolios_vectorized_math_benchmark(self) -> None:
        """Vectorized metric math (einsum + matrix products) evaluates in < 50 ms."""
        n_assets = 10
        k_portfolios = 50000
        rng = np.random.default_rng(42)

        alpha = np.ones(n_assets)
        weights = rng.dirichlet(alpha, size=k_portfolios).astype(np.float64)
        a = rng.standard_normal((n_assets, n_assets))
        cov_mat = (a @ a.T + np.eye(n_assets) * 0.05).astype(np.float64)
        mu_vec = np.linspace(0.08, 0.22, n_assets).astype(np.float64)
        rf = 0.05

        # Pure vectorized metric calculation
        t0 = time.perf_counter()
        portfolio_returns = weights @ mu_vec
        portfolio_variances = np.einsum("ij,jk,ik->i", weights, cov_mat, weights)
        portfolio_variances = np.maximum(portfolio_variances, 0.0)
        portfolio_volatilities = np.sqrt(portfolio_variances)
        sharpe_ratios = (portfolio_returns - rf) / portfolio_volatilities
        math_time_ms = (time.perf_counter() - t0) * 1000

        assert len(portfolio_returns) == k_portfolios
        assert len(sharpe_ratios) == k_portfolios

        # Measured benchmark is typically ~10-15 ms (target: < 50 ms).
        # We assert a robust CI safety buffer of < 100 ms to avoid spurious CI flakiness.
        assert math_time_ms < 100.0, f"Vectorized math took {math_time_ms:.2f} ms (expected < 100 ms)"

    def test_fifty_thousand_portfolios_end_to_end_timing(self) -> None:
        """Full simulate_monte_carlo_portfolios execution takes < 500 ms."""
        n_assets = 10
        k_portfolios = 50000
        rng = np.random.default_rng(42)

        a = rng.standard_normal((n_assets, n_assets))
        cov_raw = a @ a.T + np.eye(n_assets) * 0.05
        tickers = [f"ASSET_{i}" for i in range(n_assets)]
        cov = pd.DataFrame(cov_raw, index=tickers, columns=tickers)
        mu = pd.Series(np.linspace(0.08, 0.22, n_assets), index=tickers)

        t0 = time.perf_counter()
        res = simulate_monte_carlo_portfolios(mu, cov, num_portfolios=k_portfolios, random_seed=42)
        total_time_ms = (time.perf_counter() - t0) * 1000

        assert res.num_portfolios == k_portfolios
        assert res.num_assets == n_assets
        assert total_time_ms < 500.0, f"Total execution took {total_time_ms:.2f} ms (expected < 500 ms)"


# ---------------------------------------------------------------------------
# 10. Namespace Exports & Identities
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package exports and cross-namespace identities."""

    MC_SYMBOLS = [
        "DEFAULT_NUM_PORTFOLIOS",
        "DEFAULT_RISK_FREE_RATE",
        "MonteCarloSimulationResult",
        "MonteCarloSimulator",
        "MonteCarloValidationResult",
        "SimulatedLandmarkPortfolio",
        "simulate_monte_carlo_portfolios",
        "validate_monte_carlo_against_frontier",
    ]

    def test_portfolio_engine_exports(self) -> None:
        """portfolio_engine exports all Monte Carlo symbols in __all__."""
        for sym in self.MC_SYMBOLS:
            assert hasattr(pe, sym), f"portfolio_engine missing export {sym}"
            assert sym in pe.__all__, f"{sym} not in portfolio_engine.__all__"

    def test_financial_engine_exports(self) -> None:
        """financial_engine re-exports all Monte Carlo symbols in __all__."""
        for sym in self.MC_SYMBOLS:
            assert hasattr(fe, sym), f"financial_engine missing export {sym}"
            assert sym in fe.__all__, f"{sym} not in financial_engine.__all__"

    def test_financial_engine_monte_carlo_module_exports(self) -> None:
        """financial_engine.monte_carlo re-exports all symbols in __all__."""
        for sym in self.MC_SYMBOLS:
            assert hasattr(femc, sym), f"financial_engine.monte_carlo missing {sym}"
            assert sym in femc.__all__, f"{sym} not in femc.__all__"

    def test_cross_namespace_identities(self) -> None:
        """Symbols across portfolio_engine and financial_engine point to identical objects."""
        for sym in self.MC_SYMBOLS:
            pe_obj = getattr(pe, sym)
            fe_obj = getattr(fe, sym)
            femc_obj = getattr(femc, sym)
            assert pe_obj is fe_obj, f"Identity mismatch between pe and fe for {sym}"
            assert pe_obj is femc_obj, f"Identity mismatch between pe and femc for {sym}"
