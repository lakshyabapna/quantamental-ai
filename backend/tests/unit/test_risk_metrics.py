"""Quantamental AI - Downside & Tail Risk Analytics Test Suite.

Phase 2 Task 7: Comprehensive unit tests verifying:
- Parametric VaR at 95% and 99% confidence.
- Historical VaR at 95% and 99% confidence.
- Modified Cornish-Fisher VaR with skewness and kurtosis expansions.
- Conditional Value at Risk (CVaR / Expected Shortfall).
- Hand-crafted analytical test vectors.
- Normal distribution identity checks (CF VaR == Parametric VaR when S=0, K=0).
- Sign conventions (positive = loss, negative = gain).
- Invariant assertions (CVaR >= Historical VaR).
- Degenerate distributions (constant return, zero volatility).
- Strict input validation (NaN, Inf, complex, boolean, sample size, dimensions).
- Defensive immutability and JSON serialization.
- Cross-namespace export identities.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
)
from app.portfolio_engine.risk_metrics import (
    ALLOWED_QUANTILE_METHODS,
    DEFAULT_CONFIDENCE_LEVELS,
    CornishFisherExpansionResult,
    RiskMetricsEngine,
    RiskMetricsResult,
    _format_confidence_key,
    calculate_cornish_fisher_expansion,
    calculate_cornish_fisher_var,
    calculate_cvar,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_risk_metrics,
)
import app.portfolio_engine as pe
import app.financial_engine as fe
import app.financial_engine.risk_metrics as ferm


# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def gaussian_sample_10k() -> np.ndarray:
    """10,000 observations from a calibrated Normal(mu=0.0005, sigma=0.015)."""
    rng = np.random.default_rng(12345)
    mu = 0.0005
    sigma = 0.015
    return rng.normal(loc=mu, scale=sigma, size=10000)


@pytest.fixture
def hand_crafted_returns() -> np.ndarray:
    """10 exact returns for step-by-step independent calculation verification.

    Sorted values: [-0.05, -0.04, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.06]
    Sum: 0.04
    Mean: 0.004
    """
    return np.array([0.01, -0.02, 0.04, -0.05, 0.02, -0.01, 0.06, 0.00, -0.04, 0.03])


# ---------------------------------------------------------------------------
# 1. Independent Hand-Calculated Test Vectors
# ---------------------------------------------------------------------------

class TestHandCalculatedVectors:
    """Verify metrics against independent, hand-crafted analytical calculations."""

    def test_hand_crafted_sample_metrics(self, hand_crafted_returns: np.ndarray) -> None:
        """Step-by-step verification of hand-crafted 10-observation return series."""
        r = hand_crafted_returns
        n = len(r)
        assert n == 10

        # Step 1: Mean & Sample Volatility (ddof=1)
        mu_expected = float(np.mean(r))  # 0.004
        var_expected = float(np.sum((r - mu_expected)**2) / (n - 1))
        sigma_expected = float(np.sqrt(var_expected))

        assert mu_expected == pytest.approx(0.004, abs=1e-12)

        # Step 2: Parametric VaR at 95%
        # z_{0.05} = stats.norm.ppf(0.05) ~= -1.6448536
        z_95 = float(stats.norm.ppf(0.05))
        pvar_95_expected = -(mu_expected + z_95 * sigma_expected)
        pvar_95_actual = calculate_parametric_var(r, confidence_level=0.95)
        assert pvar_95_actual == pytest.approx(pvar_95_expected, abs=1e-12)

        # Step 3: Historical VaR at 95%
        # alpha = 0.05. Using linear interpolation:
        q_05_expected = float(np.quantile(r, 0.05, method="linear"))
        hvar_95_expected = -q_05_expected
        hvar_95_actual = calculate_historical_var(r, confidence_level=0.95)
        assert hvar_95_actual == pytest.approx(hvar_95_expected, abs=1e-12)

        # Step 4: CVaR at 95%
        # Tail observations: r <= q_05_expected
        tail = r[r <= (q_05_expected + 1e-12)]
        cvar_95_expected = -float(np.mean(tail))
        cvar_95_actual = calculate_cvar(r, confidence_level=0.95)
        assert cvar_95_actual == pytest.approx(cvar_95_expected, abs=1e-12)

        # Mathematical invariant: CVaR >= Historical VaR
        assert cvar_95_actual >= hvar_95_actual - 1e-12


# ---------------------------------------------------------------------------
# 2. Parametric VaR & Normal Distribution Equivalence
# ---------------------------------------------------------------------------

class TestParametricVaR:
    """Verify Gaussian Parametric VaR behavior and asymptotic accuracy."""

    def test_parametric_var_normal_distribution_95_and_99(
        self, gaussian_sample_10k: np.ndarray
    ) -> None:
        """Parametric VaR on large Gaussian sample closely matches theoretical formula."""
        r = gaussian_sample_10k
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))

        pvar_95 = calculate_parametric_var(r, confidence_level=0.95)
        pvar_99 = calculate_parametric_var(r, confidence_level=0.99)

        # Theoretical values: - (mu + z * sigma)
        # 95%: z = -1.64485
        theo_pvar_95 = -(mu + stats.norm.ppf(0.05) * sigma)
        # 99%: z = -2.32635
        theo_pvar_99 = -(mu + stats.norm.ppf(0.01) * sigma)

        assert pvar_95 == pytest.approx(theo_pvar_95, abs=1e-12)
        assert pvar_99 == pytest.approx(theo_pvar_99, abs=1e-12)

        # 99% VaR must be strictly more conservative (larger loss) than 95% VaR
        assert pvar_99 > pvar_95

    def test_pandas_series_input_supported(self) -> None:
        """Parametric VaR accepts pd.Series with DatetimeIndex."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        rng = np.random.default_rng(999)
        s = pd.Series(rng.normal(0.001, 0.02, size=100), index=dates)

        pvar = calculate_parametric_var(s, confidence_level=0.95)
        assert isinstance(pvar, float)
        assert np.isfinite(pvar)


# ---------------------------------------------------------------------------
# 3. Historical VaR & Empirical Quantiles
# ---------------------------------------------------------------------------

class TestHistoricalVaR:
    """Verify non-parametric empirical Historical VaR."""

    def test_historical_var_matches_negative_empirical_quantile(self) -> None:
        """Historical VaR equals -quantile(alpha) exactly."""
        rng = np.random.default_rng(42)
        returns = rng.standard_t(df=5, size=1000) * 0.02

        for conf in [0.90, 0.95, 0.99]:
            alpha = 1.0 - conf
            expected_hvar = -float(np.quantile(returns, alpha, method="linear"))
            actual_hvar = calculate_historical_var(returns, confidence_level=conf)
            assert actual_hvar == pytest.approx(expected_hvar, abs=1e-12)

    def test_historical_var_monotonic_with_confidence(self) -> None:
        """Historical VaR increases monotonically with confidence level."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=500)

        var_90 = calculate_historical_var(returns, confidence_level=0.90)
        var_95 = calculate_historical_var(returns, confidence_level=0.95)
        var_99 = calculate_historical_var(returns, confidence_level=0.99)

        assert var_90 < var_95 < var_99


# ---------------------------------------------------------------------------
# 4. Modified Cornish-Fisher VaR & Expansion Terms
# ---------------------------------------------------------------------------

class TestCornishFisherVaR:
    """Verify Cornish-Fisher expansion and skewness/kurtosis adjustments."""

    def test_cornish_fisher_collapses_to_parametric_when_normal(self) -> None:
        """When skewness=0 and kurtosis=0, Cornish-Fisher VaR == Parametric VaR.

        Uses a calibrated symmetric sample where both sample skewness and sample
        excess kurtosis are mathematically zero to machine precision.
        """
        # For symmetric points [-b, -1.0, -0.5, 0.5, 1.0, b], odd moments are exactly 0 (S=0).
        # Calibrating b = 2.505964358491755 makes sample excess kurtosis zero (K=0).
        b = 2.505964358491755
        calibrated_sample = np.array([-b, -1.0, -0.5, 0.5, 1.0, b]) * 0.01

        # 1. Skewness is zero
        s = float(stats.skew(calibrated_sample, bias=False))
        assert s == pytest.approx(0.0, abs=1e-12)

        # 2. Excess kurtosis is zero within justified numerical tolerance
        k = float(stats.kurtosis(calibrated_sample, bias=False, fisher=True))
        assert k == pytest.approx(0.0, abs=1e-12)

        # 3. Test across standard confidence levels (95% and 99%)
        for conf in (0.95, 0.99):
            exp_res = calculate_cornish_fisher_expansion(calibrated_sample, confidence_level=conf, bias=False)
            assert exp_res.skewness == pytest.approx(0.0, abs=1e-12)
            assert exp_res.excess_kurtosis == pytest.approx(0.0, abs=1e-12)
            assert exp_res.skewness_linear_term == pytest.approx(0.0, abs=1e-12)
            assert exp_res.kurtosis_term == pytest.approx(0.0, abs=1e-12)
            assert exp_res.skewness_quadratic_term == pytest.approx(0.0, abs=1e-12)

            # 4. adjusted_z == standard_z
            assert exp_res.adjusted_z == pytest.approx(exp_res.standard_z, abs=1e-12)

            # 5. Cornish-Fisher VaR equals Parametric VaR
            cf_var = calculate_cornish_fisher_var(calibrated_sample, confidence_level=conf, bias=False)
            p_var = calculate_parametric_var(calibrated_sample, confidence_level=conf)
            assert cf_var == pytest.approx(p_var, abs=1e-12)

    def test_cornish_fisher_expansion_components_hand_calculation(self) -> None:
        """Step-by-step verification of individual Cornish-Fisher terms."""
        # Construct sample with known skewness and kurtosis
        r = np.array([-0.06, -0.04, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.08])
        res = calculate_cornish_fisher_expansion(r, confidence_level=0.95, bias=False)

        z = float(stats.norm.ppf(0.05))
        s = float(stats.skew(r, bias=False))
        k = float(stats.kurtosis(r, bias=False, fisher=True))

        expected_t1 = (1.0 / 6.0) * (z**2 - 1.0) * s
        expected_t2 = (1.0 / 24.0) * (z**3 - 3.0 * z) * k
        expected_t3 = -(1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
        expected_z_tilde = z + expected_t1 + expected_t2 + expected_t3

        assert res.standard_z == pytest.approx(z, abs=1e-12)
        assert res.skewness == pytest.approx(s, abs=1e-12)
        assert res.excess_kurtosis == pytest.approx(k, abs=1e-12)
        assert res.skewness_linear_term == pytest.approx(expected_t1, abs=1e-12)
        assert res.kurtosis_term == pytest.approx(expected_t2, abs=1e-12)
        assert res.skewness_quadratic_term == pytest.approx(expected_t3, abs=1e-12)
        assert res.adjusted_z == pytest.approx(expected_z_tilde, abs=1e-12)

        # VaR_CF = - (mu + z_tilde * sigma)
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        expected_var_cf = -(mu + expected_z_tilde * sigma)
        actual_var_cf = calculate_cornish_fisher_var(r, confidence_level=0.95, bias=False)
        assert actual_var_cf == pytest.approx(expected_var_cf, abs=1e-12)

    def test_negative_skewness_and_fat_tails_increase_var(self) -> None:
        """Negative skewness and positive excess kurtosis increase VaR relative to Gaussian."""
        # Generate left-skewed, fat-tailed distribution (e.g., standard market crash scenario)
        rng = np.random.default_rng(101)
        normal_data = rng.normal(0.0, 0.015, size=1000)
        # Add severe negative outliers to induce left skew and excess kurtosis
        crashes = np.array([-0.08, -0.09, -0.10, -0.12, -0.15])
        skewed_data = np.concatenate([normal_data, crashes])

        s = float(stats.skew(skewed_data))
        k = float(stats.kurtosis(skewed_data, fisher=True))
        assert s < 0.0, "Sample should have negative skewness"
        assert k > 0.0, "Sample should have positive excess kurtosis"

        pvar_99 = calculate_parametric_var(skewed_data, confidence_level=0.99)
        cfvar_99 = calculate_cornish_fisher_var(skewed_data, confidence_level=0.99)

        # Cornish-Fisher VaR accounts for the fat left tail -> reports higher potential loss
        assert cfvar_99 > pvar_99


# ---------------------------------------------------------------------------
# 5. Conditional Value-at-Risk (CVaR / Expected Shortfall)
# ---------------------------------------------------------------------------

class TestConditionalVaR:
    """Verify CVaR / Expected Shortfall calculations and invariants."""

    def test_cvar_equals_negative_mean_of_tail_returns(self) -> None:
        """CVaR is the negative average of returns below the historical VaR threshold."""
        returns = np.array([-0.10, -0.06, -0.04, -0.02, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
        cvar_90 = calculate_cvar(returns, confidence_level=0.90)

        # 10% quantile
        q_10 = float(np.quantile(returns, 0.10, method="linear"))
        tail = returns[returns <= (q_10 + 1e-12)]
        expected_cvar = -float(np.mean(tail))

        assert cvar_90 == pytest.approx(expected_cvar, abs=1e-12)

    def test_cvar_always_greater_than_or_equal_to_historical_var(self) -> None:
        """Theoretical Invariant: CVaR_alpha >= Historical_VaR_alpha."""
        rng = np.random.default_rng(777)
        returns = rng.standard_t(df=4, size=1000) * 0.015

        for conf in [0.90, 0.95, 0.99]:
            hvar = calculate_historical_var(returns, confidence_level=conf)
            cvar = calculate_cvar(returns, confidence_level=conf)
            assert cvar >= hvar - 1e-12, f"CVaR ({cvar}) must be >= Historical VaR ({hvar})"

    def test_cvar_with_explicit_var_threshold(self) -> None:
        """Supplying explicit positive loss var_threshold evaluates tail below -var_threshold."""
        returns = np.array([-0.08, -0.05, -0.03, -0.01, 0.02, 0.04])
        # Suppose VaR threshold is 0.04 (4% loss, so return threshold is -0.04)
        cvar = calculate_cvar(returns, var_threshold=0.04)
        # Tail observations <= -0.04 are: -0.08, -0.05
        expected_cvar = -np.mean([-0.08, -0.05])  # 0.065
        assert cvar == pytest.approx(expected_cvar, abs=1e-12)

    def test_cvar_raises_if_no_observations_below_user_threshold(self) -> None:
        """If user passes impossible var_threshold, raises FinancialCalculationError."""
        returns = np.array([-0.02, 0.01, 0.03, 0.05])
        # var_threshold = 0.50 (return <= -0.50, but min return is -0.02)
        with pytest.raises(FinancialCalculationError, match="No return observations fell in the tail"):
            calculate_cvar(returns, var_threshold=0.50)


# ---------------------------------------------------------------------------
# 6. Sign Conventions: Positive = Loss, Negative = Gain
# ---------------------------------------------------------------------------

class TestSignConventions:
    """Verify canonical sign conventions: positive VaR/CVaR = loss, negative = gain."""

    def test_all_positive_returns_yield_negative_var(self) -> None:
        """A portfolio with massive positive returns has negative VaR (gain)."""
        # All returns between +2% and +8%
        returns = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

        pvar_95 = calculate_parametric_var(returns, confidence_level=0.95)
        hvar_95 = calculate_historical_var(returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(returns, confidence_level=0.95)

        # Historical quantile is positive (>= 0.02), so Historical VaR is negative (< 0)
        assert hvar_95 < 0.0, f"Expected negative Historical VaR for gains, got {hvar_95}"
        assert cvar_95 < 0.0, f"Expected negative CVaR for gains, got {cvar_95}"

    def test_all_negative_returns_yield_positive_var(self) -> None:
        """A portfolio with severe losses has positive VaR (loss)."""
        returns = np.array([-0.08, -0.07, -0.06, -0.05, -0.04, -0.03, -0.02])

        pvar_95 = calculate_parametric_var(returns, confidence_level=0.95)
        hvar_95 = calculate_historical_var(returns, confidence_level=0.95)
        cvar_95 = calculate_cvar(returns, confidence_level=0.95)

        assert pvar_95 > 0.0, f"Expected positive Parametric VaR for losses, got {pvar_95}"
        assert hvar_95 > 0.0, f"Expected positive Historical VaR for losses, got {hvar_95}"
        assert cvar_95 > 0.0, f"Expected positive CVaR for losses, got {cvar_95}"


# ---------------------------------------------------------------------------
# 7. Degenerate Distributions & Zero Volatility
# ---------------------------------------------------------------------------

class TestDegenerateDistributions:
    """Verify deterministic handling of zero-volatility / constant-return series."""

    def test_constant_zero_returns(self) -> None:
        """Constant zero returns yield exactly 0.0 across all risk metrics."""
        returns = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        res = calculate_risk_metrics(returns)

        assert res.volatility == 0.0
        assert res.mean_return == 0.0
        assert res.parametric_var_95 == 0.0
        assert res.parametric_var_99 == 0.0
        assert res.historical_var_95 == 0.0
        assert res.historical_var_99 == 0.0
        assert res.cornish_fisher_var_95 == 0.0
        assert res.cornish_fisher_var_99 == 0.0
        assert res.cvar_95 == 0.0
        assert res.cvar_99 == 0.0
        assert res.details["is_degenerate"] is True

    def test_constant_positive_returns(self) -> None:
        """Constant positive return (+0.02) yields -0.02 (gain) across all metrics."""
        returns = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        res = calculate_risk_metrics(returns)

        assert res.volatility == 0.0
        assert res.mean_return == 0.02
        assert res.parametric_var_95 == -0.02
        assert res.historical_var_95 == -0.02
        assert res.cornish_fisher_var_95 == -0.02
        assert res.cvar_95 == -0.02

    def test_constant_negative_returns(self) -> None:
        """Constant negative return (-0.03) yields +0.03 (loss) across all metrics."""
        returns = np.array([-0.03, -0.03, -0.03, -0.03, -0.03])
        res = calculate_risk_metrics(returns)

        assert res.volatility == 0.0
        assert res.mean_return == -0.03
        assert res.parametric_var_95 == 0.03
        assert res.historical_var_95 == 0.03
        assert res.cornish_fisher_var_95 == 0.03
        assert res.cvar_95 == 0.03


# ---------------------------------------------------------------------------
# 8. Strict Input Validation & Edge Cases
# ---------------------------------------------------------------------------

class TestStrictInputValidation:
    """Verify rejection of invalid types, dimensions, non-finite values, and sample sizes."""

    def test_none_input_rejected(self) -> None:
        """None input raises FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="cannot be None"):
            calculate_parametric_var(None)  # type: ignore

    def test_boolean_scalar_rejected(self) -> None:
        """Boolean scalar raises FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="got bool"):
            calculate_parametric_var(True)  # type: ignore

    def test_boolean_array_rejected(self) -> None:
        """Boolean array or series raises FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="boolean data"):
            calculate_parametric_var(np.array([True, False, True, False]))

        with pytest.raises(FinancialCalculationError, match="boolean data"):
            calculate_parametric_var(pd.Series([True, False, True, False]))

    def test_complex_numbers_rejected(self) -> None:
        """Complex numbers in returns raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_parametric_var([1.0 + 2.0j, 2.0, 3.0, 4.0])

        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_parametric_var(np.array([1.0 + 2.0j, 2.0, 3.0, 4.0]))

    def test_string_elements_rejected(self) -> None:
        """Non-numeric string elements raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="non-numeric"):
            calculate_parametric_var(["0.01", "0.02", "0.03", "0.04"])  # type: ignore

    @pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_rejected(self, bad_val: float) -> None:
        """NaN or Inf in returns raises FinancialCalculationError."""
        arr = np.array([0.01, 0.02, bad_val, 0.04, 0.05])
        with pytest.raises(FinancialCalculationError, match="non-finite values"):
            calculate_parametric_var(arr)

        with pytest.raises(FinancialCalculationError, match="non-finite values"):
            calculate_historical_var(arr)

        with pytest.raises(FinancialCalculationError, match="non-finite values"):
            calculate_cornish_fisher_var(arr)

        with pytest.raises(FinancialCalculationError, match="non-finite values"):
            calculate_cvar(arr)

    def test_multidimensional_returns_rejected(self) -> None:
        """2D returns matrix raises FinancialCalculationError."""
        arr_2d = np.array([[0.01, 0.02], [0.03, 0.04]])
        with pytest.raises(FinancialCalculationError, match="1-dimensional"):
            calculate_parametric_var(arr_2d)

    @pytest.mark.parametrize("bad_c", [-0.05, 0.0, 1.0, 1.05, float("nan"), float("inf"), True, "0.95"])
    def test_invalid_confidence_levels_rejected(self, bad_c: Any) -> None:
        """Confidence level <= 0, >= 1, NaN, Inf, boolean, or string raises FinancialCalculationError."""
        valid_r = np.array([0.01, -0.02, 0.03, 0.04, -0.01])
        with pytest.raises(FinancialCalculationError):
            calculate_parametric_var(valid_r, confidence_level=bad_c)

        with pytest.raises(FinancialCalculationError):
            calculate_historical_var(valid_r, confidence_level=bad_c)

        with pytest.raises(FinancialCalculationError):
            calculate_cornish_fisher_var(valid_r, confidence_level=bad_c)

        with pytest.raises(FinancialCalculationError):
            calculate_cvar(valid_r, confidence_level=bad_c)

    def test_insufficient_sample_size_rejected(self) -> None:
        """Fewer than required observations raises InsufficientDataError."""
        # 1 observation: insufficient for mean/std
        with pytest.raises(InsufficientDataError, match="Insufficient return observations"):
            calculate_parametric_var([0.05])

        # 3 observations: insufficient for kurtosis in Cornish-Fisher
        with pytest.raises(InsufficientDataError, match="Insufficient return observations"):
            calculate_cornish_fisher_var([0.01, -0.02, 0.03])

        # 3 observations: insufficient for calculate_risk_metrics
        with pytest.raises(InsufficientDataError, match="Insufficient return observations"):
            calculate_risk_metrics([0.01, -0.02, 0.03])


# ---------------------------------------------------------------------------
# 9. Defensive Immutability & Result Serialization
# ---------------------------------------------------------------------------

class TestImmutabilityAndSerialization:
    """Verify frozen dataclass, defensive copying, and JSON serialization."""

    def test_risk_metrics_result_frozen(self) -> None:
        """Attribute reassignment raises FrozenInstanceError."""
        r = np.array([-0.05, -0.02, 0.01, 0.02, 0.04, 0.06])
        res = calculate_risk_metrics(r)

        with pytest.raises(FrozenInstanceError):
            res.parametric_var_95 = 99.0  # type: ignore

        with pytest.raises(FrozenInstanceError):
            res.sample_size = 0  # type: ignore

    def test_details_dictionary_defensive_copy(self) -> None:
        """Mutating returned details dict does not affect internal result state."""
        r = np.array([-0.05, -0.02, 0.01, 0.02, 0.04, 0.06])
        res = calculate_risk_metrics(r)

        det = res.details
        det["hacked_key"] = "compromised"

        assert "hacked_key" not in res.details
        assert "hacked_key" not in res.to_dict()["details"]

    def test_to_dict_json_serializable(self) -> None:
        """to_dict() produces valid JSON-serializable dictionary."""
        r = np.array([-0.05, -0.02, 0.01, 0.02, 0.04, 0.06])
        res = calculate_risk_metrics(r)

        d = res.to_dict()
        assert isinstance(d, dict)
        assert "parametric_var_95" in d
        assert "parametric_var_99" in d
        assert "historical_var_95" in d
        assert "historical_var_99" in d
        assert "cornish_fisher_var_95" in d
        assert "cornish_fisher_var_99" in d
        assert "cvar_95" in d
        assert "cvar_99" in d
        assert "mean_return" in d
        assert "volatility" in d
        assert "skewness" in d
        assert "excess_kurtosis" in d
        assert "sample_size" in d

        # Dump to JSON and parse back
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert parsed["sample_size"] == len(r)


# ---------------------------------------------------------------------------
# 10. Object-Oriented Engine Interface
# ---------------------------------------------------------------------------

class TestRiskMetricsEngineClass:
    """Verify object-oriented RiskMetricsEngine wrapper interface."""

    def test_engine_class_methods(self) -> None:
        """RiskMetricsEngine instance calculates all metrics identically to functions."""
        engine = RiskMetricsEngine(bias=False, method="linear")
        r = np.array([-0.04, -0.02, 0.01, 0.02, 0.03, 0.05])

        res = engine.calculate(r)
        assert isinstance(res, RiskMetricsResult)

        assert engine.parametric_var(r, 0.95) == pytest.approx(res.parametric_var_95)
        assert engine.historical_var(r, 0.95) == pytest.approx(res.historical_var_95)
        assert engine.cornish_fisher_var(r, 0.95) == pytest.approx(res.cornish_fisher_var_95)
        assert engine.cvar(r, 0.95) == pytest.approx(res.cvar_95)


# ---------------------------------------------------------------------------
# 11. Cross-Namespace Exports & Identities
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify package exports and cross-namespace identities."""

    RISK_SYMBOLS = [
        "ALLOWED_QUANTILE_METHODS",
        "DEFAULT_CONFIDENCE_LEVELS",
        "CornishFisherExpansionResult",
        "RiskMetricsEngine",
        "RiskMetricsResult",
        "calculate_cornish_fisher_expansion",
        "calculate_cornish_fisher_var",
        "calculate_cvar",
        "calculate_historical_var",
        "calculate_parametric_var",
        "calculate_risk_metrics",
    ]

    def test_portfolio_engine_exports(self) -> None:
        """portfolio_engine exports all risk metrics symbols in __all__."""
        for sym in self.RISK_SYMBOLS:
            assert hasattr(pe, sym), f"portfolio_engine missing export {sym}"
            assert sym in pe.__all__, f"{sym} not in portfolio_engine.__all__"

    def test_financial_engine_exports(self) -> None:
        """financial_engine re-exports all risk metrics symbols in __all__."""
        for sym in self.RISK_SYMBOLS:
            assert hasattr(fe, sym), f"financial_engine missing export {sym}"
            assert sym in fe.__all__, f"{sym} not in financial_engine.__all__"

    def test_financial_engine_risk_metrics_module_exports(self) -> None:
        """financial_engine.risk_metrics re-exports all symbols in __all__."""
        for sym in self.RISK_SYMBOLS:
            assert hasattr(ferm, sym), f"financial_engine.risk_metrics missing {sym}"
            assert sym in ferm.__all__, f"{sym} not in ferm.__all__"

    def test_cross_namespace_identities(self) -> None:
        """Symbols across portfolio_engine and financial_engine point to identical objects."""
        for sym in self.RISK_SYMBOLS:
            pe_obj = getattr(pe, sym)
            fe_obj = getattr(fe, sym)
            ferm_obj = getattr(ferm, sym)
            assert pe_obj is fe_obj, f"Identity mismatch between pe and fe for {sym}"
            assert pe_obj is ferm_obj, f"Identity mismatch between pe and ferm for {sym}"


# ---------------------------------------------------------------------------
# 12. Targeted Audit Findings Regression Tests
# ---------------------------------------------------------------------------

class TestAuditFindingsRegression:
    """Regression tests verifying targeted audit findings for Phase 2 Task 7."""

    def test_cvar_tail_membership_exact_boundary(self) -> None:
        """Finding 1: Observations marginally above threshold_return must be excluded.

        Under threshold_return = -0.05 (i.e. var_threshold = 0.05):
        -0.05 is included (<= -0.05).
        -0.05 + 1e-14 is strictly greater than -0.05 and MUST be excluded.
        -0.05 + 1e-12 is strictly greater than -0.05 and MUST be excluded.
        """
        r = np.array([
            -0.10,
            -0.05,
            -0.05 + 1e-14,
            -0.05 + 1e-12,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
        ])
        # With var_threshold = 0.05, threshold_return = -0.05.
        # Tail observations satisfying r <= -0.05 are strictly [-0.10, -0.05].
        # Mean of tail: (-0.10 + -0.05) / 2 = -0.075.
        # CVaR = -(-0.075) = 0.075.
        # If an epsilon like 1e-12 was added, -0.05 + 1e-14 and -0.05 + 1e-12 would be included,
        # changing the tail average to approx -0.0625.
        cvar = calculate_cvar(r, var_threshold=0.05)
        assert cvar == pytest.approx(0.075, abs=1e-10)

        # Confirm using RiskMetricsEngine as well
        engine = RiskMetricsEngine()
        assert engine.cvar(r, var_threshold=0.05) == pytest.approx(0.075, abs=1e-10)

    def test_strict_bias_validation(self, hand_crafted_returns: np.ndarray) -> None:
        """Finding 2: bias must be strictly a boolean (bool or np.bool_), no silent coercion."""
        r = hand_crafted_returns
        invalid_biases: List[Any] = [1, 0, "true", "false", None, [True], 1.0, 0.0]

        for inv_bias in invalid_biases:
            with pytest.raises(FinancialCalculationError, match="bias must be a boolean"):
                calculate_cornish_fisher_expansion(r, bias=inv_bias)

            with pytest.raises(FinancialCalculationError, match="bias must be a boolean"):
                calculate_cornish_fisher_var(r, bias=inv_bias)

            with pytest.raises(FinancialCalculationError, match="bias must be a boolean"):
                calculate_risk_metrics(r, bias=inv_bias)

            with pytest.raises(FinancialCalculationError, match="bias must be a boolean"):
                RiskMetricsEngine(bias=inv_bias)

    def test_strict_quantile_method_validation(self, hand_crafted_returns: np.ndarray) -> None:
        """Finding 2 & 3: method must be validated against ALLOWED_QUANTILE_METHODS."""
        r = hand_crafted_returns
        invalid_methods: List[Any] = ["unsupported_method", "spline", 123, None, False]

        for inv_method in invalid_methods:
            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                calculate_historical_var(r, method=inv_method)

            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                calculate_cvar(r, method=inv_method)

            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                calculate_risk_metrics(r, method=inv_method)

            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                RiskMetricsEngine(method=inv_method)

        # Check all allowed methods execute without error
        for valid_method in sorted(ALLOWED_QUANTILE_METHODS):
            hvar = calculate_historical_var(r, confidence_level=0.95, method=valid_method)
            cvar = calculate_cvar(r, confidence_level=0.95, method=valid_method)
            assert np.isfinite(hvar)
            assert np.isfinite(cvar)

    def test_api_consistency_confidence_levels(self, hand_crafted_returns: np.ndarray) -> None:
        """Finding 4: calculate_risk_metrics supports confidence_levels parameter."""
        r = hand_crafted_returns

        # Default confidence levels (0.95, 0.99)
        res_default = calculate_risk_metrics(r)
        assert "95%" in res_default.details["by_confidence_level"]
        assert "99%" in res_default.details["by_confidence_level"]
        assert set(res_default.details["by_confidence_level"].keys()) == {"95%", "99%"}
        assert res_default.details["confidence_levels_requested"] == [0.95, 0.99]

        # Custom confidence levels including 0.90, 0.95, 0.975
        custom_levels = (0.90, 0.95, 0.975)
        res_custom = calculate_risk_metrics(r, confidence_levels=custom_levels)

        # Top-level fixed 95 and 99 fields remain populated per canonical methodology
        assert np.isfinite(res_custom.parametric_var_95)
        assert np.isfinite(res_custom.parametric_var_99)
        assert np.isfinite(res_custom.historical_var_95)
        assert np.isfinite(res_custom.historical_var_99)
        assert np.isfinite(res_custom.cornish_fisher_var_95)
        assert np.isfinite(res_custom.cornish_fisher_var_99)
        assert np.isfinite(res_custom.cvar_95)
        assert np.isfinite(res_custom.cvar_99)

        # Details contains all requested custom levels
        by_conf = res_custom.details["by_confidence_level"]
        assert set(by_conf.keys()) == {"90%", "95%", "97.5%"}
        assert res_custom.details["confidence_levels_requested"] == [0.90, 0.95, 0.975]
        for key in ("90%", "95%", "97.5%"):
            entry = by_conf[key]
            assert "confidence_level" in entry
            assert "parametric_var" in entry
            assert "historical_var" in entry
            assert "cornish_fisher_var" in entry
            assert "cvar" in entry
            assert np.isfinite(entry["parametric_var"])
            assert np.isfinite(entry["historical_var"])
            assert np.isfinite(entry["cornish_fisher_var"])
            assert np.isfinite(entry["cvar"])

        # RiskMetricsEngine.calculate also supports custom confidence levels
        engine = RiskMetricsEngine()
        engine_res = engine.calculate(r, confidence_levels=[0.90, 0.99])
        assert set(engine_res.details["by_confidence_level"].keys()) == {"90%", "99%"}
        assert engine_res.details["confidence_levels_requested"] == [0.90, 0.99]

    def test_invalid_confidence_levels_rejected(self, hand_crafted_returns: np.ndarray) -> None:
        """Finding 4: Invalid confidence_levels sequences are rejected."""
        r = hand_crafted_returns
        invalid_cases: List[Any] = [
            (),                     # empty
            [],                     # empty
            0.95,                   # scalar float, not sequence
            "0.95",                 # string
            [0.95, "0.99"],         # non-float element
            [0.95, True],           # boolean element
            [0.0, 0.95],            # out of bounds (0.0)
            [0.95, 1.0],            # out of bounds (1.0)
            [-0.05, 0.95],          # negative
            [0.95, 1.05],           # > 1
            [0.95, np.nan],         # NaN
        ]

        for inv in invalid_cases:
            with pytest.raises(FinancialCalculationError):
                calculate_risk_metrics(r, confidence_levels=inv)

            with pytest.raises(FinancialCalculationError):
                RiskMetricsEngine().calculate(r, confidence_levels=inv)

    def test_cvar_method_validation_with_and_without_explicit_threshold(
        self, hand_crafted_returns: np.ndarray
    ) -> None:
        """Finding 1: method must be validated unconditionally in calculate_cvar().

        Covers:
        - Valid method with explicit threshold.
        - Invalid method with explicit threshold.
        - Invalid method without explicit threshold.
        """
        r = hand_crafted_returns

        # 1. Valid method with explicit threshold succeeds
        cvar_valid = calculate_cvar(r, var_threshold=0.05, method="linear")
        assert np.isfinite(cvar_valid)
        assert cvar_valid > 0.0

        # Also works with other valid methods
        cvar_nearest = calculate_cvar(r, var_threshold=0.05, method="nearest")
        assert np.isfinite(cvar_nearest)

        # 2. Invalid method with explicit threshold raises FinancialCalculationError
        invalid_methods = ["unsupported_method", "cubic_spline", 123, False, None]
        for inv_m in invalid_methods:
            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                calculate_cvar(r, var_threshold=0.05, method=inv_m)

        # 3. Invalid method without explicit threshold raises FinancialCalculationError
        for inv_m in invalid_methods:
            with pytest.raises(FinancialCalculationError, match=r"(Unsupported quantile method|quantile method must be)"):
                calculate_cvar(r, method=inv_m)

        # 4. Engine method interface also respects valid threshold
        engine = RiskMetricsEngine(method="linear")
        assert np.isfinite(engine.cvar(r, var_threshold=0.05))

    def test_confidence_levels_reject_duplicates(self, hand_crafted_returns: np.ndarray) -> None:
        """Finding 2: Duplicate confidence levels must be rejected."""
        r = hand_crafted_returns

        duplicate_cases = [
            (0.95, 0.95),
            [0.90, 0.95, 0.90],
            (0.99, 0.95, 0.99),
            [0.95, 0.95 + 1e-15],  # sub-epsilon duplicate
        ]

        for dup in duplicate_cases:
            with pytest.raises(FinancialCalculationError, match="Duplicate confidence level detected"):
                calculate_risk_metrics(r, confidence_levels=dup)

            with pytest.raises(FinancialCalculationError, match="Duplicate confidence level detected"):
                RiskMetricsEngine().calculate(r, confidence_levels=dup)

    def test_confidence_levels_deterministic_precision_and_no_collision(
        self, hand_crafted_returns: np.ndarray
    ) -> None:
        """Finding 2: Distinct levels must not collide and canonical 95%/99% keys preserved."""
        r = hand_crafted_returns

        # 1. Canonical keys preserved for standard levels
        res_standard = calculate_risk_metrics(r, confidence_levels=(0.95, 0.99))
        assert set(res_standard.details["by_confidence_level"].keys()) == {"95%", "99%"}

        # 2. Distinct levels that would have collided under old '{c*100:.2f}%' logic
        # e.g. 0.95001 and 0.95002 both mapped to '95.00%' previously
        distinct_levels = (0.95001, 0.95002)
        res_distinct = calculate_risk_metrics(r, confidence_levels=distinct_levels)

        by_conf = res_distinct.details["by_confidence_level"]
        assert len(by_conf) == 2
        assert "95.001%" in by_conf
        assert "95.002%" in by_conf
        assert by_conf["95.001%"]["confidence_level"] == 0.95001
        assert by_conf["95.002%"]["confidence_level"] == 0.95002
        assert np.isfinite(by_conf["95.001%"]["cvar"])
        assert np.isfinite(by_conf["95.002%"]["cvar"])

    def test_near_canonical_confidence_level_not_mislabelled(
        self, hand_crafted_returns: np.ndarray
    ) -> None:
        """Finding: Near-canonical levels must not be mislabelled as canonical '95%' or '99%'.

        Verifies:
        1. 0.95 maps to '95%'.
        2. 0.99 maps to '99%'.
        3. 0.950000000000001 does NOT map to '95%'.
        4. Supplying only 0.950000000000001 does not populate canonical 95% fields
           from that custom result; canonical 95% values are independently calculated.
        """
        r = hand_crafted_returns

        # 1. Exact 0.95 and 0.99 map to canonical keys
        assert _format_confidence_key(0.95) == "95%"
        assert _format_confidence_key(0.99) == "99%"
        assert _format_confidence_key(0.90) == "90%"

        # 2. Near-canonical confidence level does NOT map to '95%'
        near_level = 0.950000000000001
        near_key = _format_confidence_key(near_level)
        assert near_key != "95%"
        assert near_key == "95.0000000000001%"

        # 3. Supplying only near_level does not populate canonical fields from it
        res = calculate_risk_metrics(r, confidence_levels=[near_level])

        # '95%' key must NOT be in by_confidence_level
        assert "95%" not in res.details["by_confidence_level"]
        assert near_key in res.details["by_confidence_level"]

        # Canonical 95% values are independently calculated using exact 0.95
        expected_canonical_pvar_95 = calculate_parametric_var(r, confidence_level=0.95)
        custom_near_pvar = res.details["by_confidence_level"][near_key]["parametric_var"]

        assert res.parametric_var_95 == pytest.approx(expected_canonical_pvar_95, abs=1e-12)
        # The custom near result and the canonical result must be distinct
        assert res.parametric_var_95 != custom_near_pvar
        assert abs(res.parametric_var_95 - custom_near_pvar) > 0.0

        # Canonical 99% values are also independently calculated
        expected_canonical_pvar_99 = calculate_parametric_var(r, confidence_level=0.99)
        assert res.parametric_var_99 == pytest.approx(expected_canonical_pvar_99, abs=1e-12)

    def test_high_precision_custom_confidence_level_key(
        self, hand_crafted_returns: np.ndarray
    ) -> None:
        """Micro-audit: non-canonical confidence levels preserve shortest round-trip-safe float representation."""
        r = hand_crafted_returns

        # High-precision custom confidence level
        c_high_prec = 0.12345678901234567
        expected_key = f"{c_high_prec * 100.0}%"
        actual_key = _format_confidence_key(c_high_prec)
        assert actual_key == expected_key
        assert actual_key == "12.345678901234566%"

        # Standard and fractional verification
        assert _format_confidence_key(0.95) == "95%"
        assert _format_confidence_key(0.99) == "99%"
        assert _format_confidence_key(0.975) == "97.5%"
        assert _format_confidence_key(0.950000000000001) == "95.0000000000001%"

        # Full risk metrics calculation with high-precision confidence level
        res = calculate_risk_metrics(r, confidence_levels=[c_high_prec])
        assert expected_key in res.details["by_confidence_level"]
        entry = res.details["by_confidence_level"][expected_key]
        assert entry["confidence_level"] == c_high_prec
        assert np.isfinite(entry["parametric_var"])
        assert np.isfinite(entry["historical_var"])
        assert np.isfinite(entry["cornish_fisher_var"])
        assert np.isfinite(entry["cvar"])




