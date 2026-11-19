"""Quantamental AI - Capital Allocation Line (CAL) Engine Unit Tests.

Phase 2.3, Step 4: Verification of deterministic CAL generation, slope calculation,
coordinate mapping, input validation, immutability, JSON serialization, and integration
with the Maximum Sharpe Ratio (Tangency) optimizer.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    OptimizationException,
)
from app.portfolio_engine.cal import (
    CALCoordinate,
    CALPoint,
    CALResult,
    CapitalAllocationLineResult,
    calculate_cal_return,
    calculate_cal_slope,
    generate_cal_coordinates,
    generate_capital_allocation_line,
)
from app.portfolio_engine.tangency import (
    TangencyOptimizationResult,
    optimize_maximum_sharpe_ratio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tangency_result() -> TangencyOptimizationResult:
    """Standard converged Tangency portfolio for testing."""
    weights = pd.Series([0.6, 0.4], index=["A", "B"], name="weight")
    return TangencyOptimizationResult(
        weights=weights,
        expected_return=0.15,
        variance=0.04,
        volatility=0.20,
        risk_free_rate=0.05,
        sharpe_ratio=0.50,  # (0.15 - 0.05) / 0.20 = 0.50
        success=True,
        status=0,
        message="Optimization terminated successfully",
        iterations=7,
        metadata={"condition_number": 1.25, "annualized": True},
        fallback_executed=False,
    )


@pytest.fixture
def two_asset_cov() -> pd.DataFrame:
    """Well-conditioned positive definite 2-asset covariance matrix."""
    return pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )


@pytest.fixture
def two_asset_mu() -> pd.Series:
    """Expected return vector for 2-asset universe."""
    return pd.Series([0.12, 0.18], index=["A", "B"])


# ---------------------------------------------------------------------------
# 1. Independent Mathematical Correctness Tests
# ---------------------------------------------------------------------------

class TestMathematicalCorrectness:
    """Verify CAL formulas against independent hand calculations."""

    def test_hand_calculated_cal_slope(self) -> None:
        """S_CAL = (mu_T - r_f) / sigma_T. Hand calculation: (0.18 - 0.06) / 0.24 = 0.5."""
        mu_t = 0.18
        sigma_t = 0.24
        rf = 0.06
        expected_slope = (0.18 - 0.06) / 0.24  # exactly 0.5

        slope = calculate_cal_slope(
            tangency_expected_return=mu_t,
            tangency_volatility=sigma_t,
            risk_free_rate=rf,
        )
        assert slope == pytest.approx(expected_slope, rel=1e-12)

    def test_hand_calculated_cal_slope_from_tangency_result(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Slope from TangencyOptimizationResult must match (0.15 - 0.05) / 0.20 = 0.50."""
        slope = calculate_cal_slope(mock_tangency_result)
        assert slope == pytest.approx(0.50, rel=1e-12)

    def test_hand_calculated_cal_return(self) -> None:
        """E[R_c] = r_f + S_CAL * sigma_c. Hand calculation: 0.05 + 0.50 * 0.10 = 0.10."""
        ret = calculate_cal_return(volatility=0.10, slope=0.50, risk_free_rate=0.05)
        assert ret == pytest.approx(0.10, rel=1e-12)

    def test_zero_volatility_coordinate_equals_risk_free_rate(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """At sigma_c = 0.0, E[R_c] must equal r_f exactly, with w_T = 0 and w_rf = 1."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.0])
        assert len(res.points) == 1
        pt = res.points[0]
        assert pt.volatility == 0.0
        assert pt.expected_return == pytest.approx(mock_tangency_result.risk_free_rate, abs=1e-12)
        assert pt.weight_tangency == pytest.approx(0.0, abs=1e-12)
        assert pt.weight_risk_free == pytest.approx(1.0, abs=1e-12)

    def test_tangency_volatility_coordinate_equals_tangency_return(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """At sigma_c = sigma_T, E[R_c] must equal mu_T exactly, with w_T = 1 and w_rf = 0."""
        res = generate_cal_coordinates(
            mock_tangency_result, volatilities=[mock_tangency_result.volatility]
        )
        assert len(res.points) == 1
        pt = res.points[0]
        assert pt.volatility == pytest.approx(mock_tangency_result.volatility, abs=1e-12)
        assert pt.expected_return == pytest.approx(mock_tangency_result.expected_return, abs=1e-12)
        assert pt.weight_tangency == pytest.approx(1.0, abs=1e-12)
        assert pt.weight_risk_free == pytest.approx(0.0, abs=1e-12)

    def test_leveraged_and_lending_portfolio_allocations(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Test lending (sigma_c < sigma_T) and borrowing (sigma_c > sigma_T) complete portfolios."""
        # sigma_T = 0.20
        # Half tangency vol (lending): sigma_c = 0.10 -> w_T = 0.5, w_rf = 0.5
        # 1.5x tangency vol (leveraged): sigma_c = 0.30 -> w_T = 1.5, w_rf = -0.5 (requires allow_leverage=True)
        res = generate_cal_coordinates(
            mock_tangency_result, volatilities=[0.10, 0.30], allow_leverage=True
        )
        lending_pt, levered_pt = res.points

        # Lending point
        assert lending_pt.weight_tangency == pytest.approx(0.5, abs=1e-12)
        assert lending_pt.weight_risk_free == pytest.approx(0.5, abs=1e-12)
        # Expected return: 0.05 + 0.50 * 0.10 = 0.10
        assert lending_pt.expected_return == pytest.approx(0.10, abs=1e-12)

        # Leveraged point
        assert levered_pt.weight_tangency == pytest.approx(1.5, abs=1e-12)
        assert levered_pt.weight_risk_free == pytest.approx(-0.5, abs=1e-12)
        # Expected return: 0.05 + 0.50 * 0.30 = 0.20
        assert levered_pt.expected_return == pytest.approx(0.20, abs=1e-12)

    def test_negative_slope_handled_correctly(self) -> None:
        """When mu_T < r_f, CAL slope is negative but mathematically consistent."""
        mu_t = 0.04
        sigma_t = 0.20
        rf = 0.06
        expected_slope = (0.04 - 0.06) / 0.20  # -0.10
        slope = calculate_cal_slope(
            tangency_expected_return=mu_t,
            tangency_volatility=sigma_t,
            risk_free_rate=rf,
        )
        assert slope == pytest.approx(expected_slope, rel=1e-12)

        ret = calculate_cal_return(volatility=0.10, slope=slope, risk_free_rate=rf)
        # 0.06 + (-0.10) * 0.10 = 0.05
        assert ret == pytest.approx(0.05, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Coordinates & Input Ordering Preservation
# ---------------------------------------------------------------------------

class TestCoordinateGenerator:
    """Verify coordinate generation, ordering preservation, and default grid."""

    def test_preserves_caller_input_ordering(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Volatilities supplied in arbitrary order must retain exact input sequence."""
        custom_vols = [0.18, 0.05, 0.20, 0.0, 0.12]
        res = generate_cal_coordinates(mock_tangency_result, volatilities=custom_vols)

        assert len(res.points) == 5
        generated_vols = [p.volatility for p in res.points]
        assert generated_vols == custom_vols

    def test_preserves_caller_input_ordering_with_leverage(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Ordering is preserved when leveraged coordinates are explicitly enabled."""
        custom_vols = [0.25, 0.05, 0.30, 0.0, 0.15]
        res = generate_cal_coordinates(mock_tangency_result, volatilities=custom_vols, allow_leverage=True)

        assert len(res.points) == 5
        generated_vols = [p.volatility for p in res.points]
        assert generated_vols == custom_vols

    def test_numpy_array_volatilities_accepted(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """NumPy 1D array of volatilities is accepted and preserves order."""
        arr = np.array([0.05, 0.10, 0.15, 0.20])
        res = generate_cal_coordinates(mock_tangency_result, volatilities=arr)
        assert len(res.points) == 4
        assert [p.volatility for p in res.points] == [0.05, 0.10, 0.15, 0.20]

    def test_pandas_series_volatilities_accepted(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Pandas Series of volatilities is accepted."""
        s = pd.Series([0.0, 0.10, 0.20])
        res = generate_cal_coordinates(mock_tangency_result, volatilities=s)
        assert len(res.points) == 3
        assert [p.volatility for p in res.points] == [0.0, 0.10, 0.20]

    def test_default_grid_generation_when_volatilities_none(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """When volatilities is None, default grid ends at 1.0 * sigma_T (MVP no-leverage)."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=None, n_points=50)
        assert len(res.points) == 50
        assert res.points[0].volatility == pytest.approx(0.0)
        # Default grid ends at 1.0 * 0.20 = 0.20 (no leverage)
        assert res.points[-1].volatility == pytest.approx(0.20)
        assert res.points[-1].weight_tangency == pytest.approx(1.0)
        assert res.points[-1].weight_risk_free == pytest.approx(0.0)

    def test_default_grid_does_not_exceed_tangency_volatility(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Verify all default grid coordinates satisfy weight_tangency <= 1.0 and weight_risk_free >= 0.0."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=None, n_points=50)
        sigma_t = mock_tangency_result.volatility
        for p in res.points:
            assert p.volatility <= sigma_t + 1e-12
            assert p.weight_tangency <= 1.0 + 1e-12
            assert p.weight_risk_free >= -1e-12

    def test_custom_n_points_and_multiplier(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Custom n_points and max_volatility_multiplier with allow_leverage=True."""
        res = generate_cal_coordinates(
            mock_tangency_result,
            volatilities=None,
            n_points=10,
            max_volatility_multiplier=3.0,
            allow_leverage=True,
        )
        assert len(res.points) == 10
        assert res.points[-1].volatility == pytest.approx(3.0 * 0.20)
        assert res.allow_leverage is True

    def test_leveraged_coordinates_rejected_by_default(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Under default allow_leverage=False, coordinates with sigma_c > sigma_T are strictly rejected."""
        # sigma_T = 0.20; passing 0.25 exceeds tangency volatility
        with pytest.raises(FinancialCalculationError, match="MVP no-leverage policy"):
            generate_cal_coordinates(mock_tangency_result, volatilities=[0.10, 0.25])

    @pytest.mark.parametrize(
        "delta",
        [1e-8, 1e-6, 1e-4],
    )
    def test_values_just_above_tangency_boundary_rejected_by_default(
        self, mock_tangency_result: TangencyOptimizationResult, delta: float
    ) -> None:
        """Coordinates strictly above sigma_T (even slightly, e.g. delta=1e-8) must be rejected under allow_leverage=False."""
        sigma_t = mock_tangency_result.volatility
        # Absolute offset
        with pytest.raises(FinancialCalculationError, match="MVP no-leverage policy"):
            generate_cal_coordinates(mock_tangency_result, volatilities=[sigma_t + delta])

        # Relative offset
        with pytest.raises(FinancialCalculationError, match="MVP no-leverage policy"):
            generate_cal_coordinates(mock_tangency_result, volatilities=[sigma_t * (1.0 + delta)])

    def test_max_volatility_multiplier_above_one_rejected_by_default(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Setting max_volatility_multiplier > 1.0 without allow_leverage=True raises FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="MVP no-leverage policy"):
            generate_cal_coordinates(mock_tangency_result, max_volatility_multiplier=1.5)

    @pytest.mark.parametrize(
        "mult",
        [1.0001, 1.0 + 1e-8, 1.001],
    )
    def test_multiplier_just_above_one_rejected_by_default(
        self, mock_tangency_result: TangencyOptimizationResult, mult: float
    ) -> None:
        """max_volatility_multiplier just above 1.0 must be rejected under allow_leverage=False."""
        with pytest.raises(FinancialCalculationError, match="MVP no-leverage policy"):
            generate_cal_coordinates(mock_tangency_result, max_volatility_multiplier=mult)

    def test_exact_tangency_boundary_and_machine_precision_remain_valid(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Exact tangency point and machine-precision roundoff remain valid without permitting leverage."""
        sigma_t = mock_tangency_result.volatility

        # Exact tangency point
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[sigma_t])
        assert len(res.points) == 1
        p = res.points[0]
        assert p.volatility == pytest.approx(sigma_t)
        assert p.weight_tangency == 1.0
        assert p.weight_risk_free == 0.0
        assert res.metadata["is_leveraged"] is False

        # Machine-precision roundoff (e.g., 0.1 + 0.2 produces 0.30000000000000004 with 1-ULP diff)
        res_float_boundary = generate_cal_coordinates(
            tangency_expected_return=0.10,
            tangency_volatility=0.30,
            risk_free_rate=0.04,
            volatilities=[0.1 + 0.2],
        )
        assert len(res_float_boundary.points) == 1
        p_bound = res_float_boundary.points[0]
        assert p_bound.weight_tangency == 1.0
        assert p_bound.weight_risk_free == 0.0
        assert res_float_boundary.metadata["is_leveraged"] is False

    def test_is_leveraged_metadata_accurately_identifies_leveraged_coordinates(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """metadata['is_leveraged'] is True only when at least one generated coordinate is genuinely leveraged."""
        sigma_t = mock_tangency_result.volatility

        # 1. Default grid (allow_leverage=False): ends at tangency, no leverage
        res_default = generate_cal_coordinates(mock_tangency_result, n_points=20)
        assert res_default.metadata["is_leveraged"] is False
        assert all(p.weight_tangency <= 1.0 for p in res_default.points)

        # 2. allow_leverage=True, but all requested volatilities are strictly unleveraged
        res_unleveraged = generate_cal_coordinates(
            mock_tangency_result, volatilities=[0.05, 0.10], allow_leverage=True
        )
        assert res_unleveraged.metadata["is_leveraged"] is False
        assert all(p.weight_tangency <= 1.0 for p in res_unleveraged.points)

        # 3. allow_leverage=True with values just above boundary (delta = 1e-8)
        res_just_above = generate_cal_coordinates(
            mock_tangency_result, volatilities=[sigma_t + 1e-8], allow_leverage=True
        )
        assert res_just_above.metadata["is_leveraged"] is True
        assert res_just_above.points[0].weight_tangency > 1.0
        assert res_just_above.points[0].weight_risk_free < 0.0

        # 4. allow_leverage=True with standard leveraged grid
        res_leveraged = generate_cal_coordinates(
            mock_tangency_result, volatilities=[0.10, 0.30], allow_leverage=True
        )
        assert res_leveraged.metadata["is_leveraged"] is True

    def test_explicit_leveraged_mode_produces_negative_risk_free_weights(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """When allow_leverage=True, coordinates > sigma_T produce mathematically correct negative risk-free weights."""
        res = generate_cal_coordinates(
            mock_tangency_result, volatilities=[0.30], allow_leverage=True
        )
        assert len(res.points) == 1
        p = res.points[0]
        # sigma_c = 0.30, sigma_T = 0.20 -> w_T = 1.5, w_rf = -0.5
        assert p.weight_tangency == pytest.approx(1.5)
        assert p.weight_risk_free == pytest.approx(-0.5)
        assert res.allow_leverage is True
        assert res.metadata["is_leveraged"] is True
        assert "not executable under MVP" in res.metadata["leverage_policy"]

    def test_alias_generate_capital_allocation_line(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """generate_capital_allocation_line is identical to generate_cal_coordinates."""
        assert generate_capital_allocation_line is generate_cal_coordinates
        res = generate_capital_allocation_line(mock_tangency_result, volatilities=[0.10])
        assert len(res.points) == 1

    def test_point_tuple_unpacking(self) -> None:
        """CALPoint supports tuple unpacking (vol, ret)."""
        pt = CALPoint(volatility=0.10, expected_return=0.12, weight_tangency=0.5, weight_risk_free=0.5)
        vol, ret = pt
        assert vol == 0.10
        assert ret == 0.12


# ---------------------------------------------------------------------------
# 3. Input Validation & Exception Tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify strict rejection of invalid, non-finite, boolean, and non-numeric inputs."""

    def test_zero_tangency_volatility_rejected(self) -> None:
        """Zero tangency volatility causes division by zero; must raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            calculate_cal_slope(tangency_expected_return=0.15, tangency_volatility=0.0, risk_free_rate=0.05)

    def test_negative_tangency_volatility_rejected(self) -> None:
        """Negative tangency volatility must raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="strictly positive"):
            calculate_cal_slope(tangency_expected_return=0.15, tangency_volatility=-0.10, risk_free_rate=0.05)

    def test_negative_volatility_coordinate_rejected(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Negative volatility coordinates must raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="non-negative"):
            generate_cal_coordinates(mock_tangency_result, volatilities=[0.10, -0.05, 0.20])

    def test_single_negative_volatility_in_calculate_cal_return(self) -> None:
        """calculate_cal_return must reject negative volatility."""
        with pytest.raises(FinancialCalculationError, match="non-negative"):
            calculate_cal_return(volatility=-0.05, slope=0.5, risk_free_rate=0.05)

    @pytest.mark.parametrize("bad_val", [True, False, np.bool_(True)])
    def test_boolean_scalar_inputs_rejected(self, bad_val: Any) -> None:
        """Booleans must not silently convert to 1.0/0.0."""
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_cal_slope(tangency_expected_return=bad_val, tangency_volatility=0.20, risk_free_rate=0.05)
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_cal_slope(tangency_expected_return=0.15, tangency_volatility=bad_val, risk_free_rate=0.05)
        with pytest.raises(FinancialCalculationError, match="bool"):
            calculate_cal_slope(tangency_expected_return=0.15, tangency_volatility=0.20, risk_free_rate=bad_val)

    @pytest.mark.parametrize("bad_seq", [
        [True, False],
        np.array([True, False]),
        pd.Series([True, False]),
        [0.10, True, 0.20],
    ])
    def test_boolean_volatilities_rejected(
        self, mock_tangency_result: TangencyOptimizationResult, bad_seq: Any
    ) -> None:
        """Boolean elements in volatilities sequence must be rejected."""
        with pytest.raises(FinancialCalculationError, match="bool"):
            generate_cal_coordinates(mock_tangency_result, volatilities=bad_seq)

    @pytest.mark.parametrize("bad_val", [complex(1.0, 2.0), np.complex128(1 + 2j)])
    def test_complex_inputs_rejected(self, bad_val: Any) -> None:
        """Complex inputs must be rejected."""
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_cal_slope(tangency_expected_return=bad_val, tangency_volatility=0.20, risk_free_rate=0.05)
        with pytest.raises(FinancialCalculationError, match="complex"):
            calculate_cal_return(volatility=bad_val, slope=0.5, risk_free_rate=0.05)

    @pytest.mark.parametrize("bad_val", ["0.15", "abc", None])
    def test_string_and_none_scalars_rejected(self, bad_val: Any) -> None:
        """Non-numeric strings and None scalars must be rejected."""
        with pytest.raises(FinancialCalculationError):
            calculate_cal_slope(tangency_expected_return=bad_val, tangency_volatility=0.20, risk_free_rate=0.05)

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scalars_rejected(self, bad_val: float) -> None:
        """NaN, +inf, -inf must be rejected."""
        with pytest.raises(FinancialCalculationError, match="finite"):
            calculate_cal_slope(tangency_expected_return=bad_val, tangency_volatility=0.20, risk_free_rate=0.05)
        with pytest.raises(FinancialCalculationError, match="finite"):
            calculate_cal_return(volatility=bad_val, slope=0.5, risk_free_rate=0.05)

    @pytest.mark.parametrize("bad_seq", [
        [0.10, float("nan"), 0.20],
        [float("inf"), 0.10],
        [0.10, "0.20"],
    ])
    def test_non_finite_and_string_in_volatilities_rejected(
        self, mock_tangency_result: TangencyOptimizationResult, bad_seq: Any
    ) -> None:
        """Non-finite or non-numeric elements inside volatilities sequence must be rejected."""
        with pytest.raises(FinancialCalculationError):
            generate_cal_coordinates(mock_tangency_result, volatilities=bad_seq)

    def test_empty_volatilities_raises_insufficient_data(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Empty sequence of volatilities raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError, match="empty"):
            generate_cal_coordinates(mock_tangency_result, volatilities=[])

    def test_scalar_passed_as_volatilities_rejected(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """A scalar float passed as volatilities (not a sequence) raises FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="scalar"):
            generate_cal_coordinates(mock_tangency_result, volatilities=0.15)  # type: ignore

    def test_risk_free_rate_mismatch_with_tangency_result_rejected(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Passing an explicit risk_free_rate that contradicts tangency_portfolio raises FinancialCalculationError."""
        # mock_tangency_result has risk_free_rate=0.05
        with pytest.raises(FinancialCalculationError, match="does not match"):
            generate_cal_coordinates(mock_tangency_result, risk_free_rate=0.08)


# ---------------------------------------------------------------------------
# 4. Fallback Handling & Solver Failure Rejection
# ---------------------------------------------------------------------------

class TestFallbackRejection:
    """Verify that CAL generator rejects unconverged or fallback Tangency portfolios."""

    def test_tangency_fallback_executed_raises_optimization_exception(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Do not silently substitute fallback portfolios: raise OptimizationException."""
        fallback_tangency = dataclasses.replace(
            mock_tangency_result,
            success=False,
            fallback_executed=True,
            message="Fallback to EQUAL_WEIGHT: solver did not converge",
        )
        with pytest.raises(OptimizationException, match="unconverged or fallback"):
            generate_cal_coordinates(fallback_tangency)

    def test_tangency_success_false_raises_optimization_exception(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Tangency with success=False must be rejected."""
        failed_tangency = dataclasses.replace(
            mock_tangency_result,
            success=False,
            message="SLSQP iteration limit reached",
        )
        with pytest.raises(OptimizationException, match="unconverged or fallback"):
            generate_cal_coordinates(failed_tangency)

    def test_slope_rejects_fallback_tangency(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """calculate_cal_slope must also reject fallback tangency results."""
        fallback_tangency = dataclasses.replace(
            mock_tangency_result,
            success=False,
            fallback_executed=True,
        )
        with pytest.raises(OptimizationException, match="unconverged or fallback"):
            calculate_cal_slope(fallback_tangency)


# ---------------------------------------------------------------------------
# 5. Result Dataclass Structure, Defensive Immutability & Serialization
# ---------------------------------------------------------------------------

class TestResultStructureAndImmutability:
    """Verify frozen dataclass behavior, defensive copying, and JSON serialization."""

    def test_capital_allocation_line_result_attributes(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Result contains all required attributes with correct types."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.0, 0.10, 0.20])
        assert isinstance(res.slope, float)
        assert isinstance(res.risk_free_rate, float)
        assert isinstance(res.tangency_expected_return, float)
        assert isinstance(res.tangency_volatility, float)
        assert isinstance(res.points, tuple)
        assert isinstance(res.coordinates, tuple)
        assert isinstance(res.metadata, dict)
        assert isinstance(res.tangency_portfolio, TangencyOptimizationResult)
        assert res.allow_leverage is False

        # Metadata requirements
        assert res.metadata["annualized_units"] == "annualized"
        assert res.metadata["allow_leverage"] is False
        assert "source_tangency_result" in res.metadata
        assert res.metadata["source_tangency_result"]["success"] is True
        assert res.metadata["input_ordering_preserved"] is True

    def test_result_immutability(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Attempting to mutate attributes on frozen dataclass must raise FrozenInstanceError."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.10])
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.slope = 999.0  # type: ignore

    def test_point_immutability(self) -> None:
        """Attempting to mutate attributes on CALPoint must raise FrozenInstanceError."""
        pt = CALPoint(volatility=0.10, expected_return=0.12, weight_tangency=0.5, weight_risk_free=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pt.expected_return = 999.0  # type: ignore

    def test_defensive_copy_tangency_portfolio_mutation(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Mutating the returned tangency_portfolio does not alter stored tangency portfolio."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.10])
        tp1 = res.tangency_portfolio
        assert tp1 is not None
        # Attempt to mutate weights
        try:
            tp1.weights.loc["A"] = 999.0
        except Exception:
            pass
        tp2 = res.tangency_portfolio
        assert tp2 is not None
        assert tp2.weights["A"] == pytest.approx(0.6)
        assert tp2.weights["A"] != 999.0

    def test_defensive_copy_metadata_mutation(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Mutating returned metadata dictionary does not affect stored metadata."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.10])
        meta = res.metadata
        meta["annualized_units"] = "MUTATED"
        assert res.metadata["annualized_units"] == "annualized"

    def test_to_dict_capital_allocation_line_result(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """to_dict() returns expected keys and values."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.0, 0.10])
        d = res.to_dict()

        assert isinstance(d, dict)
        assert "slope" in d
        assert "risk_free_rate" in d
        assert "tangency_expected_return" in d
        assert "tangency_volatility" in d
        assert "tangency_portfolio" in d
        assert "points" in d
        assert "allow_leverage" in d
        assert d["allow_leverage"] is False
        assert "metadata" in d
        assert len(d["points"]) == 2
        assert "volatility" in d["points"][0]
        assert "expected_return" in d["points"][0]
        assert "weight_tangency" in d["points"][0]
        assert "weight_risk_free" in d["points"][0]

    def test_to_dict_json_serializable(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """to_dict() is 100% JSON-serializable."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=[0.0, 0.10, 0.20])
        serialized = json.dumps(res.to_dict())
        assert serialized is not None
        deserialized = json.loads(serialized)
        assert deserialized["slope"] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# 6. Integration Consistency with Tangency Optimizer
# ---------------------------------------------------------------------------

class TestTangencyOptimizerIntegration:
    """Verify end-to-end integration with the real optimize_maximum_sharpe_ratio optimizer."""

    def test_cal_from_real_tangency_optimizer(
        self, two_asset_cov: pd.DataFrame, two_asset_mu: pd.Series
    ) -> None:
        """Solve real Maximum Sharpe Ratio portfolio and verify CAL consistency."""
        rf = 0.05
        tang = optimize_maximum_sharpe_ratio(
            expected_returns=two_asset_mu,
            covariance=two_asset_cov,
            risk_free_rate=rf,
        )
        assert tang.success is True

        # Generate CAL from real Tangency portfolio (with allow_leverage=True for leveraged 1.5x point)
        res = generate_cal_coordinates(
            tang,
            volatilities=[0.0, tang.volatility, tang.volatility * 1.5],
            allow_leverage=True,
        )

        # 1. Slope must match tangency Sharpe ratio
        assert res.slope == pytest.approx(tang.sharpe_ratio, rel=1e-7)

        # 2. Point at 0.0 must equal risk-free rate
        p0 = res.points[0]
        assert p0.volatility == 0.0
        assert p0.expected_return == pytest.approx(rf, abs=1e-7)
        assert p0.weight_tangency == pytest.approx(0.0, abs=1e-7)

        # 3. Point at tang.volatility must equal tangency expected return
        p_tang = res.points[1]
        assert p_tang.volatility == pytest.approx(tang.volatility, rel=1e-7)
        assert p_tang.expected_return == pytest.approx(tang.expected_return, rel=1e-7)
        assert p_tang.weight_tangency == pytest.approx(1.0, rel=1e-7)

        # 4. Leveraged point
        p_lev = res.points[2]
        expected_ret = rf + tang.sharpe_ratio * (tang.volatility * 1.5)
        assert p_lev.expected_return == pytest.approx(expected_ret, rel=1e-7)
        assert p_lev.weight_tangency == pytest.approx(1.5, rel=1e-7)


# ---------------------------------------------------------------------------
# 7. Package Namespace & Re-Export Tests
# ---------------------------------------------------------------------------

class TestNamespaceExports:
    """Verify exports in portfolio_engine, financial_engine, and financial_engine.cal."""

    def test_portfolio_engine_exports(self) -> None:
        import app.portfolio_engine as pe

        assert hasattr(pe, "CALPoint")
        assert hasattr(pe, "CALCoordinate")
        assert hasattr(pe, "CALResult")
        assert hasattr(pe, "CapitalAllocationLineResult")
        assert hasattr(pe, "calculate_cal_slope")
        assert hasattr(pe, "calculate_cal_return")
        assert hasattr(pe, "generate_cal_coordinates")
        assert hasattr(pe, "generate_capital_allocation_line")

    def test_financial_engine_exports(self) -> None:
        import app.financial_engine as fe

        assert hasattr(fe, "CALPoint")
        assert hasattr(fe, "CALCoordinate")
        assert hasattr(fe, "CALResult")
        assert hasattr(fe, "CapitalAllocationLineResult")
        assert hasattr(fe, "calculate_cal_slope")
        assert hasattr(fe, "calculate_cal_return")
        assert hasattr(fe, "generate_cal_coordinates")
        assert hasattr(fe, "generate_capital_allocation_line")

    def test_financial_engine_cal_module_exports(self) -> None:
        import app.financial_engine.cal as fecal

        assert hasattr(fecal, "CALPoint")
        assert hasattr(fecal, "CALCoordinate")
        assert hasattr(fecal, "CALResult")
        assert hasattr(fecal, "CapitalAllocationLineResult")
        assert hasattr(fecal, "calculate_cal_slope")
        assert hasattr(fecal, "calculate_cal_return")
        assert hasattr(fecal, "generate_cal_coordinates")
        assert hasattr(fecal, "generate_capital_allocation_line")

    def test_cross_namespace_identities(self) -> None:
        import app.financial_engine as fe
        import app.financial_engine.cal as fecal
        import app.portfolio_engine as pe
        import app.portfolio_engine.cal as pecal

        assert pe.CapitalAllocationLineResult is pecal.CapitalAllocationLineResult
        assert fe.CapitalAllocationLineResult is pecal.CapitalAllocationLineResult
        assert fecal.CapitalAllocationLineResult is pecal.CapitalAllocationLineResult
        assert pe.calculate_cal_slope is pecal.calculate_cal_slope
        assert fe.calculate_cal_slope is pecal.calculate_cal_slope
        assert fecal.calculate_cal_slope is pecal.calculate_cal_slope


# ---------------------------------------------------------------------------
# 8. Computed-Output Numerical Validation & Overflow Tests
# ---------------------------------------------------------------------------

class TestComputedOutputNumericalValidation:
    """Verify that finite inputs producing overflow/non-finite outputs are rejected."""

    def test_overflow_in_slope_rejected(self) -> None:
        """When slope computation overflows to +inf/-inf, raise FinancialCalculationError."""
        # 1e308 - (-1e308) = 2e308, divided by 1e-100 overflows float64 to inf
        with pytest.raises(FinancialCalculationError, match="overflowed or is non-finite"):
            calculate_cal_slope(
                tangency_expected_return=1e308,
                tangency_volatility=1e-100,
                risk_free_rate=-1e308,
            )

    def test_overflow_in_expected_return_rejected(self) -> None:
        """When expected return computation overflows, raise FinancialCalculationError."""
        with pytest.raises(FinancialCalculationError, match="overflowed or is non-finite"):
            calculate_cal_return(
                volatility=1e308,
                slope=1e308,
                risk_free_rate=0.05,
            )

    def test_overflow_in_generate_cal_coordinates_rejected(self) -> None:
        """When expected return computation overflows during coordinate generation, reject cleanly."""
        with pytest.raises(FinancialCalculationError, match="overflowed or is non-finite"):
            generate_cal_coordinates(
                tangency_expected_return=1e308,
                tangency_volatility=0.20,
                risk_free_rate=0.05,
                volatilities=[1e308],
                allow_leverage=True,
            )

    def test_all_generated_coordinates_remain_finite(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """All computed attributes on generated CALPoint coordinates must be finite."""
        res = generate_cal_coordinates(mock_tangency_result, volatilities=None, n_points=50)
        for p in res.points:
            assert np.isfinite(p.volatility)
            assert np.isfinite(p.expected_return)
            assert np.isfinite(p.weight_tangency)
            assert np.isfinite(p.weight_risk_free)
            assert p.volatility >= 0.0
            assert 0.0 <= p.weight_tangency <= 1.0 + 1e-12
            assert 0.0 <= p.weight_risk_free <= 1.0 + 1e-12

    def test_invalid_allow_leverage_type_rejected(
        self, mock_tangency_result: TangencyOptimizationResult
    ) -> None:
        """Non-boolean allow_leverage must be rejected."""
        with pytest.raises(FinancialCalculationError, match="allow_leverage must be a boolean"):
            generate_cal_coordinates(mock_tangency_result, allow_leverage="yes")  # type: ignore
