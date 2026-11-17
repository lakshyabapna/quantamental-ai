"""Unit tests for the Historical Returns and Expected Returns Engine (Phase 2.1).

Verifies:
- Simple return formula R_{i,t} = (P_t / P_{t-1}) - 1 against manual calculations.
- Log return formula r_{i,t} = ln(P_t / P_{t-1}) against manual calculations.
- Time-additivity of log returns vs cross-sectional linearity of simple returns.
- Annualized arithmetic expected return: mu = 252 * mean(R).
- Compound Annual Growth Rate (CAGR): (P_T / P_0)^(252 / N) - 1.
- Exact annualization over 252 trading days.
- Multi-asset listwise-complete observation window alignment.
- Validation: empty input, missing columns, non-positive prices, NaNs, Inf,
  duplicate trading dates, unsorted dates, insufficient observations (< 2 prices).
- ReturnsResult summary container consistency.
"""

from datetime import date, timedelta
import math
import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import (
    FinancialCalculationError,
    InsufficientDataError,
    InvalidPriceError,
    UnalignedObservationsError,
)
from app.data_engine.calendar import nse_calendar
from app.portfolio_engine.returns import (
    DEFAULT_TRADING_DAYS_PER_YEAR,
    ReturnsResult,
    calculate_cagr,
    calculate_expected_returns,
    calculate_log_returns,
    calculate_simple_returns,
    compute_returns_summary,
)


# ===================================================================
# 1. Manual Numerical Calculations & Formula Verification
# ===================================================================

def test_manual_simple_returns_single_asset():
    """Verify daily simple returns against hand-calculated values."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    # Day 0: 100.0, Day 1: 105.0 (+5%), Day 2: 94.50 (-10%)
    prices = pd.Series([100.0, 105.0, 94.50], index=dates, name="TCS.NS")

    ret = calculate_simple_returns(prices)

    assert isinstance(ret, pd.Series)
    assert len(ret) == 2
    assert ret.index.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert pytest.approx(ret.iloc[0], rel=1e-9) == 0.05
    assert pytest.approx(ret.iloc[1], rel=1e-9) == -0.10


def test_manual_log_returns_single_asset():
    """Verify daily log returns against hand-calculated values and check time additivity."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    prices = pd.Series([100.0, 105.0, 94.50], index=dates, name="TCS.NS")

    log_ret = calculate_log_returns(prices)

    assert isinstance(log_ret, pd.Series)
    assert len(log_ret) == 2
    expected_r1 = math.log(105.0 / 100.0)  # ln(1.05) ~ 0.04879016
    expected_r2 = math.log(94.50 / 105.0)  # ln(0.90) ~ -0.10536052
    assert pytest.approx(log_ret.iloc[0], rel=1e-9) == expected_r1
    assert pytest.approx(log_ret.iloc[1], rel=1e-9) == expected_r2

    # Verify time-additivity: r_total = r_1 + r_2 == ln(P_T / P_0)
    total_log_return = log_ret.sum()
    assert pytest.approx(total_log_return, rel=1e-9) == math.log(94.50 / 100.0)


def test_annualized_arithmetic_expected_returns_single_asset():
    """Verify mu_annual = 252 * mean(daily simple returns)."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    prices = pd.Series([100.0, 105.0, 94.50], index=dates, name="TCS.NS")
    # Returns are 0.05 and -0.10 -> mean = -0.025
    # mu_annual = 252 * (-0.025) = -6.30

    # Calculate from returns
    returns = calculate_simple_returns(prices)
    mu_from_returns = calculate_expected_returns(returns, is_returns=True, trading_days_per_year=252)
    assert pytest.approx(mu_from_returns, rel=1e-9) == -6.30

    # Calculate directly from prices
    mu_from_prices = calculate_expected_returns(prices, is_returns=False, trading_days_per_year=252)
    assert pytest.approx(mu_from_prices, rel=1e-9) == -6.30


def test_cagr_calculation_single_asset():
    """Verify CAGR = (P_T / P_0)^(252 / N) - 1 across exact 1-year and multi-period horizons."""
    # Case A: Exactly 252 daily return observations (253 official exchange trading days) growing 15%
    dates_252 = nse_calendar.get_trading_days(date(2024, 1, 1), date(2025, 1, 15))[:253]
    prices_252 = [100.0] * 253
    prices_252[-1] = 115.0  # Ended at 115.0 after 252 steps
    # For a constant growth scenario, let's create geometric compounding
    r_daily = (1.15) ** (1.0 / 252)
    prices_geo = [100.0 * (r_daily ** i) for i in range(253)]
    s_geo = pd.Series(prices_geo, index=dates_252, name="INFY.NS")

    cagr_geo = calculate_cagr(s_geo, is_prices=True, trading_days_per_year=252)
    assert pytest.approx(cagr_geo, rel=1e-6) == 0.15  # Exactly 15.0%

    # Case B: From simple returns
    ret_geo = calculate_simple_returns(s_geo)
    cagr_from_ret = calculate_cagr(ret_geo, is_prices=False, trading_days_per_year=252)
    assert pytest.approx(cagr_from_ret, rel=1e-6) == 0.15


def test_distinction_between_cagr_and_expected_returns():
    """Document and verify that CAGR != arithmetic expected return due to volatility drag."""
    # Symmetrical 2-day return: +20%, then -20%
    # P0 = 100 -> P1 = 120 -> P2 = 96
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    prices = pd.Series([100.0, 120.0, 96.0], index=dates, name="VOLATILE.NS")

    # Simple returns: +0.20, -0.20
    # Arithmetic mean = 0.0 -> Annualized expected return = 0.0
    mu = calculate_expected_returns(prices, is_returns=False, trading_days_per_year=252)
    assert pytest.approx(mu, rel=1e-9) == 0.0

    # Realized CAGR over 2 days: (96 / 100)^(252 / 2) - 1 = (0.96)^126 - 1 < 0
    cagr = calculate_cagr(prices, is_prices=True, trading_days_per_year=252)
    expected_cagr = (96.0 / 100.0) ** (252 / 2) - 1.0
    assert pytest.approx(cagr, rel=1e-9) == expected_cagr
    assert cagr < 0.0  # Negative due to volatility drag


# ===================================================================
# 2. Multi-Asset Matrix Calculations & Structure
# ===================================================================

@pytest.fixture
def multi_asset_prices_df():
    """Synthetic 4-day, 3-asset price DataFrame."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    return pd.DataFrame({
        "RELIANCE.NS": [2500.0, 2525.0, 2550.25, 2601.255],  # +1%, +1%, +2%
        "TCS.NS": [3800.0, 3762.0, 3837.24, 3875.6124],     # -1%, +2%, +1%
        "HDFCBANK.NS": [1600.0, 1600.0, 1600.0, 1616.0],     # 0%, 0%, +1%
    }, index=dates)


def test_multi_asset_simple_returns(multi_asset_prices_df):
    """Test multi-asset wide DataFrame simple returns."""
    ret = calculate_simple_returns(multi_asset_prices_df)

    assert isinstance(ret, pd.DataFrame)
    assert ret.shape == (3, 3)  # 3 return dates, 3 assets
    assert list(ret.columns) == ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

    # Check Day 1 (2024-01-02) returns
    assert pytest.approx(ret.loc[date(2024, 1, 2), "RELIANCE.NS"], rel=1e-6) == 0.01
    assert pytest.approx(ret.loc[date(2024, 1, 2), "TCS.NS"], rel=1e-6) == -0.01
    assert pytest.approx(ret.loc[date(2024, 1, 2), "HDFCBANK.NS"], rel=1e-6) == 0.0


def test_multi_asset_log_returns(multi_asset_prices_df):
    """Test multi-asset wide DataFrame log returns."""
    log_ret = calculate_log_returns(multi_asset_prices_df)

    assert isinstance(log_ret, pd.DataFrame)
    assert log_ret.shape == (3, 3)
    assert pytest.approx(log_ret.loc[date(2024, 1, 2), "RELIANCE.NS"], rel=1e-6) == math.log(1.01)
    assert pytest.approx(log_ret.loc[date(2024, 1, 2), "TCS.NS"], rel=1e-6) == math.log(0.99)


def test_multi_asset_expected_returns(multi_asset_prices_df):
    """Test multi-asset annualized arithmetic expected returns."""
    simple_ret = calculate_simple_returns(multi_asset_prices_df)
    mu = calculate_expected_returns(simple_ret, is_returns=True, trading_days_per_year=252)

    assert isinstance(mu, pd.Series)
    assert len(mu) == 3
    # RELIANCE mean: (0.01 + 0.01 + 0.02) / 3 = 0.04 / 3
    expected_rel_mu = 252.0 * (0.04 / 3.0)
    assert pytest.approx(mu["RELIANCE.NS"], rel=1e-6) == expected_rel_mu


def test_multi_asset_cagr(multi_asset_prices_df):
    """Test multi-asset CAGR calculation from prices."""
    cagr = calculate_cagr(multi_asset_prices_df, is_prices=True, trading_days_per_year=252)

    assert isinstance(cagr, pd.Series)
    assert len(cagr) == 3
    # RELIANCE CAGR: (2601.255 / 2500.0) ** (252 / 3) - 1
    expected_rel_cagr = (2601.255 / 2500.0) ** (252.0 / 3.0) - 1.0
    assert pytest.approx(cagr["RELIANCE.NS"], rel=1e-6) == expected_rel_cagr


def test_compute_returns_summary(multi_asset_prices_df):
    """Test high-level compute_returns_summary orchestrator."""
    summary = compute_returns_summary(multi_asset_prices_df, trading_days_per_year=252)

    assert isinstance(summary, ReturnsResult)
    assert summary.observation_count == 3
    assert summary.symbols == ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    assert summary.start_date == date(2024, 1, 2)
    assert summary.end_date == date(2024, 1, 4)
    assert isinstance(summary.simple_returns, pd.DataFrame)
    assert isinstance(summary.log_returns, pd.DataFrame)
    assert isinstance(summary.expected_returns, pd.Series)
    assert isinstance(summary.cagr, pd.Series)

    as_dict = summary.to_dict()
    assert as_dict["observation_count"] == 3
    assert "RELIANCE.NS" in as_dict["expected_returns"]


# ===================================================================
# 3. Long-Format DataFrame Ingestion & Dictionary Support
# ===================================================================

def test_long_format_dataframe_input():
    """Verify that normalized long-format DataFrame (symbol, trading_date, adj_close) is supported."""
    records = [
        {"symbol": "INFY.NS", "trading_date": date(2024, 1, 1), "adj_close": 1500.0},
        {"symbol": "INFY.NS", "trading_date": date(2024, 1, 2), "adj_close": 1530.0},
        {"symbol": "WIPRO.NS", "trading_date": date(2024, 1, 1), "adj_close": 450.0},
        {"symbol": "WIPRO.NS", "trading_date": date(2024, 1, 2), "adj_close": 441.0},
    ]
    long_df = pd.DataFrame(records)

    ret = calculate_simple_returns(long_df)
    assert isinstance(ret, pd.DataFrame)
    assert list(ret.columns) == ["INFY.NS", "WIPRO.NS"]
    assert len(ret) == 1
    assert pytest.approx(ret.loc[date(2024, 1, 2), "INFY.NS"], rel=1e-6) == 0.02
    assert pytest.approx(ret.loc[date(2024, 1, 2), "WIPRO.NS"], rel=1e-6) == -0.02


def test_dict_of_prices_input():
    """Verify that dictionary of price series is accepted."""
    dates = [date(2024, 1, 1), date(2024, 1, 2)]
    prices_dict = {
        "SBIN.NS": pd.Series([600.0, 630.0], index=dates),
        "ICICIBANK.NS": pd.Series([1000.0, 1020.0], index=dates),
    }

    ret = calculate_simple_returns(prices_dict)
    assert isinstance(ret, pd.DataFrame)
    assert pytest.approx(ret.loc[date(2024, 1, 2), "SBIN.NS"], rel=1e-6) == 0.05
    assert pytest.approx(ret.loc[date(2024, 1, 2), "ICICIBANK.NS"], rel=1e-6) == 0.02


# ===================================================================
# 4. Listwise-Complete Alignment & Missing Observations
# ===================================================================

def test_missing_observation_does_not_create_multisession_return():
    """Verify that a missing observation on Tuesday produces NaN and does NOT treat

    Monday-to-Wednesday as a daily return. Daily returns resume on Thursday when
    consecutive valid prices are observed.
    """
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    # Mon: A=100, B=200
    # Tue: A=105, B=NaN
    # Wed: A=110.25, B=210
    # Thu: A=115.7625, B=220.5
    df = pd.DataFrame({
        "A": [100.0, 105.0, 110.25, 115.7625],
        "B": [200.0, np.nan, 210.0, 220.5],
    }, index=dates)

    # 1. Unaligned returns (aligned=False)
    ret_unaligned = calculate_simple_returns(df, aligned=False)
    assert len(ret_unaligned) == 3
    assert ret_unaligned.index.tolist() == [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

    # Tuesday: A is 105/100-1 = +5%, B is NaN (missing Tuesday price)
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 2), "A"], rel=1e-6) == 0.05
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 2), "B"])

    # Wednesday: A is 110.25/105-1 = +5%, B is NaN because preceding (Tuesday) price was NaN!
    # Crucially: B must NOT be 210/200-1 = 0.05 (a 2-session return)!
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 3), "A"], rel=1e-6) == 0.05
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 3), "B"])

    # Thursday: Consecutive prices exist for both A (115.7625/110.25) and B (220.5/210.0) -> +5% each
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 4), "A"], rel=1e-6) == 0.05
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 4), "B"], rel=1e-6) == 0.05

    # 2. Aligned returns (aligned=True)
    # Applies listwise-complete filtering to the resulting returns matrix.
    # Drops Tuesday and Wednesday (where B is NaN).
    # Retains ONLY Thursday where both A and B have true 1-day daily returns!
    ret_aligned = calculate_simple_returns(df, aligned=True)
    assert len(ret_aligned) == 1
    assert ret_aligned.index.tolist() == [date(2024, 1, 4)]
    assert pytest.approx(ret_aligned.loc[date(2024, 1, 4), "A"], rel=1e-6) == 0.05
    assert pytest.approx(ret_aligned.loc[date(2024, 1, 4), "B"], rel=1e-6) == 0.05


def test_single_series_missing_observation_safety():
    """Verify that a single price Series with a missing observation produces NaN on both

    the missing date and the immediately following date, resuming on the next consecutive date.
    """
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    s = pd.Series([100.0, np.nan, 110.25, 115.7625], index=dates, name="TCS.NS")

    # Unaligned
    ret_unaligned = calculate_simple_returns(s, aligned=False)
    assert len(ret_unaligned) == 3
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 2)])
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 3)])  # Must NOT be 110.25/100-1
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 4)], rel=1e-6) == 0.05

    # Aligned: drops NaNs from returns
    ret_aligned = calculate_simple_returns(s, aligned=True)
    assert len(ret_aligned) == 1
    assert ret_aligned.index.tolist() == [date(2024, 1, 4)]
    assert pytest.approx(ret_aligned.iloc[0], rel=1e-6) == 0.05


def test_long_format_missing_observation_consistency():
    """Verify that long-format DataFrame with missing dates follows identical daily return rules."""
    records = [
        {"symbol": "A", "trading_date": date(2024, 1, 1), "adj_close": 100.0},
        {"symbol": "A", "trading_date": date(2024, 1, 2), "adj_close": 105.0},
        {"symbol": "A", "trading_date": date(2024, 1, 3), "adj_close": 110.25},
        {"symbol": "A", "trading_date": date(2024, 1, 4), "adj_close": 115.7625},
        {"symbol": "B", "trading_date": date(2024, 1, 1), "adj_close": 200.0},
        # B missing Jan 2!
        {"symbol": "B", "trading_date": date(2024, 1, 3), "adj_close": 210.0},
        {"symbol": "B", "trading_date": date(2024, 1, 4), "adj_close": 220.5},
    ]
    long_df = pd.DataFrame(records)

    ret_unaligned = calculate_simple_returns(long_df, aligned=False)
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 2), "B"])
    assert pd.isna(ret_unaligned.loc[date(2024, 1, 3), "B"])
    assert pytest.approx(ret_unaligned.loc[date(2024, 1, 4), "B"], rel=1e-6) == 0.05

    ret_aligned = calculate_simple_returns(long_df, aligned=True)
    assert len(ret_aligned) == 1
    assert ret_aligned.index.tolist() == [date(2024, 1, 4)]
    assert pytest.approx(ret_aligned.loc[date(2024, 1, 4), "A"], rel=1e-6) == 0.05
    assert pytest.approx(ret_aligned.loc[date(2024, 1, 4), "B"], rel=1e-6) == 0.05


def test_no_synthetic_returns_or_ffill():
    """Verify that no forward fill or synthetic zero returns are introduced for missing ticks."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    df = pd.DataFrame({"A": [100.0, np.nan, 100.0]}, index=dates)

    ret = calculate_simple_returns(df, aligned=False)
    # If ffill had occurred, day 2 price would be 100 -> return 0.0; day 3 return 0.0
    # True behavior: both return observations must be strictly NaN!
    assert pd.isna(ret.loc[date(2024, 1, 2), "A"])
    assert pd.isna(ret.loc[date(2024, 1, 3), "A"])
    assert not (ret == 0.0).any().any()


def test_expected_returns_common_listwise_complete_sample_count():
    """Verify calculate_expected_returns evaluates across common listwise-complete returns

    and does not silently calculate means over differing sample counts.
    """
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    # Daily returns:
    # A has returns on all 4 days: +1%, +1%, +1%, +1%
    # B has return on day 1 (+2%), NaN on day 2 and day 3, return on day 4 (+2%)
    ret_df = pd.DataFrame({
        "A": [0.01, 0.01, 0.01, 0.01],
        "B": [0.02, np.nan, np.nan, 0.02],
    }, index=dates[1:])

    # With aligned=True (default), common listwise-complete dates are day 1 and day 4 (2 observations)
    # On day 1 and day 4:
    # A mean = (0.01 + 0.01) / 2 = 0.01 -> mu = 252 * 0.01 = 2.52
    # B mean = (0.02 + 0.02) / 2 = 0.02 -> mu = 252 * 0.02 = 5.04
    mu = calculate_expected_returns(ret_df, is_returns=True, aligned=True, trading_days_per_year=252)

    assert pytest.approx(mu["A"], rel=1e-6) == 252.0 * 0.01
    assert pytest.approx(mu["B"], rel=1e-6) == 252.0 * 0.02


def test_trading_days_per_year_parameter_validation():
    """Verify that trading_days_per_year rejects non-positive, zero, boolean, float, and non-integer inputs."""
    valid_prices = pd.Series([100.0, 105.0], index=[date(2024, 1, 1), date(2024, 1, 2)])
    valid_returns = pd.Series([0.05], index=[date(2024, 1, 2)])

    invalid_params = [0, -252, True, False, 252.5, "252", None, [252]]

    for invalid_val in invalid_params:
        # 1. calculate_expected_returns
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_expected_returns(valid_returns, trading_days_per_year=invalid_val)  # type: ignore

        # 2. calculate_cagr
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            calculate_cagr(valid_prices, trading_days_per_year=invalid_val)  # type: ignore

        # 3. compute_returns_summary
        with pytest.raises(FinancialCalculationError, match="trading_days_per_year"):
            compute_returns_summary(valid_prices, trading_days_per_year=invalid_val)  # type: ignore


def test_disjoint_dates_raises_unaligned_error():
    """Verify that completely disjoint trading dates raise UnalignedObservationsError."""
    df_disjoint = pd.DataFrame({
        "A": [100.0, 105.0, np.nan],
        "B": [np.nan, np.nan, 200.0],
    }, index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])

    with pytest.raises(UnalignedObservationsError, match="No common trading dates"):
        calculate_simple_returns(df_disjoint, aligned=True)


# ===================================================================
# 5. Data Validation & Boundary Edge Cases
# ===================================================================

def test_empty_input_raises_insufficient_data():
    """Verify that empty inputs raise InsufficientDataError."""
    with pytest.raises(InsufficientDataError):
        calculate_simple_returns(pd.DataFrame())

    with pytest.raises(InsufficientDataError):
        calculate_simple_returns(pd.Series(dtype=float))

    with pytest.raises(InsufficientDataError):
        calculate_simple_returns({})

    with pytest.raises(InsufficientDataError):
        calculate_simple_returns(None)  # type: ignore


def test_single_observation_raises_insufficient_data():
    """At least 2 price observations are required to compute 1 return."""
    single_df = pd.DataFrame({"A": [100.0]}, index=[date(2024, 1, 1)])
    with pytest.raises(InsufficientDataError, match="[Aa]t least 2 price observations"):
        calculate_simple_returns(single_df)


def test_non_positive_prices_rejected():
    """Prices <= 0 must be rejected with InvalidPriceError."""
    # Zero price
    df_zero = pd.DataFrame({"A": [100.0, 0.0, 105.0]}, index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(InvalidPriceError, match="strictly positive"):
        calculate_simple_returns(df_zero)

    # Negative price
    df_neg = pd.DataFrame({"A": [100.0, -10.0, 105.0]}, index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(InvalidPriceError, match="strictly positive"):
        calculate_simple_returns(df_neg)


def test_infinite_prices_rejected():
    """Infinite prices must raise InvalidPriceError."""
    df_inf = pd.DataFrame({"A": [100.0, np.inf, 105.0]}, index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(InvalidPriceError, match="infinite price"):
        calculate_simple_returns(df_inf)


def test_non_numeric_prices_rejected():
    """Non-numeric price columns must raise InvalidPriceError."""
    df_str = pd.DataFrame({"A": [100.0, "invalid_price", 105.0]}, index=[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
    with pytest.raises(InvalidPriceError, match="non-numeric"):
        calculate_simple_returns(df_str)


def test_duplicate_trading_dates_rejected():
    """Duplicate dates in wide index or long format must raise FinancialCalculationError."""
    # Wide index duplicate
    dup_index = [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)]
    df_dup = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=dup_index)
    with pytest.raises(FinancialCalculationError, match="[Dd]uplicate trading dates"):
        calculate_simple_returns(df_dup)

    # Long format duplicate per symbol
    long_dup = pd.DataFrame([
        {"symbol": "A", "trading_date": date(2024, 1, 1), "adj_close": 100.0},
        {"symbol": "A", "trading_date": date(2024, 1, 1), "adj_close": 101.0},
    ])
    with pytest.raises(FinancialCalculationError, match="[Dd]uplicate trading date"):
        calculate_simple_returns(long_dup)


def test_missing_required_column_in_long_format_rejected():
    """Long-format DataFrame missing price column raises FinancialCalculationError."""
    bad_long = pd.DataFrame([
        {"symbol": "A", "trading_date": date(2024, 1, 1), "wrong_col": 100.0},
        {"symbol": "A", "trading_date": date(2024, 1, 2), "wrong_col": 102.0},
    ])
    with pytest.raises(FinancialCalculationError, match="missing required price column"):
        calculate_simple_returns(bad_long)


def test_unsorted_dates_are_sorted_chronologically():
    """Input with unordered dates is sorted chronologically before return computation."""
    unordered_dates = [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)]
    df_unordered = pd.DataFrame({"A": [110.0, 100.0, 105.0]}, index=unordered_dates)

    ret = calculate_simple_returns(df_unordered)
    # Sorted dates are Jan 1 (100) -> Jan 2 (105, +5%) -> Jan 3 (110, +4.7619%)
    assert ret.index.tolist() == [date(2024, 1, 2), date(2024, 1, 3)]
    assert pytest.approx(ret.iloc[0], rel=1e-6) == 0.05
    assert pytest.approx(ret.iloc[1], rel=1e-6) == (110.0 / 105.0) - 1.0


def test_flat_prices_produce_exact_zero_returns():
    """Constant flat prices yield exactly 0.0 simple return, log return, expected return, and CAGR."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    flat_prices = pd.Series([100.0, 100.0, 100.0], index=dates, name="FLAT.NS")

    s_ret = calculate_simple_returns(flat_prices)
    assert (s_ret == 0.0).all()

    l_ret = calculate_log_returns(flat_prices)
    assert (l_ret == 0.0).all()

    mu = calculate_expected_returns(s_ret, is_returns=True)
    assert mu == 0.0

    cagr = calculate_cagr(flat_prices, is_prices=True)
    assert cagr == 0.0


# ===================================================================
# 6. Mathematical Axioms & Extended Edge Cases
# ===================================================================

def test_portfolio_simple_returns_linear_additivity():
    """Verify simple returns are strictly linear across assets, whereas log returns are NOT."""
    # Day 0: Asset A = 100, Asset B = 100 -> Portfolio (50/50) = 100
    # Day 1: Asset A = 120 (+20%), Asset B = 80 (-20%) -> Portfolio (50/50) = 100 (0%)
    prices = pd.DataFrame({
        "A": [100.0, 120.0],
        "B": [100.0, 80.0],
    }, index=[date(2024, 1, 1), date(2024, 1, 2)])

    s_ret = calculate_simple_returns(prices)
    weights = np.array([0.5, 0.5])
    # Cross-sectional linear combination: 0.5 * 0.20 + 0.5 * (-0.20) = 0.0
    portfolio_simple_return = float(np.dot(s_ret.iloc[0].values, weights))
    assert pytest.approx(portfolio_simple_return, rel=1e-9) == 0.0

    # Actual portfolio price moves from 0.5*100 + 0.5*100 = 100 to 0.5*120 + 0.5*80 = 100
    # True portfolio return is (100 / 100) - 1 = 0.0 -> matches simple return!
    true_port_ret = (100.0 / 100.0) - 1.0
    assert pytest.approx(portfolio_simple_return, rel=1e-9) == true_port_ret

    # Now verify log returns: ln(1.20) ~ 0.1823, ln(0.80) ~ -0.2231
    l_ret = calculate_log_returns(prices)
    weighted_log_return = float(np.dot(l_ret.iloc[0].values, weights))
    # True portfolio log return is ln(100 / 100) = 0.0
    # But weighted sum of log returns is 0.5 * 0.1823 + 0.5 * (-0.2231) = -0.0204 != 0.0!
    assert weighted_log_return != 0.0
    assert pytest.approx(weighted_log_return, abs=1e-3) == -0.0204
    assert weighted_log_return != true_port_ret  # Confirms log returns break portfolio additivity


def test_expected_returns_error_cases():
    """Verify input validation and edge cases in calculate_expected_returns."""
    # Non-numeric returns
    with pytest.raises(FinancialCalculationError, match="non-numeric"):
        calculate_expected_returns(pd.Series(["a", "b"]))

    # Infinite returns
    with pytest.raises(FinancialCalculationError, match="infinite"):
        calculate_expected_returns(pd.Series([0.05, np.inf]))

    with pytest.raises(FinancialCalculationError, match="infinite"):
        calculate_expected_returns(pd.DataFrame({"A": [0.05, np.inf]}))

    # All NaN returns
    with pytest.raises(InsufficientDataError, match="No valid return observations"):
        calculate_expected_returns(pd.Series([np.nan, np.nan]))


def test_cagr_error_cases():
    """Verify input validation and edge cases in calculate_cagr."""
    # Total loss: return <= -100%
    with pytest.raises(InvalidPriceError, match="[Tt]otal loss"):
        calculate_cagr(pd.Series([0.10, -1.0]), is_prices=False)

    # Infinite return in series
    with pytest.raises(FinancialCalculationError, match="infinite"):
        calculate_cagr(pd.Series([0.10, np.inf]), is_prices=False)

    with pytest.raises(FinancialCalculationError, match="infinite"):
        calculate_cagr(pd.DataFrame({"A": [0.10, np.inf]}), is_prices=False)


def test_single_asset_long_format_without_symbol_column():
    """Verify single-asset long format DataFrame with only trading_date and adj_close."""
    long_df = pd.DataFrame({
        "trading_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "adj_close": [100.0, 105.0, 110.25],
    })

    ret = calculate_simple_returns(long_df)
    assert isinstance(ret, pd.Series)
    assert len(ret) == 2
    assert pytest.approx(ret.iloc[0], rel=1e-6) == 0.05
    assert pytest.approx(ret.iloc[1], rel=1e-6) == 0.05


def test_financial_engine_namespace_alias():
    """Verify financial_engine re-exports from portfolio_engine identically."""
    import app.financial_engine.returns as fe_returns
    import app.portfolio_engine.returns as pe_returns

    assert fe_returns.calculate_simple_returns is pe_returns.calculate_simple_returns
    assert fe_returns.calculate_log_returns is pe_returns.calculate_log_returns
    assert fe_returns.calculate_expected_returns is pe_returns.calculate_expected_returns
    assert fe_returns.calculate_cagr is pe_returns.calculate_cagr
    assert fe_returns.compute_returns_summary is pe_returns.compute_returns_summary
    assert fe_returns.ReturnsResult is pe_returns.ReturnsResult


# ===================================================================
# 7. Targeted CAGR Quantitative Audit Regression Tests
# ===================================================================

def test_cagr_discontinuous_window_rejected():
    """Verify that price inputs with unacceptable discontinuity or insufficient coverage

    are strictly rejected with InsufficientDataError.
    """
    # Case A: Low coverage (< 95%)
    # 10 trading sessions, but 2 are missing -> coverage = 8 / 10 = 80.0% < 95.0%
    trading_days_10 = nse_calendar.get_trading_days(date(2024, 1, 1), date(2024, 1, 15))[:10]
    # Drop 2 days in the middle (indices 3 and 4)
    dropped_days = [trading_days_10[i] for i in [0, 1, 2, 5, 6, 7, 8, 9]]
    prices_low_cov = pd.Series(
        [100.0 + i for i in range(len(dropped_days))],
        index=dropped_days,
        name="LOW_COV.NS",
    )
    with pytest.raises(InsufficientDataError, match="coverage ratio .* below minimum"):
        calculate_cagr(prices_low_cov, is_prices=True)

    # Case B: Excessive consecutive missing trading sessions (> 3 days)
    # 100 trading sessions, but 4 consecutive sessions missing in the middle (coverage = 96% >= 95%)
    trading_days_100 = nse_calendar.get_trading_days(date(2024, 1, 1), date(2024, 6, 30))[:100]
    # Drop indices 20, 21, 22, 23 (4 consecutive missing)
    retained_days = [d for i, d in enumerate(trading_days_100) if i not in (20, 21, 22, 23)]
    prices_consec_missing = pd.Series(
        [100.0 * (1.0005 ** i) for i in range(len(retained_days))],
        index=retained_days,
        name="GAP.NS",
    )
    with pytest.raises(InsufficientDataError, match="consecutive missing trading sessions"):
        calculate_cagr(prices_consec_missing, is_prices=True)

    # Case C: Discontinuous non-calendar / non-date index with gaps
    non_consecutive_series = pd.Series([100.0, 105.0, 110.0], index=[0, 2, 5])
    with pytest.raises(InsufficientDataError, match="Cannot establish elapsed trading sessions"):
        calculate_cagr(non_consecutive_series, is_prices=True)


def test_cagr_elapsed_trading_days_consistency():
    """Verify CAGR uses elapsed trading sessions T, not the count of observed prices or valid returns."""
    # 253 trading sessions (T = 252 elapsed sessions)
    full_trading_days = nse_calendar.get_trading_days(date(2024, 1, 1), date(2025, 1, 15))[:253]

    # Baseline: 15% appreciation from 100.0 to 115.0 with full observations
    baseline_prices = [100.0 * ((1.15) ** (i / 252.0)) for i in range(253)]
    baseline_series = pd.Series(baseline_prices, index=full_trading_days, name="TEST.NS")

    cagr_full = calculate_cagr(baseline_series, is_prices=True, trading_days_per_year=252)
    assert pytest.approx(cagr_full, rel=1e-6) == 0.15

    # Introduce 2 isolated missing sessions in the middle (indices 50 and 150)
    # Valid count = 251 / 253 = 99.2% coverage (>= 95%), max consecutive = 1 (<= 3)
    retained_indices = [i for i in range(253) if i not in (50, 150)]
    partial_days = [full_trading_days[i] for i in retained_indices]
    partial_prices = [baseline_prices[i] for i in retained_indices]
    partial_series = pd.Series(partial_prices, index=partial_days, name="TEST.NS")

    # The elapsed trading sessions T remains exactly 252 (full exchange calendar between t_0 and t_T)
    # CAGR must remain EXACTLY 15.0%, preserving elapsed-time consistency!
    cagr_partial = calculate_cagr(partial_series, is_prices=True, trading_days_per_year=252)
    assert pytest.approx(cagr_partial, rel=1e-6) == 0.15


def test_cagr_boundary_prices_required():
    """Verify calculate_cagr strictly requires valid, positive, finite starting and ending prices."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

    # 1. Missing (NaN) starting price P_0
    s_nan_start = pd.Series([np.nan, 105.0, 110.0], index=dates)
    with pytest.raises(InsufficientDataError, match="[Ss]tarting price .* is missing"):
        calculate_cagr(s_nan_start, is_prices=True)

    # 2. Missing (NaN) ending price P_T
    s_nan_end = pd.Series([100.0, 105.0, np.nan], index=dates)
    with pytest.raises(InsufficientDataError, match="[Ee]nding price .* is missing"):
        calculate_cagr(s_nan_end, is_prices=True)

    # 3. Non-positive starting price P_0 <= 0
    s_zero_start = pd.Series([0.0, 105.0, 110.0], index=dates)
    with pytest.raises(InvalidPriceError, match="strictly positive"):
        calculate_cagr(s_zero_start, is_prices=True)

    # 4. Non-positive ending price P_T <= 0
    s_neg_end = pd.Series([100.0, 105.0, -10.0], index=dates)
    with pytest.raises(InvalidPriceError, match="strictly positive"):
        calculate_cagr(s_neg_end, is_prices=True)

    # 5. Infinite price values
    s_inf_start = pd.Series([np.inf, 105.0, 110.0], index=dates)
    with pytest.raises(InvalidPriceError, match="infinite price"):
        calculate_cagr(s_inf_start, is_prices=True)


def test_cagr_multi_asset_aligned_continuity():
    """Verify multi-asset aligned CAGR requires common evaluation boundaries and consistent elapsed horizons."""
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

    # Case A: Asset B is missing starting price on common start date Jan 1
    df_unaligned_start = pd.DataFrame({
        "A": [100.0, 105.0, 110.25, 115.7625],
        "B": [np.nan, 200.0, 210.0, 220.5],
    }, index=dates)

    with pytest.raises(InsufficientDataError, match="Starting price for asset 'B' is missing at common evaluation boundary"):
        calculate_cagr(df_unaligned_start, is_prices=True, aligned=True)

    # Case B: Asset B is missing ending price on common end date Jan 4
    df_unaligned_end = pd.DataFrame({
        "A": [100.0, 105.0, 110.25, 115.7625],
        "B": [200.0, 210.0, 220.5, np.nan],
    }, index=dates)

    with pytest.raises(InsufficientDataError, match="Ending price for asset 'B' is missing at common evaluation boundary"):
        calculate_cagr(df_unaligned_end, is_prices=True, aligned=True)

    # Case C: Both assets valid across common boundaries -> identical elapsed periods T = 3
    df_valid = pd.DataFrame({
        "A": [100.0, 105.0, 110.25, 115.7625],  # 15.7625% growth over 3 sessions
        "B": [200.0, 210.0, 220.5, 231.525],    # 15.7625% growth over 3 sessions
    }, index=dates)

    cagr_res = calculate_cagr(df_valid, is_prices=True, aligned=True, trading_days_per_year=252)
    assert isinstance(cagr_res, pd.Series)
    assert len(cagr_res) == 2
    expected_cagr_val = (115.7625 / 100.0) ** (252.0 / 3.0) - 1.0
    assert pytest.approx(cagr_res["A"], rel=1e-6) == expected_cagr_val
    assert pytest.approx(cagr_res["B"], rel=1e-6) == expected_cagr_val


def test_cagr_return_input_rejects_internal_nan():
    """Verify that is_prices=False strictly rejects missing observations (NaN) and does not compress time."""
    dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]

    # 1. Single Series with internal NaN
    ret_series_with_nan = pd.Series([0.01, np.nan, 0.02, 0.01], index=dates)
    with pytest.raises(InsufficientDataError, match="Returns sequence contains missing observations"):
        calculate_cagr(ret_series_with_nan, is_prices=False)

    # 2. DataFrame with internal NaN
    ret_df_with_nan = pd.DataFrame({
        "A": [0.01, 0.01, 0.01, 0.01],
        "B": [0.02, np.nan, 0.02, 0.02],
    }, index=dates)
    with pytest.raises(InsufficientDataError, match="Returns DataFrame contains missing observations"):
        calculate_cagr(ret_df_with_nan, is_prices=False)

    # 3. Clean continuous returns succeed
    ret_clean = pd.Series([0.01, 0.01, 0.01, 0.01], index=dates)
    cagr_clean = calculate_cagr(ret_clean, is_prices=False, trading_days_per_year=252)
    expected_cagr = ((1.01 ** 4) ** (252.0 / 4.0)) - 1.0
    assert pytest.approx(cagr_clean, rel=1e-6) == expected_cagr


def test_cagr_return_input_rejects_missing_nse_session_without_nan():
    """Verify that returns without NaNs but missing an intermediate NSE session are rejected."""
    # Jan 1 (Mon), Jan 2 (Tue), Jan 4 (Thu) 2024: Jan 3 (Wed) is missing from index, zero NaNs present!
    dates_missing_wed = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)]

    # 1. Single-asset Series
    s_ret = pd.Series([0.01, 0.02, 0.03], index=dates_missing_wed, name="GAP.NS")
    with pytest.raises(InsufficientDataError, match="missing expected trading session"):
        calculate_cagr(s_ret, is_prices=False)

    # 2. Multi-asset DataFrame
    df_ret = pd.DataFrame({
        "A": [0.01, 0.02, 0.03],
        "B": [0.02, 0.01, 0.02],
    }, index=dates_missing_wed)
    with pytest.raises(InsufficientDataError, match="missing expected trading session"):
        calculate_cagr(df_ret, is_prices=False)


def test_cagr_return_input_accepts_complete_nse_session_sequence():
    """Verify that a complete, unbroken sequence of NSE trading sessions succeeds for returns."""
    # Jan 1 (Mon), Jan 2 (Tue), Jan 3 (Wed), Jan 4 (Thu) 2024
    dates_complete = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

    # 1. Single-asset Series
    s_ret = pd.Series([0.01, 0.02, 0.01, 0.02], index=dates_complete, name="INFY.NS")
    cagr_s = calculate_cagr(s_ret, is_prices=False, trading_days_per_year=252)
    expected_s = ((1.01 * 1.02 * 1.01 * 1.02) ** (252.0 / 4.0)) - 1.0
    assert isinstance(cagr_s, float)
    assert pytest.approx(cagr_s, rel=1e-6) == expected_s

    # 2. Multi-asset DataFrame
    df_ret = pd.DataFrame({
        "A": [0.01, 0.02, 0.01, 0.02],
        "B": [0.02, 0.01, 0.02, 0.01],
    }, index=dates_complete)
    cagr_df = calculate_cagr(df_ret, is_prices=False, trading_days_per_year=252)
    assert isinstance(cagr_df, pd.Series)
    assert len(cagr_df) == 2
    assert pytest.approx(cagr_df["A"], rel=1e-6) == expected_s
    expected_b = ((1.02 * 1.01 * 1.02 * 1.01) ** (252.0 / 4.0)) - 1.0
    assert pytest.approx(cagr_df["B"], rel=1e-6) == expected_b


def test_cagr_return_input_rejects_nonconsecutive_numeric_index():
    """Verify that non-date numeric indices must have consecutive increments of exactly 1."""
    # 1. Series with non-consecutive indices [0, 1, 3]
    s_discont = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 3])
    with pytest.raises(InsufficientDataError, match="discontinuous; increments must be exactly 1"):
        calculate_cagr(s_discont, is_prices=False)

    # 2. DataFrame with non-consecutive indices [0, 2, 4]
    df_discont = pd.DataFrame({
        "A": [0.01, 0.02, 0.03],
        "B": [0.02, 0.01, 0.02],
    }, index=[0, 2, 4])
    with pytest.raises(InsufficientDataError, match="discontinuous; increments must be exactly 1"):
        calculate_cagr(df_discont, is_prices=False)

    # 3. Consecutive integer index [0, 1, 2] succeeds
    s_cont = pd.Series([0.01, 0.02, 0.01], index=[0, 1, 2])
    cagr_cont = calculate_cagr(s_cont, is_prices=False, trading_days_per_year=252)
    expected_cont = ((1.01 * 1.02 * 1.01) ** (252.0 / 3.0)) - 1.0
    assert pytest.approx(cagr_cont, rel=1e-6) == expected_cont


def test_cagr_return_input_rejects_unverified_calendar_range():
    """Verify that returns with dates outside verified NSE calendar circulars are rejected."""
    # Year 2010 is not covered by verified NSE circulars (which cover 2019-2026)
    unverified_dates = [date(2010, 1, 4), date(2010, 1, 5), date(2010, 1, 6)]

    # 1. Single-asset Series
    s_unverified = pd.Series([0.01, 0.02, 0.01], index=unverified_dates)
    with pytest.raises(InsufficientDataError, match="Cannot establish verified exchange calendar coverage"):
        calculate_cagr(s_unverified, is_prices=False)

    # 2. Multi-asset DataFrame
    df_unverified = pd.DataFrame({
        "A": [0.01, 0.02, 0.01],
        "B": [0.02, 0.01, 0.02],
    }, index=unverified_dates)
    with pytest.raises(InsufficientDataError, match="Cannot establish verified exchange calendar coverage"):
        calculate_cagr(df_unverified, is_prices=False)


def test_cagr_price_input_rejects_unverified_calendar_range_even_if_consecutive_days():
    """Verify that date-based price CAGR outside verified NSE circulars is rejected,

    even when the supplied dates are strictly consecutive calendar days.
    """
    # 2010 is not covered by verified NSE circulars (BUILTIN_CIRCULAR_METADATA covers 2019-2026).
    # Jan 1, 2, 3, 4, 2010 are consecutive calendar days (day_diffs == 1).
    consecutive_calendar_dates_2010 = [
        date(2010, 1, 1),
        date(2010, 1, 2),
        date(2010, 1, 3),
        date(2010, 1, 4),
    ]

    # 1. Single-asset Series
    s_prices_2010 = pd.Series([100.0, 101.0, 102.0, 103.0], index=consecutive_calendar_dates_2010, name="INFY.NS")
    with pytest.raises(InsufficientDataError, match="Cannot establish verified exchange calendar coverage"):
        calculate_cagr(s_prices_2010, is_prices=True)

    # 2. Multi-asset DataFrame
    df_prices_2010 = pd.DataFrame({
        "A": [100.0, 101.0, 102.0, 103.0],
        "B": [200.0, 202.0, 204.0, 206.0],
    }, index=consecutive_calendar_dates_2010)
    with pytest.raises(InsufficientDataError, match="Cannot establish verified exchange calendar coverage"):
        calculate_cagr(df_prices_2010, is_prices=True)


def test_cagr_price_input_inside_verified_calendar_works():
    """Verify that date-based price CAGR within verified NSE circular coverage operates as intended."""
    # Jan 1 (Mon), Jan 2 (Tue), Jan 3 (Wed), Jan 4 (Thu) 2024 are verified active NSE sessions
    verified_dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]

    # 1. Single-asset Series (T = 3 elapsed sessions)
    s_prices = pd.Series([100.0, 105.0, 110.25, 115.7625], index=verified_dates, name="TCS.NS")
    cagr_s = calculate_cagr(s_prices, is_prices=True, trading_days_per_year=252)
    expected_cagr = (115.7625 / 100.0) ** (252.0 / 3.0) - 1.0
    assert isinstance(cagr_s, float)
    assert pytest.approx(cagr_s, rel=1e-6) == expected_cagr

    # 2. Multi-asset DataFrame
    df_prices = pd.DataFrame({
        "TCS.NS": [100.0, 105.0, 110.25, 115.7625],
        "INFY.NS": [200.0, 210.0, 220.5, 231.525],
    }, index=verified_dates)
    cagr_df = calculate_cagr(df_prices, is_prices=True, trading_days_per_year=252)
    assert isinstance(cagr_df, pd.Series)
    assert len(cagr_df) == 2
    assert pytest.approx(cagr_df["TCS.NS"], rel=1e-6) == expected_cagr
    assert pytest.approx(cagr_df["INFY.NS"], rel=1e-6) == expected_cagr


def test_cagr_price_input_synthetic_numeric_index_behaviour():
    """Verify that synthetic integer-index price data is supported when strictly consecutive,

    and rejected when discontinuous.
    """
    # 1. Consecutive integer index [0, 1, 2] succeeds (T = 2)
    s_int = pd.Series([100.0, 105.0, 110.25], index=[0, 1, 2])
    cagr_int = calculate_cagr(s_int, is_prices=True, trading_days_per_year=252)
    expected_int = (110.25 / 100.0) ** (252.0 / 2.0) - 1.0
    assert isinstance(cagr_int, float)
    assert pytest.approx(cagr_int, rel=1e-6) == expected_int

    # 2. Multi-asset consecutive integer index succeeds
    df_int = pd.DataFrame({
        "A": [100.0, 105.0, 110.25],
        "B": [200.0, 210.0, 220.5],
    }, index=[0, 1, 2])
    cagr_df_int = calculate_cagr(df_int, is_prices=True, trading_days_per_year=252)
    assert isinstance(cagr_df_int, pd.Series)
    assert pytest.approx(cagr_df_int["A"], rel=1e-6) == expected_int
    assert pytest.approx(cagr_df_int["B"], rel=1e-6) == expected_int

    # 3. Discontinuous integer index [0, 1, 3] is strictly rejected
    s_discont_int = pd.Series([100.0, 105.0, 110.25], index=[0, 1, 3])
    with pytest.raises(InsufficientDataError, match="non-consecutive numeric index"):
        calculate_cagr(s_discont_int, is_prices=True)


def test_cagr_price_input_rejects_weekend_dates():
    """Verify that price input containing a weekend date is strictly rejected."""
    # Jan 5, 2024 (Fri), Jan 6, 2024 (Sat), Jan 8, 2024 (Mon)
    dates_with_weekend = [date(2024, 1, 5), date(2024, 1, 6), date(2024, 1, 8)]
    s_weekend = pd.Series([100.0, 101.0, 102.0], index=dates_with_weekend, name="WEEKEND.NS")

    with pytest.raises(InsufficientDataError, match="not active exchange trading sessions"):
        calculate_cagr(s_weekend, is_prices=True)


def test_cagr_price_input_rejects_exchange_holiday_dates():
    """Verify that price input containing an exchange holiday is strictly rejected."""
    # Jan 25, 2024 (Thu), Jan 26, 2024 (Fri - Republic Day, holiday), Jan 29, 2024 (Mon)
    dates_with_holiday = [date(2024, 1, 25), date(2024, 1, 26), date(2024, 1, 29)]
    s_holiday = pd.Series([100.0, 101.0, 102.0], index=dates_with_holiday, name="HOLIDAY.NS")

    with pytest.raises(InsufficientDataError, match="not active exchange trading sessions"):
        calculate_cagr(s_holiday, is_prices=True)


def test_cagr_price_input_rejects_duplicate_timestamps_normalizing_to_same_date():
    """Verify that multiple intraday timestamps on the same calendar date are rejected."""
    # Two distinct timestamps on Jan 1, 2024 and one on Jan 2, 2024
    timestamps = [
        pd.Timestamp("2024-01-01 09:15:00"),
        pd.Timestamp("2024-01-01 15:30:00"),
        pd.Timestamp("2024-01-02 15:30:00"),
    ]
    s_timestamps = pd.Series([100.0, 101.0, 105.0], index=timestamps, name="INTRADAY.NS")

    with pytest.raises(FinancialCalculationError, match="Duplicate trading dates detected after date normalization"):
        calculate_cagr(s_timestamps, is_prices=True)


def test_cagr_price_input_valid_date_based_series_calculates_correctly():
    """Verify that valid date-based price series within NSE calendar coverage calculate accurately."""
    # Jan 1 (Mon), Jan 2 (Tue), Jan 3 (Wed), Jan 4 (Thu) 2024
    valid_dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    s_valid = pd.Series([100.0, 105.0, 110.25, 115.7625], index=valid_dates, name="VALID.NS")

    cagr_val = calculate_cagr(s_valid, is_prices=True, trading_days_per_year=252)
    expected_cagr = (115.7625 / 100.0) ** (252.0 / 3.0) - 1.0
    assert isinstance(cagr_val, float)
    assert pytest.approx(cagr_val, rel=1e-6) == expected_cagr





