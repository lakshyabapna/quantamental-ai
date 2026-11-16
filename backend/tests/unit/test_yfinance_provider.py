"""Unit tests for the Yahoo Finance Data Provider Adapter."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app.core.exceptions import (
    ProviderConnectionError,
    ProviderValidationError,
)
from app.data_engine.base import AdjustmentMode, CorporateActionType
from app.data_engine.yfinance_provider import YFinanceDataProvider


@pytest.fixture
def yf_provider():
    return YFinanceDataProvider()


def _mock_history_df(auto_adjust: bool = False) -> pd.DataFrame:
    """Construct mock DataFrame returned by yf.Ticker.history."""
    dates = pd.date_range("2024-01-10", periods=3, freq="B", tz="Asia/Kolkata")
    if not auto_adjust:
        return pd.DataFrame({
            "Open": [2700.0, 2710.0, 2720.0],
            "High": [2750.0, 2760.0, 2770.0],
            "Low": [2690.0, 2700.0, 2710.0],
            "Close": [2740.0, 2750.0, 2760.0],
            "Adj Close": [2740.0, 2750.0, 2760.0],
            "Volume": [1000000, 1100000, 1200000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        }, index=dates)
    else:
        return pd.DataFrame({
            "Open": [2700.0, 2710.0, 2720.0],
            "High": [2750.0, 2760.0, 2770.0],
            "Low": [2690.0, 2700.0, 2710.0],
            "Close": [2740.0, 2750.0, 2760.0],
            "Volume": [1000000, 1100000, 1200000],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        }, index=dates)


def test_provider_metadata(yf_provider: YFinanceDataProvider):
    """Test provider metadata and licensing disclosure."""
    meta = yf_provider.get_provider_metadata()
    assert meta.provider_name == "yfinance"
    assert AdjustmentMode.AUTO_ADJUST_FALSE in meta.supported_adjustment_modes
    assert AdjustmentMode.AUTO_ADJUST_TRUE in meta.supported_adjustment_modes
    assert meta.supports_corporate_actions is True
    assert meta.supports_intraday is False
    assert meta.rate_limit_requests_per_minute is None
    assert "personal" in meta.licensing_notice.lower()


def test_fetch_historical_ohlcv_auto_adjust_false(yf_provider: YFinanceDataProvider):
    """Test fetching historical OHLCV in AUTO_ADJUST_FALSE mode."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history_df(auto_adjust=False)

    with patch("yfinance.Ticker", return_value=mock_ticker):
        df = yf_provider.fetch_historical_ohlcv(
            symbol="RELIANCE.NS",
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 12),
            adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
        )

    assert not df.empty
    assert len(df) == 3
    assert df["open_nominal"].iloc[0] == 2700.0
    assert df["adj_close"].iloc[0] == 2740.0
    assert df["adjustment_mode"].iloc[0] == AdjustmentMode.AUTO_ADJUST_FALSE.value
    assert isinstance(df["trading_date"].iloc[0], date)
    assert yf_provider.validate_normalized_dataframe(df) is True


def test_fetch_historical_ohlcv_auto_adjust_true(yf_provider: YFinanceDataProvider):
    """Test fetching historical OHLCV in AUTO_ADJUST_TRUE mode."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history_df(auto_adjust=True)

    with patch("yfinance.Ticker", return_value=mock_ticker):
        df = yf_provider.fetch_historical_ohlcv(
            symbol="RELIANCE.NS",
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 12),
            adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE,
        )

    assert not df.empty
    assert len(df) == 3
    # Nominal fields must remain None when auto_adjust=True
    assert df["open_nominal"].isna().all()
    assert df["high_nominal"].isna().all()
    assert df["low_nominal"].isna().all()
    assert df["close_nominal"].isna().all()
    assert df["adj_close"].iloc[0] == 2740.0
    assert df["adjustment_mode"].iloc[0] == AdjustmentMode.AUTO_ADJUST_TRUE.value
    assert yf_provider.validate_normalized_dataframe(df) is True


def test_fetch_historical_ohlcv_invalid_date_range(yf_provider: YFinanceDataProvider):
    """Test start_date > end_date raises ProviderValidationError."""
    with pytest.raises(ProviderValidationError):
        yf_provider.fetch_historical_ohlcv(
            symbol="TCS.NS",
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 10),
        )


def test_fetch_historical_ohlcv_empty_response(yf_provider: YFinanceDataProvider):
    """Test handling of empty response from provider."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        df = yf_provider.fetch_historical_ohlcv(
            symbol="NONEXISTENT.NS",
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 12),
        )

    assert df.empty
    assert list(df.columns) == yf_provider.REQUIRED_NORMALIZED_COLUMNS


def test_fetch_historical_ohlcv_network_failure(yf_provider: YFinanceDataProvider):
    """Test handling of network failure raises ProviderConnectionError."""
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = RuntimeError("Network timeout")

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderConnectionError):
            yf_provider.fetch_historical_ohlcv(
                symbol="RELIANCE.NS",
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 12),
            )


def test_fetch_corporate_actions(yf_provider: YFinanceDataProvider):
    """Test fetching dividends and splits from dedicated ticker streams."""
    mock_ticker = MagicMock()
    dates = pd.date_range("2024-01-10", periods=2, freq="30D", tz="Asia/Kolkata")
    mock_ticker.dividends = pd.Series([10.0, 15.0], index=dates)
    mock_ticker.splits = pd.Series([2.0], index=[dates[0]])

    with patch("yfinance.Ticker", return_value=mock_ticker):
        actions = yf_provider.fetch_corporate_actions(
            symbol="INFY.NS",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 1),
        )

    assert len(actions) == 3
    div_actions = [a for a in actions if a.action_type == CorporateActionType.DIVIDEND]
    split_actions = [a for a in actions if a.action_type == CorporateActionType.SPLIT]
    assert len(div_actions) == 2
    assert len(split_actions) == 1
    assert split_actions[0].value == 2.0
    assert div_actions[0].value == 10.0


# ===================================================================
# Additional Validation and Edge Cases
# ===================================================================

def test_fetch_historical_ohlcv_invalid_adjustment_mode(yf_provider: YFinanceDataProvider):
    """Test that an invalid adjustment mode raises ProviderValidationError."""
    with pytest.raises(ProviderValidationError, match="Invalid adjustment_mode"):
        yf_provider.fetch_historical_ohlcv(
            symbol="TCS.NS",
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 12),
            adjustment_mode="INVALID_MODE",  # type: ignore
        )


def test_fetch_historical_missing_volume_raises(yf_provider: YFinanceDataProvider):
    """Test that missing Volume column or NaN volume raises ProviderValidationError."""
    mock_ticker = MagicMock()
    df_no_vol = _mock_history_df(auto_adjust=False).drop(columns=["Volume"])
    mock_ticker.history.return_value = df_no_vol

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderValidationError, match="Missing mandatory 'Volume'"):
            yf_provider.fetch_historical_ohlcv(
                symbol="RELIANCE.NS",
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 12),
            )


def test_fetch_historical_missing_adj_close_in_auto_adjust_false_raises(yf_provider: YFinanceDataProvider):
    """Test that missing Adj Close in AUTO_ADJUST_FALSE raises ProviderValidationError."""
    mock_ticker = MagicMock()
    df_no_adj = _mock_history_df(auto_adjust=False).drop(columns=["Adj Close"])
    mock_ticker.history.return_value = df_no_adj

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderValidationError, match="Missing mandatory 'Adj Close'"):
            yf_provider.fetch_historical_ohlcv(
                symbol="RELIANCE.NS",
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 12),
                adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )


def test_fetch_historical_nan_nominal_price_in_auto_adjust_false_raises(yf_provider: YFinanceDataProvider):
    """Test that NaN nominal price in AUTO_ADJUST_FALSE raises ProviderValidationError."""
    mock_ticker = MagicMock()
    df_bad_price = _mock_history_df(auto_adjust=False)
    df_bad_price.loc[df_bad_price.index[0], "Open"] = float("nan")
    mock_ticker.history.return_value = df_bad_price

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderValidationError, match="nominal price"):
            yf_provider.fetch_historical_ohlcv(
                symbol="RELIANCE.NS",
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 12),
                adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )


def test_fetch_historical_missing_close_in_auto_adjust_true_raises(yf_provider: YFinanceDataProvider):
    """Test that missing Close in AUTO_ADJUST_TRUE raises ProviderValidationError."""
    mock_ticker = MagicMock()
    df_bad = _mock_history_df(auto_adjust=True).drop(columns=["Close"])
    mock_ticker.history.return_value = df_bad

    with patch("yfinance.Ticker", return_value=mock_ticker):
        with pytest.raises(ProviderValidationError, match="Missing mandatory 'Close'"):
            yf_provider.fetch_historical_ohlcv(
                symbol="RELIANCE.NS",
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 12),
                adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE,
            )


def test_fetch_corporate_actions_empty_streams(yf_provider: YFinanceDataProvider):
    """Test that empty corporate action streams return an empty list."""
    mock_ticker = MagicMock()
    mock_ticker.dividends = pd.Series(dtype=float)
    mock_ticker.splits = pd.Series(dtype=float)

    with patch("yfinance.Ticker", return_value=mock_ticker):
        actions = yf_provider.fetch_corporate_actions(
            symbol="INFY.NS",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 1),
        )

    assert actions == []


def test_fetch_corporate_actions_timezone_normalization(yf_provider: YFinanceDataProvider):
    """Test that corporate action timestamps are converted cleanly to Asia/Kolkata date objects."""
    mock_ticker = MagicMock()
    # UTC timestamp representing 2024-01-10 18:30 UTC -> 2024-01-11 00:00 IST
    idx = pd.to_datetime(["2024-01-10 18:30:00+00:00"])
    mock_ticker.dividends = pd.Series([25.0], index=idx)
    mock_ticker.splits = pd.Series(dtype=float)

    with patch("yfinance.Ticker", return_value=mock_ticker):
        actions = yf_provider.fetch_corporate_actions(
            symbol="INFY.NS",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

    assert len(actions) == 1
    assert actions[0].ex_date == date(2024, 1, 11)  # Converted to IST date
    assert actions[0].value == 25.0
