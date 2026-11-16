"""Unit tests for Data Provider Base Abstraction and Normalized Schemas.

Covers:
- ABC enforcement (direct instantiation, incomplete subclasses).
- Concrete provider contract execution.
- NormalizedOHLCVRecord Pydantic validation (mode consistency, OHLC ordering).
- validate_normalized_dataframe() comprehensive checks:
    * Missing columns
    * Empty DataFrame passes
    * Invalid trading_date types
    * Duplicate (symbol, trading_date) pairs
    * NaN in mandatory numeric fields
    * Inf in mandatory numeric fields
    * Non-positive adj_close
    * Negative volume
    * Non-positive split_factor
    * Negative dividend_amount
    * Full OHLC consistency (high >= low, high >= open, high >= close, low <= open, low <= close)
    * AUTO_ADJUST_TRUE with non-null nominal prices
    * AUTO_ADJUST_FALSE with null/non-positive nominal prices
    * Non-DataFrame input
"""

from datetime import date, datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.exceptions import ProviderValidationError
from app.data_engine.base import (
    AdjustmentMode,
    BaseDataProvider,
    CorporateActionRecord,
    CorporateActionType,
    NormalizedOHLCVRecord,
    ProviderMetadata,
)


# ---------------------------------------------------------------------------
# Shared helper: build a valid DataFrame row
# ---------------------------------------------------------------------------

def _valid_row(
    symbol: str = "RELIANCE.NS",
    trading_date: object = date(2024, 1, 10),
    adj_close: float = 2800.0,
    volume: int = 1000000,
    adjustment_mode: str = AdjustmentMode.AUTO_ADJUST_FALSE.value,
    open_nominal: object = 2750.0,
    high_nominal: object = 2820.0,
    low_nominal: object = 2740.0,
    close_nominal: object = 2800.0,
    split_factor: float = 1.0,
    dividend_amount: float = 0.0,
) -> dict:
    return {
        "symbol": symbol,
        "trading_date": trading_date,
        "open_nominal": open_nominal,
        "high_nominal": high_nominal,
        "low_nominal": low_nominal,
        "close_nominal": close_nominal,
        "adj_close": adj_close,
        "volume": volume,
        "split_factor": split_factor,
        "dividend_amount": dividend_amount,
        "is_stale": False,
        "data_provider": "yfinance",
        "adjustment_mode": adjustment_mode,
        "retrieval_timestamp": datetime.now(timezone.utc),
    }


def _valid_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame([_valid_row(**kwargs)])


# ---------------------------------------------------------------------------
# Concrete dummy provider
# ---------------------------------------------------------------------------

class DummyValidProvider(BaseDataProvider):
    """Concrete provider implementing all abstract methods for testing."""

    def fetch_historical_ohlcv(self, symbol, start_date, end_date,
                               adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE):
        is_false = adjustment_mode == AdjustmentMode.AUTO_ADJUST_FALSE
        return pd.DataFrame([{
            "symbol": symbol,
            "trading_date": start_date,
            "open_nominal": 100.0 if is_false else None,
            "high_nominal": 105.0 if is_false else None,
            "low_nominal": 98.0 if is_false else None,
            "close_nominal": 102.0 if is_false else None,
            "adj_close": 101.5,
            "volume": 500000,
            "split_factor": 1.0,
            "dividend_amount": 0.0,
            "is_stale": False,
            "data_provider": "dummy",
            "adjustment_mode": adjustment_mode.value,
            "retrieval_timestamp": datetime.now(timezone.utc),
        }])

    def fetch_corporate_actions(self, symbol, start_date, end_date):
        return [CorporateActionRecord(
            symbol=symbol, ex_date=start_date,
            action_type=CorporateActionType.DIVIDEND, value=10.0, source="test",
        )]

    def fetch_trading_calendar(self, start_date, end_date):
        return [start_date, end_date]

    def get_provider_metadata(self):
        return ProviderMetadata(
            provider_name="dummy_test_provider",
            supported_adjustment_modes=[AdjustmentMode.AUTO_ADJUST_FALSE, AdjustmentMode.AUTO_ADJUST_TRUE],
            supports_corporate_actions=True,
            supports_intraday=False,
            licensing_notice="For testing purposes only.",
        )


# ===================================================================
# 1. ABC enforcement
# ===================================================================

class TestABCEnforcement:

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError) as exc_info:
            BaseDataProvider()  # type: ignore
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_incomplete_subclass_rejected(self):
        class Incomplete(BaseDataProvider):
            def fetch_historical_ohlcv(self, symbol, start_date, end_date,
                                       adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE):
                return pd.DataFrame()
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore


# ===================================================================
# 2. Concrete provider contract
# ===================================================================

class TestConcreteProviderContract:

    def test_executes_full_contract(self):
        provider = DummyValidProvider()
        meta = provider.get_provider_metadata()
        assert meta.provider_name == "dummy_test_provider"
        assert meta.supports_corporate_actions is True

        df = provider.fetch_historical_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 1))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert provider.validate_normalized_dataframe(df) is True

        actions = provider.fetch_corporate_actions("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 1))
        assert len(actions) == 1
        assert actions[0].action_type == CorporateActionType.DIVIDEND


# ===================================================================
# 3. NormalizedOHLCVRecord Pydantic validation
# ===================================================================

class TestNormalizedOHLCVPydantic:

    def test_auto_adjust_false_valid(self):
        record = NormalizedOHLCVRecord(
            symbol="TCS.NS", trading_date=date(2024, 1, 15),
            open_nominal=3800.0, high_nominal=3850.0, low_nominal=3780.0, close_nominal=3820.0,
            adj_close=3820.0, volume=1200000,
            data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
        )
        assert record.adj_close == 3820.0

    def test_auto_adjust_false_missing_nominal_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedOHLCVRecord(
                symbol="TCS.NS", trading_date=date(2024, 1, 15),
                open_nominal=None, high_nominal=3850.0, low_nominal=3780.0, close_nominal=3820.0,
                adj_close=3820.0, volume=1200000,
                data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )

    def test_high_lower_than_low_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedOHLCVRecord(
                symbol="TCS.NS", trading_date=date(2024, 1, 15),
                open_nominal=3800.0, high_nominal=3750.0, low_nominal=3780.0, close_nominal=3770.0,
                adj_close=3770.0, volume=1200000,
                data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )

    def test_auto_adjust_true_valid(self):
        record = NormalizedOHLCVRecord(
            symbol="INFY.NS", trading_date=date(2024, 1, 15),
            open_nominal=None, high_nominal=None, low_nominal=None, close_nominal=None,
            adj_close=1650.0, volume=2500000,
            data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE,
        )
        assert record.open_nominal is None

    def test_auto_adjust_true_with_nominal_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedOHLCVRecord(
                symbol="INFY.NS", trading_date=date(2024, 1, 15),
                open_nominal=1650.0, high_nominal=None, low_nominal=None, close_nominal=None,
                adj_close=1650.0, volume=2500000,
                data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE,
            )

    def test_non_positive_adj_close_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedOHLCVRecord(
                symbol="LT.NS", trading_date=date(2024, 1, 15),
                open_nominal=3500.0, high_nominal=3550.0, low_nominal=3480.0, close_nominal=3520.0,
                adj_close=0.0, volume=100000,
                data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )

    def test_negative_volume_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedOHLCVRecord(
                symbol="LT.NS", trading_date=date(2024, 1, 15),
                open_nominal=3500.0, high_nominal=3550.0, low_nominal=3480.0, close_nominal=3520.0,
                adj_close=3520.0, volume=-10,
                data_provider="yfinance", adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
            )


# ===================================================================
# 4. validate_normalized_dataframe — existing + new checks
# ===================================================================

class TestDataFrameValidation:

    def _provider(self):
        return DummyValidProvider()

    # --- Basic checks ---

    def test_valid_df_passes(self):
        assert self._provider().validate_normalized_dataframe(_valid_df()) is True

    def test_empty_df_with_columns_passes(self):
        """Empty DataFrame with correct columns passes immediately."""
        df = pd.DataFrame(columns=BaseDataProvider.REQUIRED_NORMALIZED_COLUMNS)
        assert self._provider().validate_normalized_dataframe(df) is True

    def test_non_dataframe_raises(self):
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe({"not": "df"})  # type: ignore
        assert "not a pandas DataFrame" in str(exc.value)

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"symbol": ["X"], "trading_date": [date(2024, 1, 1)]})
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(bad_df)
        assert "missing required normalized columns" in str(exc.value)

    # --- trading_date type ---

    def test_string_trading_date_raises(self):
        df = _valid_df(trading_date="2024-01-10")
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "trading_date" in str(exc.value)

    def test_datetime_trading_date_raises(self):
        """Verify that trading_date strictly requires datetime.date and rejects datetime."""
        df = _valid_df(trading_date=datetime(2024, 1, 10, 10, 0))
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "trading_date" in str(exc.value)

    # --- Duplicates ---

    def test_duplicate_symbol_trading_date_raises(self):
        row = _valid_row()
        df = pd.DataFrame([row, row])
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "duplicate" in str(exc.value).lower()

    # --- NaN in mandatory columns ---

    def test_nan_in_adj_close_raises(self):
        df = _valid_df()
        df["adj_close"] = float("nan")
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "NaN" in str(exc.value)

    def test_nan_in_volume_raises(self):
        df = _valid_df()
        df["volume"] = pd.array([pd.NA], dtype=pd.Int64Dtype())
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "NaN" in str(exc.value) or "volume" in str(exc.value)

    # --- Inf ---

    def test_positive_inf_in_adj_close_raises(self):
        df = _valid_df()
        df["adj_close"] = float("inf")
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "infinite" in str(exc.value).lower() or "inf" in str(exc.value).lower()

    def test_negative_inf_in_adj_close_raises(self):
        df = _valid_df()
        df["adj_close"] = float("-inf")
        with pytest.raises(ProviderValidationError):
            self._provider().validate_normalized_dataframe(df)

    def test_inf_in_optional_nominal_raises(self):
        df = _valid_df()
        df["open_nominal"] = float("inf")
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "infinite" in str(exc.value).lower() or "inf" in str(exc.value).lower()

    # --- Value constraints ---

    def test_non_positive_adj_close_raises(self):
        df = _valid_df(adj_close=-5.0)
        with pytest.raises(ProviderValidationError):
            self._provider().validate_normalized_dataframe(df)

    def test_negative_volume_raises(self):
        df = _valid_df(volume=-100)
        with pytest.raises(ProviderValidationError):
            self._provider().validate_normalized_dataframe(df)

    def test_non_positive_split_factor_raises(self):
        df = _valid_df(split_factor=0.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "split_factor" in str(exc.value)

    def test_negative_dividend_amount_raises(self):
        df = _valid_df(dividend_amount=-1.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "dividend_amount" in str(exc.value)

    # --- Full OHLC consistency ---

    def test_ohlc_high_less_than_low_raises(self):
        """high_nominal < low_nominal is rejected."""
        df = _valid_df(high_nominal=2700.0, low_nominal=2800.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "high_nominal" in str(exc.value) and "low_nominal" in str(exc.value)

    def test_ohlc_high_less_than_open_raises(self):
        """high_nominal < open_nominal is rejected."""
        # open=2800, high=2790 (invalid), low=2740, close=2780
        df = _valid_df(open_nominal=2800.0, high_nominal=2790.0, low_nominal=2740.0, close_nominal=2780.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "high_nominal" in str(exc.value) and "open_nominal" in str(exc.value)

    def test_ohlc_high_less_than_close_raises(self):
        """high_nominal < close_nominal is rejected."""
        # open=2750, high=2790, low=2740, close=2800 (close > high, invalid)
        df = _valid_df(open_nominal=2750.0, high_nominal=2790.0, low_nominal=2740.0, close_nominal=2800.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "high_nominal" in str(exc.value) and "close_nominal" in str(exc.value)

    def test_ohlc_low_greater_than_open_raises(self):
        """low_nominal > open_nominal is rejected."""
        # open=2740, high=2820, low=2750 (low > open, invalid), close=2800
        df = _valid_df(open_nominal=2740.0, high_nominal=2820.0, low_nominal=2750.0, close_nominal=2800.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "low_nominal" in str(exc.value) and "open_nominal" in str(exc.value)

    def test_ohlc_low_greater_than_close_raises(self):
        """low_nominal > close_nominal is rejected."""
        # open=2800, high=2820, low=2790 (low > close), close=2780
        df = _valid_df(open_nominal=2800.0, high_nominal=2820.0, low_nominal=2790.0, close_nominal=2780.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "low_nominal" in str(exc.value) and "close_nominal" in str(exc.value)

    def test_ohlc_valid_flat_bar_passes(self):
        """Open == High == Low == Close (flat bar) is valid."""
        df = _valid_df(open_nominal=2800.0, high_nominal=2800.0, low_nominal=2800.0, close_nominal=2800.0)
        assert self._provider().validate_normalized_dataframe(df) is True

    # --- Adjustment-mode consistency ---

    def test_auto_adjust_true_with_nominal_raises(self):
        """AUTO_ADJUST_TRUE rows must have all nominal fields as null."""
        df = _valid_df(
            adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE.value,
            open_nominal=100.0,
            high_nominal=None,
            low_nominal=None,
            close_nominal=None,
        )
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "AUTO_ADJUST_TRUE" in str(exc.value)

    def test_auto_adjust_true_null_nominals_passes(self):
        df = _valid_df(
            adjustment_mode=AdjustmentMode.AUTO_ADJUST_TRUE.value,
            open_nominal=None, high_nominal=None, low_nominal=None, close_nominal=None,
        )
        assert self._provider().validate_normalized_dataframe(df) is True

    def test_auto_adjust_false_null_nominal_raises(self):
        df = _valid_df(open_nominal=None)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "AUTO_ADJUST_FALSE" in str(exc.value)

    def test_auto_adjust_false_zero_nominal_raises(self):
        """Zero nominal price in AUTO_ADJUST_FALSE is rejected (must be strictly positive)."""
        # Set all OHLC consistently so OHLC checks pass but mode check catches zero close.
        df = _valid_df(open_nominal=100.0, high_nominal=110.0, low_nominal=0.0, close_nominal=0.0)
        with pytest.raises(ProviderValidationError) as exc:
            self._provider().validate_normalized_dataframe(df)
        assert "AUTO_ADJUST_FALSE" in str(exc.value) or "non-positive" in str(exc.value).lower()
