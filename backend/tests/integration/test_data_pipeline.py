"""Integration tests for the Incremental Ingestion and Storage Pipeline."""

from datetime import date, datetime, timezone
import tempfile
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from app.core.exceptions import CalendarDateRangeError

from app.data_engine.base import (
    AdjustmentMode,
    BaseDataProvider,
    CorporateActionRecord,
    CorporateActionType,
    ProviderMetadata,
)
from app.data_engine.calendar import NSECalendar
from app.data_engine.clean_pipeline import DataCleansingPipeline, QualityThresholds
from app.data_engine.incremental import IncrementalHistoricalEngine
from app.data_engine.storage import ParquetStorageEngine


class MockIntegrationProvider(BaseDataProvider):
    """Configurable provider for pipeline integration tests."""

    def __init__(self) -> None:
        self.data_by_symbol = {}
        self.actions_by_symbol = {}

    def set_historical_data(self, symbol: str, df: pd.DataFrame) -> None:
        self.data_by_symbol[symbol] = df

    def set_corporate_actions(self, symbol: str, actions: list) -> None:
        self.actions_by_symbol[symbol] = actions

    def fetch_historical_ohlcv(self, symbol, start_date, end_date, adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE):
        if symbol not in self.data_by_symbol:
            return pd.DataFrame(columns=self.REQUIRED_NORMALIZED_COLUMNS)
        df = self.data_by_symbol[symbol].copy()
        mask = (df["trading_date"] >= start_date) & (df["trading_date"] <= end_date)
        return df.loc[mask].reset_index(drop=True)

    def fetch_corporate_actions(self, symbol, start_date, end_date):
        return self.actions_by_symbol.get(symbol, [])

    def fetch_trading_calendar(self, start_date, end_date):
        return []

    def get_provider_metadata(self):
        return ProviderMetadata(
            provider_name="mock_integration_provider",
            supported_adjustment_modes=[AdjustmentMode.AUTO_ADJUST_FALSE],
            supports_corporate_actions=True,
            supports_intraday=False,
            licensing_notice="Test only",
        )


def _generate_ohlcv_batch(symbol: str, dates: list, base_price: float = 1000.0) -> pd.DataFrame:
    records = []
    for i, d in enumerate(dates):
        p = base_price + i * 5.0
        records.append({
            "symbol": symbol,
            "trading_date": d,
            "open_nominal": p - 2.0,
            "high_nominal": p + 5.0,
            "low_nominal": p - 4.0,
            "close_nominal": p,
            "adj_close": p,
            "volume": 2000000,
            "split_factor": 1.0,
            "dividend_amount": 0.0,
            "is_stale": False,
            "data_provider": "mock",
            "adjustment_mode": AdjustmentMode.AUTO_ADJUST_FALSE.value,
            "retrieval_timestamp": datetime.now(timezone.utc),
        })
    return pd.DataFrame(records)


@pytest.fixture
def test_pipeline_env():
    """Setup isolated pipeline components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ParquetStorageEngine(storage_dir=tmpdir)
        provider = MockIntegrationProvider()
        calendar = NSECalendar(load_builtin=True)
        cleanser = DataCleansingPipeline(
            default_thresholds=QualityThresholds(
                min_history_coverage_ratio=0.80,
                max_consecutive_missing_days=5,
                max_single_day_return_spike=0.50,
            )
        )
        engine = IncrementalHistoricalEngine(
            provider=provider,
            calendar=calendar,
            storage=storage,
            cleanser=cleanser,
        )
        yield {
            "engine": engine,
            "provider": provider,
            "storage": storage,
            "calendar": calendar,
        }


def test_initial_ingestion_and_incremental_overlap(test_pipeline_env):
    """Test full cycle: initial load, then incremental update with overlap."""
    engine = test_pipeline_env["engine"]
    provider = test_pipeline_env["provider"]
    storage = test_pipeline_env["storage"]

    symbol = "RELIANCE.NS"
    # Days in Jan 2024 (weekdays)
    batch_1_dates = [
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
        date(2024, 1, 11),
        date(2024, 1, 12),
    ]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, batch_1_dates, 2500.0))

    # Initial refresh as of 12 Jan 2024
    audit_1 = engine.refresh_symbol(
        symbol=symbol,
        as_of=date(2024, 1, 12),
        overlap_days=2,
        start_date=batch_1_dates[0],
    )
    assert audit_1.status == "SUCCESS"
    assert audit_1.records_added == 5
    assert audit_1.records_updated == 0
    assert storage.get_latest_stored_date(symbol) == date(2024, 1, 12)

    # Prepare second batch: 11th, 12th (overlap), and 15th, 16th (new)
    batch_2_dates = [
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
        date(2024, 1, 11),
        date(2024, 1, 12),
        date(2024, 1, 15),
        date(2024, 1, 16),
    ]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, batch_2_dates, 2500.0))

    # Second refresh as of 16 Jan 2024 with 2 days overlap
    audit_2 = engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 16), overlap_days=2)
    assert audit_2.status == "SUCCESS"
    assert audit_2.records_added == 2  # 15th, 16th
    assert audit_2.records_updated > 0  # overlap days reconciled
    assert storage.get_latest_stored_date(symbol) == date(2024, 1, 16)

    # Verify stored data has all 7 unique days sorted
    stored_df = storage.load_symbol_data(symbol)
    assert len(stored_df) == 7
    assert stored_df["trading_date"].tolist() == batch_2_dates


def test_quarantine_preserves_existing_archive(test_pipeline_env):
    """Test that a quarantined batch does not corrupt existing stored historical data."""
    engine = test_pipeline_env["engine"]
    provider = test_pipeline_env["provider"]
    storage = test_pipeline_env["storage"]

    symbol = "TCS.NS"
    initial_dates = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, initial_dates, 3800.0))

    # Initial valid load
    engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 10), start_date=initial_dates[0])
    initial_stored = storage.load_symbol_data(symbol)
    assert len(initial_stored) == 3

    # Create corrupted incremental batch with unexplained price spike on 11 Jan
    corrupted_dates = [date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]
    bad_df = _generate_ohlcv_batch(symbol, corrupted_dates, 3800.0)
    # Set price to 7000 (huge jump, no split)
    bad_df.loc[bad_df["trading_date"] == date(2024, 1, 11), "adj_close"] = 7000.0
    bad_df.loc[bad_df["trading_date"] == date(2024, 1, 11), "high_nominal"] = 7050.0
    bad_df.loc[bad_df["trading_date"] == date(2024, 1, 11), "close_nominal"] = 7000.0
    provider.set_historical_data(symbol, bad_df)

    audit_bad = engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 11), overlap_days=2)
    assert audit_bad.status == "QUARANTINED"
    assert any("QUARANTINE" in w for w in audit_bad.warnings)

    # Check that storage was preserved intact (no rows added or modified)
    preserved_df = storage.load_symbol_data(symbol)
    assert len(preserved_df) == 3
    assert preserved_df["trading_date"].max() == date(2024, 1, 10)


def test_get_active_research_window(test_pipeline_env):
    """Test slicing active research window from storage."""
    engine = test_pipeline_env["engine"]
    provider = test_pipeline_env["provider"]
    storage = test_pipeline_env["storage"]

    symbol = "HDFCBANK.NS"
    dates = [
        date(2019, 1, 10),
        date(2021, 6, 15),
        date(2024, 1, 10),
    ]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, dates, 1500.0))
    storage.save_symbol_data(symbol, _generate_ohlcv_batch(symbol, dates, 1500.0))

    # Active 5-year window as of 10 Jan 2024
    # 10 Jan 2024 minus 5 years = 10 Jan 2019 (Thursday, active session)
    research_df = engine.get_active_research_window(symbol, as_of=date(2024, 1, 10), years=5)
    assert not research_df.empty
    assert research_df["trading_date"].min() >= date(2019, 1, 10)
    assert research_df["trading_date"].max() <= date(2024, 1, 10)


# ===================================================================
# Incremental Engine Status, Overlap, and Cutoff Handling
# ===================================================================

def test_negative_overlap_days_raises(test_pipeline_env):
    """Test that overlap_days < 0 raises CalendarDateRangeError."""
    engine = test_pipeline_env["engine"]
    with pytest.raises(CalendarDateRangeError, match="overlap_days must be non-negative"):
        engine.refresh_symbol(symbol="INFY.NS", overlap_days=-1)


def test_zero_overlap_days_starts_at_next_trading_day(test_pipeline_env):
    """Test that overlap_days == 0 starts fetching from the day after latest_stored."""
    engine = test_pipeline_env["engine"]
    provider = test_pipeline_env["provider"]
    storage = test_pipeline_env["storage"]

    symbol = "SBIN.NS"
    # Initial batch: 8, 9, 10 Jan 2024
    initial_dates = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, initial_dates, 600.0))
    engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 10), start_date=initial_dates[0])

    # Next batch has 11, 12 Jan 2024
    full_dates = initial_dates + [date(2024, 1, 11), date(2024, 1, 12)]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, full_dates, 600.0))

    # Incremental refresh with overlap_days = 0
    audit = engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 12), overlap_days=0)
    assert audit.status == "SUCCESS"
    assert audit.records_added == 2  # exactly 11th and 12th
    assert audit.records_updated == 0  # no overlap, 0 updated

    stored = storage.load_symbol_data(symbol)
    assert len(stored) == 5
    assert stored["trading_date"].tolist() == full_dates


def test_excluded_batch_preserves_archive(test_pipeline_env):
    """Test that an EXCLUDED batch (coverage < 95% but non-consecutive) preserves existing archive."""
    engine = test_pipeline_env["engine"]
    provider = test_pipeline_env["provider"]
    storage = test_pipeline_env["storage"]

    symbol = "WIPRO.NS"
    # Initial batch: 8, 9, 10 Jan 2024
    initial_dates = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, initial_dates, 450.0))
    engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 10), start_date=initial_dates[0])

    # Sized batch where coverage is < 80% (the pipeline's min_history_coverage_ratio in fixture)
    # Target range: Jan 11 to Jan 25, 2024 (11 trading days)
    # We only supply 5 non-consecutive days (coverage 5/11 = 45% < 80%)
    # Max consecutive missing is 2 <= 5 (not quarantined)
    sparse_dates = [
        date(2024, 1, 11),
        date(2024, 1, 15),
        date(2024, 1, 17),
        date(2024, 1, 19),
        date(2024, 1, 23),
    ]
    provider.set_historical_data(symbol, _generate_ohlcv_batch(symbol, initial_dates + sparse_dates, 450.0))

    # Refresh as of 25 Jan 2024
    audit = engine.refresh_symbol(symbol=symbol, as_of=date(2024, 1, 25), overlap_days=0)
    assert audit.status == "EXCLUDED"
    assert audit.records_added == 0
    assert audit.records_updated == 0
    assert any("exclusion threshold" in w for w in audit.warnings)

    # Verify storage remains untouched
    stored = storage.load_symbol_data(symbol)
    assert len(stored) == 3
    assert stored["trading_date"].max() == date(2024, 1, 10)


def test_resolve_latest_completed_session_cutoff(test_pipeline_env):
    """Test resolve_latest_completed_session uses 16:00 IST cutoff when as_of is None."""
    engine = test_pipeline_env["engine"]
    tz = ZoneInfo("Asia/Kolkata")

    # Case 1: Wednesday 10 Jan 2024 at 15:30 IST (before 16:00 cutoff)
    # Session is NOT completed yet -> returns previous trading day (9 Jan 2024)
    dt_before_cutoff = datetime(2024, 1, 10, 15, 30, tzinfo=tz)
    with patch("app.data_engine.incremental.datetime") as mock_dt:
        mock_dt.now.return_value = dt_before_cutoff
        session = engine.resolve_latest_completed_session(as_of=None)
        assert session == date(2024, 1, 9)

    # Case 2: Wednesday 10 Jan 2024 at 16:05 IST (after 16:00 cutoff)
    # Session IS completed -> returns 10 Jan 2024
    dt_after_cutoff = datetime(2024, 1, 10, 16, 5, tzinfo=tz)
    with patch("app.data_engine.incremental.datetime") as mock_dt:
        mock_dt.now.return_value = dt_after_cutoff
        session = engine.resolve_latest_completed_session(as_of=None)
        assert session == date(2024, 1, 10)

    # Case 3: Sunday 14 Jan 2024 at 18:00 IST (weekend closure)
    # Weekend is not a trading day -> returns previous completed trading session (Friday 12 Jan 2024)
    dt_weekend = datetime(2024, 1, 14, 18, 0, tzinfo=tz)
    with patch("app.data_engine.incremental.datetime") as mock_dt:
        mock_dt.now.return_value = dt_weekend
        session = engine.resolve_latest_completed_session(as_of=None)
        assert session == date(2024, 1, 12)


def test_resolve_latest_completed_session_explicit_datetime(test_pipeline_env):
    """Test resolve_latest_completed_session with explicit as_of datetime enforces 16:00 cutoff."""
    engine = test_pipeline_env["engine"]
    tz = ZoneInfo("Asia/Kolkata")

    # 1. Trading day at 15:30 IST (before 16:00 cutoff) -> NOT completed -> prior day (9 Jan)
    dt_before = datetime(2024, 1, 10, 15, 30, tzinfo=tz)
    assert engine.resolve_latest_completed_session(as_of=dt_before) == date(2024, 1, 9)

    # 2. Trading day at 16:00 IST (exact cutoff) -> completed -> today (10 Jan)
    dt_exact = datetime(2024, 1, 10, 16, 0, tzinfo=tz)
    assert engine.resolve_latest_completed_session(as_of=dt_exact) == date(2024, 1, 10)

    # 3. Trading day at 16:30 IST (after cutoff) -> completed -> today (10 Jan)
    dt_after = datetime(2024, 1, 10, 16, 30, tzinfo=tz)
    assert engine.resolve_latest_completed_session(as_of=dt_after) == date(2024, 1, 10)

    # 4. Weekend datetime (Sunday 14 Jan 2024 at 18:00 IST) -> prior completed session (Friday 12 Jan)
    dt_weekend = datetime(2024, 1, 14, 18, 0, tzinfo=tz)
    assert engine.resolve_latest_completed_session(as_of=dt_weekend) == date(2024, 1, 12)

    # 5. Deterministic historical date: date(2024, 1, 10) always resolves to date(2024, 1, 10)
    assert engine.resolve_latest_completed_session(as_of=date(2024, 1, 10)) == date(2024, 1, 10)
