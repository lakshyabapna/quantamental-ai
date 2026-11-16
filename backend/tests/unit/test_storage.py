"""Unit tests for the Parquet Time-Series Storage Engine."""

from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from app.core.exceptions import ProviderValidationError
from app.data_engine.base import AdjustmentMode, BaseDataProvider
from app.data_engine.storage import ParquetStorageEngine


def _sample_df(symbol: str, dates_list: list) -> pd.DataFrame:
    """Helper to create a valid normalized DataFrame for storage tests."""
    records = []
    for d in dates_list:
        records.append({
            "symbol": symbol,
            "trading_date": d,
            "open_nominal": 2000.0,
            "high_nominal": 2050.0,
            "low_nominal": 1980.0,
            "close_nominal": 2020.0,
            "adj_close": 2020.0,
            "volume": 500000,
            "split_factor": 1.0,
            "dividend_amount": 0.0,
            "is_stale": False,
            "data_provider": "yfinance",
            "adjustment_mode": AdjustmentMode.AUTO_ADJUST_FALSE.value,
            "retrieval_timestamp": datetime.now(timezone.utc),
        })
    return pd.DataFrame(records)


@pytest.fixture
def temp_storage():
    """Provide a ParquetStorageEngine rooted in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield ParquetStorageEngine(storage_dir=tmpdir)


def test_save_and_load_symbol_data(temp_storage: ParquetStorageEngine):
    """Test saving and loading symbol time-series."""
    dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)]
    df = _sample_df("TCS.NS", dates)

    temp_storage.save_symbol_data("TCS.NS", df)
    assert temp_storage.file_exists("TCS.NS") is True

    loaded = temp_storage.load_symbol_data("TCS.NS")
    assert len(loaded) == 3
    assert loaded["trading_date"].tolist() == dates


def test_load_symbol_data_date_filters(temp_storage: ParquetStorageEngine):
    """Test date filtering on load."""
    dates = [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 15)]
    df = _sample_df("INFY.NS", dates)
    temp_storage.save_symbol_data("INFY.NS", df)

    filtered = temp_storage.load_symbol_data(
        "INFY.NS",
        start_date=date(2024, 1, 11),
        end_date=date(2024, 1, 12),
    )
    assert len(filtered) == 2
    assert filtered["trading_date"].tolist() == [date(2024, 1, 11), date(2024, 1, 12)]


def test_get_earliest_and_latest_stored_date(temp_storage: ParquetStorageEngine):
    """Test retrieving earliest and latest dates."""
    dates = [date(2024, 1, 10), date(2024, 1, 15), date(2024, 1, 20)]
    df = _sample_df("HDFCBANK.NS", dates)
    temp_storage.save_symbol_data("HDFCBANK.NS", df)

    assert temp_storage.get_earliest_stored_date("HDFCBANK.NS") == date(2024, 1, 10)
    assert temp_storage.get_latest_stored_date("HDFCBANK.NS") == date(2024, 1, 20)


def test_upsert_symbol_data(temp_storage: ParquetStorageEngine):
    """Test incremental upsert logic (merging new dates and updating existing dates)."""
    # Initial batch: 10th and 11th
    initial_dates = [date(2024, 1, 10), date(2024, 1, 11)]
    df_init = _sample_df("ICICIBANK.NS", initial_dates)
    added, updated = temp_storage.upsert_symbol_data("ICICIBANK.NS", df_init)
    assert added == 2
    assert updated == 0

    # Second batch: 11th (overlap/update) and 12th (new)
    second_dates = [date(2024, 1, 11), date(2024, 1, 12)]
    df_second = _sample_df("ICICIBANK.NS", second_dates)
    # Modify adj_close on 11th to verify update
    df_second.loc[df_second["trading_date"] == date(2024, 1, 11), "adj_close"] = 999.0

    added, updated = temp_storage.upsert_symbol_data("ICICIBANK.NS", df_second)
    assert added == 1
    assert updated == 1

    loaded = temp_storage.load_symbol_data("ICICIBANK.NS")
    assert len(loaded) == 3
    # Check that update took effect
    row_11 = loaded[loaded["trading_date"] == date(2024, 1, 11)]
    assert row_11["adj_close"].iloc[0] == 999.0


def test_list_stored_symbols(temp_storage: ParquetStorageEngine):
    """Test listing symbols in storage."""
    temp_storage.save_symbol_data("RELIANCE.NS", _sample_df("RELIANCE.NS", [date(2024, 1, 10)]))
    temp_storage.save_symbol_data("^NSEI", _sample_df("^NSEI", [date(2024, 1, 10)]))

    symbols = temp_storage.list_stored_symbols()
    assert "RELIANCE.NS" in symbols
    assert "^NSEI" in symbols


# ===================================================================
# Path Traversal and Symbol Security
# ===================================================================

def test_symbol_validation_path_traversal(temp_storage: ParquetStorageEngine):
    """Test that path traversal attempts in symbols are strictly blocked."""
    invalid_symbols = [
        "../../etc/passwd",
        "../secret",
        "nested/sub/dir",
        "C:\\Windows\\System32",
        "TCS\0NS",
        "",
        " ",
    ]
    for sym in invalid_symbols:
        with pytest.raises(ProviderValidationError):
            temp_storage.validate_symbol(sym)
        with pytest.raises(ProviderValidationError):
            temp_storage._get_file_path(sym)


def test_symbol_validation_invalid_chars(temp_storage: ParquetStorageEngine):
    """Test that disallowed special characters raise ProviderValidationError."""
    disallowed = ["TCS;DROP", "RELIANCE$NS", "INFY|NSE", "WIPRO NS", "HDFC<NS>"]
    for sym in disallowed:
        with pytest.raises(ProviderValidationError):
            temp_storage.validate_symbol(sym)


def test_symbol_collision_prevention(temp_storage: ParquetStorageEngine):
    """Test that special characters like '^' in ^NSEI do not collide with _NSEI."""
    d = [date(2024, 1, 10)]
    df_caret = _sample_df("^NSEI", d)
    df_under = _sample_df("_NSEI", d)

    path_caret = temp_storage._get_file_path("^NSEI")
    path_under = temp_storage._get_file_path("_NSEI")

    # Distinct filenames
    assert path_caret.name == "%5ENSEI.parquet"
    assert path_under.name == "_NSEI.parquet"
    assert path_caret != path_under

    temp_storage.save_symbol_data("^NSEI", df_caret)
    temp_storage.save_symbol_data("_NSEI", df_under)

    assert temp_storage.file_exists("^NSEI") is True
    assert temp_storage.file_exists("_NSEI") is True

    loaded_caret = temp_storage.load_symbol_data("^NSEI")
    loaded_under = temp_storage.load_symbol_data("_NSEI")
    assert loaded_caret["symbol"].iloc[0] == "^NSEI"
    assert loaded_under["symbol"].iloc[0] == "_NSEI"

    symbols = temp_storage.list_stored_symbols()
    assert "^NSEI" in symbols
    assert "_NSEI" in symbols


# ===================================================================
# Pre-Write Schema and Data Hygiene Validation Gates
# ===================================================================

def test_write_validation_mixed_symbols(temp_storage: ParquetStorageEngine):
    """Test that attempting to write a DataFrame with rows for another symbol fails."""
    d = [date(2024, 1, 10), date(2024, 1, 11)]
    df = _sample_df("TCS.NS", d)
    # Corrupt row 1 symbol
    df.loc[1, "symbol"] = "INFY.NS"

    with pytest.raises(ProviderValidationError, match="Symbol mismatch"):
        temp_storage.save_symbol_data("TCS.NS", df)

    with pytest.raises(ProviderValidationError, match="Symbol mismatch"):
        temp_storage.upsert_symbol_data("TCS.NS", df)


def test_write_validation_missing_columns(temp_storage: ParquetStorageEngine):
    """Test that missing required columns raises ProviderValidationError."""
    d = [date(2024, 1, 10)]
    df = _sample_df("TCS.NS", d)
    df = df.drop(columns=["adj_close", "volume"])

    with pytest.raises(ProviderValidationError, match="missing required normalized columns"):
        temp_storage.save_symbol_data("TCS.NS", df)

    with pytest.raises(ProviderValidationError, match="missing required normalized columns"):
        temp_storage.upsert_symbol_data("TCS.NS", df)


def test_write_validation_invalid_date_type(temp_storage: ParquetStorageEngine):
    """Test that non-date trading_date (e.g. str) raises ProviderValidationError."""
    d = [date(2024, 1, 10)]
    df = _sample_df("TCS.NS", d)
    df["trading_date"] = "2024-01-10"  # str instead of datetime.date

    with pytest.raises(ProviderValidationError, match="non-date objects"):
        temp_storage.save_symbol_data("TCS.NS", df)


def test_write_validation_rejects_datetime_objects(temp_storage: ParquetStorageEngine):
    """Test that datetime.datetime values in trading_date are strictly rejected (only date allowed)."""
    d = [date(2024, 1, 10)]
    df = _sample_df("TCS.NS", d)
    df["trading_date"] = [datetime(2024, 1, 10, 12, 0, 0)]  # datetime instead of date

    with pytest.raises(ProviderValidationError, match="strictly datetime.date required"):
        temp_storage.save_symbol_data("TCS.NS", df)

    with pytest.raises(ProviderValidationError, match="strictly datetime.date required"):
        temp_storage.upsert_symbol_data("TCS.NS", df)


def test_write_validation_duplicate_composite_keys(temp_storage: ParquetStorageEngine):
    """Test that duplicate (symbol, trading_date) in input raises ProviderValidationError."""
    d = [date(2024, 1, 10), date(2024, 1, 10)]  # duplicate date
    df = _sample_df("TCS.NS", d)

    with pytest.raises(ProviderValidationError, match="duplicate composite keys"):
        temp_storage.save_symbol_data("TCS.NS", df)

    with pytest.raises(ProviderValidationError, match="duplicate composite keys"):
        temp_storage.upsert_symbol_data("TCS.NS", df)


# ===================================================================
# Atomic Writes and Archive Preservation
# ===================================================================

def test_atomic_write_preserves_archive_on_failure(temp_storage: ParquetStorageEngine):
    """Test that an invalid write attempt leaves existing archive on disk completely intact."""
    initial_dates = [date(2024, 1, 10), date(2024, 1, 11)]
    df_valid = _sample_df("TCS.NS", initial_dates)
    temp_storage.save_symbol_data("TCS.NS", df_valid)

    # Attempt to upsert invalid data (duplicate keys)
    bad_df = _sample_df("TCS.NS", [date(2024, 1, 12), date(2024, 1, 12)])
    with pytest.raises(ProviderValidationError):
        temp_storage.upsert_symbol_data("TCS.NS", bad_df)

    # Verify existing archive is intact
    preserved = temp_storage.load_symbol_data("TCS.NS")
    assert len(preserved) == 2
    assert preserved["trading_date"].tolist() == initial_dates


def test_invalid_datetime_write_preserves_archive(temp_storage: ParquetStorageEngine):
    """Test that attempting to save or upsert datetime values leaves existing archive completely intact."""
    initial_dates = [date(2024, 1, 10), date(2024, 1, 11)]
    df_valid = _sample_df("INFY.NS", initial_dates)
    temp_storage.save_symbol_data("INFY.NS", df_valid)

    # Attempt to overwrite with datetime objects in trading_date
    bad_df = _sample_df("INFY.NS", [date(2024, 1, 12)])
    bad_df["trading_date"] = [datetime(2024, 1, 12, 10, 0, 0)]

    with pytest.raises(ProviderValidationError):
        temp_storage.save_symbol_data("INFY.NS", bad_df)

    with pytest.raises(ProviderValidationError):
        temp_storage.upsert_symbol_data("INFY.NS", bad_df)

    preserved = temp_storage.load_symbol_data("INFY.NS")
    assert len(preserved) == 2
    assert preserved["trading_date"].tolist() == initial_dates
