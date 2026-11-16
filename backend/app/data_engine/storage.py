"""Quantamental AI - Parquet Time-Series Storage Layer.

Provides high-performance columnar persistence for normalized OHLCV time-series.
Implements path-traversal protection, collision-free symbol encoding, atomic writes,
composite-key deduplication, and schema validation gates before persistence.
"""

from datetime import date, datetime
import os
from pathlib import Path
import re
import tempfile
from typing import List, Optional, Tuple
from urllib.parse import quote, unquote

import pandas as pd

from app.config import settings
from app.core.exceptions import DataProviderException, ProviderValidationError
from app.core.logger import get_logger
from app.data_engine.base import BaseDataProvider

logger = get_logger("quantamental.storage")

# Valid NSE/benchmark symbol pattern: alphanumeric, dot, caret, hyphen, underscore
SYMBOL_REGEX = re.compile(r"^[A-Za-z0-9\.\^\-_]+$")


class ParquetStorageEngine:
    """Manages persistent Parquet time-series storage for normalized equity data."""

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        """Initialize storage engine with target storage directory."""
        self.storage_dir = Path(storage_dir or settings.PARQUET_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_symbol(symbol: str) -> None:
        """Validate symbol to prevent path traversal and malformed identifiers."""
        if not symbol or not isinstance(symbol, str):
            raise ProviderValidationError(
                "Symbol must be a non-empty string",
                details={"received": symbol, "type": type(symbol).__name__},
            )

        # Check for path traversal sequences and null bytes
        if ".." in symbol or "/" in symbol or "\\" in symbol or "\0" in symbol:
            raise ProviderValidationError(
                f"Path traversal or illegal path characters detected in symbol: {symbol}",
                details={"symbol": symbol},
            )

        # Validate against allowed character set
        if not SYMBOL_REGEX.match(symbol):
            raise ProviderValidationError(
                f"Symbol '{symbol}' contains invalid characters (allowed: alphanumeric, '.', '^', '-', '_')",
                details={"symbol": symbol},
            )

    def _get_file_path(self, symbol: str) -> Path:
        """Return the Parquet file path for a given symbol with collision-free encoding."""
        self.validate_symbol(symbol)
        # quote encodes special characters (e.g. ^NSEI -> %5ENSEI) collision-free
        safe_name = quote(symbol, safe=".-_")
        target_path = (self.storage_dir / f"{safe_name}.parquet").resolve()
        storage_root = self.storage_dir.resolve()

        # Enforce that the target file path resides strictly inside storage_dir
        if storage_root not in target_path.parents and target_path != storage_root:
            raise ProviderValidationError(
                f"Path traversal attempt blocked for symbol: {symbol}",
                details={"symbol": symbol, "resolved_path": str(target_path)},
            )
        return target_path

    def file_exists(self, symbol: str) -> bool:
        """Check if persistent Parquet file exists for symbol."""
        return self._get_file_path(symbol).is_file()

    def _validate_dataframe_before_write(self, symbol: str, df: pd.DataFrame) -> None:
        """Enforce strict data hygiene, schema compliance, and symbol matching before write."""
        if not isinstance(df, pd.DataFrame):
            raise ProviderValidationError(
                "Object to persist must be a pandas DataFrame",
                details={"received_type": type(df).__name__},
            )

        if df.empty:
            return

        # 1. Verify all required columns are present
        missing_cols = set(BaseDataProvider.REQUIRED_NORMALIZED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ProviderValidationError(
                f"DataFrame missing required normalized columns: {sorted(missing_cols)}",
                details={"missing_columns": sorted(list(missing_cols))},
            )

        # 2. Verify all rows match the requested symbol (no mixed-symbol DataFrames)
        symbol_series = df["symbol"].astype(str)
        if (symbol_series != symbol).any():
            mismatched = df.loc[symbol_series != symbol, "symbol"].unique().tolist()
            raise ProviderValidationError(
                f"Symbol mismatch in DataFrame: expected '{symbol}', found rows with {mismatched}",
                details={"expected_symbol": symbol, "found_symbols": mismatched},
            )

        # 3. Verify trading_date contains valid date objects (strictly datetime.date, rejecting datetime)
        invalid_dates = df["trading_date"].apply(lambda v: not isinstance(v, date) or isinstance(v, datetime))
        if invalid_dates.any():
            bad_types = df.loc[invalid_dates, "trading_date"].apply(lambda v: type(v).__name__).unique().tolist()
            raise ProviderValidationError(
                "Column 'trading_date' contains non-date objects (strictly datetime.date required)",
                details={"unexpected_types": bad_types},
            )

        # 4. Verify no duplicate (symbol, trading_date) composite keys
        dup_mask = df.duplicated(subset=["symbol", "trading_date"], keep=False)
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            raise ProviderValidationError(
                f"DataFrame contains {dup_count} rows with duplicate composite keys (symbol, trading_date)",
                details={"duplicate_row_count": dup_count},
            )

    def load_symbol_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load historical normalized data for a symbol with optional date filtering.

        Args:
            symbol: Equity ticker symbol (e.g. 'RELIANCE.NS').
            start_date: Optional inclusive start date.
            end_date: Optional inclusive end date.

        Returns:
            pd.DataFrame sorted chronologically by trading_date.
            Empty DataFrame if no data exists.
        """
        file_path = self._get_file_path(symbol)
        if not file_path.is_file():
            return pd.DataFrame(columns=BaseDataProvider.REQUIRED_NORMALIZED_COLUMNS)

        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            raise DataProviderException(
                f"Failed to read Parquet file for {symbol} at {file_path}",
                details={"symbol": symbol, "path": str(file_path), "error": str(exc)},
            ) from exc

        if df.empty:
            return df

        # Ensure trading_date is datetime.date
        if not isinstance(df["trading_date"].iloc[0], date):
            df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.date

        # Filter by start_date and end_date if requested
        if start_date is not None:
            df = df[df["trading_date"] >= start_date]
        if end_date is not None:
            df = df[df["trading_date"] <= end_date]

        return df.sort_values("trading_date").reset_index(drop=True)

    def get_latest_stored_date(self, symbol: str) -> Optional[date]:
        """Return the latest available trading date in storage for a symbol."""
        df = self.load_symbol_data(symbol)
        if df.empty:
            return None
        return df["trading_date"].max()

    def get_earliest_stored_date(self, symbol: str) -> Optional[date]:
        """Return the earliest available trading date in storage for a symbol."""
        df = self.load_symbol_data(symbol)
        if df.empty:
            return None
        return df["trading_date"].min()

    def save_symbol_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Atomically overwrite or initialize the storage file for a symbol.

        Validates data schema, symbol matching, and composite keys before writing.
        Invalid inputs fail immediately, preserving existing archives untouched.

        Args:
            symbol: Equity ticker symbol.
            df: Validated DataFrame conforming to normalized schema.
        """
        self.validate_symbol(symbol)
        self._validate_dataframe_before_write(symbol, df)

        if df.empty:
            logger.warning("Attempted to save empty DataFrame for %s; skipping.", symbol)
            return

        file_path = self._get_file_path(symbol)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        clean_df = df.sort_values("trading_date").drop_duplicates(
            subset=["symbol", "trading_date"], keep="last"
        ).reset_index(drop=True)

        # Atomic write: write to temp file on same filesystem, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f"{file_path.stem}_", suffix=".tmp", dir=str(file_path.parent)
        )
        os.close(temp_fd)

        try:
            clean_df.to_parquet(temp_path, engine="pyarrow", index=False)
            os.replace(temp_path, file_path)
            logger.info("Successfully saved %d records for %s to %s", len(clean_df), symbol, file_path)
        except Exception as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise DataProviderException(
                f"Failed to atomically write Parquet file for {symbol}",
                details={"symbol": symbol, "target_path": str(file_path), "error": str(exc)},
            ) from exc

    def upsert_symbol_data(
        self, symbol: str, new_df: pd.DataFrame
    ) -> Tuple[int, int]:
        """Incrementally upsert new observations into existing storage.

        Validates all incoming data before mutation.
        Calculates added and updated counts using the composite key (symbol, trading_date).

        Args:
            symbol: Equity ticker symbol.
            new_df: New observations DataFrame.

        Returns:
            Tuple of (records_added, records_updated).
        """
        self.validate_symbol(symbol)
        self._validate_dataframe_before_write(symbol, new_df)

        if new_df.empty:
            return 0, 0

        # Load existing data
        existing_df = self.load_symbol_data(symbol)

        if existing_df.empty:
            self.save_symbol_data(symbol, new_df)
            return len(new_df), 0

        # Calculate added vs updated counts using composite key (symbol, trading_date)
        existing_keys = set(zip(existing_df["symbol"], existing_df["trading_date"]))
        new_keys = set(zip(new_df["symbol"], new_df["trading_date"]))

        records_updated = len(existing_keys.intersection(new_keys))
        records_added = len(new_keys - existing_keys)

        # Combine, with new rows taking precedence on matching (symbol, trading_date)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df = combined_df.drop_duplicates(
            subset=["symbol", "trading_date"], keep="last"
        ).sort_values("trading_date").reset_index(drop=True)

        self.save_symbol_data(symbol, merged_df)
        return records_added, records_updated

    def list_stored_symbols(self) -> List[str]:
        """List all symbols currently present in storage using collision-free decoding."""
        if not self.storage_dir.is_dir():
            return []
        symbols = []
        for p in self.storage_dir.glob("*.parquet"):
            sym = unquote(p.stem)
            symbols.append(sym)
        return sorted(symbols)


# Singleton instance
storage_engine = ParquetStorageEngine()
