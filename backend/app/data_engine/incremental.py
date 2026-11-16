"""Quantamental AI - Incremental Historical Data Lifecycle Engine.

Coordinates persistent historical storage, incremental fetches with configurable
overlap reconciliation, quality cleansing gates, and dynamic rolling 5-year slicing.
Conforms to Data Strategy Sections 2, 3, and 4.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Union
import zoneinfo
import pandas as pd
from pydantic import BaseModel, Field

from app.config import settings
from app.core.exceptions import CalendarDateRangeError, DataQualityException, QuarantineAssetError
from app.core.logger import get_logger
from app.data_engine.base import AdjustmentMode, BaseDataProvider
from app.data_engine.calendar import NSECalendar, nse_calendar
from app.data_engine.clean_pipeline import (
    DataCleansingPipeline,
    QualityStatus,
    cleansing_pipeline,
)
from app.data_engine.storage import ParquetStorageEngine, storage_engine
from app.data_engine.yfinance_provider import YFinanceDataProvider

logger = get_logger("quantamental.incremental")

KOLKATA_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")
# NSE Equity session completes at 15:30 IST; official closing settlement/Bhavcopy published by 16:00 IST
MARKET_CLOSE_TIME = time(16, 0, 0)


class RefreshAuditLog(BaseModel):
    """Audit log schema recording data refresh execution details."""

    symbol: str = Field(..., description="Equity ticker symbol")
    provider_name: str = Field(..., description="Data provider identifier")
    refresh_start: datetime = Field(..., description="Timestamp when refresh started")
    refresh_end: datetime = Field(..., description="Timestamp when refresh completed")
    latest_stored_date: Optional[date] = Field(default=None, description="Latest session stored in Parquet")
    records_fetched: int = Field(default=0, description="Count of raw records fetched from provider")
    records_added: int = Field(default=0, description="Count of new records appended")
    records_updated: int = Field(default=0, description="Count of existing records updated during overlap")
    corporate_actions_detected: int = Field(default=0, description="Count of corporate actions identified")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic data-quality warnings")
    status: str = Field(..., description="Refresh status: SUCCESS, PARTIAL, FAILED, EXCLUDED, QUARANTINED")


class IncrementalHistoricalEngine:
    """Orchestrates persistent retention, incremental refresh, and dynamic slicing."""

    def __init__(
        self,
        provider: Optional[BaseDataProvider] = None,
        calendar: Optional[NSECalendar] = None,
        storage: Optional[ParquetStorageEngine] = None,
        cleanser: Optional[DataCleansingPipeline] = None,
    ) -> None:
        self.provider = provider or YFinanceDataProvider()
        self.calendar = calendar or nse_calendar
        self.storage = storage or storage_engine
        self.cleanser = cleanser or cleansing_pipeline

    def resolve_latest_completed_session(self, as_of: Optional[Union[date, datetime]] = None) -> date:
        """Resolve the latest available completed NSE trading session date.

        Explicit Cutoff vs Real-Time Execution:
        - If as_of is explicitly provided as a datetime:
          Evaluates cutoff in Asia/Kolkata timezone. A datetime before 16:00 IST on an active
          trading day is NOT yet completed, and resolves to the previous completed trading day.
          A datetime at or after 16:00 IST on an active trading day resolves to that day.
        - If as_of is explicitly provided as a date (deterministic historical execution):
          Treated as an already-settled session date. If as_of is an active trading day, returns as_of.
          If not (e.g. weekend/holiday), returns the nearest previous completed trading session.
        - If as_of is None (real-time execution):
          Evaluates current time in Asia/Kolkata against 16:00 IST cutoff.
        """
        if as_of is not None:
            if isinstance(as_of, datetime):
                if as_of.tzinfo is None:
                    dt_kolkata = as_of.replace(tzinfo=KOLKATA_TZ)
                else:
                    dt_kolkata = as_of.astimezone(KOLKATA_TZ)

                target_date = dt_kolkata.date()
                if self.calendar.is_trading_day(target_date) and dt_kolkata.time() >= MARKET_CLOSE_TIME:
                    return target_date
                return self.calendar.get_previous_trading_day(target_date)

            elif isinstance(as_of, date):
                if self.calendar.is_trading_day(as_of):
                    return as_of
                return self.calendar.get_previous_trading_day(as_of)

            else:
                raise CalendarDateRangeError(
                    f"as_of must be a date, datetime, or None (received {type(as_of).__name__})",
                    details={"received_type": type(as_of).__name__},
                )

        # Real-time execution
        now_kolkata = datetime.now(KOLKATA_TZ)
        today = now_kolkata.date()
        if self.calendar.is_trading_day(today) and now_kolkata.time() >= MARKET_CLOSE_TIME:
            return today
        return self.calendar.get_previous_trading_day(today)

    def refresh_symbol(
        self,
        symbol: str,
        overlap_days: int = 5,
        start_date: Optional[date] = None,
        as_of: Optional[Union[date, datetime]] = None,
        adjustment_mode: AdjustmentMode = AdjustmentMode.AUTO_ADJUST_FALSE,
        listing_date: Optional[date] = None,
    ) -> RefreshAuditLog:
        """Execute incremental historical refresh for an equity symbol.

        Steps:
        1. Validate overlap_days is non-negative.
        2. Resolve latest completed trading session T_end.
        3. Determine fetch start date:
           - If start_date explicitly supplied: use start_date.
           - If no stored data exists: 5-year rolling start date from T_end.
           - If stored data exists:
             * If overlap_days == 0: next trading day after latest_stored.
             * If overlap_days > 0: overlap_days sessions prior to latest_stored.
        4. Fetch incremental batch from provider.
        5. Evaluate series through cleansing pipeline gates.
        6. If QUARANTINED or EXCLUDED: preserve existing storage, abort update.
        7. If clean/warning: atomically upsert into storage.
        8. Return immutable RefreshAuditLog.
        """
        if overlap_days < 0:
            raise CalendarDateRangeError(
                f"overlap_days must be non-negative (received {overlap_days})",
                details={"overlap_days": overlap_days},
            )

        start_ts = datetime.now(timezone.utc)
        warnings: List[str] = []

        t_end = self.resolve_latest_completed_session(as_of)
        latest_stored = self.storage.get_latest_stored_date(symbol)

        if start_date is None:
            if latest_stored is None:
                # First ingestion: fetch full 5-year rolling window
                start_date = self.calendar.get_rolling_window_start(
                    t_end, years=settings.DEFAULT_ROLLING_WINDOW_YEARS
                )
                logger.info("Initializing historical archive for %s from %s to %s", symbol, start_date, t_end)
            else:
                if overlap_days == 0:
                    # Zero overlap: begin at next trading day after latest_stored
                    start_date = self.calendar.get_next_trading_day(latest_stored)
                else:
                    cur = latest_stored
                    for _ in range(overlap_days):
                        cur = self.calendar.get_previous_trading_day(cur)
                    start_date = cur

                logger.info(
                    "Incremental refresh for %s from %s to %s (overlap: %d days)",
                    symbol,
                    start_date,
                    t_end,
                    overlap_days,
                )

        if start_date > t_end:
            return RefreshAuditLog(
                symbol=symbol,
                provider_name=self.provider.get_provider_metadata().provider_name,
                refresh_start=start_ts,
                refresh_end=datetime.now(timezone.utc),
                latest_stored_date=latest_stored,
                records_fetched=0,
                records_added=0,
                records_updated=0,
                corporate_actions_detected=0,
                warnings=["Symbol already synchronized through target date."],
                status="SUCCESS",
            )

        try:
            # 1. Fetch incremental OHLCV
            df_fetched = self.provider.fetch_historical_ohlcv(
                symbol=symbol,
                start_date=start_date,
                end_date=t_end,
                adjustment_mode=adjustment_mode,
            )

            # 2. Fetch corporate actions
            corp_actions = self.provider.fetch_corporate_actions(
                symbol=symbol,
                start_date=start_date,
                end_date=t_end,
            )

            # 3. Cleanse & Quality Gate Evaluation
            expected_days = self.calendar.get_trading_days(start_date, t_end)
            clean_res = self.cleanser.evaluate_series(
                symbol=symbol,
                df=df_fetched,
                expected_trading_days=expected_days,
                listing_date=listing_date,
            )

            warnings.extend(clean_res.warnings)

            # 4. Handle EXCLUDED and QUARANTINED batches
            # An EXCLUDED or QUARANTINED batch must NEVER be persisted to storage.
            # Existing archive is preserved intact.
            if clean_res.status == QualityStatus.QUARANTINED:
                logger.error(
                    "Quarantine triggered for %s: %s. Preserving existing storage.",
                    symbol,
                    clean_res.quarantine_reason,
                )
                warnings.append(f"QUARANTINE: {clean_res.quarantine_reason}")
                return RefreshAuditLog(
                    symbol=symbol,
                    provider_name=self.provider.get_provider_metadata().provider_name,
                    refresh_start=start_ts,
                    refresh_end=datetime.now(timezone.utc),
                    latest_stored_date=latest_stored,
                    records_fetched=len(df_fetched),
                    records_added=0,
                    records_updated=0,
                    corporate_actions_detected=len(corp_actions),
                    warnings=warnings,
                    status="QUARANTINED",
                )

            if clean_res.status == QualityStatus.EXCLUDED:
                logger.warning(
                    "Batch excluded for %s due to insufficient coverage. Preserving existing storage.",
                    symbol,
                )
                warnings.append("EXCLUDED: Batch coverage does not meet minimum threshold.")
                return RefreshAuditLog(
                    symbol=symbol,
                    provider_name=self.provider.get_provider_metadata().provider_name,
                    refresh_start=start_ts,
                    refresh_end=datetime.now(timezone.utc),
                    latest_stored_date=latest_stored,
                    records_fetched=len(df_fetched),
                    records_added=0,
                    records_updated=0,
                    corporate_actions_detected=len(corp_actions),
                    warnings=warnings,
                    status="EXCLUDED",
                )

            # 5. Upsert into Parquet storage for CLEAN or WARNING batches
            records_added, records_updated = self.storage.upsert_symbol_data(
                symbol=symbol,
                new_df=clean_res.cleaned_df,
            )

            new_latest = self.storage.get_latest_stored_date(symbol)
            status = "SUCCESS" if not warnings else "PARTIAL"

            return RefreshAuditLog(
                symbol=symbol,
                provider_name=self.provider.get_provider_metadata().provider_name,
                refresh_start=start_ts,
                refresh_end=datetime.now(timezone.utc),
                latest_stored_date=new_latest,
                records_fetched=len(df_fetched),
                records_added=records_added,
                records_updated=records_updated,
                corporate_actions_detected=len(corp_actions),
                warnings=warnings,
                status=status,
            )

        except Exception as exc:
            logger.error("Failed refresh for %s: %s", symbol, str(exc), exc_info=True)
            return RefreshAuditLog(
                symbol=symbol,
                provider_name=self.provider.get_provider_metadata().provider_name,
                refresh_start=start_ts,
                refresh_end=datetime.now(timezone.utc),
                latest_stored_date=latest_stored,
                records_fetched=0,
                records_added=0,
                records_updated=0,
                corporate_actions_detected=0,
                warnings=[str(exc)],
                status="FAILED",
            )

    def refresh_universe(
        self,
        symbols: List[str],
        overlap_days: int = 5,
        as_of: Optional[Union[date, datetime]] = None,
        adjustment_mode: AdjustmentMode = AdjustmentMode.AUTO_ADJUST_FALSE,
    ) -> Dict[str, RefreshAuditLog]:
        """Execute incremental refresh across a universe of equity symbols."""
        results = {}
        for sym in symbols:
            results[sym] = self.refresh_symbol(
                symbol=sym,
                overlap_days=overlap_days,
                as_of=as_of,
                adjustment_mode=adjustment_mode,
            )
        return results

    def get_active_research_window(
        self,
        symbol: str,
        as_of: Optional[Union[date, datetime]] = None,
        years: int = 5,
    ) -> pd.DataFrame:
        """Extract the dynamic rolling 5-year active research dataset from storage.

        Resolves:
        - T_end: Latest completed NSE trading session.
        - T_start: T_end - years calendar years, adjusted forward to active session.
        """
        t_end = self.resolve_latest_completed_session(as_of)
        t_start = self.calendar.get_rolling_window_start(t_end, years=years)
        return self.storage.load_symbol_data(symbol, start_date=t_start, end_date=t_end)


# Singleton instance
incremental_engine = IncrementalHistoricalEngine()
