"""Quantamental AI - Data Cleansing & Quality Gate Pipeline.

Enforces configurable, asset-specific missing-data thresholds, anomaly detection,
and decision hierarchies (warning, exclusion, quarantine) per Data Strategy Section 4.
Strictly prohibits backward-filling (bfill) and synthetic price fabrication.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from app.core.exceptions import DataQualityException, QuarantineAssetError
from app.core.logger import get_logger

logger = get_logger("quantamental.clean_pipeline")


class QualityStatus(str, Enum):
    """Quality classification for asset time-series.

    Decision Hierarchy & Deterministic Semantics:
    - CLEAN: Coverage meets or exceeds warning_coverage_ratio (>=98% default);
      zero data integrity anomalies. Fully eligible for research and estimation.
    - WARNING: Coverage is between min_history_coverage_ratio and warning_coverage_ratio
      (e.g. 95% to <98%), or minor non-fatal gaps. Eligible for research with logged diagnostics.
    - EXCLUDED: History coverage is strictly below min_history_coverage_ratio (<95% default)
      or series is completely empty. Insufficient history for reliable covariance estimation;
      MUST NOT be persisted or included in portfolio estimation window.
    - QUARANTINED: Data integrity or statistical discontinuity failure (consecutive missing
      sessions > threshold, consecutive zero volume >= threshold, or single-day price jump
      > threshold without corporate action). Existing archive is preserved; updates halted.
    """

    CLEAN = "CLEAN"
    WARNING = "WARNING"
    EXCLUDED = "EXCLUDED"
    QUARANTINED = "QUARANTINED"


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for data integrity checks."""

    warning_coverage_ratio: float = 0.98  # Coverage below this triggers WARNING
    min_history_coverage_ratio: float = 0.95  # Coverage below this triggers EXCLUDED
    max_consecutive_missing_days: int = 3  # Consecutive missing sessions exceeding this triggers QUARANTINED
    max_single_day_return_spike: float = 0.50  # 50% absolute move without corporate action triggers QUARANTINED
    max_consecutive_zero_volume_days: int = 5  # Zero volume >= this triggers QUARANTINED


@dataclass
class CleansingResult:
    """Outcome of cleansing and quality evaluation for a symbol's series."""

    symbol: str
    status: QualityStatus
    cleaned_df: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    quarantine_reason: Optional[str] = None
    missing_trading_days: List[date] = field(default_factory=list)
    total_expected_days: int = 0
    observed_days: int = 0
    coverage_ratio: float = 1.0


class DataCleansingPipeline:
    """Validates historical series against calendar expectations and quality constraints."""

    def __init__(self, default_thresholds: Optional[QualityThresholds] = None) -> None:
        self.default_thresholds = default_thresholds or QualityThresholds()
        self.asset_thresholds: Dict[str, QualityThresholds] = {}

    def set_asset_thresholds(self, symbol: str, thresholds: QualityThresholds) -> None:
        """Configure asset-specific thresholds (e.g. for newer listings or lower-beta assets)."""
        self.asset_thresholds[symbol] = thresholds

    def get_thresholds(self, symbol: str) -> QualityThresholds:
        """Retrieve applicable thresholds for symbol."""
        return self.asset_thresholds.get(symbol, self.default_thresholds)

    def evaluate_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        expected_trading_days: List[date],
        listing_date: Optional[date] = None,
    ) -> CleansingResult:
        """Evaluate a series against calendar trading days and numerical sanity gates.

        Evaluates missing sessions through the requested end date across the complete
        expected trading calendar, while respecting asset listing date (pre-listing
        sessions are never classified as missing).

        Decision Hierarchy:
        1. QUARANTINED: Unexplained gaps (> max_consecutive_missing_days), prolonged zero volume,
           or unexplained price spikes.
        2. EXCLUDED: Empty dataset or coverage < min_history_coverage_ratio.
        3. WARNING: Coverage between min_history_coverage_ratio and warning_coverage_ratio.
        4. CLEAN: Coverage >= warning_coverage_ratio with zero warnings.

        Args:
            symbol: Equity ticker symbol.
            df: Normalized OHLCV DataFrame.
            expected_trading_days: Authoritative expected trading days from NSECalendar.
            listing_date: Optional verified listing date of the asset.

        Returns:
            CleansingResult with status, diagnostic warnings, and cleaned DataFrame.
        """
        thresholds = self.get_thresholds(symbol)
        warnings: List[str] = []

        # Sort expected trading days chronologically
        sorted_expected = sorted(expected_trading_days)

        # Filter out pre-listing sessions if listing_date is known
        if listing_date is not None:
            in_scope_expected = [d for d in sorted_expected if d >= listing_date]
        else:
            in_scope_expected = sorted_expected

        total_expected = len(in_scope_expected)

        if df.empty:
            return CleansingResult(
                symbol=symbol,
                status=QualityStatus.EXCLUDED,
                cleaned_df=df,
                warnings=["Received empty dataset for evaluation"],
                missing_trading_days=in_scope_expected,
                total_expected_days=total_expected,
                observed_days=0,
                coverage_ratio=0.0,
            )

        # Sort chronologically
        clean_df = df.sort_values("trading_date").reset_index(drop=True)

        observed_dates: Set[date] = set(clean_df["trading_date"])
        expected_set: Set[date] = set(in_scope_expected)
        missing_days = sorted(list(expected_set - observed_dates))

        observed_count = len(observed_dates.intersection(expected_set))
        coverage_ratio = observed_count / total_expected if total_expected > 0 else 1.0

        # Check 1: Consecutive missing sessions across complete in-scope calendar
        # This catches missing initial, middle, and trailing sessions up to T_end
        if missing_days and in_scope_expected:
            idx_map = {d: i for i, d in enumerate(in_scope_expected)}
            consecutive = 1
            max_consecutive = 1
            for i in range(1, len(missing_days)):
                prev_idx = idx_map.get(missing_days[i - 1], -1)
                cur_idx = idx_map.get(missing_days[i], -1)
                if cur_idx == prev_idx + 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1

            if max_consecutive > thresholds.max_consecutive_missing_days:
                return CleansingResult(
                    symbol=symbol,
                    status=QualityStatus.QUARANTINED,
                    cleaned_df=clean_df,
                    warnings=warnings,
                    quarantine_reason=(
                        f"Unexplained trading gap: {max_consecutive} consecutive unobserved sessions "
                        f"(threshold: {thresholds.max_consecutive_missing_days})"
                    ),
                    missing_trading_days=missing_days,
                    total_expected_days=total_expected,
                    observed_days=observed_count,
                    coverage_ratio=coverage_ratio,
                )

        # Check 2: Zero-volume sessions
        zero_vol_mask = clean_df["volume"] == 0
        if zero_vol_mask.any():
            zero_vol_runs = (zero_vol_mask != zero_vol_mask.shift()).cumsum()
            max_zero_vol = int(zero_vol_mask.groupby(zero_vol_runs).sum().max())
            if max_zero_vol >= thresholds.max_consecutive_zero_volume_days:
                return CleansingResult(
                    symbol=symbol,
                    status=QualityStatus.QUARANTINED,
                    cleaned_df=clean_df,
                    warnings=warnings,
                    quarantine_reason=(
                        f"Prolonged zero trading volume: {max_zero_vol} consecutive sessions with volume=0 "
                        f"(threshold: {thresholds.max_consecutive_zero_volume_days})"
                    ),
                    missing_trading_days=missing_days,
                    total_expected_days=total_expected,
                    observed_days=observed_count,
                    coverage_ratio=coverage_ratio,
                )
            else:
                zero_count = int(zero_vol_mask.sum())
                warnings.append(
                    f"Detected {zero_count} zero-volume session(s) (max consecutive: {max_zero_vol}, "
                    f"quarantine threshold: {thresholds.max_consecutive_zero_volume_days})"
                )

        # Check 3: Unexplained return spikes
        if len(clean_df) > 1:
            clean_df = clean_df.copy()
            clean_df["daily_ret"] = clean_df["adj_close"].pct_change().abs()
            # Spikes without corporate action
            spike_mask = (clean_df["daily_ret"] > thresholds.max_single_day_return_spike) & (
                clean_df["split_factor"] == 1.0
            ) & (clean_df["dividend_amount"] == 0.0)
            if spike_mask.any():
                bad_row = clean_df.loc[spike_mask].iloc[0]
                bad_ret = float(bad_row["daily_ret"])
                bad_date = bad_row["trading_date"]
                return CleansingResult(
                    symbol=symbol,
                    status=QualityStatus.QUARANTINED,
                    cleaned_df=clean_df.drop(columns=["daily_ret"]),
                    warnings=warnings,
                    quarantine_reason=(
                        f"Unexplained price discontinuity: {bad_ret:.1%} price move on {bad_date} "
                        f"without recorded corporate action"
                    ),
                    missing_trading_days=missing_days,
                    total_expected_days=total_expected,
                    observed_days=observed_count,
                    coverage_ratio=coverage_ratio,
                )

        if "daily_ret" in clean_df.columns:
            clean_df = clean_df.drop(columns=["daily_ret"])

        # Check 4: Deterministic coverage policy boundaries
        if coverage_ratio < thresholds.min_history_coverage_ratio:
            warnings.append(
                f"History coverage ratio {coverage_ratio:.2%} is below exclusion threshold "
                f"{thresholds.min_history_coverage_ratio:.2%} ({len(missing_days)} missing sessions)"
            )
            return CleansingResult(
                symbol=symbol,
                status=QualityStatus.EXCLUDED,
                cleaned_df=clean_df,
                warnings=warnings,
                missing_trading_days=missing_days,
                total_expected_days=total_expected,
                observed_days=observed_count,
                coverage_ratio=coverage_ratio,
            )

        if coverage_ratio < thresholds.warning_coverage_ratio:
            warnings.append(
                f"History coverage ratio {coverage_ratio:.2%} is below warning threshold "
                f"{thresholds.warning_coverage_ratio:.2%} ({len(missing_days)} missing sessions)"
            )
            return CleansingResult(
                symbol=symbol,
                status=QualityStatus.WARNING,
                cleaned_df=clean_df,
                warnings=warnings,
                missing_trading_days=missing_days,
                total_expected_days=total_expected,
                observed_days=observed_count,
                coverage_ratio=coverage_ratio,
            )

        final_status = QualityStatus.WARNING if warnings else QualityStatus.CLEAN

        return CleansingResult(
            symbol=symbol,
            status=final_status,
            cleaned_df=clean_df,
            warnings=warnings,
            missing_trading_days=missing_days,
            total_expected_days=total_expected,
            observed_days=observed_count,
            coverage_ratio=coverage_ratio,
        )


# Singleton instance
cleansing_pipeline = DataCleansingPipeline()
