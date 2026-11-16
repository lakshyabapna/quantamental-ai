"""Unit tests for the Data Cleansing and Quality Gate Pipeline.

Covers:
- Deterministic decision boundaries: CLEAN, WARNING, EXCLUDED, QUARANTINED.
- Missing final 1, 2, and 4 expected sessions.
- Pre-listing date awareness: sessions before listing_date are never classified as missing.
- Missing sessions in the middle.
- Completely empty response handling.
- Zero-volume and price-spike quarantine triggers.
"""

from datetime import date, timedelta
import pandas as pd
import pytest

from app.data_engine.clean_pipeline import (
    DataCleansingPipeline,
    QualityStatus,
    QualityThresholds,
)


def _make_series(dates: list, adj_close: list, volume: list, split: float = 1.0, div: float = 0.0) -> pd.DataFrame:
    """Helper to build test series DataFrame."""
    return pd.DataFrame({
        "symbol": ["TEST.NS"] * len(dates),
        "trading_date": dates,
        "adj_close": adj_close,
        "volume": volume,
        "split_factor": [split] * len(dates),
        "dividend_amount": [div] * len(dates),
    })


@pytest.fixture
def pipeline():
    return DataCleansingPipeline(
        default_thresholds=QualityThresholds(
            warning_coverage_ratio=0.98,
            min_history_coverage_ratio=0.95,
            max_consecutive_missing_days=3,
            max_single_day_return_spike=0.50,
            max_consecutive_zero_volume_days=5,
        )
    )


# ===================================================================
# 1. Deterministic Decision Boundaries
# ===================================================================

def test_boundary_clean_status(pipeline: DataCleansingPipeline):
    """CLEAN: 100% coverage (>= warning_coverage_ratio 98%) with zero anomalies."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    df = _make_series(days, [100.0] * 100, [500000] * 100)

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=days)
    assert res.status == QualityStatus.CLEAN
    assert len(res.warnings) == 0
    assert res.coverage_ratio == 1.0


def test_boundary_warning_status(pipeline: DataCleansingPipeline):
    """WARNING: Coverage between min_history_coverage_ratio (95%) and warning_coverage_ratio (98%)."""
    # 100 expected days, 96 observed (96% -> between 95% and 98%)
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    # Drop isolated non-consecutive days: indices 10, 20, 30, 40 (isolated, so consecutive gap = 1 <= 3)
    drop_indices = {10, 20, 30, 40}
    observed = [d for idx, d in enumerate(expected) if idx not in drop_indices]
    df = _make_series(observed, [100.0] * 96, [500000] * 96)

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    assert res.status == QualityStatus.WARNING
    assert any("warning threshold" in w for w in res.warnings)
    assert res.coverage_ratio == 0.96


def test_boundary_excluded_status(pipeline: DataCleansingPipeline):
    """EXCLUDED: Coverage strictly below min_history_coverage_ratio (< 95%)."""
    # 100 expected days, 90 observed (90% < 95%), isolated missing days
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    drop_indices = {i for i in range(2, 22, 2)}  # 10 isolated days
    observed = [d for idx, d in enumerate(expected) if idx not in drop_indices]
    df = _make_series(observed, [100.0] * len(observed), [500000] * len(observed))

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    assert res.status == QualityStatus.EXCLUDED
    assert any("exclusion threshold" in w for w in res.warnings)
    assert res.coverage_ratio == 0.90


def test_boundary_empty_dataset_is_excluded(pipeline: DataCleansingPipeline):
    """Completely empty response returns EXCLUDED."""
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    empty_df = pd.DataFrame()

    res = pipeline.evaluate_series("TEST.NS", empty_df, expected_trading_days=expected)
    assert res.status == QualityStatus.EXCLUDED
    assert res.observed_days == 0
    assert res.coverage_ratio == 0.0
    assert len(res.missing_trading_days) == 10


# ===================================================================
# 2. Missing Trailing Sessions (Final 1, 2, 4 days)
# ===================================================================

def test_missing_final_1_session(pipeline: DataCleansingPipeline):
    """Missing final 1 session out of 50 (49/50 = 98% -> CLEAN or WARNING border)."""
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    # All observed except the last day
    observed = expected[:-1]
    df = _make_series(observed, [100.0] * len(observed), [500000] * len(observed))

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    # Consecutive missing is 1 <= 3, not quarantined
    assert res.status != QualityStatus.QUARANTINED
    assert res.missing_trading_days == [expected[-1]]


def test_missing_final_2_sessions(pipeline: DataCleansingPipeline):
    """Missing final 2 sessions: consecutive = 2 <= 3 (not quarantined, but detected)."""
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    observed = expected[:-2]
    df = _make_series(observed, [100.0] * len(observed), [500000] * len(observed))

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    # 48/50 = 96% -> between 95% and 98% -> WARNING
    assert res.status == QualityStatus.WARNING
    assert res.missing_trading_days == expected[-2:]


def test_missing_final_4_sessions_quarantined(pipeline: DataCleansingPipeline):
    """Missing final 4 sessions exceeds max_consecutive_missing_days (3) -> QUARANTINED."""
    expected = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
    observed = expected[:-4]
    df = _make_series(observed, [100.0] * len(observed), [500000] * len(observed))

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    assert res.status == QualityStatus.QUARANTINED
    assert "consecutive unobserved sessions" in (res.quarantine_reason or "")
    assert res.missing_trading_days == expected[-4:]


# ===================================================================
# 3. Listing Date Awareness
# ===================================================================

def test_sessions_before_listing_date_not_classified_as_missing(pipeline: DataCleansingPipeline):
    """Sessions before the known listing date are filtered out and not counted as missing."""
    # Calendar window is 1 Jan to 10 Jan, but asset listed on 6 Jan
    expected = [date(2024, 1, d) for d in range(1, 11)]
    listing_date = date(2024, 1, 6)

    # Asset observed on all days from listing_date onwards (6 Jan to 10 Jan = 5 days)
    observed = [date(2024, 1, d) for d in range(6, 11)]
    df = _make_series(observed, [100.0] * 5, [500000] * 5)

    res = pipeline.evaluate_series("NEWIPO.NS", df, expected_trading_days=expected, listing_date=listing_date)
    # Days 1–5 must NOT be marked missing; total expected is 5, observed is 5
    assert res.total_expected_days == 5
    assert res.observed_days == 5
    assert res.coverage_ratio == 1.0
    assert len(res.missing_trading_days) == 0
    assert res.status == QualityStatus.CLEAN


# ===================================================================
# 4. Missing Sessions in the Middle
# ===================================================================

def test_missing_sessions_in_middle_detected(pipeline: DataCleansingPipeline):
    """Missing 2 sessions in the middle: correctly reported in missing_trading_days."""
    expected = [date(2024, 1, d) for d in range(1, 21)]
    # Missing days 10 and 11
    observed = [d for d in expected if d.day not in (10, 11)]
    df = _make_series(observed, [100.0] * len(observed), [500000] * len(observed))

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=expected)
    assert date(2024, 1, 10) in res.missing_trading_days
    assert date(2024, 1, 11) in res.missing_trading_days
    assert len(res.missing_trading_days) == 2


# ===================================================================
# 5. Quarantine Triggers: Zero-Volume and Price Spikes
# ===================================================================

def test_isolated_zero_volume_generates_warning(pipeline: DataCleansingPipeline):
    """Isolated zero-volume session (1 or 2 days) must generate a WARNING, not CLEAN."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    vols = [500000] * 100
    vols[10] = 0  # 1 isolated zero-volume session
    df = _make_series(days, [100.0] * 100, vols)

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=days)
    assert res.status == QualityStatus.WARNING
    assert any("zero-volume session" in w for w in res.warnings)
    assert res.coverage_ratio == 1.0


def test_exact_boundary_consecutive_zero_volume(pipeline: DataCleansingPipeline):
    """Exact boundary: 4 consecutive zero-volume days -> WARNING; 5 consecutive -> QUARANTINED."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]

    # Exactly 4 consecutive zero-volume days (< threshold 5)
    vols_4 = [500000] * 50
    for i in range(10, 14):
        vols_4[i] = 0
    df_4 = _make_series(days, [100.0] * 50, vols_4)
    res_4 = pipeline.evaluate_series("TEST.NS", df_4, expected_trading_days=days)
    assert res_4.status == QualityStatus.WARNING
    assert any("zero-volume session" in w for w in res_4.warnings)

    # Exactly 5 consecutive zero-volume days (>= threshold 5)
    vols_5 = [500000] * 50
    for i in range(10, 15):
        vols_5[i] = 0
    df_5 = _make_series(days, [100.0] * 50, vols_5)
    res_5 = pipeline.evaluate_series("TEST.NS", df_5, expected_trading_days=days)
    assert res_5.status == QualityStatus.QUARANTINED
    assert "Prolonged zero trading volume" in (res_5.quarantine_reason or "")


def test_clean_status_strictly_requires_zero_warnings(pipeline: DataCleansingPipeline):
    """CLEAN status is returned strictly and only when zero quality warnings exist."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    df = _make_series(days, [100.0] * 100, [500000] * 100)

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=days)
    assert res.status == QualityStatus.CLEAN
    assert len(res.warnings) == 0


def test_quarantine_on_prolonged_zero_volume(pipeline: DataCleansingPipeline):
    """5 consecutive zero-volume days triggers QUARANTINED."""
    days = [date(2024, 1, d) for d in range(1, 6)]
    df = _make_series(days, [100.0] * 5, [0, 0, 0, 0, 0])

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=days)
    assert res.status == QualityStatus.QUARANTINED
    assert "Prolonged zero trading volume" in (res.quarantine_reason or "")


def test_quarantine_on_unexplained_price_spike(pipeline: DataCleansingPipeline):
    """Price jump > 50% without split/dividend triggers QUARANTINED."""
    days = [date(2024, 1, 10), date(2024, 1, 11)]
    df = _make_series(days, [100.0, 160.0], [500000, 500000])

    res = pipeline.evaluate_series("TEST.NS", df, expected_trading_days=days)
    assert res.status == QualityStatus.QUARANTINED
    assert "Unexplained price discontinuity" in (res.quarantine_reason or "")
