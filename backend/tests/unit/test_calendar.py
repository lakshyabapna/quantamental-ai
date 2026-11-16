"""Unit tests for the Official NSE Trading Calendar Module.

Covers:
- Weekday / weekend detection.
- Official NSE holidays (2024, 2026 from NSE/CMTR/71775).
- Regression tests for every previously-incorrect 2026 date.
- Diwali Muhurat special sessions (including 2026 14:20–15:40 IST).
- Multi-day range retrieval and counting.
- Next / previous trading day navigation.
- Rolling 5-year window calculation (including leap-year edge case).
- Unknown-year rejection (HolidayCircularNotFoundError).
- Multi-year range coverage enforcement.
- Annual circular replacement (safe year-scoped replace, no cross-year corruption).
- Amended circulars.
- Invalid circular update rollback.
- Dynamic circular loading for future years.
- Date normalisation (ISO string, datetime, invalid types, None).
"""

from datetime import date, datetime
import pytest

from app.core.exceptions import (
    CalendarDateRangeError,
    HolidayCircularNotFoundError,
    InvalidCalendarDateError,
)
from app.data_engine.calendar import NSECalendar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_calendar(*years: int) -> NSECalendar:
    """Return a fresh calendar pre-loaded with only the specified year schedules."""
    cal = NSECalendar(load_builtin=False)
    from app.data_engine.calendar import (
        BUILTIN_CIRCULAR_METADATA,
        BUILTIN_NSE_HOLIDAYS,
        BUILTIN_SPECIAL_SESSIONS,
    )
    for year in years:
        cal._holidays_by_year[year] = dict(BUILTIN_NSE_HOLIDAYS.get(year, {}))
        cal._special_sessions_by_year[year] = dict(BUILTIN_SPECIAL_SESSIONS.get(year, {}))
        cal._circular_metadata[year] = dict(BUILTIN_CIRCULAR_METADATA[year])
    return cal


# ===================================================================
# 1. Weekday / weekend detection
# ===================================================================

class TestWeekdayWeekend:

    def test_standard_weekday_is_trading_day(self, calendar: NSECalendar):
        """Regular non-holiday weekday within a covered year is an active session."""
        assert calendar.is_trading_day(date(2024, 1, 10)) is True  # Wednesday
        assert calendar.is_holiday(date(2024, 1, 10)) is False

    def test_saturday_is_not_trading_day(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2024, 1, 13)) is False

    def test_sunday_is_not_trading_day(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2024, 1, 14)) is False


# ===================================================================
# 2. Official NSE holidays — 2024
# ===================================================================

class TestOfficialHolidays2024:

    def test_republic_day(self, calendar: NSECalendar):
        d = date(2024, 1, 26)
        assert calendar.is_trading_day(d) is False
        assert calendar.is_holiday(d) is True
        assert calendar.get_holiday_reason(d) == "Republic Day"

    def test_independence_day(self, calendar: NSECalendar):
        d = date(2024, 8, 15)
        assert calendar.is_trading_day(d) is False
        assert calendar.is_holiday(d) is True

    def test_gandhi_jayanti(self, calendar: NSECalendar):
        d = date(2024, 10, 2)
        assert calendar.is_trading_day(d) is False
        assert calendar.is_holiday(d) is True

    def test_pran_pratishtha_special_holiday(self, calendar: NSECalendar):
        d = date(2024, 1, 22)
        assert calendar.is_trading_day(d) is False
        assert calendar.is_holiday(d) is True


# ===================================================================
# 3. Official NSE holidays — 2026 (NSE/CMTR/71775)
#    These test the EXACT dates from the circular, and serve as
#    regression tests for every previously-incorrect date.
# ===================================================================

class TestOfficialHolidays2026:
    """Each holiday is independently verified against circular NSE/CMTR/71775."""

    def test_republic_day_jan_26(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 1, 26)) is False
        assert calendar.get_holiday_reason(date(2026, 1, 26)) == "Republic Day"

    def test_holi_mar_3_not_mar_20(self, calendar: NSECalendar):
        """Regression: Holi is 3 March, not 20 March."""
        assert calendar.is_trading_day(date(2026, 3, 3)) is False
        assert "Holi" in (calendar.get_holiday_reason(date(2026, 3, 3)) or "")
        # 20 March must NOT be a holiday (it was incorrectly set before)
        assert calendar.is_trading_day(date(2026, 3, 20)) is True

    def test_ram_navami_mar_26(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 3, 26)) is False
        assert "Ram Navami" in (calendar.get_holiday_reason(date(2026, 3, 26)) or "")

    def test_mahavir_jayanti_mar_31(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 3, 31)) is False
        assert "Mahavir" in (calendar.get_holiday_reason(date(2026, 3, 31)) or "")

    def test_good_friday_apr_3(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 4, 3)) is False

    def test_ambedkar_jayanti_apr_14(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 4, 14)) is False

    def test_maharashtra_day_may_1(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 5, 1)) is False

    def test_bakri_id_may_28_not_jun_19(self, calendar: NSECalendar):
        """Regression: Bakri Id is 28 May, not 19 June."""
        assert calendar.is_trading_day(date(2026, 5, 28)) is False
        assert "Bakri" in (calendar.get_holiday_reason(date(2026, 5, 28)) or "")
        # 19 June must NOT be a holiday
        assert calendar.is_trading_day(date(2026, 6, 19)) is True

    def test_muharram_jun_26_not_jul_9(self, calendar: NSECalendar):
        """Regression: Muharram is 26 June, not 9 July."""
        assert calendar.is_trading_day(date(2026, 6, 26)) is False
        assert "Muharram" in (calendar.get_holiday_reason(date(2026, 6, 26)) or "")
        # 9 July must NOT be a holiday
        assert calendar.is_trading_day(date(2026, 7, 9)) is True

    def test_independence_day_aug_15_is_saturday_no_observed_holiday(self, calendar: NSECalendar):
        """Regression: 15 Aug 2026 is Saturday. NSE did NOT declare a compensatory closure.
        No Aug 17 entry should exist."""
        # 15 Aug is Saturday — not a trading day, but also not a holiday entry
        assert calendar.is_trading_day(date(2026, 8, 15)) is False  # Weekend
        assert calendar.get_holiday_reason(date(2026, 8, 15)) is None  # Not in holiday list
        # 17 Aug is Monday — must be a normal trading day
        assert calendar.is_trading_day(date(2026, 8, 17)) is True

    def test_ganesh_chaturthi_sep_14(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 9, 14)) is False
        assert "Ganesh" in (calendar.get_holiday_reason(date(2026, 9, 14)) or "")

    def test_gandhi_jayanti_oct_2(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 10, 2)) is False

    def test_dussehra_oct_20_not_oct_22(self, calendar: NSECalendar):
        """Regression: Dussehra is 20 October, not 22 October."""
        assert calendar.is_trading_day(date(2026, 10, 20)) is False
        assert "Dussehra" in (calendar.get_holiday_reason(date(2026, 10, 20)) or "")
        # 22 October must NOT be a holiday
        assert calendar.is_trading_day(date(2026, 10, 22)) is True

    def test_diwali_balipratipada_nov_10_not_nov_11(self, calendar: NSECalendar):
        """Regression: Diwali-Balipratipada is 10 November, not 11 November."""
        assert calendar.is_trading_day(date(2026, 11, 10)) is False
        reason = calendar.get_holiday_reason(date(2026, 11, 10)) or ""
        assert "Diwali" in reason or "Balipratipada" in reason
        # 11 November must NOT be a holiday
        assert calendar.is_trading_day(date(2026, 11, 11)) is True

    def test_guru_nanak_dev_nov_24_not_nov_25(self, calendar: NSECalendar):
        """Regression: Guru Nanak Dev is 24 November, not 25 November."""
        assert calendar.is_trading_day(date(2026, 11, 24)) is False
        reason = calendar.get_holiday_reason(date(2026, 11, 24)) or ""
        assert "Guru Nanak" in reason or "Gurpurb" in reason
        # 25 November must NOT be a holiday
        assert calendar.is_trading_day(date(2026, 11, 25)) is True

    def test_christmas_dec_25(self, calendar: NSECalendar):
        assert calendar.is_trading_day(date(2026, 12, 25)) is False

    def test_2026_holiday_count(self, calendar: NSECalendar):
        """NSE/CMTR/71775 declares exactly 15 weekday holidays for 2026."""
        holidays = calendar.get_all_holidays_for_year(2026)
        assert len(holidays) == 15


class TestCircularMetadata2026:

    def test_circular_ref(self, calendar: NSECalendar):
        meta = calendar.get_circular_metadata(2026)
        assert meta["circular_ref"] == "NSE/CMTR/71775"

    def test_publication_date(self, calendar: NSECalendar):
        meta = calendar.get_circular_metadata(2026)
        assert meta["publication_date"] == date(2025, 12, 12)


# ===================================================================
# 4. Diwali Muhurat special sessions
# ===================================================================

class TestMuhuratSessions:

    def test_muhurat_2026_date_and_timings(self, calendar: NSECalendar):
        """2026 Diwali Muhurat: 8 Nov 2026 (Sunday) 14:20–15:40 IST."""
        muhurat = date(2026, 11, 8)
        assert calendar.is_special_session(muhurat) is True
        assert calendar.is_trading_day(muhurat) is True  # Sunday overridden
        reason = calendar.get_special_session_reason(muhurat)
        assert reason is not None
        assert "14:20" in reason
        assert "15:40" in reason

    def test_muhurat_2024_friday(self, calendar: NSECalendar):
        d = date(2024, 11, 1)
        assert calendar.is_special_session(d) is True
        assert calendar.is_trading_day(d) is True
        assert "Muhurat" in (calendar.get_special_session_reason(d) or "")

    def test_muhurat_2023_overrides_sunday(self, calendar: NSECalendar):
        d = date(2023, 11, 12)
        assert calendar.is_special_session(d) is True
        assert calendar.is_trading_day(d) is True

    def test_muhurat_2020_overrides_saturday(self, calendar: NSECalendar):
        d = date(2020, 11, 14)
        assert calendar.is_special_session(d) is True
        assert calendar.is_trading_day(d) is True

    def test_ordinary_weekend_remains_non_trading(self, calendar: NSECalendar):
        """Regular weekends (no special session) are not trading days."""
        # Sunday 4 Jan 2026 — no special session
        assert calendar.is_trading_day(date(2026, 1, 4)) is False
        assert calendar.is_special_session(date(2026, 1, 4)) is False


# ===================================================================
# 5. Date range retrieval & counting
# ===================================================================

class TestDateRanges:

    def test_get_trading_days_range(self, calendar: NSECalendar):
        # Week 22–26 Jan 2024:
        # 22 Mon = Pran Pratishtha holiday, 23 Tue/24 Wed/25 Thu = trading, 26 Fri = Republic Day
        days = calendar.get_trading_days(date(2024, 1, 22), date(2024, 1, 26))
        assert days == [date(2024, 1, 23), date(2024, 1, 24), date(2024, 1, 25)]
        assert calendar.count_trading_days(date(2024, 1, 22), date(2024, 1, 26)) == 3

    def test_invalid_range(self, calendar: NSECalendar):
        with pytest.raises(CalendarDateRangeError):
            calendar.get_trading_days(date(2024, 1, 25), date(2024, 1, 20))

    def test_same_day_trading(self, calendar: NSECalendar):
        assert calendar.get_trading_days(date(2024, 1, 10), date(2024, 1, 10)) == [date(2024, 1, 10)]

    def test_same_day_non_trading(self, calendar: NSECalendar):
        assert calendar.get_trading_days(date(2024, 1, 26), date(2024, 1, 26)) == []


# ===================================================================
# 6. Next / previous trading day navigation
# ===================================================================

class TestNavigation:

    def test_next_across_holiday_and_weekend(self, calendar: NSECalendar):
        # Fri 19 Jan → Sat 20 (wknd) → Sun 21 (wknd) → Mon 22 (holiday) → Tue 23
        assert calendar.get_next_trading_day(date(2024, 1, 19)) == date(2024, 1, 23)

    def test_previous_across_holiday_and_weekend(self, calendar: NSECalendar):
        # Tue 23 ← Mon 22 (holiday) ← Sun 21 (wknd) ← Sat 20 (wknd) ← Fri 19
        assert calendar.get_previous_trading_day(date(2024, 1, 23)) == date(2024, 1, 19)


# ===================================================================
# 7. Rolling window calculation
# ===================================================================

class TestRollingWindow:

    def test_active_start(self, calendar: NSECalendar):
        start = calendar.get_rolling_window_start(date(2024, 3, 15), years=5)
        assert start == date(2019, 3, 15)
        assert calendar.is_trading_day(start) is True

    def test_leap_year_edge_case(self, calendar: NSECalendar):
        start = calendar.get_rolling_window_start(date(2024, 2, 29), years=5)
        assert start == date(2019, 2, 28)
        assert calendar.is_trading_day(start) is True

    def test_adjusts_forward_over_weekend(self, calendar: NSECalendar):
        # 26 Jan 2024 − 5y = 26 Jan 2019 (Sat). Next active = Mon 28 Jan 2019.
        start = calendar.get_rolling_window_start(date(2024, 1, 26), years=5)
        assert start == date(2019, 1, 28)
        assert calendar.is_trading_day(start) is True

    def test_invalid_years_zero(self, calendar: NSECalendar):
        with pytest.raises(CalendarDateRangeError):
            calendar.get_rolling_window_start(date(2024, 1, 1), years=0)

    def test_invalid_years_negative(self, calendar: NSECalendar):
        with pytest.raises(CalendarDateRangeError):
            calendar.get_rolling_window_start(date(2024, 1, 1), years=-1)


# ===================================================================
# 8. Unknown-year rejection
# ===================================================================

class TestUnknownYearRejection:

    def test_is_trading_day_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError) as exc_info:
            cal.is_trading_day(date(2030, 6, 15))
        assert "2030" in str(exc_info.value)

    def test_is_holiday_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.is_holiday(date(2030, 1, 1))

    def test_is_special_session_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.is_special_session(date(2030, 11, 8))

    def test_get_trading_days_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.get_trading_days(date(2030, 1, 1), date(2030, 1, 10))

    def test_get_holiday_reason_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.get_holiday_reason(date(2030, 1, 1))

    def test_get_special_session_reason_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.get_special_session_reason(date(2030, 1, 1))

    def test_get_next_trading_day_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.get_next_trading_day(date(2030, 1, 1))

    def test_get_previous_trading_day_raises(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.get_previous_trading_day(date(2030, 1, 1))

    def test_known_year_weekday_does_not_raise(self, calendar: NSECalendar):
        """A covered year's weekday must not raise."""
        assert calendar.is_trading_day(date(2024, 6, 3)) is True


# ===================================================================
# 9. Multi-year range coverage
# ===================================================================

class TestMultiYearCoverage:

    def test_missing_coverage_raises(self):
        cal = _minimal_calendar(2024)
        with pytest.raises(HolidayCircularNotFoundError) as exc_info:
            cal.get_trading_days(date(2024, 12, 1), date(2025, 1, 10))
        assert "2025" in str(exc_info.value)

    def test_all_covered_succeeds(self):
        cal = _minimal_calendar(2023, 2024)
        days = cal.get_trading_days(date(2023, 12, 28), date(2024, 1, 5))
        assert date(2023, 12, 28) in days
        assert date(2024, 1, 2) in days
        for d in days:
            assert d.weekday() < 5

    def test_rolling_window_missing_coverage_raises(self):
        cal = _minimal_calendar(2024)
        with pytest.raises(HolidayCircularNotFoundError) as exc_info:
            cal.get_rolling_window_start(date(2024, 6, 1), years=5)
        assert "2019" in str(exc_info.value)


# ===================================================================
# 10. Annual circular replacement
# ===================================================================

class TestCircularReplacement:

    def test_replaces_year_schedule(self):
        cal = _minimal_calendar(2024)
        # Republic Day was a holiday
        assert cal.is_trading_day(date(2024, 1, 26)) is False

        cal.load_holiday_circular(
            year=2024,
            holidays={date(2024, 7, 4): "Dummy Holiday"},
            circular_ref="NSE/TEST/00001",
            publication_date=date(2024, 1, 1),
        )
        # Republic Day removed by replacement
        assert cal.is_trading_day(date(2024, 1, 26)) is True
        assert cal.get_holiday_reason(date(2024, 1, 26)) is None
        # New dummy holiday active
        assert cal.is_trading_day(date(2024, 7, 4)) is False

    def test_does_not_affect_other_years(self):
        cal = _minimal_calendar(2023, 2024)
        assert cal.is_trading_day(date(2023, 12, 25)) is False  # 2023 Christmas

        cal.load_holiday_circular(
            year=2024,
            holidays={date(2024, 7, 4): "Dummy"},
            circular_ref="NSE/TEST/00002",
            publication_date=date(2024, 1, 1),
        )
        # 2023 Christmas must still be a holiday
        assert cal.is_trading_day(date(2023, 12, 25)) is False
        assert cal.get_holiday_reason(date(2023, 12, 25)) == "Christmas"

    def test_replaces_special_sessions(self):
        cal = _minimal_calendar(2024)
        assert cal.is_special_session(date(2024, 11, 1)) is True

        cal.load_holiday_circular(
            year=2024,
            holidays={},
            circular_ref="NSE/TEST/00003",
            publication_date=date(2024, 6, 1),
            special_sessions={date(2024, 5, 15): "Test Session"},
        )
        assert cal.is_special_session(date(2024, 11, 1)) is False
        assert cal.is_special_session(date(2024, 5, 15)) is True

    def test_amended_circular_takes_effect(self):
        cal = _minimal_calendar(2026)
        assert cal.get_circular_metadata(2026)["circular_ref"] == "NSE/CMTR/71775"

        cal.load_holiday_circular(
            year=2026,
            holidays={
                date(2026, 1, 26): "Republic Day",
                date(2026, 4, 3): "Good Friday",
                date(2026, 10, 2): "Mahatma Gandhi Jayanti",
            },
            circular_ref="NSE/CMTR/71775-AMD1",
            publication_date=date(2026, 1, 15),
        )
        meta = cal.get_circular_metadata(2026)
        assert meta["circular_ref"] == "NSE/CMTR/71775-AMD1"
        assert meta["holiday_count"] == 3
        # Christmas removed in amendment
        assert cal.get_holiday_reason(date(2026, 12, 25)) is None


# ===================================================================
# 11. Invalid circular update rollback
# ===================================================================

class TestInvalidCircularRollback:

    def test_rejects_wrong_year_holiday(self):
        cal = _minimal_calendar(2024)
        original = cal.get_holiday_reason(date(2024, 1, 26))

        with pytest.raises(CalendarDateRangeError):
            cal.load_holiday_circular(
                year=2024,
                holidays={
                    date(2024, 7, 4): "Valid",
                    date(2025, 1, 26): "WRONG YEAR",
                },
                circular_ref="NSE/TEST/INVALID",
                publication_date=date(2024, 1, 1),
            )
        # State unchanged
        assert cal.is_trading_day(date(2024, 1, 26)) is False
        assert cal.get_holiday_reason(date(2024, 1, 26)) == original

    def test_rejects_wrong_year_special_session(self):
        cal = _minimal_calendar(2024)
        with pytest.raises(CalendarDateRangeError):
            cal.load_holiday_circular(
                year=2024,
                holidays={},
                circular_ref="NSE/TEST/INVALID2",
                publication_date=date(2024, 1, 1),
                special_sessions={date(2025, 11, 8): "WRONG YEAR"},
            )
        # Original 2024 Muhurat preserved
        assert cal.is_special_session(date(2024, 11, 1)) is True


# ===================================================================
# 12. Dynamic circular loading for future years
# ===================================================================

class TestDynamicLoading:

    def test_future_year(self):
        cal = NSECalendar(load_builtin=False)
        cal.load_holiday_circular(
            year=2027,
            holidays={
                date(2027, 1, 26): "Republic Day",
                date(2027, 8, 15): "Independence Day",
            },
            circular_ref="NSE/CMTR/FUTURE/2027",
            publication_date=date(2026, 12, 1),
            special_sessions={date(2027, 10, 20): "Muhurat 2027"},
        )
        assert cal.get_circular_metadata(2027)["circular_ref"] == "NSE/CMTR/FUTURE/2027"
        assert cal.is_trading_day(date(2027, 1, 26)) is False
        assert cal.is_special_session(date(2027, 10, 20)) is True
        assert cal.is_trading_day(date(2027, 3, 15)) is True

    def test_unknown_before_load(self):
        cal = NSECalendar(load_builtin=False)
        with pytest.raises(HolidayCircularNotFoundError):
            cal.is_trading_day(date(2027, 1, 1))
        with pytest.raises(HolidayCircularNotFoundError):
            cal.is_trading_day(date(2028, 6, 15))


# ===================================================================
# 13. Date normalisation
# ===================================================================

class TestDateNormalisation:

    def test_iso_string(self, calendar: NSECalendar):
        assert calendar.is_trading_day("2024-01-10") is True
        assert calendar.is_trading_day("2024-01-26") is False

    def test_datetime_object(self, calendar: NSECalendar):
        assert calendar.is_trading_day(datetime(2024, 1, 10, 15, 30)) is True

    def test_invalid_int_raises(self, calendar: NSECalendar):
        with pytest.raises(InvalidCalendarDateError):
            calendar.is_trading_day(12345)  # type: ignore

    def test_malformed_string_raises(self, calendar: NSECalendar):
        with pytest.raises(InvalidCalendarDateError):
            calendar.is_trading_day("not-a-date")

    def test_none_raises(self, calendar: NSECalendar):
        with pytest.raises(InvalidCalendarDateError):
            calendar.is_trading_day(None)  # type: ignore


# ===================================================================
# 14. Covered years property
# ===================================================================

class TestCoveredYears:

    def test_sorted(self, calendar: NSECalendar):
        years = calendar.covered_years
        assert years == sorted(years)
        assert 2019 in years
        assert 2026 in years
