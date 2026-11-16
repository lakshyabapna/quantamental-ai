"""Quantamental AI - Official NSE Trading Calendar Module.

Maintains verified, versioned exchange circular holiday schedules, special trading
sessions (such as Diwali Muhurat trading), and calendar arithmetic for Indian equities.

Strict Coverage Policy
----------------------
A date may only be evaluated when the calendar schedule for its year has been
registered via a verified NSE circular.  Any operation on a date whose year is
not covered raises ``HolidayCircularNotFoundError`` immediately.  This prevents
unknown years from silently falling through to weekend-only logic.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from app.core.exceptions import (
    CalendarDateRangeError,
    HolidayCircularNotFoundError,
    InvalidCalendarDateError,
)
from app.core.logger import get_logger

logger = get_logger("quantamental.calendar")


# ---------------------------------------------------------------------------
# Built-in verified NSE Circular Metadata (2019 – 2026)
# Source: National Stock Exchange of India Ltd.
# ---------------------------------------------------------------------------
BUILTIN_CIRCULAR_METADATA: Dict[int, Dict[str, Any]] = {
    2019: {
        "circular_ref": "NSE/CMTR/39600",
        "publication_date": date(2018, 12, 10),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2019",
    },
    2020: {
        "circular_ref": "NSE/CMTR/42838",
        "publication_date": date(2019, 12, 3),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2020",
    },
    2021: {
        "circular_ref": "NSE/CMTR/46788",
        "publication_date": date(2020, 12, 23),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2021",
    },
    2022: {
        "circular_ref": "NSE/CMTR/50529",
        "publication_date": date(2021, 12, 7),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2022",
    },
    2023: {
        "circular_ref": "NSE/CMTR/54756",
        "publication_date": date(2022, 12, 8),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2023",
    },
    2024: {
        "circular_ref": "NSE/CMTR/59737",
        "publication_date": date(2023, 12, 13),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2024",
    },
    2025: {
        "circular_ref": "NSE/CMTR/65688",
        "publication_date": date(2024, 12, 16),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2025",
    },
    # Circular NSE/CMTR/71775, published 12 December 2025.
    2026: {
        "circular_ref": "NSE/CMTR/71775",
        "publication_date": date(2025, 12, 12),
        "source": "National Stock Exchange of India Ltd",
        "description": "NSE Trading Holidays for the Calendar Year 2026",
    },
}

# ---------------------------------------------------------------------------
# Verified weekday exchange holidays per year
# Source: official NSE circulars referenced in BUILTIN_CIRCULAR_METADATA.
# Saturdays and Sundays are excluded from this table (handled structurally).
# ---------------------------------------------------------------------------
BUILTIN_NSE_HOLIDAYS: Dict[int, Dict[date, str]] = {
    2019: {
        date(2019, 3, 4): "Mahashivratri",
        date(2019, 3, 21): "Holi",
        date(2019, 4, 17): "Mahavir Jayanti",
        date(2019, 4, 19): "Good Friday",
        date(2019, 4, 29): "General Parliamentary Elections (Maharashtra)",
        date(2019, 5, 1): "Maharashtra Day",
        date(2019, 6, 5): "Id-Ul-Fitr (Ramzan Id)",
        date(2019, 8, 12): "Bakri Id",
        date(2019, 8, 15): "Independence Day",
        date(2019, 9, 2): "Ganesh Chaturthi",
        date(2019, 9, 10): "Muharram",
        date(2019, 10, 2): "Mahatma Gandhi Jayanti",
        date(2019, 10, 8): "Dussehra",
        date(2019, 10, 21): "Maharashtra Assembly Elections",
        date(2019, 10, 28): "Diwali Balipratipada",
        date(2019, 11, 12): "Gurunanak Jayanti",
        date(2019, 12, 25): "Christmas",
    },
    2020: {
        date(2020, 2, 21): "Mahashivratri",
        date(2020, 3, 10): "Holi",
        date(2020, 4, 2): "Ram Navami",
        date(2020, 4, 6): "Mahavir Jayanti",
        date(2020, 4, 10): "Good Friday",
        date(2020, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2020, 5, 1): "Maharashtra Day",
        date(2020, 5, 25): "Id-Ul-Fitr",
        date(2020, 10, 2): "Mahatma Gandhi Jayanti",
        date(2020, 11, 16): "Diwali Balipratipada",
        date(2020, 11, 30): "Gurunanak Jayanti",
        date(2020, 12, 25): "Christmas",
    },
    2021: {
        date(2021, 1, 26): "Republic Day",
        date(2021, 3, 11): "Mahashivratri",
        date(2021, 3, 29): "Holi",
        date(2021, 4, 2): "Good Friday",
        date(2021, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2021, 4, 21): "Ram Navami",
        date(2021, 5, 13): "Id-Ul-Fitr",
        date(2021, 7, 21): "Bakri Id",
        date(2021, 8, 19): "Muharram",
        date(2021, 10, 15): "Dussehra",
        date(2021, 11, 5): "Diwali Balipratipada",
        date(2021, 11, 19): "Gurunanak Jayanti",
    },
    2022: {
        date(2022, 1, 26): "Republic Day",
        date(2022, 3, 1): "Mahashivratri",
        date(2022, 3, 18): "Holi",
        date(2022, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti / Mahavir Jayanti",
        date(2022, 4, 15): "Good Friday",
        date(2022, 5, 3): "Id-Ul-Fitr",
        date(2022, 8, 9): "Muharram",
        date(2022, 8, 15): "Independence Day",
        date(2022, 8, 31): "Ganesh Chaturthi",
        date(2022, 10, 5): "Dussehra",
        date(2022, 10, 26): "Diwali Balipratipada",
        date(2022, 11, 8): "Gurunanak Jayanti",
    },
    2023: {
        date(2023, 1, 26): "Republic Day",
        date(2023, 3, 7): "Holi",
        date(2023, 3, 30): "Ram Navami",
        date(2023, 4, 4): "Mahavir Jayanti",
        date(2023, 4, 7): "Good Friday",
        date(2023, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2023, 5, 1): "Maharashtra Day",
        date(2023, 6, 29): "Bakri Id",
        date(2023, 8, 15): "Independence Day",
        date(2023, 9, 19): "Ganesh Chaturthi",
        date(2023, 10, 2): "Mahatma Gandhi Jayanti",
        date(2023, 10, 24): "Dussehra",
        date(2023, 11, 14): "Diwali Balipratipada",
        date(2023, 11, 27): "Gurunanak Jayanti",
        date(2023, 12, 25): "Christmas",
    },
    2024: {
        date(2024, 1, 22): "Special Holiday (Ayodhya Ram Mandir Pran Pratishtha)",
        date(2024, 1, 26): "Republic Day",
        date(2024, 3, 8): "Mahashivratri",
        date(2024, 3, 25): "Holi",
        date(2024, 3, 29): "Good Friday",
        date(2024, 4, 11): "Id-Ul-Fitr",
        date(2024, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2024, 4, 17): "Ram Navami",
        date(2024, 4, 21): "Mahavir Jayanti",
        date(2024, 5, 20): "General Parliamentary Elections (Mumbai)",
        date(2024, 6, 17): "Bakri Id",
        date(2024, 7, 17): "Muharram",
        date(2024, 8, 15): "Independence Day",
        date(2024, 10, 2): "Mahatma Gandhi Jayanti",
        date(2024, 11, 15): "Gurunanak Jayanti",
        date(2024, 11, 20): "Maharashtra Assembly Elections",
        date(2024, 12, 25): "Christmas",
    },
    2025: {
        date(2025, 2, 26): "Mahashivratri",
        date(2025, 3, 14): "Holi",
        date(2025, 3, 31): "Id-Ul-Fitr",
        date(2025, 4, 10): "Mahavir Jayanti",
        date(2025, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
        date(2025, 4, 18): "Good Friday",
        date(2025, 5, 1): "Maharashtra Day",
        date(2025, 8, 15): "Independence Day",
        date(2025, 8, 27): "Ganesh Chaturthi",
        date(2025, 10, 2): "Mahatma Gandhi Jayanti / Dussehra",
        date(2025, 10, 22): "Diwali Balipratipada",
        date(2025, 11, 5): "Prakash Gurpurb Sri Guru Nanak Dev",
        date(2025, 12, 25): "Christmas",
    },
    # 2026 — per NSE Circular NSE/CMTR/71775 dated 12 December 2025.
    # This table lists exactly the 15 weekday holidays declared in that circular.
    # Notes:
    #   - Independence Day (15 Aug) falls on a Saturday — not a weekday; NSE
    #     did NOT declare a compensatory weekday closure. No entry added.
    #   - Mahashivratri is NOT listed in this circular for 2026.
    #   - Id-Ul-Fitr is NOT listed in this circular for 2026.
    #   - Diwali Muhurat Trading (8 Nov 2026, Sunday, 14:20–15:40 IST) is a
    #     special session, not a closure; it is recorded in BUILTIN_SPECIAL_SESSIONS.
    2026: {
        date(2026, 1, 26): "Republic Day",
        date(2026, 3, 3):  "Holi",
        date(2026, 3, 26): "Shri Ram Navami",
        date(2026, 3, 31): "Shri Mahavir Jayanti",
        date(2026, 4, 3):  "Good Friday",
        date(2026, 4, 14): "Dr. Babasaheb Ambedkar Jayanti",
        date(2026, 5, 1):  "Maharashtra Day",
        date(2026, 5, 28): "Bakri Id",
        date(2026, 6, 26): "Muharram",
        date(2026, 9, 14): "Ganesh Chaturthi",
        date(2026, 10, 2): "Mahatma Gandhi Jayanti",
        date(2026, 10, 20): "Dussehra",
        date(2026, 11, 10): "Diwali-Balipratipada",
        date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
        date(2026, 12, 25): "Christmas",
    },
}

# ---------------------------------------------------------------------------
# Special Trading Sessions (e.g. Diwali Muhurat Trading)
# These override weekend / holiday closures — the date IS an active session.
# Stored per year for consistent year-scoped management.
# ---------------------------------------------------------------------------
BUILTIN_SPECIAL_SESSIONS: Dict[int, Dict[date, str]] = {
    2019: {
        date(2019, 10, 27): "Diwali Muhurat Trading (Sunday 18:15–19:15 IST)",
    },
    2020: {
        date(2020, 11, 14): "Diwali Muhurat Trading (Saturday 18:15–19:15 IST)",
    },
    2021: {
        date(2021, 11, 4): "Diwali Muhurat Trading (Thursday 18:15–19:15 IST)",
    },
    2022: {
        date(2022, 10, 24): "Diwali Muhurat Trading (Monday 18:15–19:15 IST)",
    },
    2023: {
        date(2023, 11, 12): "Diwali Muhurat Trading (Sunday 18:15–19:15 IST)",
    },
    2024: {
        date(2024, 11, 1): "Diwali Muhurat Trading (Friday 18:00–19:00 IST)",
    },
    2025: {
        date(2025, 10, 21): "Diwali Muhurat Trading (Tuesday 18:15–19:15 IST)",
    },
    # 2026 Muhurat session — per subsequent official NSE circular.
    # Date: Sunday, 8 November 2026; Timings: 14:20–15:40 IST.
    2026: {
        date(2026, 11, 8): "Diwali Muhurat Trading (Sunday 14:20–15:40 IST)",
    },
}


def _flatten_year_dict(by_year: Dict[int, Dict[date, str]]) -> Dict[date, str]:
    """Merge year-keyed nested dicts into a single flat date→reason dict."""
    result: Dict[date, str] = {}
    for year_entries in by_year.values():
        result.update(year_entries)
    return result


class NSECalendar:
    """Official National Stock Exchange of India (NSE) Trading Calendar.

    Coverage Policy
    ---------------
    Every date query first checks whether the year of that date has a registered
    circular.  If the year is not covered, ``HolidayCircularNotFoundError`` is
    raised.  This prevents unknown years from silently returning weekday-based
    answers without verified holiday data.

    Multi-year range operations (``get_trading_days``, ``count_trading_days``,
    ``get_rolling_window_start``) verify coverage for every calendar year
    spanned by the requested range before iterating.
    """

    def __init__(self, load_builtin: bool = True) -> None:
        """Initialize the calendar with optional built-in verified schedules."""
        # Per-year holiday store: year → {date: reason}
        self._holidays_by_year: Dict[int, Dict[date, str]] = {}
        # Per-year special-session store: year → {date: reason}
        self._special_sessions_by_year: Dict[int, Dict[date, str]] = {}
        # Circular metadata: year → metadata dict
        self._circular_metadata: Dict[int, Dict[str, Any]] = {}

        if load_builtin:
            for year, entries in BUILTIN_NSE_HOLIDAYS.items():
                self._holidays_by_year[year] = dict(entries)
            for year, entries in BUILTIN_SPECIAL_SESSIONS.items():
                self._special_sessions_by_year[year] = dict(entries)
            self._circular_metadata.update(BUILTIN_CIRCULAR_METADATA)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_date(self, d: Any) -> date:
        """Ensure input is converted strictly to a ``datetime.date`` instance."""
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                return date.fromisoformat(d)
            except ValueError as exc:
                raise InvalidCalendarDateError(
                    f"Could not parse ISO date string '{d}'",
                    details={"input": d, "error": str(exc)},
                ) from exc
        raise InvalidCalendarDateError(
            f"Expected date, datetime, or ISO string; received {type(d).__name__}",
            details={"input": repr(d), "received_type": type(d).__name__},
        )

    def _require_year_coverage(self, year: int) -> None:
        """Raise ``HolidayCircularNotFoundError`` when a year has no registered circular."""
        if year not in self._circular_metadata:
            raise HolidayCircularNotFoundError(
                f"No verified NSE holiday circular registered for calendar year {year}. "
                "Load the official circular via load_holiday_circular() before querying dates in this year.",
                details={
                    "requested_year": year,
                    "covered_years": sorted(self._circular_metadata.keys()),
                },
            )

    def _require_range_coverage(self, start: date, end: date) -> None:
        """Raise ``HolidayCircularNotFoundError`` if any year in [start, end] is uncovered."""
        for year in range(start.year, end.year + 1):
            self._require_year_coverage(year)

    def _holidays_flat(self) -> Dict[date, str]:
        """Return a flat merged view of all registered holidays (read-only helper)."""
        return _flatten_year_dict(self._holidays_by_year)

    def _special_sessions_flat(self) -> Dict[date, str]:
        """Return a flat merged view of all registered special sessions."""
        return _flatten_year_dict(self._special_sessions_by_year)

    # ------------------------------------------------------------------
    # Public calendar query API
    # ------------------------------------------------------------------

    def is_trading_day(self, as_of: Any) -> bool:
        """Determine whether the specified date is an active NSE trading session.

        Evaluation order:
        1. Raise ``HolidayCircularNotFoundError`` if the year is not covered.
        2. If ``as_of`` is in special sessions → True (overrides weekend/holiday).
        3. If ``as_of`` is a registered holiday → False.
        4. If ``as_of`` is Saturday or Sunday → False.
        5. Otherwise → True.
        """
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)

        year_sessions = self._special_sessions_by_year.get(d.year, {})
        if d in year_sessions:
            return True

        year_holidays = self._holidays_by_year.get(d.year, {})
        if d in year_holidays:
            return False

        if d.weekday() >= 5:
            return False

        return True

    def is_holiday(self, as_of: Any) -> bool:
        """Check whether the date is a registered exchange holiday (excluding special sessions)."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        year_holidays = self._holidays_by_year.get(d.year, {})
        year_sessions = self._special_sessions_by_year.get(d.year, {})
        return d in year_holidays and d not in year_sessions

    def is_special_session(self, as_of: Any) -> bool:
        """Check whether the date has a designated special trading session."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        return d in self._special_sessions_by_year.get(d.year, {})

    def get_holiday_reason(self, as_of: Any) -> Optional[str]:
        """Return the festival or reason label for a registered holiday, or None."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        return self._holidays_by_year.get(d.year, {}).get(d)

    def get_special_session_reason(self, as_of: Any) -> Optional[str]:
        """Return the description of a special trading session, or None."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        return self._special_sessions_by_year.get(d.year, {}).get(d)

    def get_trading_days(self, start_date: Any, end_date: Any) -> List[date]:
        """Return all active trading sessions in the closed range [start_date, end_date].

        Raises:
            CalendarDateRangeError: If ``start_date`` > ``end_date``.
            HolidayCircularNotFoundError: If any calendar year in the range lacks coverage.
        """
        start = self._normalize_date(start_date)
        end = self._normalize_date(end_date)

        if start > end:
            raise CalendarDateRangeError(
                f"start_date ({start}) cannot be strictly greater than end_date ({end})",
                details={"start_date": str(start), "end_date": str(end)},
            )

        self._require_range_coverage(start, end)

        trading_days: List[date] = []
        cur = start
        while cur <= end:
            if self.is_trading_day(cur):
                trading_days.append(cur)
            cur += timedelta(days=1)

        return trading_days

    def count_trading_days(self, start_date: Any, end_date: Any) -> int:
        """Count the number of active NSE trading sessions between start_date and end_date."""
        return len(self.get_trading_days(start_date, end_date))

    def get_next_trading_day(self, as_of: Any) -> date:
        """Return the first active trading day strictly after ``as_of``."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        cur = d + timedelta(days=1)
        # Ensure coverage as we advance into the next year if necessary
        while True:
            self._require_year_coverage(cur.year)
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)

    def get_previous_trading_day(self, as_of: Any) -> date:
        """Return the latest active trading day strictly before ``as_of``."""
        d = self._normalize_date(as_of)
        self._require_year_coverage(d.year)
        cur = d - timedelta(days=1)
        while True:
            self._require_year_coverage(cur.year)
            if self.is_trading_day(cur):
                return cur
            cur -= timedelta(days=1)

    def get_rolling_window_start(self, end_date: Any, years: int = 5) -> date:
        """Compute rolling window start date per Data Strategy § 2.

        Algorithm:
        1. ``T_start_raw`` = ``end_date`` minus ``years`` calendar years
           (leap-year safe: Feb 29 → Feb 28 on non-leap target).
        2. Advance ``T_start_raw`` forward to the nearest subsequent active
           trading session.
        3. All years between ``T_start_raw`` and ``end_date`` must be covered.

        Raises:
            CalendarDateRangeError: If ``years`` ≤ 0.
            HolidayCircularNotFoundError: If any spanned year lacks coverage.
        """
        end = self._normalize_date(end_date)
        if years <= 0:
            raise CalendarDateRangeError(
                f"Rolling window years must be strictly positive (got {years})",
                details={"years": years},
            )

        try:
            raw_start = end.replace(year=end.year - years)
        except ValueError:
            # Feb 29 on a non-leap target year
            raw_start = end.replace(year=end.year - years, day=28)

        self._require_range_coverage(raw_start, end)

        cur = raw_start
        while cur <= end:
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)

        raise CalendarDateRangeError(
            f"No active trading days found between {raw_start} and {end}",
            details={"raw_start": str(raw_start), "end_date": str(end)},
        )

    # ------------------------------------------------------------------
    # Circular management
    # ------------------------------------------------------------------

    def load_holiday_circular(
        self,
        year: int,
        holidays: Dict[Any, str],
        circular_ref: str,
        publication_date: Any,
        special_sessions: Optional[Dict[Any, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Load or replace the complete holiday schedule for a given calendar year.

        Behaviour
        ---------
        * **Replaces** (not merges) all holidays registered for ``year``.
        * **Replaces** all special sessions registered for ``year``.
        * **Preserves** schedules for all other years unchanged.
        * Validates that every supplied holiday date and every special-session date
          belongs to ``year``; rejects the entire update on any violation.
        * Validates circular metadata before modifying the active schedule so that
          invalid updates do not partially corrupt existing data.

        Raises:
            InvalidCalendarDateError: If any date cannot be parsed.
            CalendarDateRangeError: If any supplied date does not belong to ``year``.
        """
        pub_date = self._normalize_date(publication_date)

        # --- Phase 1: normalize and validate all inputs before touching state ---
        normalized_holidays: Dict[date, str] = {}
        for raw_key, reason in holidays.items():
            d = self._normalize_date(raw_key)
            if d.year != year:
                raise CalendarDateRangeError(
                    f"Holiday date {d} does not belong to year {year}",
                    details={"offending_date": str(d), "expected_year": year},
                )
            normalized_holidays[d] = reason

        normalized_special: Dict[date, str] = {}
        if special_sessions:
            for raw_key, reason in special_sessions.items():
                d = self._normalize_date(raw_key)
                if d.year != year:
                    raise CalendarDateRangeError(
                        f"Special-session date {d} does not belong to year {year}",
                        details={"offending_date": str(d), "expected_year": year},
                    )
                normalized_special[d] = reason

        new_metadata: Dict[str, Any] = {
            "circular_ref": circular_ref,
            "publication_date": pub_date,
            "source": "National Stock Exchange of India Ltd",
            "description": description or f"NSE Trading Holidays for the Calendar Year {year}",
            "holiday_count": len(normalized_holidays),
            "special_session_count": len(normalized_special),
        }

        # --- Phase 2: atomically replace state only after all validation passes ---
        self._holidays_by_year[year] = normalized_holidays
        self._special_sessions_by_year[year] = normalized_special
        self._circular_metadata[year] = new_metadata

        logger.info(
            "Loaded circular %s for year %d: %d holidays, %d special sessions",
            circular_ref,
            year,
            len(normalized_holidays),
            len(normalized_special),
        )

    def get_circular_metadata(self, year: int) -> Dict[str, Any]:
        """Retrieve stored circular metadata for a specific calendar year.

        Raises:
            HolidayCircularNotFoundError: If no circular is registered for ``year``.
        """
        if year not in self._circular_metadata:
            raise HolidayCircularNotFoundError(
                f"No circular metadata found for calendar year {year}",
                details={
                    "requested_year": year,
                    "available_years": sorted(self._circular_metadata.keys()),
                },
            )
        return dict(self._circular_metadata[year])

    def get_all_holidays_for_year(self, year: int) -> Dict[date, str]:
        """Return a copy of all registered holidays for a specific year."""
        return dict(self._holidays_by_year.get(year, {}))

    def get_all_special_sessions_for_year(self, year: int) -> Dict[date, str]:
        """Return a copy of all registered special sessions for a specific year."""
        return dict(self._special_sessions_by_year.get(year, {}))

    @property
    def covered_years(self) -> List[int]:
        """Return a sorted list of calendar years with registered circulars."""
        return sorted(self._circular_metadata.keys())


# Singleton calendar instance (built-in schedules pre-loaded)
nse_calendar = NSECalendar(load_builtin=True)
