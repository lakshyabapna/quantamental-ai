"""Quantamental AI - Data Provider Base Abstraction & Normalized Schemas.

Defines the abstract interface for all market data providers, normalized price
and corporate action schemas, adjustment modes, and provider metadata.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class AdjustmentMode(str, Enum):
    """Data adjustment mode configuration."""

    AUTO_ADJUST_FALSE = "auto_adjust_false"
    AUTO_ADJUST_TRUE = "auto_adjust_true"


class CorporateActionType(str, Enum):
    """Types of corporate actions affecting share counts or nominal prices."""

    DIVIDEND = "DIVIDEND"
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    RIGHTS = "RIGHTS"


class CorporateActionRecord(BaseModel):
    """Normalized corporate action event schema."""

    symbol: str = Field(..., description="NSE equity ticker symbol (e.g. 'RELIANCE.NS')")
    ex_date: date = Field(..., description="Ex-dividend or ex-split execution date")
    action_type: CorporateActionType = Field(..., description="Category of corporate action")
    value: float = Field(..., description="Numerical value (e.g. cash dividend in INR or split multiplier)")
    ratio: Optional[str] = Field(default=None, description="Optional ratio string (e.g. '1:1', '5:1')")
    currency: str = Field(default="INR", description="Currency code")
    description: Optional[str] = Field(default=None, description="Descriptive notes or circular details")
    source: str = Field(..., description="Origin of corporate action event (e.g. 'yfinance', 'nse_circular')")


class NormalizedOHLCVRecord(BaseModel):
    """Provider-agnostic normalized price and volume record.

    Conforms to Data Strategy Section 5.1 Internal Normalized Price Record.
    """

    symbol: str = Field(..., description="NSE equity ticker symbol")
    trading_date: date = Field(..., description="Trading date in ISO format YYYY-MM-DD")
    open_nominal: Optional[float] = Field(default=None, description="Nominal unadjusted open price")
    high_nominal: Optional[float] = Field(default=None, description="Nominal unadjusted high price")
    low_nominal: Optional[float] = Field(default=None, description="Nominal unadjusted low price")
    close_nominal: Optional[float] = Field(default=None, description="Nominal unadjusted close price")
    adj_close: float = Field(..., gt=0.0, description="Normalized back-adjusted close price (splits + dividends)")
    volume: int = Field(..., ge=0, description="Trading volume in shares")
    split_factor: Optional[float] = Field(default=1.0, gt=0.0, description="Split multiplier")
    dividend_amount: Optional[float] = Field(default=0.0, ge=0.0, description="Cash dividend amount in INR")
    is_stale: bool = Field(default=False, description="Flag indicating carried or unobserved session")
    data_provider: str = Field(..., description="Originating data provider identifier")
    adjustment_mode: AdjustmentMode = Field(..., description="Adjustment mode used for ingestion")
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when data was fetched",
    )

    @model_validator(mode="after")
    def validate_mode_consistency(self) -> "NormalizedOHLCVRecord":
        """Enforce strict separation between nominal and adjusted values per Data Strategy 5.2."""
        if self.adjustment_mode == AdjustmentMode.AUTO_ADJUST_FALSE:
            # When auto_adjust=False, nominal prices must be provided and positive
            nominal_fields = [
                ("open_nominal", self.open_nominal),
                ("high_nominal", self.high_nominal),
                ("low_nominal", self.low_nominal),
                ("close_nominal", self.close_nominal),
            ]
            for field_name, value in nominal_fields:
                if value is None or value <= 0.0:
                    raise ValueError(
                        f"Field '{field_name}' must be strictly positive when "
                        f"adjustment_mode is AUTO_ADJUST_FALSE (got {value})"
                    )

            # High must be >= Low
            if self.high_nominal is not None and self.low_nominal is not None:
                if self.high_nominal < self.low_nominal:
                    raise ValueError(
                        f"high_nominal ({self.high_nominal}) cannot be less than "
                        f"low_nominal ({self.low_nominal})"
                    )

        elif self.adjustment_mode == AdjustmentMode.AUTO_ADJUST_TRUE:
            # When auto_adjust=True, nominal prices are NOT available and must remain None
            if any(
                v is not None
                for v in (self.open_nominal, self.high_nominal, self.low_nominal, self.close_nominal)
            ):
                raise ValueError(
                    "Nominal fields (open_nominal, high_nominal, low_nominal, close_nominal) "
                    "must remain None when adjustment_mode is AUTO_ADJUST_TRUE. "
                    "Never populate nominal fields with adjusted prices."
                )

        return self


class ProviderMetadata(BaseModel):
    """Metadata detailing data provider capabilities and constraints."""

    provider_name: str = Field(..., description="Identifier name of the data provider")
    supported_adjustment_modes: List[AdjustmentMode] = Field(
        ..., description="List of adjustment modes supported by the provider"
    )
    supports_corporate_actions: bool = Field(
        ..., description="Whether provider exposes separate corporate actions stream"
    )
    supports_intraday: bool = Field(
        ..., description="Whether provider supports intraday time-series"
    )
    rate_limit_requests_per_minute: Optional[int] = Field(
        default=None, description="Provider rate limit requests per minute if applicable"
    )
    licensing_notice: str = Field(
        ..., description="Provider terms of service and compliance licensing disclosure"
    )


class BaseDataProvider(ABC):
    """Abstract Base Class defining the contract for all financial data providers."""

    REQUIRED_NORMALIZED_COLUMNS = [
        "symbol",
        "trading_date",
        "open_nominal",
        "high_nominal",
        "low_nominal",
        "close_nominal",
        "adj_close",
        "volume",
        "split_factor",
        "dividend_amount",
        "is_stale",
        "data_provider",
        "adjustment_mode",
        "retrieval_timestamp",
    ]

    @abstractmethod
    def fetch_historical_ohlcv(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        adjustment_mode: AdjustmentMode = AdjustmentMode.AUTO_ADJUST_FALSE,
    ) -> pd.DataFrame:
        """Fetch historical daily OHLCV series for an equity symbol over a date range.

        Args:
            symbol: Equity ticker symbol (e.g. 'RELIANCE.NS').
            start_date: Inclusive start date of historical observation window.
            end_date: Inclusive end date of historical observation window.
            adjustment_mode: Adjustment mode flag (AUTO_ADJUST_FALSE or AUTO_ADJUST_TRUE).

        Returns:
            pd.DataFrame conforming to NormalizedOHLCVRecord schema with columns:
            ['symbol', 'trading_date', 'open_nominal', 'high_nominal', 'low_nominal',
             'close_nominal', 'adj_close', 'volume', 'split_factor', 'dividend_amount',
             'is_stale', 'data_provider', 'adjustment_mode', 'retrieval_timestamp'].

        Raises:
            DataProviderException: If network, parsing, or data retrieval fails.
            ProviderValidationError: If data fails schema or non-negativity validation gates.
        """
        pass

    @abstractmethod
    def fetch_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[CorporateActionRecord]:
        """Fetch historical corporate actions (dividends, splits) for an equity symbol.

        Args:
            symbol: Equity ticker symbol.
            start_date: Inclusive start date of corporate action window.
            end_date: Inclusive end date of corporate action window.

        Returns:
            List of CorporateActionRecord instances sorted by ex_date ascending.

        Raises:
            DataProviderException: If corporate action retrieval fails.
        """
        pass

    @abstractmethod
    def fetch_trading_calendar(
        self,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """Fetch active trading dates from provider or exchange schedule.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            Sorted list of active trading dates.
        """
        pass

    @abstractmethod
    def get_provider_metadata(self) -> ProviderMetadata:
        """Return provider metadata, capability flags, and compliance notices."""
        pass

    def validate_normalized_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that a DataFrame conforms to the internal normalized schema.

        Checks (in order):
        1. Must be a pandas DataFrame.
        2. All REQUIRED_NORMALIZED_COLUMNS must be present.
        3. Empty DataFrames pass immediately after column checks — there
           is no data to validate.
        4. ``trading_date`` column must contain only ``datetime.date`` objects.
        5. No duplicate ``(symbol, trading_date)`` pairs.
        6. Mandatory numeric columns (``adj_close``, ``volume``) must not
           contain NaN or infinite values.
        7. Optional numeric columns (nominal prices, ``split_factor``,
           ``dividend_amount``) must not contain infinite values where present.
        8. ``adj_close`` must be strictly positive (> 0).
        9. ``volume`` must be non-negative (>= 0).
        10. ``split_factor`` must be strictly positive where non-null.
        11. ``dividend_amount`` must be non-negative where non-null.
        12. Full OHLC consistency (for rows where all four nominal prices
            are present):
            - high_nominal >= low_nominal
            - high_nominal >= open_nominal
            - high_nominal >= close_nominal
            - low_nominal  <= open_nominal
            - low_nominal  <= close_nominal
        13. Adjustment-mode consistency: AUTO_ADJUST_TRUE rows must have all
            nominal price columns as NaN/None; AUTO_ADJUST_FALSE rows must
            have all nominal price columns as finite positive numbers.

        Does NOT fill, forward-fill, or fabricate missing values.

        Args:
            df: DataFrame to validate.

        Returns:
            True when all checks pass.

        Raises:
            ProviderValidationError: On any schema or constraint violation.
        """
        import math
        from app.core.exceptions import ProviderValidationError

        if not isinstance(df, pd.DataFrame):
            raise ProviderValidationError(
                "Provided object is not a pandas DataFrame",
                details={"received_type": type(df).__name__},
            )

        # 1. Required columns
        missing_cols = set(self.REQUIRED_NORMALIZED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ProviderValidationError(
                f"DataFrame is missing required normalized columns: {sorted(missing_cols)}",
                details={"missing_columns": sorted(list(missing_cols))},
            )

        if df.empty:
            return True

        # 2. trading_date must contain datetime.date objects (strictly datetime.date, rejecting datetime/strings/NaT)
        invalid_date_mask = df["trading_date"].apply(
            lambda v: not isinstance(v, date) or isinstance(v, datetime)
        )
        if invalid_date_mask.any():
            bad_types = df.loc[invalid_date_mask, "trading_date"].apply(
                lambda v: type(v).__name__
            ).unique().tolist()
            raise ProviderValidationError(
                "Column 'trading_date' contains non-date values",
                details={"unexpected_types": bad_types},
            )

        # 3. No duplicate (symbol, trading_date) pairs
        dup_mask = df.duplicated(subset=["symbol", "trading_date"], keep=False)
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            raise ProviderValidationError(
                f"DataFrame contains {dup_count} rows with duplicate (symbol, trading_date) pairs",
                details={"duplicate_row_count": dup_count},
            )

        # 4. NaN / Inf checks on mandatory numeric columns
        mandatory_numeric = ["adj_close", "volume"]
        for col in mandatory_numeric:
            series = df[col]
            nan_count = int(series.isna().sum())
            if nan_count:
                raise ProviderValidationError(
                    f"Column '{col}' contains {nan_count} NaN values",
                    details={"column": col, "nan_count": nan_count},
                )
            # Inf check only for float columns
            if series.dtype.kind == "f":
                inf_count = int((~series.isfinite()).sum()) if hasattr(series, "isfinite") else int(
                    series.apply(lambda v: math.isinf(v) if v == v else False).sum()
                )
                try:
                    import numpy as np
                    inf_count = int(np.isinf(series).sum())
                except ImportError:
                    pass
                if inf_count:
                    raise ProviderValidationError(
                        f"Column '{col}' contains {inf_count} infinite values",
                        details={"column": col, "inf_count": inf_count},
                    )

        # Optional nominal price columns — check NaN/Inf only where values are present
        optional_numeric = ["open_nominal", "high_nominal", "low_nominal", "close_nominal",
                            "split_factor", "dividend_amount"]
        try:
            import numpy as np
            for col in optional_numeric:
                if col not in df.columns:
                    continue
                present = df[col].dropna()
                if not present.empty and present.dtype.kind == "f":
                    inf_count = int(np.isinf(present).sum())
                    if inf_count:
                        raise ProviderValidationError(
                            f"Column '{col}' contains {inf_count} infinite values",
                            details={"column": col, "inf_count": inf_count},
                        )
        except ImportError:
            pass  # numpy not available; skip inf check on optional columns

        # 5. adj_close strictly positive
        non_pos = int((df["adj_close"] <= 0.0).sum())
        if non_pos:
            raise ProviderValidationError(
                f"Found {non_pos} non-positive adj_close records in DataFrame",
                details={"non_positive_adj_close_count": non_pos},
            )

        # 6. volume non-negative
        neg_vol = int((df["volume"] < 0).sum())
        if neg_vol:
            raise ProviderValidationError(
                f"Found {neg_vol} negative volume records in DataFrame",
                details={"negative_volume_count": neg_vol},
            )

        # 7. split_factor strictly positive where present
        if "split_factor" in df.columns:
            sf_present = df["split_factor"].dropna()
            if not sf_present.empty and (sf_present <= 0.0).any():
                bad_sf = int((sf_present <= 0.0).sum())
                raise ProviderValidationError(
                    f"Found {bad_sf} non-positive split_factor records",
                    details={"non_positive_split_factor_count": bad_sf},
                )

        # 8. dividend_amount non-negative where present
        if "dividend_amount" in df.columns:
            da_present = df["dividend_amount"].dropna()
            if not da_present.empty and (da_present < 0.0).any():
                bad_da = int((da_present < 0.0).sum())
                raise ProviderValidationError(
                    f"Found {bad_da} negative dividend_amount records",
                    details={"negative_dividend_amount_count": bad_da},
                )

        # 9. OHLC consistency checks for rows where nominal prices are present.
        #    - high_nominal must be >= low_nominal
        #    - high_nominal must be >= open_nominal
        #    - high_nominal must be >= close_nominal
        #    - low_nominal  must be <= open_nominal
        #    - low_nominal  must be <= close_nominal
        nominal_cols = {"open_nominal", "high_nominal", "low_nominal", "close_nominal"}
        if nominal_cols.issubset(df.columns):
            all_present = (
                df["open_nominal"].notna()
                & df["high_nominal"].notna()
                & df["low_nominal"].notna()
                & df["close_nominal"].notna()
            )
            if all_present.any():
                ohlc = df.loc[all_present]

                bad_hl = int((ohlc["high_nominal"] < ohlc["low_nominal"]).sum())
                if bad_hl:
                    raise ProviderValidationError(
                        f"Found {bad_hl} rows where high_nominal < low_nominal",
                        details={"ohlc_violation": "high < low", "count": bad_hl},
                    )

                bad_ho = int((ohlc["high_nominal"] < ohlc["open_nominal"]).sum())
                if bad_ho:
                    raise ProviderValidationError(
                        f"Found {bad_ho} rows where high_nominal < open_nominal",
                        details={"ohlc_violation": "high < open", "count": bad_ho},
                    )

                bad_hc = int((ohlc["high_nominal"] < ohlc["close_nominal"]).sum())
                if bad_hc:
                    raise ProviderValidationError(
                        f"Found {bad_hc} rows where high_nominal < close_nominal",
                        details={"ohlc_violation": "high < close", "count": bad_hc},
                    )

                bad_lo = int((ohlc["low_nominal"] > ohlc["open_nominal"]).sum())
                if bad_lo:
                    raise ProviderValidationError(
                        f"Found {bad_lo} rows where low_nominal > open_nominal",
                        details={"ohlc_violation": "low > open", "count": bad_lo},
                    )

                bad_lc = int((ohlc["low_nominal"] > ohlc["close_nominal"]).sum())
                if bad_lc:
                    raise ProviderValidationError(
                        f"Found {bad_lc} rows where low_nominal > close_nominal",
                        details={"ohlc_violation": "low > close", "count": bad_lc},
                    )

        # 10. Adjustment-mode consistency
        if "adjustment_mode" in df.columns:
            auto_true_mask = df["adjustment_mode"] == AdjustmentMode.AUTO_ADJUST_TRUE.value
            auto_false_mask = df["adjustment_mode"] == AdjustmentMode.AUTO_ADJUST_FALSE.value

            # AUTO_ADJUST_TRUE rows must NOT have nominal prices populated
            if auto_true_mask.any() and nominal_cols.issubset(df.columns):
                at_rows = df.loc[auto_true_mask]
                has_nominal = (
                    at_rows["open_nominal"].notna()
                    | at_rows["high_nominal"].notna()
                    | at_rows["low_nominal"].notna()
                    | at_rows["close_nominal"].notna()
                )
                bad_at = int(has_nominal.sum())
                if bad_at:
                    raise ProviderValidationError(
                        f"Found {bad_at} AUTO_ADJUST_TRUE rows with non-null nominal price fields. "
                        "Nominal prices must remain null when adjustment_mode is AUTO_ADJUST_TRUE.",
                        details={"auto_adjust_true_nominal_violation_count": bad_at},
                    )

            # AUTO_ADJUST_FALSE rows must have all nominal prices as finite positive numbers
            if auto_false_mask.any() and nominal_cols.issubset(df.columns):
                af_rows = df.loc[auto_false_mask]
                for col in ["open_nominal", "high_nominal", "low_nominal", "close_nominal"]:
                    missing_or_bad = af_rows[col].isna() | (af_rows[col] <= 0.0)
                    bad_af = int(missing_or_bad.sum())
                    if bad_af:
                        raise ProviderValidationError(
                            f"Found {bad_af} AUTO_ADJUST_FALSE rows where '{col}' "
                            "is null or non-positive. All nominal prices must be strictly "
                            "positive when adjustment_mode is AUTO_ADJUST_FALSE.",
                            details={
                                "column": col,
                                "auto_adjust_false_violation_count": bad_af,
                            },
                        )

        return True
