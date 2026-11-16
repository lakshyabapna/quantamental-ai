"""Quantamental AI - Yahoo Finance Data Provider Implementation.

Implements BaseDataProvider using yfinance, providing normalized OHLCV time-series,
corporate action extraction, and compliance licensing disclosures.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Any, List, Optional
import zoneinfo
import numpy as np
import pandas as pd
import yfinance as yf

from app.core.exceptions import (
    DataProviderException,
    ProviderConnectionError,
    ProviderDataNotFoundError,
    ProviderValidationError,
)
from app.core.logger import get_logger
from app.data_engine.base import (
    AdjustmentMode,
    BaseDataProvider,
    CorporateActionRecord,
    CorporateActionType,
    ProviderMetadata,
)

logger = get_logger("quantamental.yfinance")

KOLKATA_TZ = zoneinfo.ZoneInfo("Asia/Kolkata")


class YFinanceDataProvider(BaseDataProvider):
    """Yahoo Finance market data provider adapter.

    Enforces strict separation between nominal and adjusted values per Data Strategy 5.2.
    """

    PROVIDER_NAME = "yfinance"
    LICENSING_NOTICE = (
        "Yahoo Finance data is accessed via unofficial API integration and utilized "
        "strictly for personal, individual, non-commercial quantitative finance research. "
        "Redistribution, public serving, or commercial deployment requires formal licensing "
        "from relevant exchange and data vendors."
    )

    def __init__(self) -> None:
        """Initialize YFinance provider adapter."""
        pass

    def get_provider_metadata(self) -> ProviderMetadata:
        """Return provider metadata, capability flags, and compliance notices."""
        return ProviderMetadata(
            provider_name=self.PROVIDER_NAME,
            supported_adjustment_modes=[
                AdjustmentMode.AUTO_ADJUST_FALSE,
                AdjustmentMode.AUTO_ADJUST_TRUE,
            ],
            supports_corporate_actions=True,
            supports_intraday=False,  # Intraday functionality not implemented in this phase
            rate_limit_requests_per_minute=None,  # Provider rate limit is undocumented; do not advertise arbitrary values
            licensing_notice=self.LICENSING_NOTICE,
        )

    def _validate_adjustment_mode(self, mode: Any) -> AdjustmentMode:
        """Explicitly validate that adjustment_mode is a supported AdjustmentMode."""
        if isinstance(mode, AdjustmentMode):
            return mode
        if isinstance(mode, str):
            try:
                return AdjustmentMode(mode)
            except ValueError:
                pass
        raise ProviderValidationError(
            f"Invalid adjustment_mode '{mode}'; expected AdjustmentMode.AUTO_ADJUST_FALSE or AUTO_ADJUST_TRUE",
            details={"received_mode": str(mode)},
        )

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
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            adjustment_mode: AUTO_ADJUST_FALSE (default) or AUTO_ADJUST_TRUE.

        Returns:
            pd.DataFrame conforming to the normalized schema.
        """
        adj_mode = self._validate_adjustment_mode(adjustment_mode)

        if start_date > end_date:
            raise ProviderValidationError(
                f"start_date ({start_date}) cannot be greater than end_date ({end_date})",
                details={"start_date": str(start_date), "end_date": str(end_date)},
            )

        # yfinance historical end date parameter is exclusive; add 1 day to include end_date
        yf_start = start_date.isoformat()
        yf_end = (end_date + timedelta(days=1)).isoformat()
        is_auto_adjust = adj_mode == AdjustmentMode.AUTO_ADJUST_TRUE

        try:
            ticker = yf.Ticker(symbol)
            raw_df = ticker.history(
                start=yf_start,
                end=yf_end,
                auto_adjust=is_auto_adjust,
                actions=True,
            )
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to fetch historical data for {symbol} from Yahoo Finance",
                details={"symbol": symbol, "error": str(exc)},
            ) from exc

        if raw_df is None or raw_df.empty:
            logger.warning("No data returned by Yahoo Finance for symbol %s in [%s, %s]", symbol, start_date, end_date)
            return pd.DataFrame(columns=self.REQUIRED_NORMALIZED_COLUMNS)

        # Normalize dates consistently to Asia/Kolkata exchange timezone
        raw_df = raw_df.copy()
        if hasattr(raw_df.index, "tz") and raw_df.index.tz is not None:
            raw_dates = raw_df.index.tz_convert(KOLKATA_TZ).date
        else:
            raw_dates = pd.to_datetime(raw_df.index).date

        raw_df["trading_date"] = raw_dates
        mask = (raw_df["trading_date"] >= start_date) & (raw_df["trading_date"] <= end_date)
        raw_df = raw_df.loc[mask]

        if raw_df.empty:
            return pd.DataFrame(columns=self.REQUIRED_NORMALIZED_COLUMNS)

        records = []
        retrieval_ts = datetime.now(timezone.utc)

        for _, row in raw_df.iterrows():
            t_date = row["trading_date"]

            # Mandatory Volume check: never substitute 0 for missing volume
            if "Volume" not in row or pd.isna(row["Volume"]):
                raise ProviderValidationError(
                    f"Missing mandatory 'Volume' field on {t_date} for {symbol}",
                    details={"symbol": symbol, "date": str(t_date)},
                )
            try:
                raw_vol = float(row["Volume"])
                if np.isnan(raw_vol) or np.isinf(raw_vol) or raw_vol < 0:
                    raise ValueError()
                vol = int(raw_vol)
            except Exception as exc:
                raise ProviderValidationError(
                    f"Invalid non-integer or negative volume on {t_date} for {symbol}: {row['Volume']}",
                    details={"symbol": symbol, "date": str(t_date), "volume": str(row["Volume"])},
                ) from exc

            # Corporate action fields
            raw_split = row.get("Stock Splits")
            if pd.isna(raw_split):
                raise ProviderValidationError(
                    f"Corrupted or missing 'Stock Splits' entry on {t_date} for {symbol}",
                    details={"symbol": symbol, "date": str(t_date)},
                )
            split_val = float(raw_split)
            split_factor = split_val if split_val > 0.0 else 1.0

            raw_div = row.get("Dividends")
            if pd.isna(raw_div):
                raise ProviderValidationError(
                    f"Corrupted or missing 'Dividends' entry on {t_date} for {symbol}",
                    details={"symbol": symbol, "date": str(t_date)},
                )
            div_val = float(raw_div)
            dividend_amount = div_val if div_val >= 0.0 else 0.0

            # Mandatory Price validation
            if adj_mode == AdjustmentMode.AUTO_ADJUST_FALSE:
                # Nominal fields must be present, finite, and strictly positive
                for price_col in ["Open", "High", "Low", "Close"]:
                    val = row.get(price_col)
                    if val is None or pd.isna(val) or np.isinf(val) or float(val) <= 0.0:
                        raise ProviderValidationError(
                            f"Missing, non-positive or infinite nominal price '{price_col}' "
                            f"on {t_date} for {symbol}: {val}",
                            details={"symbol": symbol, "date": str(t_date), "field": price_col, "value": str(val)},
                        )

                open_nom = float(row["Open"])
                high_nom = float(row["High"])
                low_nom = float(row["Low"])
                close_nom = float(row["Close"])

                # In AUTO_ADJUST_FALSE: Adj Close is mandatory; do NOT substitute Close
                if "Adj Close" not in row or pd.isna(row["Adj Close"]):
                    raise ProviderValidationError(
                        f"Missing mandatory 'Adj Close' on {t_date} for {symbol} in AUTO_ADJUST_FALSE mode",
                        details={"symbol": symbol, "date": str(t_date)},
                    )
                raw_adj = row["Adj Close"]
                if np.isinf(raw_adj) or float(raw_adj) <= 0.0:
                    raise ProviderValidationError(
                        f"Invalid or non-positive 'Adj Close' on {t_date} for {symbol}: {raw_adj}",
                        details={"symbol": symbol, "date": str(t_date), "adj_close": str(raw_adj)},
                    )
                adj_close = float(raw_adj)

            else:
                # AUTO_ADJUST_TRUE: nominal fields must remain None
                open_nom = None
                high_nom = None
                low_nom = None
                close_nom = None

                # In AUTO_ADJUST_TRUE, Close is the back-adjusted price
                if "Close" not in row or pd.isna(row["Close"]):
                    raise ProviderValidationError(
                        f"Missing mandatory 'Close' on {t_date} for {symbol} in AUTO_ADJUST_TRUE mode",
                        details={"symbol": symbol, "date": str(t_date)},
                    )
                raw_close = row["Close"]
                if np.isinf(raw_close) or float(raw_close) <= 0.0:
                    raise ProviderValidationError(
                        f"Invalid or non-positive 'Close' on {t_date} for {symbol}: {raw_close}",
                        details={"symbol": symbol, "date": str(t_date), "close": str(raw_close)},
                    )
                adj_close = float(raw_close)

            records.append({
                "symbol": symbol,
                "trading_date": t_date,
                "open_nominal": open_nom,
                "high_nominal": high_nom,
                "low_nominal": low_nom,
                "close_nominal": close_nom,
                "adj_close": adj_close,
                "volume": vol,
                "split_factor": split_factor,
                "dividend_amount": dividend_amount,
                "is_stale": False,
                "data_provider": self.PROVIDER_NAME,
                "adjustment_mode": adj_mode.value,
                "retrieval_timestamp": retrieval_ts,
            })

        df = pd.DataFrame(records)
        # Provider-agnostic validation gate enforces OHLC consistency and schema invariants
        self.validate_normalized_dataframe(df)
        return df

    def fetch_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[CorporateActionRecord]:
        """Fetch historical corporate actions (dividends and splits) from dedicated streams.

        Corporate actions are strictly captured from dedicated provider streams,
        never inferred from adjusted OHLC alone.
        Dates are normalized consistently to Asia/Kolkata exchange timezone.
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            splits = ticker.splits
        except Exception as exc:
            raise ProviderConnectionError(
                f"Failed to fetch corporate actions for {symbol}",
                details={"symbol": symbol, "error": str(exc)},
            ) from exc

        actions: List[CorporateActionRecord] = []

        def _to_kolkata_date(ts: Any) -> date:
            if hasattr(ts, "tz") and ts.tz is not None:
                return ts.tz_convert(KOLKATA_TZ).date()
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                return ts.astimezone(KOLKATA_TZ).date()
            return pd.to_datetime(ts).date()

        # Process dividends
        if dividends is not None and not dividends.empty:
            for ts, value in dividends.items():
                if pd.isna(value) or np.isinf(value):
                    continue
                d = _to_kolkata_date(ts)
                if start_date <= d <= end_date and float(value) > 0:
                    actions.append(
                        CorporateActionRecord(
                            symbol=symbol,
                            ex_date=d,
                            action_type=CorporateActionType.DIVIDEND,
                            value=float(value),
                            currency="INR",
                            description=f"Cash Dividend of ₹{float(value):.2f}",
                            source=self.PROVIDER_NAME,
                        )
                    )

        # Process splits
        if splits is not None and not splits.empty:
            for ts, ratio in splits.items():
                if pd.isna(ratio) or np.isinf(ratio):
                    continue
                d = _to_kolkata_date(ts)
                if start_date <= d <= end_date and float(ratio) > 0:
                    actions.append(
                        CorporateActionRecord(
                            symbol=symbol,
                            ex_date=d,
                            action_type=CorporateActionType.SPLIT,
                            value=float(ratio),
                            ratio=f"{float(ratio):.1f}:1",
                            description=f"Stock Split factor {float(ratio):.2f}",
                            source=self.PROVIDER_NAME,
                        )
                    )

        actions.sort(key=lambda a: a.ex_date)
        return actions

    def fetch_trading_calendar(
        self,
        start_date: date,
        end_date: date,
    ) -> List[date]:
        """Fetch active trading dates using benchmark symbol (^NSEI)."""
        benchmark_df = self.fetch_historical_ohlcv(
            symbol="^NSEI",
            start_date=start_date,
            end_date=end_date,
            adjustment_mode=AdjustmentMode.AUTO_ADJUST_FALSE,
        )
        if benchmark_df.empty:
            return []
        return sorted(benchmark_df["trading_date"].tolist())
