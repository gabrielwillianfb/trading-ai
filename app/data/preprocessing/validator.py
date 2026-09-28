"""Data integrity validation and time-series gap detection."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from app.data.collectors.historical import timeframe_to_seconds


@dataclass
class ValidationReport:
    """Detailed summary of data validation checks."""

    is_valid: bool
    total_candles: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    gaps_count: int = 0
    gaps_details: List[Dict[str, Any]] = field(default_factory=list)
    anomalies_count: int = 0
    anomalies_details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format human-readable validation summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"Validation Status: {status}",
            f"Total Candles: {self.total_candles}",
            f"Time Range: {self.start_time} -> {self.end_time}",
            f"Gaps Detected: {self.gaps_count}",
            f"Anomalies Detected: {self.anomalies_count}",
        ]
        if self.gaps_details:
            lines.append("Gap Details (first 5):")
            for g in self.gaps_details[:5]:
                lines.append(f"  - Missing {g['missing_candles']} bars between {g['gap_start']} and {g['gap_end']}")
        if self.anomalies_details:
            lines.append("Anomalies (first 5):")
            for a in self.anomalies_details[:5]:
                lines.append(f"  - {a}")
        return "\n".join(lines)


class DataValidator:
    """Validates candlestick series for gaps, missing data, and price anomalies."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, timeframe: str) -> ValidationReport:
        """Validate candle DataFrame for integrity, ordering, and time gaps.

        Args:
            df: DataFrame with datetime index (UTC) and OHLCV columns.
            timeframe: Candle interval (e.g. '15m').

        Returns:
            ValidationReport detailing health of dataset.
        """
        if df.empty:
            return ValidationReport(
                is_valid=False,
                total_candles=0,
                anomalies_details=["DataFrame is empty."],
                anomalies_count=1,
            )

        anomalies: List[str] = []
        gaps: List[Dict[str, Any]] = []
        expected_step_seconds = timeframe_to_seconds(timeframe)
        expected_delta = timedelta(seconds=expected_step_seconds)

        # 1. Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            anomalies.append("DataFrame index must be a DatetimeIndex.")
            return ValidationReport(
                is_valid=False,
                total_candles=len(df),
                anomalies_details=anomalies,
                anomalies_count=len(anomalies),
            )

        # 2. Check for NaN values
        null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                anomalies.append(f"Column '{col}' contains {count} NaN/null values.")

        # 3. Check price sanity
        invalid_prices = df[(df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)]
        if len(invalid_prices) > 0:
            anomalies.append(f"Found {len(invalid_prices)} candles with non-positive prices.")

        invalid_hl = df[df["high"] < df["low"]]
        if len(invalid_hl) > 0:
            anomalies.append(f"Found {len(invalid_hl)} candles where High < Low.")

        invalid_high = df[(df["high"] < df["open"]) | (df["high"] < df["close"])]
        if len(invalid_high) > 0:
            anomalies.append(f"Found {len(invalid_high)} candles where High < Open or High < Close.")

        invalid_low = df[(df["low"] > df["open"]) | (df["low"] > df["close"])]
        if len(invalid_low) > 0:
            anomalies.append(f"Found {len(invalid_low)} candles where Low > Open or Low > Close.")

        invalid_volume = df[df["volume"] < 0]
        if len(invalid_volume) > 0:
            anomalies.append(f"Found {len(invalid_volume)} candles with negative volume.")

        # 4. Check for duplicate timestamps
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            anomalies.append(f"Found {duplicates} duplicate timestamps in index.")

        # 5. Check for time gaps
        timestamps = df.index.sort_values()
        for i in range(1, len(timestamps)):
            diff = timestamps[i] - timestamps[i - 1]
            diff_seconds = diff.total_seconds()

            if diff_seconds > expected_step_seconds:
                missing_count = int(round(diff_seconds / expected_step_seconds)) - 1
                gaps.append(
                    {
                        "gap_start": timestamps[i - 1].isoformat(),
                        "gap_end": timestamps[i].isoformat(),
                        "duration_seconds": diff_seconds,
                        "missing_candles": missing_count,
                    }
                )

        is_valid = len(anomalies) == 0 and len(gaps) == 0

        return ValidationReport(
            is_valid=is_valid,
            total_candles=len(df),
            start_time=df.index.min().to_pydatetime() if len(df) > 0 else None,
            end_time=df.index.max().to_pydatetime() if len(df) > 0 else None,
            gaps_count=len(gaps),
            gaps_details=gaps,
            anomalies_count=len(anomalies),
            anomalies_details=anomalies,
        )
