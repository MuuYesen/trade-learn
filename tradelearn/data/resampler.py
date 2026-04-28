"""Data resampling logic for multi-timeframe backtesting."""

from __future__ import annotations

import pandas as pd

_MICROSECONDS = 1
_SECONDS = 2
_MINUTES = 3
_DAYS = 4
_WEEKS = 5
_MONTHS = 6
_YEARS = 7


def resample_frame(
    df: pd.DataFrame, timeframe: int, compression: int = 1
) -> pd.DataFrame:
    """Resample a OHLCV DataFrame into a higher timeframe."""
    rule = _get_pandas_rule(timeframe, compression)
    
    # Define aggregation mapping
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Include other columns if they exist
    for col in df.columns:
        if col not in agg_map:
            agg_map[col] = "last"

    resampled = df.resample(rule, label='right', closed='right').agg(agg_map)
    
    # Drop rows with no data (NaNs in OHLC)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    
    return resampled


def _get_pandas_rule(timeframe: int, compression: int) -> str:
    """Convert TimeFrame constant to pandas frequency string."""
    base = ""
    if timeframe == _MICROSECONDS:
        base = "us"
    elif timeframe == _SECONDS:
        base = "s"
    elif timeframe == _MINUTES:
        base = "min"
    elif timeframe == _DAYS:
        base = "D"
    elif timeframe == _WEEKS:
        base = "W"
    elif timeframe == _MONTHS:
        base = "ME"  # Modern pandas uses ME for Month End
    elif timeframe == _YEARS:
        base = "YE"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return f"{compression}{base}"
