"""Data resampling logic for multi-timeframe backtesting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradelearn.backtest.engine import TimeFrame


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
    if timeframe == TimeFrame.MicroSeconds:
        base = "us"
    elif timeframe == TimeFrame.Seconds:
        base = "s"
    elif timeframe == TimeFrame.Minutes:
        base = "min"
    elif timeframe == TimeFrame.Days:
        base = "D"
    elif timeframe == TimeFrame.Weeks:
        base = "W"
    elif timeframe == TimeFrame.Months:
        base = "ME"  # Modern pandas uses ME for Month End
    elif timeframe == TimeFrame.Years:
        base = "YE"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    return f"{compression}{base}"
