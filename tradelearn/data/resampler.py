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
    rust_result = _try_rust_resample_frame(df, timeframe, compression)
    if rust_result is not None:
        return rust_result
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


def _try_rust_resample_frame(
    df: pd.DataFrame,
    timeframe: int,
    compression: int,
) -> pd.DataFrame | None:
    """Use the Rust fast path for plain fixed-width OHLCV frames."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_frame requires a DatetimeIndex")
    required_columns = ["open", "high", "low", "close", "volume"]
    if list(df.columns) != required_columns:
        return None
    period_seconds = _fixed_period_seconds(timeframe, compression)
    if period_seconds is None:
        return None
    if df.empty:
        return df.copy()

    timestamps = df.index
    if timestamps.tz is None:
        timestamp_seconds = timestamps.tz_localize("UTC").asi8 // 1_000_000_000
        tz = "UTC"
    else:
        timestamp_seconds = timestamps.tz_convert("UTC").asi8 // 1_000_000_000
        tz = timestamps.tz
    try:
        labels, opens, highs, lows, closes, volumes = _rust_resample_ohlcv(
            timestamp_seconds.astype("int64", copy=False).tolist(),
            df["open"].to_numpy(dtype="float64", copy=False).tolist(),
            df["high"].to_numpy(dtype="float64", copy=False).tolist(),
            df["low"].to_numpy(dtype="float64", copy=False).tolist(),
            df["close"].to_numpy(dtype="float64", copy=False).tolist(),
            df["volume"].to_numpy(dtype="float64", copy=False).tolist(),
            period_seconds,
        )
    except (ImportError, AttributeError):
        return None
    index = pd.to_datetime(labels, unit="s", utc=True)
    if tz != "UTC":
        index = index.tz_convert(tz)
    result = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )
    try:
        result["volume"] = result["volume"].astype(df["volume"].dtype, copy=False)
    except (TypeError, ValueError):
        pass
    return result


def _rust_resample_ohlcv(
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    period_seconds: int,
):
    from tradelearn._rust import resample_ohlcv

    return resample_ohlcv(timestamps, opens, highs, lows, closes, volumes, period_seconds)


def _fixed_period_seconds(timeframe: int, compression: int) -> int | None:
    if compression <= 0:
        raise ValueError("compression must be a positive integer")
    if timeframe == _MICROSECONDS:
        return None
    if timeframe == _SECONDS:
        return compression
    if timeframe == _MINUTES:
        return compression * 60
    if timeframe == _DAYS:
        return compression * 86_400
    return None


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
