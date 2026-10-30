"""PyneCore-backed TradingView indicator adapters."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from threading import RLock

import pandas as pd
import pynecore.lib as pyne_lib
import pynecore.lib.ta as pyne_ta
from pynecore.core.function_isolation import isolate_function
from pynecore.core.function_isolation import reset as reset_pyne_functions
from pynecore.core.series import inline_series
from pynecore.types.na import NA

_PYNE_LOCK = RLock()


def _isolated_run(func):
    """Serialize access to PyneCore's global bar and function-isolation state."""

    @wraps(func)
    def run(*args, **kwargs):
        """Run one adapter while restoring PyneCore's shared state."""
        with _PYNE_LOCK:
            fields = ("bar_index", "open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4")
            previous = {key: getattr(pyne_lib, key) for key in fields}
            try:
                return func(*args, **kwargs)
            finally:
                for key, value in previous.items():
                    setattr(pyne_lib, key, value)

    return run


def _aligned(reference: pd.Series, **inputs: pd.Series) -> None:
    """Require each input series to share the reference index."""
    for name, value in inputs.items():
        if not pd.Series(value).index.equals(reference.index):
            raise ValueError(f"{name} index must exactly match the reference index")


def _anchor_series(reference: pd.Series, anchor: pd.Series | None) -> pd.Series:
    """Validate an anchor series or construct its default initial anchor."""
    if anchor is None:
        return (
            pd.Series([True] + [False] * (len(reference) - 1), index=reference.index, dtype=bool)
            if len(reference)
            else pd.Series(index=reference.index, dtype=bool)
        )
    anchor = pd.Series(anchor)
    _aligned(reference, anchor=anchor)
    if not pd.api.types.is_bool_dtype(anchor.dtype) or anchor.isna().any():
        raise ValueError("anchor must contain nonmissing bool values")
    return anchor


def _accdist(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute the accumulation/distribution line from OHLCV inputs."""
    return _run_ohlcv_indicator(high, low, close, volume, "accdist", lambda fn: fn())


def _pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute price-volume trend from close and volume inputs."""
    return _run_ohlcv_indicator(close, close, close, volume, "pvt", lambda fn: fn())


def _wpr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Compute Williams percent range for the requested window."""
    return _run_ohlcv_indicator(high, low, close, None, "wpr", lambda fn: fn(length))


def _vwma(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """Compute the volume-weighted moving average."""
    return _run_ohlcv_indicator(
        close, close, close, volume, "vwma", lambda fn: fn(pyne_lib.close, length)
    )


def _correlation(source1: pd.Series, source2: pd.Series, length: int = 20) -> pd.Series:
    """Compute rolling correlation for two aligned input series."""
    source1, source2 = pd.Series(source1), pd.Series(source2)
    _aligned(source1, source2=source2)
    return _run_source_indicator(
        source1,
        "correlation",
        lambda source, fn: fn(source, _pyne_value(source2.iloc[pyne_lib.bar_index]), length),
    )


def _pivot_point_levels(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    anchor: pd.Series,
    type: str = "Traditional",
    developing: bool = False,
) -> pd.DataFrame:
    """Native PyneCore pivot levels from completed periods (unless developing).

    Unsupported native levels are NaN. The first completed period becomes
    available only at the next anchor. Woodie uses the current period's open.
    """
    if type.lower() not in {"traditional", "fibonacci", "woodie", "classic", "dm", "camarilla"}:
        raise ValueError("unknown pivot type")
    close = pd.Series(close)
    anchor = _anchor_series(close, anchor)
    rows = _run_ohlcv_values(
        high,
        low,
        close,
        None,
        "pivot_point_levels",
        lambda fn: fn(type, bool(anchor.iloc[pyne_lib.bar_index]), developing),
        open=open,
    )
    return pd.DataFrame(
        rows,
        index=close.index,
        columns=("p", "r1", "s1", "r2", "s2", "r3", "s3", "r4", "s4", "r5", "s5"),
        dtype=float,
    )


def _sma(close: pd.Series, length: int = 20) -> pd.Series:
    """Compute a simple moving average for the source series."""
    return _run_source_indicator(close, "sma", lambda source, fn: fn(source, length))


def _ema(close: pd.Series, length: int = 20) -> pd.Series:
    """Compute an exponential moving average for the source series."""
    return _run_source_indicator(close, "ema", lambda source, fn: fn(source, length))


def _wma(close: pd.Series, length: int = 20) -> pd.Series:
    """Compute a weighted moving average for the source series."""
    return _run_source_indicator(close, "wma", lambda source, fn: fn(source, length))


def _rma(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute a running moving average for the source series."""
    return _run_source_indicator(close, "rma", lambda source, fn: fn(source, length))


def _hma(close: pd.Series, length: int = 20) -> pd.Series:
    """Compute a Hull moving average for the source series."""
    return _run_source_indicator(close, "hma", lambda source, fn: fn(source, length))


def _swma(close: pd.Series) -> pd.Series:
    """Compute a symmetrically weighted moving average."""
    return _run_source_indicator(close, "swma", lambda source, fn: fn(source))


def _alma(
    close: pd.Series,
    length: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
    floor: bool = False,
) -> pd.Series:
    """Compute an Arnaud Legoux moving average with the supplied parameters."""
    return _run_source_indicator(
        close,
        "alma",
        lambda source, fn: fn(source, length, offset, sigma, floor),
    )


def _stdev(close: pd.Series, length: int = 20, biased: bool = True) -> pd.Series:
    """Compute rolling standard deviation for the source series."""
    return _run_source_indicator(close, "stdev", lambda source, fn: fn(source, length, biased))


def _variance(close: pd.Series, length: int = 20, biased: bool = True) -> pd.Series:
    """Compute rolling variance for the source series."""
    return _run_source_indicator(close, "variance", lambda source, fn: fn(source, length, biased))


def _roc(close: pd.Series, length: int = 10) -> pd.Series:
    """Compute rate of change over the requested lookback."""
    return _run_source_indicator(close, "roc", lambda source, fn: fn(source, length))


def _mom(close: pd.Series, length: int = 10) -> pd.Series:
    """Compute momentum over the requested lookback."""
    return _run_source_indicator(close, "mom", lambda source, fn: fn(source, length))


def _cmo(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute the Chande momentum oscillator."""
    return _run_source_indicator(close, "cmo", lambda source, fn: fn(source, length))


def _tsi(close: pd.Series, short_length: int = 13, long_length: int = 25) -> pd.Series:
    """Compute the true strength index for the source series."""
    return _run_source_indicator(
        close,
        "tsi",
        lambda source, fn: fn(source, short_length, long_length),
    )


def _change(close: pd.Series, length: int = 1) -> pd.Series:
    """Compute the source change over the requested lookback."""
    return _run_source_indicator(close, "change", lambda source, fn: fn(source, length))


def _cum(close: pd.Series) -> pd.Series:
    """Compute the cumulative total of the source series."""
    return _run_source_indicator(close, "cum", lambda source, fn: fn(source))


def _linreg(close: pd.Series, length: int = 14, offset: int = 0) -> pd.Series:
    """Compute a rolling linear-regression estimate."""
    return _run_source_indicator(close, "linreg", lambda source, fn: fn(source, length, offset))


def _bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger band values for the source series."""
    frame = _run_source_frame(
        close,
        "bb",
        lambda source, fn: fn(source, length, std),
        columns=("mid", "upper", "lower"),
    )
    return frame[["lower", "mid", "upper"]]


def _bb(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Return Bollinger band values using the requested parameters."""
    return _run_source_frame(
        close,
        "bb",
        lambda source, fn: fn(source, length, mult),
        columns=("mid", "upper", "lower"),
    )


def _bbw(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    """Compute Bollinger band width for the source series."""
    return _run_source_indicator(close, "bbw", lambda source, fn: fn(source, length, mult))


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Compute relative strength index for the source series."""
    return _run_source_indicator(close, "rsi", lambda source, fn: fn(source, length))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD, signal, and histogram series."""
    return _run_source_frame(
        close,
        "macd",
        lambda source, fn: fn(source, fast, slow, signal),
        columns=("macd", "signal", "hist"),
    )


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Compute average true range from aligned price inputs."""
    return _run_ohlcv_indicator(high, low, close, None, "atr", lambda fn: fn(length))


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.DataFrame:
    """Compute average directional movement strength."""
    frame = _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "dmi",
        lambda fn: fn(length, length),
        columns=("dmp", "dmn", "adx"),
    )
    return frame[["adx", "dmp", "dmn"]]


def _dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smoothing: int = 14,
) -> pd.DataFrame:
    """Compute directional movement and ADX outputs."""
    return _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "dmi",
        lambda fn: fn(length, smoothing),
        columns=("dmp", "dmn", "adx"),
    )


def _tr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    handle_na: bool = False,
) -> pd.Series:
    """Compute true range from aligned high, low, and close inputs."""
    return _run_ohlcv_indicator(high, low, close, None, "tr", lambda fn: fn(handle_na))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute on-balance volume from price and volume inputs."""
    close_series = pd.Series(close)
    return _run_ohlcv_indicator(
        close_series,
        close_series,
        close_series,
        volume,
        "obv",
        lambda fn: fn(),
    )


def _sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    start: float = 0.02,
    inc: float = 0.02,
    max: float = 0.2,
) -> pd.Series:
    """Compute parabolic stop-and-reverse values."""
    return _run_ohlcv_indicator(high, low, close, None, "sar", lambda fn: fn(start, inc, max))


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Compute the stochastic oscillator from price inputs."""
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "stoch",
        lambda fn: fn(pyne_lib.close, pyne_lib.high, pyne_lib.low, length),
    )


def _kc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    use_true_range: bool = True,
) -> pd.DataFrame:
    """Compute Keltner channel bands from price inputs."""
    return _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "kc",
        lambda fn: fn(pyne_lib.close, length, mult, use_true_range),
        columns=("mid", "upper", "lower"),
    )


def _kcw(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    use_true_range: bool = True,
) -> pd.Series:
    """Compute Keltner channel width from price inputs."""
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "kcw",
        lambda fn: fn(pyne_lib.close, length, mult, use_true_range),
    )


def _cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
) -> pd.Series:
    """Compute commodity channel index from aligned price inputs."""
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        None,
        "cci",
        lambda fn: fn(pyne_lib.hlc3, length),
    )


def _mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Compute money flow index from aligned OHLCV inputs."""
    return _run_ohlcv_indicator(
        high,
        low,
        close,
        volume,
        "mfi",
        lambda fn: fn(pyne_lib.hlc3, length),
    )


def _vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    source: pd.Series | None = None,
    anchor: pd.Series | None = None,
    stdev_mult: float | None = None,
) -> pd.Series | pd.DataFrame:
    """VWAP with optional explicit source, reset bars, and weighted bands.

    Defaults retain close-price accumulation starting at the first bar.
    Explicit anchors produce NaN until the first True bar.
    """
    close = pd.Series(close)
    source = close if source is None else pd.Series(source)
    _aligned(close, source=source)
    anchor = _anchor_series(close, anchor)
    rows = _run_ohlcv_values(
        high,
        low,
        close,
        volume,
        "vwap",
        lambda fn: fn(
            _pyne_value(source.iloc[pyne_lib.bar_index]),
            anchor=bool(anchor.iloc[pyne_lib.bar_index]),
            stdev_mult=stdev_mult,
        ),
    )
    if stdev_mult is not None:
        return pd.DataFrame(
            rows, index=close.index, columns=("vwap", "upper", "lower"), dtype=float
        )
    return pd.Series(rows, index=close.index, name="vwap", dtype=float)


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Compute supertrend values and direction from price inputs."""
    frame = _run_ohlcv_frame(
        high,
        low,
        close,
        None,
        "supertrend",
        lambda fn: fn(multiplier, length),
        columns=("supertrend", "direction"),
    )
    direction = frame["direction"]
    frame["long"] = frame["supertrend"].where(direction < 0)
    frame["short"] = frame["supertrend"].where(direction > 0)
    return frame[["supertrend", "direction", "long", "short"]]


def _ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
) -> pd.DataFrame:
    """Compute the supported Ichimoku indicator outputs."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_line + kijun_line) / 2).shift(kijun)
    span_b = ((high.rolling(senkou).max() + low.rolling(senkou).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    return pd.DataFrame(
        {
            "span_a": span_a,
            "span_b": span_b,
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "chikou": chikou,
        },
        index=close.index,
    )


def _run_source_indicator(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
) -> pd.Series:
    """Run a source-series indicator and return its primary output."""
    values = _run_source_values(source, name, call)
    return pd.Series(
        values, index=pd.Series(source).index, name=getattr(source, "name", None), dtype=float
    )


def _run_source_frame(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Run a source-series indicator and return aligned output columns."""
    rows = _run_source_values(source, name, call)
    return pd.DataFrame(rows, index=pd.Series(source).index, columns=list(columns), dtype=float)


@_isolated_run
def _run_source_values(
    source: pd.Series,
    name: str,
    call: Callable[[object, Callable], object],
) -> list[object]:
    """Execute a source-series indicator over successive input bars."""
    series = pd.Series(source)
    reset_pyne_functions()
    fn = isolate_function(getattr(pyne_ta, name), name, f"tradelearn.tv.{name}")
    values: list[object] = []
    for index, value in enumerate(series):
        pyne_lib.bar_index = index
        source_value = _pyne_value(value)
        source_series = inline_series(source_value, 0)
        values.append(_to_pandas_value(call(source_series, fn)))
    return values


def _run_ohlcv_indicator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
) -> pd.Series:
    """Run an OHLCV indicator and return its primary output."""
    values = _run_ohlcv_values(high, low, close, volume, name, call)
    return pd.Series(values, index=pd.Series(close).index, name=name, dtype=float)


def _run_ohlcv_frame(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Run an OHLCV indicator and return aligned output columns."""
    rows = _run_ohlcv_values(high, low, close, volume, name, call)
    return pd.DataFrame(rows, index=pd.Series(close).index, columns=list(columns), dtype=float)


@_isolated_run
def _run_ohlcv_values(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series | None,
    name: str,
    call: Callable[[Callable], object],
    *,
    open: pd.Series | None = None,
) -> list[object]:
    """Execute an OHLCV indicator over successive input bars."""
    high_series, low_series, close_series = pd.Series(high), pd.Series(low), pd.Series(close)
    volume_series = (
        pd.Series(1.0, index=close_series.index) if volume is None else pd.Series(volume)
    )
    open_series = close_series if open is None else pd.Series(open)
    _aligned(close_series, high=high_series, low=low_series, volume=volume_series, open=open_series)
    reset_pyne_functions()
    fn = isolate_function(getattr(pyne_ta, name), name, f"tradelearn.tv.{name}")
    values: list[object] = []
    for index, (high_value, low_value, close_value, volume_value, open_value) in enumerate(
        zip(high_series, low_series, close_series, volume_series, open_series, strict=True)
    ):
        _set_pyne_ohlcv(index, high_value, low_value, close_value, volume_value, open_value)
        values.append(_to_pandas_value(call(fn)))
    return values


def _set_pyne_ohlcv(
    index: int,
    high: object,
    low: object,
    close: object,
    volume: object,
    open: object,
) -> None:
    """Set PyneCore's current bar fields from aligned input values."""
    open_value = _pyne_value(open)
    high_value = _pyne_value(high)
    low_value = _pyne_value(low)
    close_value = _pyne_value(close)
    volume_value = _pyne_value(volume)
    pyne_lib.bar_index = index
    pyne_lib.open = open_value
    pyne_lib.high = high_value
    pyne_lib.low = low_value
    pyne_lib.close = close_value
    pyne_lib.volume = volume_value
    pyne_lib.hl2 = (
        NA(float)
        if isinstance(high_value, NA) or isinstance(low_value, NA)
        else (high_value + low_value) / 2
    )
    pyne_lib.hlc3 = (
        NA(float)
        if isinstance(high_value, NA) or isinstance(low_value, NA) or isinstance(close_value, NA)
        else (high_value + low_value + close_value) / 3
    )
    pyne_lib.ohlc4 = (
        NA(float)
        if isinstance(high_value, NA)
        or isinstance(low_value, NA)
        or isinstance(close_value, NA)
        or isinstance(open_value, NA)
        else (open_value + high_value + low_value + close_value) / 4
    )


def _pyne_value(value: object) -> float | NA:
    """Convert a missing input to PyneCore's missing-value representation."""
    if pd.isna(value):
        return NA(float)
    return float(value)


def _to_pandas_value(value: object) -> object:
    """Convert a PyneCore output to a pandas-compatible value."""
    if isinstance(value, NA):
        return None
    if isinstance(value, list):
        return [_to_pandas_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_pandas_value(item) for item in value)
    return value


__all__ = [
    "_accdist",
    "_pvt",
    "_wpr",
    "_vwma",
    "_correlation",
    "_pivot_point_levels",
    "_adx",
    "_alma",
    "_atr",
    "_bb",
    "_bbands",
    "_bbw",
    "_cci",
    "_change",
    "_cmo",
    "_cum",
    "_dmi",
    "_ema",
    "_hma",
    "_ichimoku",
    "_kc",
    "_kcw",
    "_linreg",
    "_macd",
    "_mfi",
    "_mom",
    "_obv",
    "_rma",
    "_roc",
    "_rsi",
    "_run_source_indicator",
    "_sar",
    "_sma",
    "_stdev",
    "_stoch",
    "_supertrend",
    "_swma",
    "_tr",
    "_tsi",
    "_variance",
    "_vwap",
    "_wma",
]
