"""Backtrader-style indicator aliases backed by pandas computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradelearn.backtest import DataFeed, LineSeries

SUPPORTED_INDICATOR_ALIASES = (
    "SMA",
    "SimpleMovingAverage",
    "MovingAverageSimple",
    "EMA",
    "ExponentialMovingAverage",
    "WMA",
    "WeightedMovingAverage",
    "RSI",
    "RelativeStrengthIndex",
    "MACD",
    "BollingerBands",
    "BBands",
    "Highest",
    "Lowest",
    "ATR",
    "AverageTrueRange",
    "TrueRange",
    "CrossOver",
    "CrossUp",
    "CrossDown",
    "Stochastic",
)


class IndicatorLine(LineSeries):
    """Computed line whose cursor follows a source line."""

    def __init__(self, values: pd.Series, source: LineSeries | None = None) -> None:
        super().__init__(values.tolist())
        self._source = source
        self.index = values.index

    @property
    def _effective_cursor(self) -> int:
        return self._source._cursor if self._source is not None else self._cursor

    def __getitem__(self, ago: int) -> Any:
        index = self._effective_cursor + ago
        if index < 0 or index >= len(self._values):
            raise IndexError(
                f"tried to access line[{ago}] but current bar index is {self._effective_cursor}"
            )
        return self._values[index]

    def get(self, ago: int = 0, size: int = 1) -> list[Any]:
        end = self._effective_cursor + ago + 1
        start = max(0, end - size)
        if end < 0:
            return []
        return self._values[start:end]


@dataclass
class MACDLines:
    macd: IndicatorLine | pd.Series
    signal: IndicatorLine | pd.Series
    histo: IndicatorLine | pd.Series

    @property
    def hist(self) -> IndicatorLine | pd.Series:
        return self.histo


@dataclass
class BollingerBandLines:
    top: IndicatorLine | pd.Series
    mid: IndicatorLine | pd.Series
    bot: IndicatorLine | pd.Series


@dataclass
class StochasticLines:
    percK: IndicatorLine | pd.Series
    percD: IndicatorLine | pd.Series


def _line_source(data: Any) -> LineSeries | None:
    return data if isinstance(data, LineSeries) else None


def _close_line(data: Any) -> Any:
    return data.close if isinstance(data, DataFeed) else data


def _series(data: Any) -> pd.Series:
    if isinstance(data, LineSeries):
        return pd.Series(data._values, dtype="float64")
    if isinstance(data, pd.Series):
        return data.astype("float64")
    return pd.Series(data, dtype="float64")


def _wrap(data: Any, values: pd.Series) -> IndicatorLine | pd.Series:
    source = _line_source(data)
    if source is None:
        return values
    return IndicatorLine(values.reset_index(drop=True), source=source)


def _period(kwargs: dict[str, Any], default: int = 30) -> int:
    return int(kwargs.pop("period", kwargs.pop("length", default)))


def SMA(data: Any, period: int = 30, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    return _wrap(source, _series(source).rolling(period).mean())


SimpleMovingAverage = SMA
MovingAverageSimple = SMA


def EMA(data: Any, period: int = 30, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    return _wrap(source, _series(source).ewm(span=period, adjust=False, min_periods=period).mean())


ExponentialMovingAverage = EMA


def WMA(data: Any, period: int = 30, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    weights = pd.Series(range(1, period + 1), dtype="float64")
    values = _series(source).rolling(period).apply(
        lambda window: float((window * weights).sum() / weights.sum()),
        raw=False,
    )
    return _wrap(source, values)


WeightedMovingAverage = WMA


def RSI(data: Any, period: int = 14, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    close = _series(source)
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    values = 100.0 - (100.0 / (1.0 + gain / loss))
    return _wrap(source, values)


RelativeStrengthIndex = RSI


def MACD(
    data: Any,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **kwargs: Any,
) -> MACDLines:
    source = _close_line(data)
    close = _series(source)
    fast_line = close.ewm(span=int(fast), adjust=False, min_periods=int(fast)).mean()
    slow_line = close.ewm(span=int(slow), adjust=False, min_periods=int(slow)).mean()
    macd_line = fast_line - slow_line
    signal_line = macd_line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()
    hist_line = macd_line - signal_line
    return MACDLines(
        macd=_wrap(source, macd_line),
        signal=_wrap(source, signal_line),
        histo=_wrap(source, hist_line),
    )


def BollingerBands(
    data: Any,
    period: int = 20,
    devfactor: float = 2.0,
    **kwargs: Any,
) -> BollingerBandLines:
    period = _period(kwargs, period)
    source = _close_line(data)
    close = _series(source)
    mid = close.rolling(period).mean()
    deviation = close.rolling(period).std()
    return BollingerBandLines(
        top=_wrap(source, mid + float(devfactor) * deviation),
        mid=_wrap(source, mid),
        bot=_wrap(source, mid - float(devfactor) * deviation),
    )


BBands = BollingerBands


def Highest(data: Any, period: int = 30, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    return _wrap(source, _series(source).rolling(period).max())


def Lowest(data: Any, period: int = 30, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    source = _close_line(data)
    return _wrap(source, _series(source).rolling(period).min())


def TrueRange(data: DataFeed, **kwargs: Any) -> IndicatorLine | pd.Series:
    high = _series(data.high)
    low = _series(data.low)
    previous_close = _series(data.close).shift(1)
    values = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return _wrap(data.close, values)


def ATR(data: DataFeed, period: int = 14, **kwargs: Any) -> IndicatorLine | pd.Series:
    period = _period(kwargs, period)
    true_range = _series(TrueRange(data))
    return _wrap(data.close, true_range.rolling(period).mean())


AverageTrueRange = ATR


def _compare_values(left: Any, right: Any) -> tuple[pd.Series, pd.Series, LineSeries | None]:
    source = _line_source(left) or _line_source(right)
    return _series(left), _series(right), source


def CrossOver(left: Any, right: Any) -> IndicatorLine | pd.Series:
    left_values, right_values, source = _compare_values(left, right)
    diff = left_values - right_values
    previous = diff.shift(1)
    values = pd.Series(0.0, index=diff.index)
    values[(previous <= 0.0) & (diff > 0.0)] = 1.0
    values[(previous >= 0.0) & (diff < 0.0)] = -1.0
    return IndicatorLine(values, source=source) if source is not None else values


def CrossUp(left: Any, right: Any) -> IndicatorLine | pd.Series:
    values = _series(CrossOver(left, right)).eq(1.0).astype("float64")
    source = _line_source(left) or _line_source(right)
    return IndicatorLine(values, source=source) if source is not None else values


def CrossDown(left: Any, right: Any) -> IndicatorLine | pd.Series:
    values = _series(CrossOver(left, right)).eq(-1.0).astype("float64")
    source = _line_source(left) or _line_source(right)
    return IndicatorLine(values, source=source) if source is not None else values


def Stochastic(
    data: DataFeed,
    period: int = 14,
    period_dfast: int = 3,
    **kwargs: Any,
) -> StochasticLines:
    period = _period(kwargs, period)
    low = _series(data.low).rolling(period).min()
    high = _series(data.high).rolling(period).max()
    close = _series(data.close)
    k_line = 100.0 * (close - low) / (high - low)
    d_line = k_line.rolling(int(period_dfast)).mean()
    return StochasticLines(
        percK=_wrap(data.close, k_line),
        percD=_wrap(data.close, d_line),
    )


__all__ = [
    "ATR",
    "AverageTrueRange",
    "BBands",
    "BollingerBands",
    "CrossDown",
    "CrossOver",
    "CrossUp",
    "EMA",
    "ExponentialMovingAverage",
    "Highest",
    "IndicatorLine",
    "Lowest",
    "MACD",
    "MovingAverageSimple",
    "RSI",
    "RelativeStrengthIndex",
    "SMA",
    "SUPPORTED_INDICATOR_ALIASES",
    "SimpleMovingAverage",
    "Stochastic",
    "TrueRange",
    "WMA",
    "WeightedMovingAverage",
]
