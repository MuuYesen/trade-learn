"""Backtrader-style indicator aliases backed by pandas computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import inspect

import pandas as pd

from tradelearn.backtest import DataFeed, LineSeries

# Global context to track current data feed during strategy initialization
_CURRENT_DATA = None
_CURRENT_STRATEGY = None

def set_current_data(data):
    global _CURRENT_DATA
    _CURRENT_DATA = data

def set_current_strategy(strategy):
    global _CURRENT_STRATEGY
    _CURRENT_STRATEGY = strategy

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


class MetaIndicator(type):
    """Metaclass to handle Backtrader-style parameter stripping."""
    def __call__(cls, *args, **kwargs):
        # Check if __init__ accepts the arguments
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())
        
        # If __init__ only takes 'self', we need to strip args/kwargs
        # and store them on the instance instead
        if len(params) == 1:
            instance = cls.__new__(cls)
            # Standard Backtrader behavior: first arg is often data
            if args:
                instance.data = args[0]
            # Store kwargs as params (p)
            class Params: pass
            instance.p = Params()
            
            # Support self.lines and self.l
            class Lines: pass
            instance.lines = instance.l = Lines()
            
            # If class has params defined, use them as defaults

            base_params = getattr(cls, 'params', ())
            if isinstance(base_params, tuple):
                for k, v in base_params:
                    setattr(instance.p, k, v)
            for k, v in kwargs.items():
                setattr(instance.p, k, v)
            
            instance.__init__()
            return instance
        
        return super().__call__(*args, **kwargs)


class Indicator(LineSeries, metaclass=MetaIndicator):
    """Base class for all Backtrader-style indicators."""
    params = ()

    def __init__(self, *args, **kwargs):
        super().__init__([])

    def __getattr__(self, name: str) -> Any:
        # Support accessing lines directly as attributes (e.g. indicator.dch)
        if hasattr(self, "lines") and hasattr(self.lines, name):
            return getattr(self.lines, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")



class IndicatorLine(LineSeries):
    """Computed line whose cursor follows a source line."""

    def __init__(self, values: pd.Series, source: LineSeries | None = None) -> None:
        import numpy as np
        # Convert to raw numpy array immediately
        super().__init__(values.to_numpy(dtype=np.float64))
        self._source = source

    @property
    def _cursor(self) -> int:
        if self._source is None:
            return self.__dict__.get('_cursor', -1)
        return self._source._cursor

    @_cursor.setter
    def _cursor(self, value: int):
        self.__dict__['_cursor'] = value

    def _advance(self, cursor: int) -> None:
        # Cursor is synced via the property linked to source
        pass

    def _to_series(self) -> pd.Series:
        return pd.Series(self._values, dtype="float64")

    def _get_other_vals(self, other: Any) -> Any:
        from tradelearn.backtest import LineSeries
        if isinstance(other, LineSeries):
            return pd.Series(other._values, dtype="float64")
        return other

    def __add__(self, other: Any) -> IndicatorLine:
        return IndicatorLine(self._to_series() + self._get_other_vals(other), source=self._source)

    def __sub__(self, other: Any) -> IndicatorLine:
        return IndicatorLine(self._to_series() - self._get_other_vals(other), source=self._source)

    def __mul__(self, other: Any) -> IndicatorLine:
        return IndicatorLine(self._to_series() * self._get_other_vals(other), source=self._source)

    def __truediv__(self, other: Any) -> IndicatorLine:
        return IndicatorLine(self._to_series() / self._get_other_vals(other), source=self._source)



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
    line = IndicatorLine(values.reset_index(drop=True), source=source)
    # Auto-register min_period with the strategy being constructed
    if _CURRENT_STRATEGY is not None and hasattr(_CURRENT_STRATEGY, 'addminperiod'):
        import numpy as np
        arr = line._values
        non_nan = np.where(~np.isnan(arr))[0]
        if len(non_nan) > 0:
            _CURRENT_STRATEGY.addminperiod(int(non_nan[0]))
    return line


def _period(kwargs: dict[str, Any], default: int = 30) -> int:
    return int(kwargs.pop("period", kwargs.pop("length", default)))


def _get_data(data: Any, kwargs: dict[str, Any]) -> tuple[Any, int]:
    """Smartly extract data and period from args."""
    period = _period(kwargs)
    
    # Check if first arg is an integer (period)
    if isinstance(data, (int, float)):
        period = int(data)
        data = None

    if data is None:
        # Auto-bind to current data context
        data = _CURRENT_DATA

    return data, period


def SMA(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    source = _close_line(data)
    return _wrap(source, _series(source).rolling(period).mean())


SimpleMovingAverage = SMA
MovingAverageSimple = SMA


def EMA(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    source = _close_line(data)
    return _wrap(source, _series(source).ewm(span=period, adjust=False, min_periods=period).mean())


ExponentialMovingAverage = EMA


def WMA(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    source = _close_line(data)
    weights = pd.Series(range(1, period + 1), dtype="float64")
    values = _series(source).rolling(period).apply(
        lambda window: float((window * weights).sum() / weights.sum()),
        raw=False,
    )
    return _wrap(source, values)


WeightedMovingAverage = WMA


def RSI(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    source = _close_line(data)
    close = _series(source)
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    values = 100.0 - (100.0 / (1.0 + gain / loss))
    return _wrap(source, values)


RelativeStrengthIndex = RSI
RSI_SMA = RSI


def MACD(
    data: Any = None,
    **kwargs: Any,
) -> MACDLines:
    fast = int(kwargs.pop("period_me1", kwargs.pop("fast", 12)))
    slow = int(kwargs.pop("period_me2", kwargs.pop("slow", 26)))
    signal = int(kwargs.pop("period_signal", kwargs.pop("signal", 9)))
    data, _ = _get_data(data, kwargs)
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
    data: Any = None,
    **kwargs: Any,
) -> BollingerBandLines:
    period = int(kwargs.pop("period", 20))
    devfactor = float(kwargs.pop("devfactor", 2.0))
    data, _ = _get_data(data, kwargs)
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


def Highest(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    source = _close_line(data)
    return _wrap(source, _series(source).rolling(period).max())


def Lowest(data: Any = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
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


def ATR(data: DataFeed = None, **kwargs: Any) -> IndicatorLine | pd.Series:
    data, period = _get_data(data, kwargs)
    true_range = _series(TrueRange(data))
    # Wilder's smoothing used in BT's ATR is an EMA with span = 2*period - 1
    return _wrap(data.close, true_range.ewm(span=2 * period - 1, adjust=False, min_periods=period).mean())


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
    data: DataFeed = None,
    **kwargs: Any,
) -> StochasticLines:
    period = int(kwargs.pop("period", 14))
    period_dfast = int(kwargs.pop("period_dfast", 3))
    data, _ = _get_data(data, kwargs)
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
    "Indicator",
    "IndicatorLine",
    "Lowest",
    "MACD",
    "MovingAverageSimple",
    "RSI",
    "RSI_SMA",
    "RelativeStrengthIndex",
    "SMA",
    "SUPPORTED_INDICATOR_ALIASES",
    "SimpleMovingAverage",
    "Stochastic",
    "TrueRange",
    "WMA",
    "WeightedMovingAverage",
    "set_current_data",
]
