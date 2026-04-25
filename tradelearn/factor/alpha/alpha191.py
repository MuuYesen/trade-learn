"""Alpha191 formulas migrated into the v2 factor layer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ALPHA191_SUPPORTED = frozenset(
    {
        "alpha001",
        "alpha002",
        "alpha003",
        "alpha004",
        "alpha005",
        "alpha006",
        "alpha007",
        "alpha008",
        "alpha009",
        "alpha010",
    }
)


def alpha191(
    stock_data: pd.DataFrame,
    bench_data: pd.DataFrame,
    names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return selected Alpha191 factors in legacy Query-compatible long form."""
    selected = list(names or sorted(ALPHA191_SUPPORTED))
    unsupported = sorted(set(selected).difference(ALPHA191_SUPPORTED))
    if unsupported:
        raise ValueError(f"unsupported Alpha191 formulas: {unsupported}")

    factors = Alpha191Factors(_pivot_stock_data(stock_data), bench_data)
    result = pd.DataFrame({"date": [], "code": []})
    for name in selected:
        frame = getattr(factors, name)().copy()
        frame["date"] = frame.index
        frame = frame.melt(
            id_vars="date",
            value_vars=frame.columns.drop("date"),
            value_name=name,
        )
        frame.rename(columns={name: f"{name}_191"}, inplace=True)
        result = pd.merge(result, frame, how="outer", on=["date", "code"])
    return result


class Alpha191Factors:
    """Compute the migrated subset of Alpha191 formulas on pivoted OHLCV data."""

    def __init__(self, data: pd.DataFrame, bench_data: pd.DataFrame) -> None:
        """Create a factor calculator from ``data.pivot(index='date', columns='code')``."""
        self.open = data["open"]
        self.high = data["high"]
        self.low = data["low"]
        self.close = data["close"]
        self.volume = data["volume"]
        self.returns = _returns(data["close"])
        self.vwap = data["vwap"]
        self.close_prev = data["close"].shift(1)
        self.amount = data["amount"]
        self.benchmark_open = bench_data["open"]
        self.benchmark_close = bench_data["close"]

    def alpha001(self) -> pd.DataFrame:
        """Return Alpha#1."""
        return -1 * _correlation(
            _rank(_delta(np.log(self.volume), 1)),
            _rank((self.close - self.open) / self.open),
            6,
        )

    def alpha002(self) -> pd.DataFrame:
        """Return Alpha#2."""
        return -1 * _delta(
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low),
            1,
        )

    def alpha003(self) -> pd.DataFrame:
        """Return Alpha#3."""
        cond1 = self.close == _delay(self.close, 1)
        cond2 = self.close > _delay(self.close, 1)
        cond3 = self.close < _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = 0
        part[cond2] = self.close - _elementwise_min(self.low, _delay(self.close, 1))
        part[cond3] = self.close - _elementwise_max(self.high, _delay(self.close, 1))
        return _ts_sum(part, 6)

    def alpha004(self) -> pd.DataFrame:
        """Return Alpha#4."""
        cond1 = (_ts_sum(self.close, 8) / 8 + _stddev(self.close, 8)) < (
            _ts_sum(self.close, 2) / 2
        )
        cond2 = (_ts_sum(self.close, 8) / 8 + _stddev(self.close, 8)) > (
            _ts_sum(self.close, 2) / 2
        )
        cond3 = (_ts_sum(self.close, 8) / 8 + _stddev(self.close, 8)) == (
            _ts_sum(self.close, 2) / 2
        )
        cond4 = self.volume / _sma(self.volume, 20) >= 1
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1
        part[cond3 & cond4] = 1
        return part

    def alpha005(self) -> pd.DataFrame:
        """Return Alpha#5."""
        return -1 * _ts_max(
            _correlation(_ts_rank(self.volume, 5), _ts_rank(self.high, 5), 5),
            3,
        )

    def alpha006(self) -> pd.DataFrame:
        """Return Alpha#6."""
        return -1 * _rank(
            np.sign(_delta((self.open * 0.85) + (self.high * 0.15), 4))
        )

    def alpha007(self) -> pd.DataFrame:
        """Return Alpha#7."""
        spread = self.vwap - self.close
        return (
            _rank(_ts_max(spread, 3)) + _rank(_ts_min(spread, 3))
        ) * _rank(_delta(self.volume, 3))

    def alpha008(self) -> pd.DataFrame:
        """Return Alpha#8."""
        return _rank(
            _delta(((self.high + self.low) / 2 * 0.2) + (self.vwap * 0.8), 4)
            * -1
        )

    def alpha009(self) -> pd.DataFrame:
        """Return Alpha#9."""
        return _sma(
            (
                (self.high + self.low) / 2
                - (_delay(self.high, 1) + _delay(self.low, 1)) / 2
            )
            * (self.high - self.low)
            / self.volume,
            7,
            2,
        )

    def alpha010(self) -> pd.DataFrame:
        """Return Alpha#10."""
        cond = self.returns < 0
        part = self.returns.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = _stddev(self.returns, 20)
        part[~cond] = self.close
        return _rank(_ts_max(part**2, 5))


def _pivot_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Return stock data pivoted to the Alpha191 formula layout."""
    required = {
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "amount",
    }
    missing = required.difference(stock_data.columns)
    if missing:
        raise ValueError(f"stock_data is missing required columns: {sorted(missing)}")
    return stock_data.pivot(index="date", columns="code")


def _returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-period simple returns."""
    return frame.rolling(2).apply(lambda values: values.iloc[-1] / values.iloc[0]) - 1


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cross-sectional percentile ranks."""
    return frame.rank(axis=1, method="min", pct=True)


def _delta(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return period difference."""
    return frame.diff(period)


def _delay(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return lagged values."""
    return frame.shift(period)


def _correlation(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling correlation matching the legacy Alpha191 warmup behavior."""
    result = left.rolling(window).corr(right).fillna(0)
    result.iloc[: (window - 1), :] = None
    return result


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sum."""
    return frame.rolling(window).sum()


def _stddev(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sample standard deviation."""
    return frame.rolling(window).std()


def _sma(frame: pd.DataFrame, window: int, weight: int = 1) -> pd.DataFrame:
    """Return Alpha191 SMA implemented with exponentially weighted mean."""
    return frame.ewm(alpha=weight / window, adjust=False).mean()


def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling rank of the latest value."""
    return frame.rolling(window).apply(lambda values: rankdata(values)[-1])


def _ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling maximum."""
    return frame.rolling(window).max()


def _ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling minimum."""
    return frame.rolling(window).min()


def _elementwise_max(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise maximum."""
    return np.maximum(left, right)


def _elementwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise minimum."""
    return np.minimum(left, right)
