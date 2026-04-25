"""WorldQuant Alpha101 formulas migrated into the v2 factor layer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ALPHA101_SUPPORTED = frozenset(
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
        "alpha011",
        "alpha012",
    }
)


def alpha101(stock_data: pd.DataFrame, names: Iterable[str] | None = None) -> pd.DataFrame:
    """Return selected Alpha101 factors in legacy Query-compatible long form."""
    selected = list(names or sorted(ALPHA101_SUPPORTED))
    unsupported = sorted(set(selected).difference(ALPHA101_SUPPORTED))
    if unsupported:
        raise ValueError(f"unsupported Alpha101 formulas: {unsupported}")

    factors = Alpha101Factors(_pivot_stock_data(stock_data))
    result = pd.DataFrame({"date": [], "code": []})
    for name in selected:
        frame = getattr(factors, name)().copy()
        frame["date"] = frame.index
        frame = frame.melt(
            id_vars="date",
            value_vars=frame.columns.drop("date"),
            value_name=name,
        )
        frame.rename(columns={name: f"{name}_101"}, inplace=True)
        result = pd.merge(result, frame, how="outer", on=["date", "code"])
    return result


class Alpha101Factors:
    """Compute the migrated subset of Alpha101 formulas on pivoted OHLCV data."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Create a factor calculator from ``data.pivot(index='date', columns='code')``."""
        self.open = data["open"]
        self.low = data["low"]
        self.close = data["close"]
        self.volume = data["volume"]
        self.returns = _returns(data["close"])
        self.vwap = data["vwap"]

    def alpha001(self) -> pd.DataFrame:
        """Return Alpha#1."""
        inner = self.close.copy()
        inner[self.returns < 0] = _stddev(self.returns, 20)
        return _rank(_ts_argmax(inner**2, 5)) - 0.5

    def alpha002(self) -> pd.DataFrame:
        """Return Alpha#2."""
        values = -1 * _correlation(
            _rank(_delta(np.log(self.volume), 2)),
            _rank((self.close - self.open) / self.open),
            6,
        )
        return values.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self) -> pd.DataFrame:
        """Return Alpha#3."""
        values = -1 * _correlation(_rank(self.open), _rank(self.volume), 10)
        return values.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self) -> pd.DataFrame:
        """Return Alpha#4."""
        return -1 * _ts_rank(_rank(self.low), 9)

    def alpha005(self) -> pd.DataFrame:
        """Return Alpha#5."""
        return _rank(self.open - (_ts_sum(self.vwap, 10) / 10)) * (
            -1 * np.abs(_rank(self.close - self.vwap))
        )

    def alpha006(self) -> pd.DataFrame:
        """Return Alpha#6."""
        values = -1 * _correlation(self.open, self.volume, 10)
        return values.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self) -> pd.DataFrame:
        """Return Alpha#7."""
        adv20 = _sma(self.volume, 20)
        delta_close = _delta(self.close, 7)
        values = -1 * _ts_rank(np.abs(delta_close), 60) * np.sign(delta_close)
        values[adv20 >= self.volume] = -1
        return values

    def alpha008(self) -> pd.DataFrame:
        """Return Alpha#8."""
        product = _ts_sum(self.open, 5) * _ts_sum(self.returns, 5)
        return -1 * _rank(product - _delay(product, 10))

    def alpha009(self) -> pd.DataFrame:
        """Return Alpha#9."""
        delta_close = _delta(self.close, 1)
        cond_1 = _ts_min(delta_close, 5) > 0
        cond_2 = _ts_max(delta_close, 5) < 0
        values = -1 * delta_close
        values[cond_1 | cond_2] = delta_close
        return values

    def alpha010(self) -> pd.DataFrame:
        """Return Alpha#10."""
        delta_close = _delta(self.close, 1)
        cond_1 = _ts_min(delta_close, 4) > 0
        cond_2 = _ts_max(delta_close, 4) < 0
        values = -1 * delta_close
        values[cond_1 | cond_2] = delta_close
        return _rank(values)

    def alpha011(self) -> pd.DataFrame:
        """Return Alpha#11."""
        vwap_close_spread = self.vwap - self.close
        return (
            _rank(_ts_max(vwap_close_spread, 3)) + _rank(_ts_min(vwap_close_spread, 3))
        ) * _rank(_delta(self.volume, 3))

    def alpha012(self) -> pd.DataFrame:
        """Return Alpha#12."""
        return np.sign(_delta(self.volume, 1)) * (-1 * _delta(self.close, 1))


def _pivot_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Return stock data pivoted to the Alpha101 formula layout."""
    required = {"date", "code", "open", "low", "close", "volume", "vwap"}
    missing = required.difference(stock_data.columns)
    if missing:
        raise ValueError(f"stock_data is missing required columns: {sorted(missing)}")
    return stock_data.pivot(index="date", columns="code")


def _returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-period simple returns."""
    return frame.rolling(2).apply(lambda values: values.iloc[-1] / values.iloc[0]) - 1


def _stddev(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sample standard deviation."""
    return frame.rolling(window).std()


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sum."""
    return frame.rolling(window).sum()


def _sma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling mean."""
    return frame.rolling(window).mean()


def _correlation(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling correlation with Alpha101 legacy missing-value handling."""
    return left.rolling(window).corr(right).fillna(0).replace([np.inf, -np.inf], 0)


def _delta(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return period difference."""
    return frame.diff(period)


def _delay(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return lagged values."""
    return frame.shift(period)


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cross-sectional percentile ranks."""
    return frame.rank(axis=1, method="min", pct=True)


def _rolling_rank(values: np.ndarray) -> float:
    """Return the rank of the most recent value in a rolling window."""
    return float(rankdata(values, method="min")[-1])


def _ts_argmax(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the one-based position of the rolling maximum."""
    return frame.rolling(window).apply(np.argmax) + 1


def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling rank of the latest value."""
    return frame.rolling(window).apply(_rolling_rank)


def _ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling minimum."""
    return frame.rolling(window).min()


def _ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling maximum."""
    return frame.rolling(window).max()
