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
        "alpha011",
        "alpha012",
        "alpha013",
        "alpha014",
        "alpha015",
        "alpha016",
        "alpha017",
        "alpha018",
        "alpha019",
        "alpha020",
        "alpha021",
        "alpha022",
        "alpha023",
        "alpha024",
        "alpha025",
        "alpha026",
        "alpha027",
        "alpha028",
        "alpha029",
        "alpha031",
        "alpha032",
        "alpha033",
        "alpha034",
        "alpha035",
        "alpha036",
        "alpha037",
        "alpha038",
        "alpha039",
        "alpha040",
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

    def alpha011(self) -> pd.DataFrame:
        """Return Alpha#11."""
        return _ts_sum(
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low)
            * self.volume,
            6,
        )

    def alpha012(self) -> pd.DataFrame:
        """Return Alpha#12."""
        return _rank(self.open - (_ts_sum(self.vwap, 10) / 10)) * (
            -1 * _rank(np.abs(self.close - self.vwap))
        )

    def alpha013(self) -> pd.DataFrame:
        """Return Alpha#13."""
        return ((self.high * self.low) ** 0.5) - self.vwap

    def alpha014(self) -> pd.DataFrame:
        """Return Alpha#14."""
        return self.close - _delay(self.close, 5)

    def alpha015(self) -> pd.DataFrame:
        """Return Alpha#15."""
        return self.open / _delay(self.close, 1) - 1

    def alpha016(self) -> pd.DataFrame:
        """Return Alpha#16."""
        return -1 * _ts_max(_rank(_correlation(_rank(self.volume), _rank(self.vwap), 5)), 5)

    def alpha017(self) -> pd.DataFrame:
        """Return Alpha#17."""
        return _rank(self.vwap - _ts_max(self.vwap, 15)) ** _delta(self.close, 5)

    def alpha018(self) -> pd.DataFrame:
        """Return Alpha#18."""
        return self.close / _delay(self.close, 5)

    def alpha019(self) -> pd.DataFrame:
        """Return Alpha#19."""
        delayed_close = _delay(self.close, 5)
        cond1 = self.close < delayed_close
        cond2 = self.close == delayed_close
        cond3 = self.close > delayed_close
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = (self.close - delayed_close) / delayed_close
        part[cond2] = 0
        part[cond3] = (self.close - delayed_close) / self.close
        return part

    def alpha020(self) -> pd.DataFrame:
        """Return Alpha#20."""
        delayed_close = _delay(self.close, 6)
        return (self.close - delayed_close) / delayed_close * 100

    def alpha021(self) -> pd.DataFrame:
        """Return Alpha#21."""
        return _regbeta(_mean(self.close, 6), _sequence(6))

    def alpha022(self) -> pd.DataFrame:
        """Return Alpha#22."""
        close_mean = _mean(self.close, 6)
        ratio = (self.close - close_mean) / close_mean
        return _sma(ratio - _delay(ratio, 3), 12, 1)

    def alpha023(self) -> pd.DataFrame:
        """Return Alpha#23."""
        cond = self.close > _delay(self.close, 1)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond] = _stddev(self.close, 20)
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[~cond] = _stddev(self.close, 20)
        part2[cond] = 0
        return 100 * _sma(part1, 20, 1) / (_sma(part1, 20, 1) + _sma(part2, 20, 1))

    def alpha024(self) -> pd.DataFrame:
        """Return Alpha#24."""
        return _sma(self.close - _delay(self.close, 5), 5, 1)

    def alpha025(self) -> pd.DataFrame:
        """Return Alpha#25."""
        return (
            -1
            * _rank(
                _delta(self.close, 7)
                * (1 - _rank(_decay_linear(self.volume / _mean(self.volume, 20), 9)))
            )
            * (1 + _rank(_ts_sum(self.returns, 250)))
        )

    def alpha026(self) -> pd.DataFrame:
        """Return Alpha#26."""
        return ((_ts_sum(self.close, 7) / 7) - self.close) + _correlation(
            self.vwap,
            _delay(self.close, 5),
            230,
        )

    def alpha027(self) -> pd.DataFrame:
        """Return Alpha#27."""
        values = (
            (self.close - _delay(self.close, 3)) / _delay(self.close, 3) * 100
            + (self.close - _delay(self.close, 6)) / _delay(self.close, 6) * 100
        )
        return _wma(values, 12)

    def alpha028(self) -> pd.DataFrame:
        """Return Alpha#28."""
        return 3 * _sma(
            (self.close - _ts_min(self.low, 9))
            / (_ts_max(self.high, 9) - _ts_min(self.low, 9))
            * 100,
            3,
            1,
        ) - 2 * _sma(
            _sma(
                (self.close - _ts_min(self.low, 9))
                / (_ts_max(self.high, 9) - _ts_max(self.low, 9))
                * 100,
                3,
                1,
            ),
            3,
            1,
        )

    def alpha029(self) -> pd.DataFrame:
        """Return Alpha#29."""
        delayed_close = _delay(self.close, 6)
        return (self.close - delayed_close) / delayed_close * self.volume

    def alpha031(self) -> pd.DataFrame:
        """Return Alpha#31."""
        close_mean = _mean(self.close, 12)
        return (self.close - close_mean) / close_mean * 100

    def alpha032(self) -> pd.DataFrame:
        """Return Alpha#32."""
        return -1 * _ts_sum(
            _rank(_correlation(_rank(self.high), _rank(self.volume), 3)),
            3,
        )

    def alpha033(self) -> pd.DataFrame:
        """Return Alpha#33."""
        low_min = _ts_min(self.low, 5)
        return (
            ((-1 * low_min) + _delay(low_min, 5))
            * _rank((_ts_sum(self.returns, 240) - _ts_sum(self.returns, 20)) / 220)
            * _ts_rank(self.volume, 5)
        )

    def alpha034(self) -> pd.DataFrame:
        """Return Alpha#34."""
        return _mean(self.close, 12) / self.close

    def alpha035(self) -> pd.DataFrame:
        """Return Alpha#35."""
        left = _rank(_decay_linear(_delta(self.open, 1), 15))
        right = _rank(
            _decay_linear(
                _correlation(self.volume, (self.open * 0.65) + (self.open * 0.35), 17),
                7,
            )
        )
        return _elementwise_min(left, right) * -1

    def alpha036(self) -> pd.DataFrame:
        """Return Alpha#36."""
        return _rank(_ts_sum(_correlation(_rank(self.volume), _rank(self.vwap), 6), 2))

    def alpha037(self) -> pd.DataFrame:
        """Return Alpha#37."""
        product = _ts_sum(self.open, 5) * _ts_sum(self.returns, 5)
        return -1 * _rank(product - _delay(product, 10))

    def alpha038(self) -> pd.DataFrame:
        """Return Alpha#38."""
        cond = (_ts_sum(self.high, 20) / 20) < self.high
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = -1 * _delta(self.high, 2)
        part[~cond] = 0
        return part

    def alpha039(self) -> pd.DataFrame:
        """Return Alpha#39."""
        left = _rank(_decay_linear(_delta(self.close, 2), 8))
        right = _rank(
            _decay_linear(
                _correlation(
                    (self.vwap * 0.3) + (self.open * 0.7),
                    _ts_sum(_mean(self.volume, 180), 37),
                    14,
                ),
                12,
            )
        )
        return (left - right) * -1

    def alpha040(self) -> pd.DataFrame:
        """Return Alpha#40."""
        cond = self.close > _delay(self.close, 1)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond] = self.volume
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[~cond] = self.volume
        part2[cond] = 0
        return _ts_sum(part1, 26) / _ts_sum(part2, 26) * 100


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
    result.iloc[: (window - 1), :] = np.nan
    return result


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sum."""
    return frame.rolling(window).sum()


def _stddev(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sample standard deviation."""
    return frame.rolling(window).std()


def _mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling mean."""
    return frame.rolling(window).mean()


def _sma(frame: pd.DataFrame, window: int, weight: int = 1) -> pd.DataFrame:
    """Return Alpha191 SMA implemented with exponentially weighted mean."""
    return frame.ewm(alpha=weight / window, adjust=False).mean()


def _sequence(size: int) -> np.ndarray:
    """Return the 1-based sequence used by legacy regression formulas."""
    return np.arange(1, size + 1)


def _regbeta(frame: pd.DataFrame, x_values: np.ndarray) -> pd.DataFrame:
    """Return rolling linear-regression slope against ``x_values``."""
    window = len(x_values)
    return frame.rolling(window).apply(lambda values: np.polyfit(x_values, values, 1)[0])


def _decay_linear(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling linear weighted average."""
    weights = np.arange(1, window + 1)
    weight_sum = np.sum(weights)
    return frame.rolling(window).apply(lambda values: np.sum(weights * values) / weight_sum)


def _wma(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return legacy exponentially decayed weighted moving average."""
    weights = np.power(0.9, np.arange(window - 1, -1, -1))
    weight_sum = np.sum(weights)
    return frame.rolling(window).apply(lambda values: np.sum(weights * values) / weight_sum)


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
