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
        "alpha030",
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
        "alpha041",
        "alpha042",
        "alpha043",
        "alpha044",
        "alpha045",
        "alpha046",
        "alpha047",
        "alpha049",
        "alpha050",
        "alpha051",
        "alpha052",
        "alpha053",
        "alpha054",
        "alpha055",
        "alpha057",
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
        self.high = data["high"]
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

    def alpha013(self) -> pd.DataFrame:
        """Return Alpha#13."""
        return -1 * _rank(_covariance(_rank(self.close), _rank(self.volume), 5))

    def alpha014(self) -> pd.DataFrame:
        """Return Alpha#14."""
        values = _correlation(self.open, self.volume, 10)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _rank(_delta(self.returns, 3)) * values

    def alpha015(self) -> pd.DataFrame:
        """Return Alpha#15."""
        values = _correlation(_rank(self.high), _rank(self.volume), 3)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _ts_sum(_rank(values), 3)

    def alpha016(self) -> pd.DataFrame:
        """Return Alpha#16."""
        return -1 * _rank(_covariance(_rank(self.high), _rank(self.volume), 5))

    def alpha017(self) -> pd.DataFrame:
        """Return Alpha#17."""
        adv20 = _sma(self.volume, 20)
        return -1 * (
            _rank(_ts_rank(self.close, 10))
            * _rank(_delta(_delta(self.close, 1), 1))
            * _rank(_ts_rank(self.volume / adv20, 5))
        )

    def alpha018(self) -> pd.DataFrame:
        """Return Alpha#18."""
        values = _correlation(self.close, self.open, 10)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _rank(
            _stddev(np.abs(self.close - self.open), 5) + (self.close - self.open) + values
        )

    def alpha019(self) -> pd.DataFrame:
        """Return Alpha#19."""
        return (
            -1
            * np.sign((self.close - _delay(self.close, 7)) + _delta(self.close, 7))
            * (1 + _rank(1 + _ts_sum(self.returns, 250)))
        )

    def alpha020(self) -> pd.DataFrame:
        """Return Alpha#20."""
        return -1 * (
            _rank(self.open - _delay(self.high, 1))
            * _rank(self.open - _delay(self.close, 1))
            * _rank(self.open - _delay(self.low, 1))
        )

    def alpha021(self) -> pd.DataFrame:
        """Return Alpha#21."""
        cond_1 = _sma(self.close, 8) + _stddev(self.close, 8) < _sma(self.close, 2)
        cond_2 = _sma(self.close, 2) < _sma(self.close, 8) - _stddev(self.close, 8)
        cond_3 = _sma(self.volume, 20) / self.volume < 1
        return (cond_1 | ((~cond_1) & (~cond_2) & (~cond_3))).astype("int") * (-2) + 1

    def alpha022(self) -> pd.DataFrame:
        """Return Alpha#22."""
        values = _correlation(self.high, self.volume, 5)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _delta(values, 5) * _rank(_stddev(self.close, 20))

    def alpha023(self) -> pd.DataFrame:
        """Return Alpha#23."""
        cond = _sma(self.high, 20) < self.high
        values = self.close.copy(deep=True)
        values[cond] = -1 * _delta(self.high, 2).fillna(value=0)
        values[~cond] = 0
        return values

    def alpha024(self) -> pd.DataFrame:
        """Return Alpha#24."""
        cond = _delta(_sma(self.close, 100), 100) / _delay(self.close, 100) <= 0.05
        values = -1 * _delta(self.close, 3)
        values[cond] = -1 * (self.close - _ts_min(self.close, 100))
        return values

    def alpha025(self) -> pd.DataFrame:
        """Return Alpha#25."""
        adv20 = _sma(self.volume, 20)
        return _rank(((-1 * self.returns) * adv20) * self.vwap * (self.high - self.close))

    def alpha026(self) -> pd.DataFrame:
        """Return Alpha#26."""
        values = _correlation(_ts_rank(self.volume, 5), _ts_rank(self.high, 5), 5)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * _ts_max(values, 3)

    def alpha027(self) -> pd.DataFrame:
        """Return Alpha#27."""
        values = _rank(_sma(_correlation(_rank(self.volume), _rank(self.vwap), 6), 2) / 2.0)
        return np.sign((values - 0.5) * (-2))

    def alpha028(self) -> pd.DataFrame:
        """Return Alpha#28."""
        adv20 = _sma(self.volume, 20)
        values = _correlation(adv20, self.low, 5)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return _scale((values + ((self.high + self.low) / 2)) - self.close)

    def alpha029(self) -> pd.DataFrame:
        """Return Alpha#29."""
        return _ts_min(
            _rank(
                _rank(
                    _scale(
                        np.log(
                            _ts_sum(
                                _rank(_rank(-1 * _rank(_delta((self.close - 1), 5)))),
                                2,
                            )
                        )
                    )
                )
            ),
            5,
        ) + _ts_rank(_delay((-1 * self.returns), 6), 5)

    def alpha030(self) -> pd.DataFrame:
        """Return Alpha#30."""
        delta_close = _delta(self.close, 1)
        inner = np.sign(delta_close) + np.sign(_delay(delta_close, 1)) + np.sign(
            _delay(delta_close, 2)
        )
        return ((1.0 - _rank(inner)) * _ts_sum(self.volume, 5)) / _ts_sum(self.volume, 20)

    def alpha031(self) -> pd.DataFrame:
        """Return Alpha#31."""
        adv20 = _sma(self.volume, 20)
        values = _correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(
            value=0
        )
        return (
            _rank(_rank(_rank(_decay_linear(-1 * _rank(_rank(_delta(self.close, 10))), 10))))
            + _rank(-1 * _delta(self.close, 3))
            + np.sign(_scale(values))
        )

    def alpha032(self) -> pd.DataFrame:
        """Return Alpha#32."""
        return _scale((_sma(self.close, 7) / 7) - self.close) + (
            20 * _scale(_correlation(self.vwap, _delay(self.close, 5), 230))
        )

    def alpha033(self) -> pd.DataFrame:
        """Return Alpha#33."""
        return _rank(-1 + (self.open / self.close))

    def alpha034(self) -> pd.DataFrame:
        """Return Alpha#34."""
        inner = (_stddev(self.returns, 2) / _stddev(self.returns, 5)).replace(
            [-np.inf, np.inf], 1
        ).fillna(value=1)
        return _rank(2 - _rank(inner) - _rank(_delta(self.close, 1)))

    def alpha035(self) -> pd.DataFrame:
        """Return Alpha#35."""
        return (
            _ts_rank(self.volume, 32)
            * (1 - _ts_rank(self.close + self.high - self.low, 16))
            * (1 - _ts_rank(self.returns, 32))
        )

    def alpha036(self) -> pd.DataFrame:
        """Return Alpha#36."""
        adv20 = _sma(self.volume, 20)
        return (
            (
                (
                    (
                        (
                            2.21
                            * _rank(
                                _correlation(
                                    self.close - self.open,
                                    _delay(self.volume, 1),
                                    15,
                                )
                            )
                        )
                        + (0.7 * _rank(self.open - self.close))
                    )
                    + (0.73 * _rank(_ts_rank(_delay(-1 * self.returns, 6), 5)))
                )
                + _rank(np.abs(_correlation(self.vwap, adv20, 6)))
            )
            + (
                0.6
                * _rank(((_sma(self.close, 200) / 200) - self.open) * (self.close - self.open))
            )
        )

    def alpha037(self) -> pd.DataFrame:
        """Return Alpha#37."""
        return _rank(_correlation(_delay(self.open - self.close, 1), self.close, 200)) + _rank(
            self.open - self.close
        )

    def alpha038(self) -> pd.DataFrame:
        """Return Alpha#38."""
        inner = (self.close / self.open).replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * _rank(_ts_rank(self.open, 10)) * _rank(inner)

    def alpha039(self) -> pd.DataFrame:
        """Return Alpha#39."""
        adv20 = _sma(self.volume, 20)
        return (
            -1
            * _rank(_delta(self.close, 7) * (1 - _rank(_decay_linear(self.volume / adv20, 9))))
            * (1 + _rank(_sma(self.returns, 250)))
        )

    def alpha040(self) -> pd.DataFrame:
        """Return Alpha#40."""
        return -1 * _rank(_stddev(self.high, 10)) * _correlation(self.high, self.volume, 10)

    def alpha041(self) -> pd.DataFrame:
        """Return Alpha#41."""
        return (self.high * self.low).pow(0.5) - self.vwap

    def alpha042(self) -> pd.DataFrame:
        """Return Alpha#42."""
        return _rank(self.vwap - self.close) / _rank(self.vwap + self.close)

    def alpha043(self) -> pd.DataFrame:
        """Return Alpha#43."""
        adv20 = _sma(self.volume, 20)
        return _ts_rank(self.volume / adv20, 20) * _ts_rank(-1 * _delta(self.close, 7), 8)

    def alpha044(self) -> pd.DataFrame:
        """Return Alpha#44."""
        values = _correlation(self.high, _rank(self.volume), 5)
        return -1 * values.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha045(self) -> pd.DataFrame:
        """Return Alpha#45."""
        values = _correlation(self.close, self.volume, 2)
        values = values.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (
            _rank(_sma(_delay(self.close, 5), 20))
            * values
            * _rank(_correlation(_ts_sum(self.close, 5), _ts_sum(self.close, 20), 2))
        )

    def alpha046(self) -> pd.DataFrame:
        """Return Alpha#46."""
        inner = ((_delay(self.close, 20) - _delay(self.close, 10)) / 10) - (
            (_delay(self.close, 10) - self.close) / 10
        )
        values = -1 * _delta(self.close, 1)
        values[inner < 0] = 1
        values[inner > 0.25] = -1
        return values

    def alpha047(self) -> pd.DataFrame:
        """Return Alpha#47."""
        adv20 = _sma(self.volume, 20)
        return (
            (
                ((_rank(1 / self.close) * self.volume) / adv20)
                * ((self.high * _rank(self.high - self.close)) / (_sma(self.high, 5) / 5))
            )
            - _rank(self.vwap - _delay(self.vwap, 5))
        )

    def alpha049(self) -> pd.DataFrame:
        """Return Alpha#49."""
        inner = ((_delay(self.close, 20) - _delay(self.close, 10)) / 10) - (
            (_delay(self.close, 10) - self.close) / 10
        )
        values = -1 * _delta(self.close, 1)
        values[inner < -0.1] = 1
        return values

    def alpha050(self) -> pd.DataFrame:
        """Return Alpha#50."""
        return -1 * _ts_max(_rank(_correlation(_rank(self.volume), _rank(self.vwap), 5)), 5)

    def alpha051(self) -> pd.DataFrame:
        """Return Alpha#51."""
        inner = ((_delay(self.close, 20) - _delay(self.close, 10)) / 10) - (
            (_delay(self.close, 10) - self.close) / 10
        )
        values = -1 * _delta(self.close, 1)
        values[inner < -0.05] = 1
        return values

    def alpha052(self) -> pd.DataFrame:
        """Return Alpha#52."""
        return (
            -1
            * _delta(_ts_min(self.low, 5), 5)
            * _rank((_ts_sum(self.returns, 240) - _ts_sum(self.returns, 20)) / 220)
            * _ts_rank(self.volume, 5)
        )

    def alpha053(self) -> pd.DataFrame:
        """Return Alpha#53."""
        denominator = (self.close - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) / denominator
        return -1 * _delta(inner, 9)

    def alpha054(self) -> pd.DataFrame:
        """Return Alpha#54."""
        denominator = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open**5) / (
            denominator * (self.close**5)
        )

    def alpha055(self) -> pd.DataFrame:
        """Return Alpha#55."""
        denominator = (_ts_max(self.high, 12) - _ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - _ts_min(self.low, 12)) / denominator
        values = _correlation(_rank(inner), _rank(self.volume), 6)
        return -1 * values.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha057(self) -> pd.DataFrame:
        """Return Alpha#57."""
        denominator = _decay_linear(_rank(_ts_argmax(self.close, 30)), 2)
        return -1 * ((self.close - self.vwap) / denominator)


def _pivot_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Return stock data pivoted to the Alpha101 formula layout."""
    required = {"date", "code", "open", "high", "low", "close", "volume", "vwap"}
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


def _covariance(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling covariance."""
    return left.rolling(window).cov(right)


def _delta(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return period difference."""
    return frame.diff(period)


def _delay(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return lagged values."""
    return frame.shift(period)


def _rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cross-sectional percentile ranks."""
    return frame.rank(axis=1, method="min", pct=True)


def _scale(frame: pd.DataFrame, k: float = 1) -> pd.DataFrame:
    """Return frame scaled so the sum of absolute values equals ``k``."""
    return frame.mul(k).div(np.abs(frame).sum())


def _decay_linear(frame: pd.DataFrame, period: int) -> pd.DataFrame:
    """Return rolling linearly weighted moving averages."""
    weights = np.arange(1, period + 1)
    sum_weights = np.sum(weights)
    return frame.rolling(period).apply(lambda values: np.sum(weights * values) / sum_weights)


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
