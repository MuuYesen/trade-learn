"""WorldQuant Alpha101 formulas migrated into the v2 factor layer."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_ALPHA101_NAMES = frozenset(f"alpha{index:03d}" for index in range(1, 102))


def alpha101(stock_data: pd.DataFrame, names: Iterable[str] | None = None) -> pd.DataFrame:
    """Return selected Alpha101 factors in Tradelearn long form."""
    selected = list(names or sorted(_ALPHA101_NAMES))
    unknown = sorted(set(selected).difference(_ALPHA101_NAMES))
    if unknown:
        raise ValueError(f"unknown Alpha101 formulas: {unknown}")

    factors = Alpha101Factors(_pivot_stock_data(stock_data))
    result = pd.DataFrame({"date": [], "symbol": []})
    for name in selected:
        frame = getattr(factors, name)().copy()
        frame["date"] = frame.index
        frame = frame.melt(
            id_vars="date",
            value_vars=frame.columns.drop("date"),
            var_name="symbol",
            value_name=name,
        )
        frame.rename(columns={name: f"{name}_101"}, inplace=True)
        result = pd.merge(result, frame, how="outer", on=["date", "symbol"])
    return result


class Alpha101Factors:
    """Compute the migrated subset of Alpha101 formulas on pivoted OHLCV data."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Create a factor calculator from ``data.pivot(index='date', columns='symbol')``."""
        self.open = data["open"]
        self.high = data["high"]
        self.low = data["low"]
        self.close = data["close"]
        self.volume = data["volume"]
        self.returns = _returns(data["close"])
        self.vwap = data["vwap"]
        self.cap = self.vwap * self.volume

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

    def alpha048(self) -> pd.DataFrame:
        """Return Alpha#48."""
        close_delta = _delta(self.close, 1)
        numerator = (
            _correlation(close_delta, _delta(_delay(self.close, 1), 1), 250)
            * close_delta
        ) / self.close
        denominator = _ts_sum((close_delta / _delay(self.close, 1)).pow(2), 250)
        return _neutralize(numerator) / denominator

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

    def alpha056(self) -> pd.DataFrame:
        """Return Alpha#56."""
        left = _rank(_ts_sum(self.returns, 10) / _ts_sum(_ts_sum(self.returns, 2), 3))
        right = _rank(self.returns * self.cap)
        return -1 * left * right

    def alpha057(self) -> pd.DataFrame:
        """Return Alpha#57."""
        denominator = _decay_linear(_rank(_ts_argmax(self.close, 30)), 2)
        return -1 * ((self.close - self.vwap) / denominator)

    def alpha058(self) -> pd.DataFrame:
        """Return Alpha#58."""
        values = _correlation(_neutralize(self.vwap), self.volume, 4)
        return -1 * _ts_rank(_decay_linear(values, 8), 6)

    def alpha059(self) -> pd.DataFrame:
        """Return Alpha#59."""
        values = _correlation(_neutralize(self.vwap), self.volume, 4)
        return -1 * _ts_rank(_decay_linear(values, 16), 8)

    def alpha060(self) -> pd.DataFrame:
        """Return Alpha#60."""
        denominator = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume
        return -(2 * _scale(_rank(inner / denominator)) - _scale(_rank(_ts_argmax(self.close, 10))))

    def alpha061(self) -> pd.DataFrame:
        """Return Alpha#61."""
        adv180 = _sma(self.volume, 180)
        left = _rank(self.vwap - _ts_min(self.vwap, 16))
        right = _rank(_correlation(self.vwap, adv180, 18))
        return (left < right).astype("int")

    def alpha062(self) -> pd.DataFrame:
        """Return Alpha#62."""
        adv20 = _sma(self.volume, 20)
        left = _rank(_correlation(self.vwap, _sma(adv20, 22), 10))
        right = _rank(
            (_rank(self.open) + _rank(self.open))
            < (_rank((self.high + self.low) / 2) + _rank(self.high))
        )
        return (left < right) * -1

    def alpha063(self) -> pd.DataFrame:
        """Return Alpha#63."""
        adv180 = _sma(self.volume, 180)
        left = _rank(_decay_linear(_delta(_neutralize(self.close), 2), 8))
        price_mix = (self.vwap * 0.318108) + (self.open * (1 - 0.318108))
        right = _rank(_decay_linear(_correlation(price_mix, _ts_sum(adv180, 37), 14), 12))
        return (left - right) * -1

    def alpha064(self) -> pd.DataFrame:
        """Return Alpha#64."""
        adv120 = _sma(self.volume, 120)
        price_mix = (self.open * 0.178404) + (self.low * (1 - 0.178404))
        volume_avg = _sma(adv120, 13)
        left = _rank(_correlation(_sma(price_mix, 13), volume_avg, 17))
        right_mix = (((self.high + self.low) / 2) * 0.178404) + (
            self.vwap * (1 - 0.178404)
        )
        right = _rank(_delta(right_mix, 4))
        return (left < right) * -1

    def alpha065(self) -> pd.DataFrame:
        """Return Alpha#65."""
        adv60 = _sma(self.volume, 60)
        left_mix = (self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))
        left = _rank(_correlation(left_mix, _sma(adv60, 9), 6))
        right = _rank(self.open - _ts_min(self.open, 14))
        return (left < right) * -1

    def alpha066(self) -> pd.DataFrame:
        """Return Alpha#66."""
        numerator = ((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap
        denominator = self.open - ((self.high + self.low) / 2)
        term = numerator / denominator
        return (
            _rank(_decay_linear(_delta(self.vwap, 4), 7))
            + _ts_rank(_decay_linear(term, 11), 7)
        ) * -1

    def alpha067(self) -> pd.DataFrame:
        """Return Alpha#67."""
        adv20 = _sma(self.volume, 20)
        base = _rank(self.high - _ts_min(self.high, 2))
        exponent = _rank(_correlation(_neutralize(self.vwap), _neutralize(adv20), 6))
        return base.pow(exponent) * -1

    def alpha068(self) -> pd.DataFrame:
        """Return Alpha#68."""
        adv15 = _sma(self.volume, 15)
        left = _ts_rank(_correlation(_rank(self.high), _rank(adv15), 9), 14)
        right_mix = (self.close * 0.518371) + (self.low * (1 - 0.518371))
        right = _rank(_delta(right_mix, 2)) * 14
        return (left < right) * -1

    def alpha069(self) -> pd.DataFrame:
        """Return Alpha#69."""
        adv20 = _sma(self.volume, 20)
        base = _rank(_ts_max(_delta(_neutralize(self.vwap), 3), 5))
        price_mix = (self.close * 0.490655) + (self.vwap * (1 - 0.490655))
        exponent = _ts_rank(_correlation(price_mix, adv20, 5), 9)
        return base.pow(exponent) * -1

    def alpha070(self) -> pd.DataFrame:
        """Return Alpha#70."""
        adv50 = _sma(self.volume, 50)
        base = _rank(_delta(self.vwap, 1))
        exponent = _ts_rank(_correlation(_neutralize(self.close), adv50, 18), 18)
        return base.pow(exponent) * -1

    def alpha071(self) -> pd.DataFrame:
        """Return Alpha#71."""
        adv180 = _sma(self.volume, 180)
        left = _ts_rank(
            _decay_linear(
                _correlation(_ts_rank(self.close, 3), _ts_rank(adv180, 12), 18),
                4,
            ),
            16,
        )
        price_spread_rank = _rank((self.low + self.open) - (self.vwap + self.vwap))
        right = _ts_rank(_decay_linear(price_spread_rank.pow(2), 16), 4)
        return _elementwise_max(left, right)

    def alpha072(self) -> pd.DataFrame:
        """Return Alpha#72."""
        adv40 = _sma(self.volume, 40)
        numerator = _rank(
            _decay_linear(_correlation((self.high + self.low) / 2, adv40, 9), 10)
        )
        denominator = _rank(
            _decay_linear(
                _correlation(_ts_rank(self.vwap, 4), _ts_rank(self.volume, 19), 7),
                3,
            )
        )
        return numerator / denominator

    def alpha073(self) -> pd.DataFrame:
        """Return Alpha#73."""
        left = _rank(_decay_linear(_delta(self.vwap, 5), 3))
        price_mix = (self.open * 0.147155) + (self.low * (1 - 0.147155))
        right = _ts_rank(_decay_linear((_delta(price_mix, 2) / price_mix) * -1, 3), 17)
        return -1 * _elementwise_max(left, right)

    def alpha074(self) -> pd.DataFrame:
        """Return Alpha#74."""
        adv30 = _sma(self.volume, 30)
        left = _rank(_correlation(self.close, _sma(adv30, 37), 15))
        price_mix = (self.high * 0.0261661) + (self.vwap * (1 - 0.0261661))
        right = _rank(_correlation(_rank(price_mix), _rank(self.volume), 11))
        return (left < right) * -1

    def alpha075(self) -> pd.DataFrame:
        """Return Alpha#75."""
        adv50 = _sma(self.volume, 50)
        left = _rank(_correlation(self.vwap, self.volume, 4))
        right = _rank(_correlation(_rank(self.low), _rank(adv50), 12))
        return (left < right).astype("int")

    def alpha076(self) -> pd.DataFrame:
        """Return Alpha#76."""
        adv81 = _sma(self.volume, 81)
        left = _rank(_decay_linear(_delta(self.vwap, 1), 12))
        right = _ts_rank(
            _decay_linear(
                _ts_rank(_correlation(_neutralize(self.low), adv81, 8), 20),
                17,
            ),
            19,
        )
        return -1 * _elementwise_max(left, right)

    def alpha077(self) -> pd.DataFrame:
        """Return Alpha#77."""
        adv40 = _sma(self.volume, 40)
        price_spread = (((self.high + self.low) / 2) + self.high) - (
            self.vwap + self.high
        )
        left = _rank(_decay_linear(price_spread, 20))
        right = _rank(_decay_linear(_correlation((self.high + self.low) / 2, adv40, 3), 6))
        return _elementwise_min(left, right)

    def alpha078(self) -> pd.DataFrame:
        """Return Alpha#78."""
        adv40 = _sma(self.volume, 40)
        price_mix = (self.low * 0.352233) + (self.vwap * (1 - 0.352233))
        base = _rank(_correlation(_ts_sum(price_mix, 20), _ts_sum(adv40, 20), 7))
        exponent = _rank(_correlation(_rank(self.vwap), _rank(self.volume), 6))
        return base.pow(exponent)

    def alpha079(self) -> pd.DataFrame:
        """Return Alpha#79."""
        adv150 = _sma(self.volume, 150)
        price_mix = (self.close * 0.60733) + (self.open * (1 - 0.60733))
        left = _rank(_delta(_neutralize(price_mix), 1))
        right = _rank(_correlation(_ts_rank(self.vwap, 4), _ts_rank(adv150, 9), 15))
        return (left < right).astype("int")

    def alpha080(self) -> pd.DataFrame:
        """Return Alpha#80."""
        adv10 = _sma(self.volume, 10)
        price_mix = (self.open * 0.868128) + (self.high * (1 - 0.868128))
        base = _rank(np.sign(_delta(_neutralize(price_mix), 4)))
        exponent = _ts_rank(_correlation(self.high, adv10, 5), 6)
        return base.pow(exponent) * -1

    def alpha081(self) -> pd.DataFrame:
        """Return Alpha#81."""
        adv10 = _sma(self.volume, 10)
        base = _rank(_correlation(self.vwap, _ts_sum(adv10, 50), 8)).pow(4)
        left = _rank(_log(_product(_rank(base), 15)))
        right = _rank(_correlation(_rank(self.vwap), _rank(self.volume), 5))
        return (left < right) * -1

    def alpha082(self) -> pd.DataFrame:
        """Return Alpha#82."""
        left = _rank(_decay_linear(_delta(self.open, 1), 15))
        right = _ts_rank(
            _decay_linear(
                _correlation(_neutralize(self.volume), self.open, 17),
                7,
            ),
            13,
        )
        return -1 * _elementwise_min(left, right)

    def alpha083(self) -> pd.DataFrame:
        """Return Alpha#83."""
        price_range = (self.high - self.low) / (_ts_sum(self.close, 5) / 5)
        numerator = _rank(_delay(price_range, 2)) * _rank(_rank(self.volume))
        denominator = price_range / (self.vwap - self.close)
        return numerator / denominator

    def alpha084(self) -> pd.DataFrame:
        """Return Alpha#84."""
        base = _ts_rank(self.vwap - _ts_max(self.vwap, 15), 21)
        exponent = _delta(self.close, 5)
        return base.pow(exponent)

    def alpha085(self) -> pd.DataFrame:
        """Return Alpha#85."""
        adv30 = _sma(self.volume, 30)
        price_mix = (self.high * 0.876703) + (self.close * (1 - 0.876703))
        left = _rank(_correlation(price_mix, adv30, 10))
        right = _rank(
            _correlation(_ts_rank((self.high + self.low) / 2, 4), _ts_rank(self.volume, 10), 7)
        )
        return left.pow(right)

    def alpha086(self) -> pd.DataFrame:
        """Return Alpha#86."""
        adv20 = _sma(self.volume, 20)
        left = _ts_rank(_correlation(self.close, _sma(adv20, 15), 6), 20)
        right = _rank((self.open + self.close) - (self.vwap + self.open)) * 20
        return (left < right) * -1

    def alpha087(self) -> pd.DataFrame:
        """Return Alpha#87."""
        adv81 = _sma(self.volume, 81)
        price_mix = (self.close * 0.369701) + (self.vwap * (1 - 0.369701))
        left = _rank(_decay_linear(_delta(price_mix, 2), 3))
        right = _ts_rank(
            _decay_linear(_correlation(_neutralize(adv81), self.close, 13).abs(), 5),
            14,
        )
        return -1 * _elementwise_max(left, right)

    def alpha088(self) -> pd.DataFrame:
        """Return Alpha#88."""
        adv60 = _sma(self.volume, 60)
        left = _rank(
            _decay_linear(
                (_rank(self.open) + _rank(self.low))
                - (_rank(self.high) + _rank(self.close)),
                8,
            )
        )
        right = _ts_rank(
            _decay_linear(
                _correlation(_ts_rank(self.close, 8), _ts_rank(adv60, 21), 8),
                7,
            ),
            3,
        )
        return _elementwise_min(left, right)

    def alpha089(self) -> pd.DataFrame:
        """Return Alpha#89."""
        adv10 = _sma(self.volume, 10)
        left = _ts_rank(
            _decay_linear(_correlation(self.low, adv10, 7), 6),
            4,
        )
        right = _ts_rank(_decay_linear(_delta(_neutralize(self.vwap), 3), 10), 15)
        return left - right

    def alpha090(self) -> pd.DataFrame:
        """Return Alpha#90."""
        adv40 = _sma(self.volume, 40)
        base = _rank(self.close - _ts_max(self.close, 5))
        exponent = _ts_rank(_correlation(_neutralize(adv40), self.low, 5), 3)
        return base.pow(exponent) * -1

    def alpha091(self) -> pd.DataFrame:
        """Return Alpha#91."""
        adv30 = _sma(self.volume, 30)
        left = _ts_rank(
            _decay_linear(
                _decay_linear(_correlation(_neutralize(self.close), self.volume, 10), 16),
                4,
            ),
            5,
        )
        right = _rank(_decay_linear(_correlation(self.vwap, adv30, 4), 3))
        return (left - right) * -1

    def alpha092(self) -> pd.DataFrame:
        """Return Alpha#92."""
        adv30 = _sma(self.volume, 30)
        left_condition = (((self.high + self.low) / 2) + self.close) < (
            self.low + self.open
        )
        left = _ts_rank(_decay_linear(left_condition, 15), 19)
        right = _ts_rank(
            _decay_linear(_correlation(_rank(self.low), _rank(adv30), 8), 7),
            7,
        )
        return _elementwise_min(left, right)

    def alpha093(self) -> pd.DataFrame:
        """Return Alpha#93."""
        adv81 = _sma(self.volume, 81)
        numerator = _ts_rank(
            _decay_linear(_correlation(_neutralize(self.vwap), adv81, 17), 20),
            8,
        )
        price_mix = (self.close * 0.524434) + (self.vwap * (1 - 0.524434))
        denominator = _rank(_decay_linear(_delta(price_mix, 3), 16))
        return numerator / denominator

    def alpha094(self) -> pd.DataFrame:
        """Return Alpha#94."""
        adv60 = _sma(self.volume, 60)
        base = _rank(self.vwap - _ts_min(self.vwap, 12))
        exponent = _ts_rank(
            _correlation(_ts_rank(self.vwap, 20), _ts_rank(adv60, 4), 18),
            3,
        )
        return base.pow(exponent) * -1

    def alpha095(self) -> pd.DataFrame:
        """Return Alpha#95."""
        adv40 = _sma(self.volume, 40)
        left = _rank(self.open - _ts_min(self.open, 12)) * 12
        mid_price = (self.high + self.low) / 2
        right = _ts_rank(
            _rank(_correlation(_sma(mid_price, 19), _sma(adv40, 19), 13)).pow(5),
            12,
        )
        return (left < right).astype("int")

    def alpha096(self) -> pd.DataFrame:
        """Return Alpha#96."""
        adv60 = _sma(self.volume, 60)
        left = _ts_rank(
            _decay_linear(_correlation(_rank(self.vwap), _rank(self.volume), 4), 4),
            8,
        )
        right = _ts_rank(
            _decay_linear(
                _ts_argmax(
                    _correlation(_ts_rank(self.close, 7), _ts_rank(adv60, 4), 4),
                    13,
                ),
                14,
            ),
            13,
        )
        return -1 * _elementwise_max(left, right)

    def alpha098(self) -> pd.DataFrame:
        """Return Alpha#98."""
        adv5 = _sma(self.volume, 5)
        adv15 = _sma(self.volume, 15)
        left = _rank(_decay_linear(_correlation(self.vwap, _sma(adv5, 26), 5), 7))
        right = _rank(
            _decay_linear(
                _ts_rank(
                    _ts_argmin(_correlation(_rank(self.open), _rank(adv15), 21), 9),
                    7,
                ),
                8,
            )
        )
        return left - right

    def alpha099(self) -> pd.DataFrame:
        """Return Alpha#99."""
        adv60 = _sma(self.volume, 60)
        left = _rank(
            _correlation(
                _ts_sum((self.high + self.low) / 2, 20),
                _ts_sum(adv60, 20),
                9,
            )
        )
        right = _rank(_correlation(self.low, self.volume, 6))
        return (left < right) * -1

    def alpha097(self) -> pd.DataFrame:
        """Return Alpha#97."""
        adv60 = _sma(self.volume, 60)
        price_mix = (self.low * 0.721001) + (self.vwap * (1 - 0.721001))
        left = _rank(_decay_linear(_delta(_neutralize(price_mix), 3), 20))
        right = _ts_rank(
            _decay_linear(
                _ts_rank(
                    _correlation(_ts_rank(self.low, 8), _ts_rank(adv60, 17), 5),
                    19,
                ),
                16,
            ),
            7,
        )
        return (left - right) * -1

    def alpha100(self) -> pd.DataFrame:
        """Return Alpha#100."""
        adv20 = _sma(self.volume, 20)
        price_balance = (
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low).replace(0, 0.0001)
            * self.volume
        )
        left = 1.5 * _scale(_neutralize(_neutralize(_rank(price_balance))))
        right_base = _correlation(self.close, _rank(adv20), 5) - _rank(
            _ts_argmin(self.close, 30)
        )
        right = _scale(_neutralize(right_base))
        return -1 * ((left - right) * (self.volume / adv20))

    def alpha101(self) -> pd.DataFrame:
        """Return Alpha#101."""
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


def _pivot_stock_data(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Return stock data pivoted to the Alpha101 formula layout."""
    if isinstance(stock_data.index, pd.MultiIndex) and set(stock_data.index.names) >= {
        "timestamp",
        "symbol",
    }:
        stock_data = (
            stock_data.reset_index()
            .rename(columns={"timestamp": "date"})
            .copy()
        )
    if "vwap" not in stock_data.columns:
        stock_data = stock_data.copy()
        stock_data["vwap"] = stock_data[["open", "high", "low", "close"]].mean(axis=1)
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "vwap"}
    missing = required.difference(stock_data.columns)
    if missing:
        raise ValueError(f"stock_data is missing required columns: {sorted(missing)}")
    return stock_data.pivot(index="date", columns="symbol")


def _returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-period simple returns."""
    return frame.rolling(2).apply(lambda values: values.iloc[-1] / values.iloc[0]) - 1


def _stddev(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sample standard deviation."""
    return frame.rolling(window).std()


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sum."""
    return frame.rolling(window).sum()


def _product(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling product."""
    return frame.rolling(window).apply(np.prod)


def _log(frame: pd.DataFrame) -> pd.DataFrame:
    """Return natural logarithm."""
    return np.log(frame)


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


def _neutralize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic cross-sectional neutralization fallback."""
    return frame.sub(frame.mean(axis=1), axis=0)


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


def _ts_argmin(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return the one-based position of the rolling minimum."""
    return frame.rolling(window).apply(np.argmin) + 1


def _ts_rank(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling rank of the latest value."""
    return frame.rolling(window).apply(_rolling_rank)


def _ts_min(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling minimum."""
    return frame.rolling(window).min()


def _ts_max(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling maximum."""
    return frame.rolling(window).max()


def _elementwise_max(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise maximum matching the Alpha101 reference helper."""
    return pd.DataFrame(np.maximum(left, right), index=left.index, columns=left.columns)


def _elementwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise minimum matching the Alpha101 reference helper."""
    return pd.DataFrame(np.minimum(left, right), index=left.index, columns=left.columns)
