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
        "alpha041",
        "alpha042",
        "alpha043",
        "alpha044",
        "alpha045",
        "alpha046",
        "alpha047",
        "alpha048",
        "alpha049",
        "alpha050",
        "alpha051",
        "alpha052",
        "alpha053",
        "alpha054",
        "alpha055",
        "alpha056",
        "alpha057",
        "alpha058",
        "alpha059",
        "alpha060",
        "alpha061",
        "alpha062",
        "alpha063",
        "alpha064",
        "alpha065",
        "alpha066",
        "alpha067",
        "alpha068",
        "alpha069",
        "alpha070",
        "alpha071",
        "alpha072",
        "alpha073",
        "alpha074",
        "alpha075",
        "alpha076",
        "alpha077",
        "alpha078",
        "alpha079",
        "alpha080",
        "alpha081",
        "alpha082",
        "alpha083",
        "alpha084",
        "alpha085",
        "alpha086",
        "alpha087",
        "alpha088",
        "alpha089",
        "alpha090",
        "alpha091",
        "alpha092",
        "alpha093",
        "alpha094",
        "alpha095",
        "alpha096",
        "alpha097",
        "alpha098",
        "alpha099",
        "alpha100",
        "alpha101",
        "alpha102",
        "alpha103",
        "alpha104",
        "alpha105",
        "alpha106",
        "alpha107",
        "alpha108",
        "alpha109",
        "alpha110",
        "alpha111",
        "alpha112",
        "alpha113",
        "alpha114",
        "alpha115",
        "alpha116",
        "alpha117",
        "alpha118",
        "alpha119",
        "alpha120",
        "alpha121",
        "alpha122",
        "alpha123",
        "alpha124",
        "alpha125",
        "alpha126",
        "alpha127",
        "alpha128",
        "alpha129",
        "alpha130",
        "alpha131",
        "alpha132",
        "alpha133",
        "alpha134",
        "alpha135",
        "alpha136",
        "alpha137",
        "alpha138",
        "alpha139",
        "alpha140",
        "alpha141",
        "alpha142",
        "alpha144",
        "alpha145",
        "alpha146",
        "alpha147",
        "alpha148",
        "alpha150",
        "alpha151",
        "alpha152",
        "alpha153",
        "alpha154",
        "alpha155",
        "alpha156",
        "alpha157",
        "alpha158",
        "alpha159",
        "alpha160",
        "alpha161",
        "alpha162",
        "alpha163",
        "alpha164",
        "alpha165",
        "alpha166",
        "alpha167",
        "alpha168",
        "alpha169",
        "alpha170",
        "alpha171",
        "alpha172",
        "alpha173",
        "alpha174",
        "alpha175",
        "alpha176",
        "alpha177",
        "alpha178",
        "alpha179",
        "alpha180",
        "alpha181",
        "alpha182",
        "alpha183",
        "alpha184",
        "alpha185",
        "alpha186",
        "alpha187",
        "alpha188",
        "alpha189",
        "alpha191",
    }
)

ALPHA191_SKIPPED = {
    "alpha030": "requires external MKT/SMB/HML regression inputs",
    "alpha143": "legacy formula is commented placeholder",
    "alpha149": "requires benchmark filter input",
    "alpha190": "legacy formula is commented placeholder",
}


def alpha191(
    stock_data: pd.DataFrame,
    bench_data: pd.DataFrame,
    names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return selected Alpha191 factors in legacy Query-compatible long form."""
    selected = list(names or sorted(ALPHA191_SUPPORTED))
    unsupported = sorted(set(selected).difference(ALPHA191_SUPPORTED))
    if unsupported:
        skipped = [name for name in unsupported if name in ALPHA191_SKIPPED]
        unknown = [name for name in unsupported if name not in ALPHA191_SKIPPED]
        if skipped and not unknown:
            details = "; ".join(
                f"{name}: {ALPHA191_SKIPPED[name]}" for name in skipped
            )
            raise ValueError(f"skipped Alpha191 formulas are not supported: {details}")
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

    def alpha041(self) -> pd.DataFrame:
        """Return Alpha#41."""
        return _rank(_ts_max(_delta(self.vwap, 3), 5)) * -1

    def alpha042(self) -> pd.DataFrame:
        """Return Alpha#42."""
        return -1 * _rank(_stddev(self.high, 10)) * _correlation(
            self.high,
            self.volume,
            10,
        )

    def alpha043(self) -> pd.DataFrame:
        """Return Alpha#43."""
        cond1 = self.close > _delay(self.close, 1)
        cond2 = self.close < _delay(self.close, 1)
        cond3 = self.close == _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = self.volume
        part[cond2] = -self.volume
        part[cond3] = 0
        return _ts_sum(part, 6)

    def alpha044(self) -> pd.DataFrame:
        """Return Alpha#44."""
        return _ts_rank(
            _decay_linear(_correlation(self.low, _mean(self.volume, 10), 7), 6),
            4,
        ) + _ts_rank(_decay_linear(_delta(self.vwap, 3), 10), 15)

    def alpha045(self) -> pd.DataFrame:
        """Return Alpha#45."""
        return _rank(_delta((self.close * 0.6) + (self.open * 0.4), 1)) * _rank(
            _correlation(self.vwap, _mean(self.volume, 150), 15)
        )

    def alpha046(self) -> pd.DataFrame:
        """Return Alpha#46."""
        return (
            _mean(self.close, 3)
            + _mean(self.close, 6)
            + _mean(self.close, 12)
            + _mean(self.close, 24)
        ) / (4 * self.close)

    def alpha047(self) -> pd.DataFrame:
        """Return Alpha#47."""
        return _sma(
            (_ts_max(self.high, 6) - self.close)
            / (_ts_max(self.high, 6) - _ts_min(self.low, 6))
            * 100,
            9,
            1,
        )

    def alpha048(self) -> pd.DataFrame:
        """Return Alpha#48."""
        signs = (
            np.sign(self.close - _delay(self.close, 1))
            + np.sign(_delay(self.close, 1) - _delay(self.close, 2))
            + np.sign(_delay(self.close, 2) - _delay(self.close, 3))
        )
        return -1 * _rank(signs) * _ts_sum(self.volume, 5) / _ts_sum(self.volume, 20)

    def alpha049(self) -> pd.DataFrame:
        """Return Alpha#49."""
        part1, part2 = _directional_range_parts(
            self.close,
            self.high,
            self.low,
            (self.high + self.low) > (_delay(self.high, 1) + _delay(self.low, 1)),
        )
        return _ts_sum(part1, 12) / (_ts_sum(part1, 12) + _ts_sum(part2, 12))

    def alpha050(self) -> pd.DataFrame:
        """Return Alpha#50."""
        part1, part2 = _directional_range_parts(
            self.close,
            self.high,
            self.low,
            (self.high + self.low) <= (_delay(self.high, 1) + _delay(self.low, 1)),
        )
        return (_ts_sum(part1, 12) - _ts_sum(part2, 12)) / (
            _ts_sum(part1, 12) + _ts_sum(part2, 12)
        )

    def alpha051(self) -> pd.DataFrame:
        """Return Alpha#51."""
        part1, part2 = _directional_range_parts(
            self.close,
            self.high,
            self.low,
            (self.high + self.low) <= (_delay(self.high, 1) + _delay(self.low, 1)),
        )
        return _ts_sum(part1, 12) / (_ts_sum(part1, 12) + _ts_sum(part2, 12))

    def alpha052(self) -> pd.DataFrame:
        """Return Alpha#52."""
        typical_price = (self.high + self.low + self.close) / 3
        return (
            _ts_sum(_elementwise_max(self.high - _delay(typical_price, 1), 0), 26)
            / _ts_sum(_elementwise_max(_delay(typical_price, 1) - self.low, 0), 26)
            * 100
        )

    def alpha053(self) -> pd.DataFrame:
        """Return Alpha#53."""
        return _count(self.close > _delay(self.close, 1), 12) / 12 * 100

    def alpha054(self) -> pd.DataFrame:
        """Return Alpha#54."""
        spread = self.close - self.open
        return -1 * _rank(spread.abs().std() + spread + _correlation(self.close, self.open, 10))

    def alpha055(self) -> pd.DataFrame:
        """Return Alpha#55."""
        high_close = (self.high - _delay(self.close, 1)).abs()
        low_close = (self.low - _delay(self.close, 1)).abs()
        high_low = (self.high - _delay(self.low, 1)).abs()
        cond1 = (high_close > low_close) & (high_close > high_low)
        cond2 = (low_close > high_low) & (low_close > high_close)
        cond3 = (high_low >= high_close) & (high_low >= low_close)
        numerator = 16 * (self.close + (self.close - self.open) / 2 - _delay(self.open, 1))
        denominator = self.close.copy(deep=True)
        denominator.loc[:, :] = 0
        denominator[cond1] = (
            high_close
            + low_close / 2
            + (_delay(self.close, 1) - _delay(self.open, 1)).abs() / 4
        )
        denominator[cond2] = (
            low_close
            + high_close / 2
            + (_delay(self.close, 1) - _delay(self.open, 1)).abs() / 4
        )
        denominator[cond3] = (
            high_low + (_delay(self.close, 1) - _delay(self.open, 1)).abs() / 4
        )
        multiplier = _elementwise_max(high_close, low_close)
        return _ts_sum(numerator / denominator * multiplier, 20)

    def alpha056(self) -> pd.DataFrame:
        """Return Alpha#56."""
        left = _rank(self.open - _ts_min(self.open, 12))
        right = _rank(
            _rank(
                _correlation(
                    _ts_sum((self.high + self.low) / 2, 19),
                    _ts_sum(_mean(self.volume, 40), 19),
                    13,
                )
            )
            ** 5
        )
        cond = left < right
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha057(self) -> pd.DataFrame:
        """Return Alpha#57."""
        return _sma(
            (self.close - _ts_min(self.low, 9))
            / (_ts_max(self.high, 9) - _ts_min(self.low, 9))
            * 100,
            3,
            1,
        )

    def alpha058(self) -> pd.DataFrame:
        """Return Alpha#58."""
        return _count(self.close > _delay(self.close, 1), 20) / 20 * 100

    def alpha059(self) -> pd.DataFrame:
        """Return Alpha#59."""
        cond1 = self.close == _delay(self.close, 1)
        cond2 = self.close > _delay(self.close, 1)
        cond3 = self.close < _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = 0
        part[cond2] = self.close - _elementwise_min(self.low, _delay(self.close, 1))
        part[cond3] = self.close - _elementwise_max(self.low, _delay(self.close, 1))
        return _ts_sum(part, 20)

    def alpha060(self) -> pd.DataFrame:
        """Return Alpha#60."""
        return _ts_sum(
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low)
            * self.volume,
            20,
        )

    def alpha061(self) -> pd.DataFrame:
        """Return Alpha#61."""
        left = _rank(_decay_linear(_delta(self.vwap, 1), 12))
        right = _rank(
            _decay_linear(_rank(_correlation(self.low, _mean(self.volume, 80), 8)), 17)
        )
        return _elementwise_max(left, right) * -1

    def alpha062(self) -> pd.DataFrame:
        """Return Alpha#62."""
        return -1 * _correlation(self.high, _rank(self.volume), 5)

    def alpha063(self) -> pd.DataFrame:
        """Return Alpha#63."""
        close_delta = self.close - _delay(self.close, 1)
        return (
            _sma(_elementwise_max(close_delta, 0), 6, 1)
            / _sma(close_delta.abs(), 6, 1)
            * 100
        )

    def alpha064(self) -> pd.DataFrame:
        """Return Alpha#64."""
        left = _rank(
            _decay_linear(
                _correlation(_rank(self.vwap), _rank(self.volume), 4),
                4,
            )
        )
        right = _rank(
            _decay_linear(
                _ts_max(
                    _correlation(_rank(self.close), _rank(_mean(self.volume, 60)), 4),
                    13,
                ),
                14,
            )
        )
        return _elementwise_max(left, right) * -1

    def alpha065(self) -> pd.DataFrame:
        """Return Alpha#65."""
        return _mean(self.close, 6) / self.close

    def alpha066(self) -> pd.DataFrame:
        """Return Alpha#66."""
        close_mean = _mean(self.close, 6)
        return (self.close - close_mean) / close_mean * 100

    def alpha067(self) -> pd.DataFrame:
        """Return Alpha#67."""
        close_delta = self.close - _delay(self.close, 1)
        return (
            _sma(_elementwise_max(close_delta, 0), 24, 1)
            / _sma(close_delta.abs(), 24, 1)
            * 100
        )

    def alpha068(self) -> pd.DataFrame:
        """Return Alpha#68."""
        return _sma(
            (
                (self.high + self.low) / 2
                - (_delay(self.high, 1) + _delay(self.low, 1)) / 2
            )
            * (self.high - self.low)
            / self.volume,
            15,
            2,
        )

    def alpha069(self) -> pd.DataFrame:
        """Return Alpha#69."""
        cond1 = self.open <= _delay(self.open, 1)
        cond2 = self.open >= _delay(self.open, 1)

        dtm = self.close.copy(deep=True)
        dtm.loc[:, :] = np.nan
        dtm[cond1] = 0
        dtm[~cond1] = _elementwise_max(
            self.high - self.open,
            self.open - _delay(self.open, 1),
        )

        dbm = self.close.copy(deep=True)
        dbm.loc[:, :] = np.nan
        dbm[cond2] = 0
        dbm[~cond2] = _elementwise_max(
            self.open - self.low,
            self.open - _delay(self.open, 1),
        )

        dtm_sum = _ts_sum(dtm, 20)
        dbm_sum = _ts_sum(dbm, 20)
        cond3 = dtm_sum > dbm_sum
        cond4 = dtm_sum == dbm_sum
        cond5 = dtm_sum < dbm_sum
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond3] = (dtm_sum - dbm_sum) / dtm_sum
        part[cond4] = 0
        part[cond5] = (dtm_sum - dbm_sum) / dbm_sum
        return part

    def alpha070(self) -> pd.DataFrame:
        """Return Alpha#70."""
        return _stddev(self.amount, 6)

    def alpha071(self) -> pd.DataFrame:
        """Return Alpha#71."""
        close_mean = _mean(self.close, 24)
        return (self.close - close_mean) / close_mean * 100

    def alpha072(self) -> pd.DataFrame:
        """Return Alpha#72."""
        return _sma(
            (_ts_max(self.high, 6) - self.close)
            / (_ts_max(self.high, 6) - _ts_min(self.low, 6))
            * 100,
            15,
            1,
        )

    def alpha073(self) -> pd.DataFrame:
        """Return Alpha#73."""
        left = _ts_rank(
            _decay_linear(
                _decay_linear(_correlation(self.close, self.volume, 10), 16),
                4,
            ),
            5,
        )
        right = _rank(
            _decay_linear(_correlation(self.vwap, _mean(self.volume, 30), 4), 3)
        )
        return (left - right) * -1

    def alpha074(self) -> pd.DataFrame:
        """Return Alpha#74."""
        left = _rank(
            _correlation(
                _ts_sum((self.low * 0.35) + (self.vwap * 0.65), 20),
                _ts_sum(_mean(self.volume, 40), 20),
                7,
            )
        )
        right = _rank(_correlation(_rank(self.vwap), _rank(self.volume), 6))
        return left + right

    def alpha075(self) -> pd.DataFrame:
        """Return Alpha#75."""
        benchmark_down = _broadcast_series(
            self.benchmark_close < self.benchmark_open,
            self.close,
        )
        return _count((self.close > self.open) & benchmark_down, 50) / _count(
            benchmark_down, 50
        )

    def alpha076(self) -> pd.DataFrame:
        """Return Alpha#76."""
        values = ((self.close / _delay(self.close, 1) - 1).abs()) / self.volume
        return _stddev(values, 20) / _mean(values, 20)

    def alpha077(self) -> pd.DataFrame:
        """Return Alpha#77."""
        left = _rank(
            _decay_linear(
                (((self.high + self.low) / 2) + self.high) - (self.vwap + self.high),
                20,
            )
        )
        right = _rank(
            _decay_linear(
                _correlation((self.high + self.low) / 2, _mean(self.volume, 40), 3),
                6,
            )
        )
        return _elementwise_min(left, right)

    def alpha078(self) -> pd.DataFrame:
        """Return Alpha#78."""
        typical_price = (self.high + self.low + self.close) / 3
        return (typical_price - _mean(typical_price, 12)) / (
            0.015 * _mean((self.close - _mean(typical_price, 12)).abs(), 12)
        )

    def alpha079(self) -> pd.DataFrame:
        """Return Alpha#79."""
        close_delta = self.close - _delay(self.close, 1)
        return (
            _sma(_elementwise_max(close_delta, 0), 12, 1)
            / _sma(close_delta.abs(), 12, 1)
            * 100
        )

    def alpha080(self) -> pd.DataFrame:
        """Return Alpha#80."""
        delayed_volume = _delay(self.volume, 5)
        return (self.volume - delayed_volume) / delayed_volume * 100

    def alpha081(self) -> pd.DataFrame:
        """Return Alpha#81."""
        return _sma(self.volume, 21, 2)

    def alpha082(self) -> pd.DataFrame:
        """Return Alpha#82."""
        return _sma(
            (_ts_max(self.high, 6) - self.close)
            / (_ts_max(self.high, 6) - _ts_min(self.low, 6))
            * 100,
            20,
            1,
        )

    def alpha083(self) -> pd.DataFrame:
        """Return Alpha#83."""
        return -1 * _rank(_covariance(_rank(self.high), _rank(self.volume), 5))

    def alpha084(self) -> pd.DataFrame:
        """Return Alpha#84."""
        cond1 = self.close > _delay(self.close, 1)
        cond2 = self.close < _delay(self.close, 1)
        cond3 = self.close == _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = self.volume
        part[cond2] = 0
        part[cond3] = -self.volume
        return _ts_sum(part, 20)

    def alpha085(self) -> pd.DataFrame:
        """Return Alpha#85."""
        return _ts_rank(self.volume / _mean(self.volume, 20), 20) * _ts_rank(
            -1 * _delta(self.close, 7),
            8,
        )

    def alpha086(self) -> pd.DataFrame:
        """Return Alpha#86."""
        values = ((_delay(self.close, 20) - _delay(self.close, 10)) / 10) - (
            (_delay(self.close, 10) - self.close) / 10
        )
        cond1 = values > 0.25
        cond2 = values < 0.0
        cond3 = (0 <= values) & (values <= 0.25)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1 * (self.close - _delay(self.close, 1))
        return part

    def alpha087(self) -> pd.DataFrame:
        """Return Alpha#87."""
        left = _rank(_decay_linear(_delta(self.vwap, 4), 7))
        right = _ts_rank(
            _decay_linear(
                (((self.low * 0.9) + (self.low * 0.1)) - self.vwap)
                / (self.open - ((self.high + self.low) / 2)),
                11,
            ),
            7,
        )
        return (left + right) * -1

    def alpha088(self) -> pd.DataFrame:
        """Return Alpha#88."""
        delayed_close = _delay(self.close, 20)
        return (self.close - delayed_close) / delayed_close * 100

    def alpha089(self) -> pd.DataFrame:
        """Return Alpha#89."""
        macd_like = _sma(self.close, 13, 2) - _sma(self.close, 27, 2)
        return 2 * (macd_like - _sma(macd_like, 10, 2))

    def alpha090(self) -> pd.DataFrame:
        """Return Alpha#90."""
        return _rank(_correlation(_rank(self.vwap), _rank(self.volume), 5)) * -1

    def alpha091(self) -> pd.DataFrame:
        """Return Alpha#91."""
        return (
            _rank(self.close - _ts_max(self.close, 5))
            * _rank(_correlation(_mean(self.volume, 40), self.low, 5))
        ) * -1

    def alpha092(self) -> pd.DataFrame:
        """Return Alpha#92."""
        left = _rank(
            _decay_linear(
                _delta((self.close * 0.35) + (self.vwap * 0.65), 2),
                3,
            )
        )
        right = _ts_rank(
            _decay_linear(
                _correlation(_mean(self.volume, 180), self.close, 13).abs(),
                5,
            ),
            15,
        )
        return _elementwise_max(left, right) * -1

    def alpha093(self) -> pd.DataFrame:
        """Return Alpha#93."""
        cond = self.open >= _delay(self.open, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 0
        part[~cond] = _elementwise_max(
            self.open - self.low,
            self.open - _delay(self.open, 1),
        )
        return _ts_sum(part, 20)

    def alpha094(self) -> pd.DataFrame:
        """Return Alpha#94."""
        cond1 = self.close > _delay(self.close, 1)
        cond2 = self.close < _delay(self.close, 1)
        cond3 = self.close == _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond1] = self.volume
        part[cond2] = -1 * self.volume
        part[cond3] = 0
        return _ts_sum(part, 30)

    def alpha095(self) -> pd.DataFrame:
        """Return Alpha#95."""
        return _stddev(self.amount, 20)

    def alpha096(self) -> pd.DataFrame:
        """Return Alpha#96."""
        numerator = self.close - _ts_min(self.low, 9)
        denominator = _ts_max(self.high, 9) - _ts_min(self.low, 9)
        return _sma(_sma(numerator / denominator * 100, 3, 1), 3, 1)

    def alpha097(self) -> pd.DataFrame:
        """Return Alpha#97."""
        return _stddev(self.volume, 10)

    def alpha098(self) -> pd.DataFrame:
        """Return Alpha#98."""
        cond = _delta(_ts_sum(self.close, 100) / 100, 100) / _delay(
            self.close,
            100,
        ) <= 0.05
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = -1 * (self.close - _ts_min(self.close, 100))
        part[~cond] = -1 * _delta(self.close, 3)
        return part

    def alpha099(self) -> pd.DataFrame:
        """Return Alpha#99."""
        return -1 * _rank(_covariance(_rank(self.close), _rank(self.volume), 5))

    def alpha100(self) -> pd.DataFrame:
        """Return Alpha#100."""
        return _stddev(self.volume, 20)

    def alpha101(self) -> pd.DataFrame:
        """Return Alpha#101."""
        rank1 = _rank(_correlation(self.close, _ts_sum(_mean(self.volume, 30), 37), 15))
        rank2 = _rank(
            _correlation(
                _rank((self.high * 0.1) + (self.vwap * 0.9)),
                _rank(self.volume),
                11,
            )
        )
        cond = rank1 < rank2
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha102(self) -> pd.DataFrame:
        """Return Alpha#102."""
        volume_delta = self.volume - _delay(self.volume, 1)
        return (
            _sma(_elementwise_max(volume_delta, 0), 6, 1)
            / _sma(volume_delta.abs(), 6, 1)
            * 100
        )

    def alpha103(self) -> pd.DataFrame:
        """Return Alpha#103."""
        return ((20 - _lowday(self.low, 20)) / 20) * 100

    def alpha104(self) -> pd.DataFrame:
        """Return Alpha#104."""
        return -1 * (
            _delta(_correlation(self.high, self.volume, 5), 5)
            * _rank(_stddev(self.close, 20))
        )

    def alpha105(self) -> pd.DataFrame:
        """Return Alpha#105."""
        return -1 * _correlation(_rank(self.open), _rank(self.volume), 10)

    def alpha106(self) -> pd.DataFrame:
        """Return Alpha#106."""
        return self.close - _delay(self.close, 20)

    def alpha107(self) -> pd.DataFrame:
        """Return Alpha#107."""
        return (
            (-1 * _rank(self.open - _delay(self.high, 1)))
            * _rank(self.open - _delay(self.close, 1))
            * _rank(self.open - _delay(self.low, 1))
        )

    def alpha108(self) -> pd.DataFrame:
        """Return Alpha#108."""
        return (
            _rank(self.high - _ts_min(self.high, 2))
            ** _rank(_correlation(self.vwap, _mean(self.volume, 120), 6))
        ) * -1

    def alpha109(self) -> pd.DataFrame:
        """Return Alpha#109."""
        high_low = self.high - self.low
        return _sma(high_low, 10, 2) / _sma(_sma(high_low, 10, 2), 10, 2)

    def alpha110(self) -> pd.DataFrame:
        """Return Alpha#110."""
        up = _ts_sum(_elementwise_max(self.high - _delay(self.close, 1), 0), 20)
        down = _ts_sum(_elementwise_max(_delay(self.close, 1) - self.low, 0), 20)
        return up / down * 100

    def alpha111(self) -> pd.DataFrame:
        """Return Alpha#111."""
        values = self.volume * (
            ((self.close - self.low) - (self.high - self.close))
            / (self.high - self.low)
        )
        return _sma(values, 11, 2) - _sma(values, 4, 2)

    def alpha112(self) -> pd.DataFrame:
        """Return Alpha#112."""
        close_delta = self.close - _delay(self.close, 1)
        cond = close_delta > 0
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond] = close_delta
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[~cond] = close_delta.abs()
        part2[cond] = 0
        part1_sum = _ts_sum(part1, 12)
        part2_sum = _ts_sum(part2, 12)
        return (part1_sum - part2_sum) / (part1_sum + part2_sum) * 100

    def alpha113(self) -> pd.DataFrame:
        """Return Alpha#113."""
        return -1 * (
            _rank(_ts_sum(_delay(self.close, 5), 20) / 20)
            * _correlation(self.close, self.volume, 2)
            * _rank(_correlation(_ts_sum(self.close, 5), _ts_sum(self.close, 20), 2))
        )

    def alpha114(self) -> pd.DataFrame:
        """Return Alpha#114."""
        spread_ratio = (self.high - self.low) / (_ts_sum(self.close, 5) / 5)
        return (
            _rank(_delay(spread_ratio, 2)) * _rank(_rank(self.volume))
        ) / (spread_ratio / (self.vwap - self.close))

    def alpha115(self) -> pd.DataFrame:
        """Return Alpha#115."""
        left = _rank(
            _correlation((self.high * 0.9) + (self.close * 0.1), _mean(self.volume, 30), 10)
        )
        right = _rank(
            _correlation(
                _ts_rank((self.high + self.low) / 2, 4),
                _ts_rank(self.volume, 10),
                7,
            )
        )
        return left**right

    def alpha116(self) -> pd.DataFrame:
        """Return Alpha#116."""
        return _regbeta(self.close, _sequence(20))

    def alpha117(self) -> pd.DataFrame:
        """Return Alpha#117."""
        return (
            _ts_rank(self.volume, 32)
            * (1 - _ts_rank((self.close + self.high) - self.low, 16))
            * (1 - _ts_rank(self.returns, 32))
        )

    def alpha118(self) -> pd.DataFrame:
        """Return Alpha#118."""
        return _ts_sum(self.high - self.open, 20) / _ts_sum(self.open - self.low, 20) * 100

    def alpha119(self) -> pd.DataFrame:
        """Return Alpha#119."""
        left = _rank(
            _decay_linear(
                _correlation(self.vwap, _ts_sum(_mean(self.volume, 5), 26), 5),
                7,
            )
        )
        right = _rank(
            _decay_linear(
                _ts_rank(
                    _ts_min(
                        _correlation(_rank(self.open), _rank(_mean(self.volume, 15)), 21),
                        9,
                    ),
                    7,
                ),
                8,
            )
        )
        return left - right

    def alpha120(self) -> pd.DataFrame:
        """Return Alpha#120."""
        return _rank(self.vwap - self.close) / _rank(self.vwap + self.close)

    def alpha121(self) -> pd.DataFrame:
        """Return Alpha#121."""
        return (
            _rank(self.vwap - _ts_min(self.vwap, 12))
            ** _ts_rank(
                _correlation(
                    _ts_rank(self.vwap, 20),
                    _ts_rank(_mean(self.volume, 60), 2),
                    18,
                ),
                3,
            )
        ) * -1

    def alpha122(self) -> pd.DataFrame:
        """Return Alpha#122."""
        smoothed = _sma(_sma(_sma(np.log(self.close), 13, 2), 13, 2), 13, 2)
        return (smoothed - _delay(smoothed, 1)) / _delay(smoothed, 1)

    def alpha123(self) -> pd.DataFrame:
        """Return Alpha#123."""
        left = _rank(
            _correlation(
                _ts_sum((self.high + self.low) / 2, 20),
                _ts_sum(_mean(self.volume, 60), 20),
                9,
            )
        )
        right = _rank(_correlation(self.low, self.volume, 6))
        cond = left < right
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = -1
        part[~cond] = 0
        return part

    def alpha124(self) -> pd.DataFrame:
        """Return Alpha#124."""
        return (self.close - self.vwap) / _decay_linear(_rank(_ts_max(self.close, 30)), 2)

    def alpha125(self) -> pd.DataFrame:
        """Return Alpha#125."""
        left = _rank(
            _decay_linear(_correlation(self.vwap, _mean(self.volume, 80), 17), 20)
        )
        right = _rank(
            _decay_linear(_delta((self.close * 0.5) + (self.vwap * 0.5), 3), 16)
        )
        return left / right

    def alpha126(self) -> pd.DataFrame:
        """Return Alpha#126."""
        return (self.close + self.high + self.low) / 3

    def alpha127(self) -> pd.DataFrame:
        """Return Alpha#127."""
        max_close = _ts_max(self.close, 12)
        return _mean((100 * (self.close - max_close) / max_close) ** 2, 12) ** (1 / 2)

    def alpha128(self) -> pd.DataFrame:
        """Return Alpha#128."""
        typical_price = (self.high + self.low + self.close) / 3
        cond = typical_price > _delay(typical_price, 1)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond] = typical_price * self.volume
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[~cond] = typical_price * self.volume
        part2[cond] = 0
        return 100 - (100 / (1 + _ts_sum(part1, 14) / _ts_sum(part2, 14)))

    def alpha129(self) -> pd.DataFrame:
        """Return Alpha#129."""
        close_delta = self.close - _delay(self.close, 1)
        cond = close_delta < 0
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = close_delta.abs()
        part[~cond] = 0
        return _ts_sum(part, 12)

    def alpha130(self) -> pd.DataFrame:
        """Return Alpha#130."""
        left = _rank(
            _decay_linear(
                _correlation((self.high + self.low) / 2, _mean(self.volume, 40), 9),
                10,
            )
        )
        right = _rank(
            _decay_linear(_correlation(_rank(self.vwap), _rank(self.volume), 7), 3)
        )
        return left / right

    def alpha131(self) -> pd.DataFrame:
        """Return Alpha#131."""
        return _rank(_delta(self.vwap, 1)) ** _ts_rank(
            _correlation(self.close, _mean(self.volume, 50), 18),
            18,
        )

    def alpha132(self) -> pd.DataFrame:
        """Return Alpha#132."""
        return _mean(self.amount, 20)

    def alpha133(self) -> pd.DataFrame:
        """Return Alpha#133."""
        return ((20 - _highday(self.high, 20)) / 20) * 100 - (
            (20 - _lowday(self.low, 20)) / 20
        ) * 100

    def alpha134(self) -> pd.DataFrame:
        """Return Alpha#134."""
        return (self.close - _delay(self.close, 12)) / _delay(self.close, 12) * self.volume

    def alpha135(self) -> pd.DataFrame:
        """Return Alpha#135."""
        return _sma(_delay(self.close / _delay(self.close, 20), 1), 20, 1)

    def alpha136(self) -> pd.DataFrame:
        """Return Alpha#136."""
        return -1 * _rank(_delta(self.returns, 3)) * _correlation(
            self.open,
            self.volume,
            10,
        )

    def alpha137(self) -> pd.DataFrame:
        """Return Alpha#137."""
        high_close = (self.high - _delay(self.close, 1)).abs()
        low_close = (self.low - _delay(self.close, 1)).abs()
        high_low = (self.high - _delay(self.low, 1)).abs()
        close_open = (_delay(self.close, 1) - _delay(self.open, 1)).abs()
        cond1 = (high_close > low_close) & (high_close > high_low)
        cond2 = (low_close > high_low) & (low_close > high_close)
        cond3 = ~cond1 & ~cond2
        numerator = 16 * (
            self.close
            + (self.close - self.open) / 2
            - _delay(self.open, 1)
        )
        denominator = self.close.copy(deep=True)
        denominator.loc[:, :] = np.nan
        denominator[cond1] = high_close + low_close / 2 + close_open / 4
        denominator[cond2] = low_close + high_close / 2 + close_open / 4
        denominator[cond3] = high_low + close_open / 4
        denominator.replace({0: np.nan}, inplace=True)
        return numerator / denominator * _elementwise_max(high_close, low_close)

    def alpha138(self) -> pd.DataFrame:
        """Return Alpha#138."""
        left = _rank(
            _decay_linear(_delta((self.low * 0.7) + (self.vwap * 0.3), 3), 20)
        )
        right = _ts_rank(
            _decay_linear(
                _ts_rank(
                    _correlation(
                        _ts_rank(self.low, 8),
                        _ts_rank(_mean(self.volume, 60), 17),
                        5,
                    ),
                    19,
                ),
                16,
            ),
            7,
        )
        return (left - right) * -1

    def alpha139(self) -> pd.DataFrame:
        """Return Alpha#139."""
        return -1 * _correlation(self.open, self.volume, 10)

    def alpha140(self) -> pd.DataFrame:
        """Return Alpha#140."""
        left = _rank(
            _decay_linear(
                (_rank(self.open) + _rank(self.low))
                - (_rank(self.high) + _rank(self.close)),
                8,
            )
        )
        right = _ts_rank(
            _decay_linear(
                _correlation(
                    _ts_rank(self.close, 8),
                    _ts_rank(_mean(self.volume, 60), 20),
                    8,
                ),
                7,
            ),
            3,
        )
        return _elementwise_min(left, right)

    def alpha141(self) -> pd.DataFrame:
        """Return Alpha#141."""
        return _rank(_correlation(_rank(self.high), _rank(_mean(self.volume, 15)), 9)) * -1

    def alpha142(self) -> pd.DataFrame:
        """Return Alpha#142."""
        return (
            -1
            * _rank(_ts_rank(self.close, 10))
            * _rank(_delta(_delta(self.close, 1), 1))
            * _rank(_ts_rank(self.volume / _mean(self.volume, 20), 5))
        )

    def alpha144(self) -> pd.DataFrame:
        """Return Alpha#144."""
        cond = self.close < _delay(self.close, 1)
        values = (self.close / _delay(self.close, 1) - 1).abs() / self.amount
        return _sumif(values, 20, cond) / _count(cond, 20)

    def alpha145(self) -> pd.DataFrame:
        """Return Alpha#145."""
        return (_mean(self.volume, 9) - _mean(self.volume, 26)) / _mean(
            self.volume,
            12,
        ) * 100

    def alpha146(self) -> pd.DataFrame:
        """Return Alpha#146."""
        returns = (self.close - _delay(self.close, 1)) / _delay(self.close, 1)
        smoothed = _sma(returns, 61, 2)
        residual = returns - smoothed
        return _mean(residual, 20) * residual / _sma((returns - residual) ** 2, 61, 2)

    def alpha147(self) -> pd.DataFrame:
        """Return Alpha#147."""
        return _regbeta(_mean(self.close, 12), _sequence(12))

    def alpha148(self) -> pd.DataFrame:
        """Return Alpha#148."""
        cond = _rank(
            _correlation(self.open, _ts_sum(_mean(self.volume, 60), 9), 6)
        ) < _rank(self.open - _ts_min(self.open, 14))
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = -1
        part[~cond] = 0
        return part

    def alpha150(self) -> pd.DataFrame:
        """Return Alpha#150."""
        return (self.close + self.high + self.low) / 3 * self.volume

    def alpha151(self) -> pd.DataFrame:
        """Return Alpha#151."""
        return _sma(self.close - _delay(self.close, 20), 20, 1)

    def alpha152(self) -> pd.DataFrame:
        """Return Alpha#152."""
        delayed_ratio = _delay(self.close / _delay(self.close, 9), 1)
        smoothed = _sma(delayed_ratio, 9, 1)
        delayed_smoothed = _delay(smoothed, 1)
        return _sma(_mean(delayed_smoothed, 12) - _mean(delayed_smoothed, 26), 9, 1)

    def alpha153(self) -> pd.DataFrame:
        """Return Alpha#153."""
        return (
            _mean(self.close, 3)
            + _mean(self.close, 6)
            + _mean(self.close, 12)
            + _mean(self.close, 24)
        ) / 4

    def alpha154(self) -> pd.DataFrame:
        """Return Alpha#154."""
        cond = (self.vwap - _ts_min(self.vwap, 16)) < _correlation(
            self.vwap,
            _mean(self.volume, 180),
            18,
        )
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 1
        part[~cond] = 0
        return part

    def alpha155(self) -> pd.DataFrame:
        """Return Alpha#155."""
        diff = _sma(self.volume, 13, 2) - _sma(self.volume, 27, 2)
        return diff - _sma(diff, 10, 2)

    def alpha156(self) -> pd.DataFrame:
        """Return Alpha#156."""
        mixed_price = self.open * 0.15 + self.low * 0.85
        left = _rank(_decay_linear(_delta(self.vwap, 5), 3))
        right = _rank(
            _decay_linear(_delta(mixed_price, 2) / mixed_price * -1, 3),
        )
        return _elementwise_max(left, right) * -1

    def alpha157(self) -> pd.DataFrame:
        """Return Alpha#157."""
        nested_rank = _rank(
            _rank(
                -1 * _rank(_delta(self.close - 1, 5)),
            ),
        )
        logged = np.log(_ts_sum(_ts_min(nested_rank, 2), 1))
        left = _ts_min(_prod(_rank(_rank(logged)), 1), 5)
        right = _ts_rank(_delay(-1 * self.returns, 6), 5)
        return left + right

    def alpha158(self) -> pd.DataFrame:
        """Return Alpha#158."""
        smoothed_close = _sma(self.close, 15, 2)
        return ((self.high - smoothed_close) - (self.low - smoothed_close)) / self.close

    def alpha159(self) -> pd.DataFrame:
        """Return Alpha#159."""
        delayed_close = _delay(self.close, 1)
        low_or_close = _elementwise_min(self.low, delayed_close)
        high_or_close = _elementwise_max(self.high, delayed_close)
        range_sum = high_or_close - low_or_close
        part6 = (self.close - _ts_sum(low_or_close, 6)) / _ts_sum(range_sum, 6) * 12 * 24
        part12 = (
            (self.close - _ts_sum(low_or_close, 12))
            / _ts_sum(range_sum, 12)
            * 6
            * 24
        )
        part24 = (
            (self.close - _ts_sum(low_or_close, 24))
            / _ts_sum(range_sum, 24)
            * 6
            * 24
        )
        return (part6 + part12 + part24) * 100 / (6 * 12 + 6 * 24 + 12 * 24)

    def alpha160(self) -> pd.DataFrame:
        """Return Alpha#160."""
        cond = self.close <= _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = _stddev(self.close, 20)
        part[~cond] = 0
        return _sma(part, 20, 1)

    def alpha161(self) -> pd.DataFrame:
        """Return Alpha#161."""
        delayed_close = _delay(self.close, 1)
        return _mean(
            _elementwise_max(
                _elementwise_max(self.high - self.low, (delayed_close - self.high).abs()),
                (delayed_close - self.low).abs(),
            ),
            12,
        )

    def alpha162(self) -> pd.DataFrame:
        """Return Alpha#162."""
        close_delta = self.close - _delay(self.close, 1)
        ratio = (
            _sma(_elementwise_max(close_delta, close_delta * 0), 12, 1)
            / _sma(close_delta.abs(), 12, 1)
            * 100
        )
        return (ratio - _ts_min(ratio, 12)) / (_sma(ratio, 12, 1) - _ts_min(ratio, 12))

    def alpha163(self) -> pd.DataFrame:
        """Return Alpha#163."""
        return _rank(
            -1
            * self.returns
            * _mean(self.volume, 20)
            * self.vwap
            * (self.high - self.close),
        )

    def alpha164(self) -> pd.DataFrame:
        """Return Alpha#164."""
        close_delta = self.close - _delay(self.close, 1)
        cond = self.close > _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 1 / close_delta
        part[~cond] = 1
        high_low = self.high - self.low
        high_low = high_low.replace({0: np.nan})
        return _sma((part - _ts_min(part, 12)) / high_low * 100, 13, 2)

    def alpha165(self) -> pd.DataFrame:
        """Return Alpha#165."""
        centered_sum = _ts_sum(self.close - _mean(self.close, 48), 48)
        row_max = _row_max(centered_sum)
        row_min = _row_min(centered_sum)
        stddev = _stddev(self.close, 48)
        return -1 * (1 / stddev.div(row_min, axis=0)).sub(row_max, axis=0)

    def alpha166(self) -> pd.DataFrame:
        """Return Alpha#166."""
        ratio = self.close / _delay(self.close, 1)
        centered = ratio - 1 - _mean(ratio - 1, 20)
        numerator = -20 * (20 - 1) ** 1.5 * _ts_sum(centered, 20)
        denominator = (20 - 1) * (20 - 2) * _ts_sum(_mean(ratio, 20) ** 2, 20) ** 1.5
        return numerator / denominator

    def alpha167(self) -> pd.DataFrame:
        """Return Alpha#167."""
        close_delta = self.close - _delay(self.close, 1)
        cond = self.close > _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = close_delta
        part[~cond] = 0
        return _ts_sum(part, 12)

    def alpha168(self) -> pd.DataFrame:
        """Return Alpha#168."""
        return -1 * self.volume / _mean(self.volume, 20)

    def alpha169(self) -> pd.DataFrame:
        """Return Alpha#169."""
        smoothed_delta = _sma(self.close - _delay(self.close, 1), 9, 1)
        delayed = _delay(smoothed_delta, 1)
        return _sma(_mean(delayed, 12) - _mean(delayed, 26), 10, 1)

    def alpha170(self) -> pd.DataFrame:
        """Return Alpha#170."""
        left = (
            (_rank(1 / self.close) * self.volume)
            / _mean(self.volume, 20)
            * (
                self.high
                * _rank(self.high - self.close)
                / (_ts_sum(self.high, 5) / 5)
            )
        )
        return left - _rank(self.vwap - _delay(self.vwap, 5))

    def alpha171(self) -> pd.DataFrame:
        """Return Alpha#171."""
        return (
            -1
            * ((self.low - self.close) * self.open**5)
            / ((self.close - self.high) * self.close**5)
        )

    def alpha172(self) -> pd.DataFrame:
        """Return Alpha#172."""
        delayed_close = _delay(self.close, 1)
        true_range = _elementwise_max(
            _elementwise_max(self.high - self.low, (self.high - delayed_close).abs()),
            (self.low - delayed_close).abs(),
        )
        high_delta = self.high - _delay(self.high, 1)
        low_delta = _delay(self.low, 1) - self.low
        cond1 = (low_delta > 0) & (low_delta > high_delta)
        cond2 = (high_delta > 0) & (high_delta > low_delta)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond1] = low_delta
        part1[~cond1] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[cond2] = high_delta
        part2[~cond2] = 0
        down = _ts_sum(part1, 14) * 100 / _ts_sum(true_range, 14)
        up = _ts_sum(part2, 14) * 100 / _ts_sum(true_range, 14)
        return _mean((down - up).abs() / (down + up) * 100, 6)

    def alpha173(self) -> pd.DataFrame:
        """Return Alpha#173."""
        smoothed_close = _sma(self.close, 13, 2)
        return (
            3 * smoothed_close
            - 2 * _sma(smoothed_close, 13, 2)
            + _sma(_sma(_sma(np.log(self.close), 13, 2), 13, 2), 13, 2)
        )

    def alpha174(self) -> pd.DataFrame:
        """Return Alpha#174."""
        cond = self.close > _delay(self.close, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = _stddev(self.close, 20)
        part[~cond] = 0
        return _sma(part, 20, 1)

    def alpha175(self) -> pd.DataFrame:
        """Return Alpha#175."""
        delayed_close = _delay(self.close, 1)
        return _mean(
            _elementwise_max(
                _elementwise_max(self.high - self.low, (delayed_close - self.high).abs()),
                (delayed_close - self.low).abs(),
            ),
            6,
        )

    def alpha176(self) -> pd.DataFrame:
        """Return Alpha#176."""
        return _correlation(
            _rank(
                (self.close - _ts_min(self.low, 12))
                / (_ts_max(self.high, 12) - _ts_min(self.low, 12)),
            ),
            _rank(self.volume),
            6,
        )

    def alpha177(self) -> pd.DataFrame:
        """Return Alpha#177."""
        return (20 - _highday(self.high, 20)) / 20 * 100

    def alpha178(self) -> pd.DataFrame:
        """Return Alpha#178."""
        return (self.close - _delay(self.close, 1)) / _delay(self.close, 1) * self.volume

    def alpha179(self) -> pd.DataFrame:
        """Return Alpha#179."""
        return _rank(_correlation(self.vwap, self.volume, 4)) * _rank(
            _correlation(_rank(self.low), _rank(_mean(self.volume, 50)), 12),
        )

    def alpha180(self) -> pd.DataFrame:
        """Return Alpha#180."""
        cond = _mean(self.volume, 20) < self.volume
        close_delta = _delta(self.close, 7)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = -1 * _ts_rank(close_delta.abs(), 60) * np.sign(close_delta)
        part[~cond] = -1 * self.volume
        return part

    def alpha181(self) -> pd.DataFrame:
        """Return Alpha#181."""
        returns = self.close / _delay(self.close, 1) - 1
        benchmark_deviation = self.benchmark_close - self.benchmark_close.rolling(
            20
        ).mean()
        numerator = _ts_sum(
            (returns - _mean(returns, 20)).sub(benchmark_deviation.pow(2), axis=0),
            20,
        )
        denominator = benchmark_deviation.pow(3).rolling(20).sum()
        return numerator.div(denominator, axis=0)

    def alpha182(self) -> pd.DataFrame:
        """Return Alpha#182."""
        benchmark_up = _broadcast_series(
            self.benchmark_close > self.benchmark_open,
            self.close,
        )
        benchmark_down = _broadcast_series(
            self.benchmark_close < self.benchmark_open,
            self.close,
        )
        same_direction = (
            ((self.close > self.open) & benchmark_up)
            | ((self.close < self.open) & benchmark_down)
        )
        return _count(same_direction, 20) / 20

    def alpha183(self) -> pd.DataFrame:
        """Return Alpha#183."""
        centered_sum = _ts_sum(self.close - _mean(self.close, 24), 24)
        row_max = _row_max(centered_sum)
        row_min = _row_min(centered_sum)
        stddev = _stddev(self.close, 24)
        return -1 * (1 / stddev.div(row_min, axis=0)).sub(row_max, axis=0)

    def alpha184(self) -> pd.DataFrame:
        """Return Alpha#184."""
        return _rank(_correlation(_delay(self.open - self.close, 1), self.close, 200)) + _rank(
            self.open - self.close,
        )

    def alpha185(self) -> pd.DataFrame:
        """Return Alpha#185."""
        return _rank(-1 * (1 - self.open / self.close) ** 2)

    def alpha186(self) -> pd.DataFrame:
        """Return Alpha#186."""
        indicator = self._directional_indicator(14, 6)
        return (indicator + _delay(indicator, 6)) / 2

    def alpha187(self) -> pd.DataFrame:
        """Return Alpha#187."""
        cond = self.open <= _delay(self.open, 1)
        part = self.close.copy(deep=True)
        part.loc[:, :] = np.nan
        part[cond] = 0
        part[~cond] = _elementwise_max(self.high - self.open, self.open - _delay(self.open, 1))
        return _ts_sum(part, 20)

    def alpha188(self) -> pd.DataFrame:
        """Return Alpha#188."""
        high_low = self.high - self.low
        return (high_low - _sma(high_low, 11, 2)) / _sma(high_low, 11, 2) * 100

    def alpha189(self) -> pd.DataFrame:
        """Return Alpha#189."""
        return _mean((self.close - _mean(self.close, 6)).abs(), 6)

    def alpha191(self) -> pd.DataFrame:
        """Return Alpha#191."""
        return _correlation(_mean(self.volume, 20), self.low, 5) + (
            self.high + self.low
        ) / 2 - self.close

    def _directional_indicator(self, sum_window: int, mean_window: int) -> pd.DataFrame:
        """Return the directional movement indicator used by Alpha#172 and Alpha#186."""
        delayed_close = _delay(self.close, 1)
        true_range = _elementwise_max(
            _elementwise_max(self.high - self.low, (self.high - delayed_close).abs()),
            (self.low - delayed_close).abs(),
        )
        high_delta = self.high - _delay(self.high, 1)
        low_delta = _delay(self.low, 1) - self.low
        cond1 = (low_delta > 0) & (low_delta > high_delta)
        cond2 = (high_delta > 0) & (high_delta > low_delta)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = np.nan
        part1[cond1] = low_delta
        part1[~cond1] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = np.nan
        part2[cond2] = high_delta
        part2[~cond2] = 0
        down = _ts_sum(part1, sum_window) * 100 / _ts_sum(true_range, sum_window)
        up = _ts_sum(part2, sum_window) * 100 / _ts_sum(true_range, sum_window)
        return _mean((down - up).abs() / (down + up) * 100, mean_window)


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


def _covariance(left: pd.DataFrame, right: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling covariance."""
    return left.rolling(window).cov(right)


def _ts_sum(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling sum."""
    return frame.rolling(window).sum()


def _prod(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling product."""
    return frame.rolling(window).apply(lambda values: np.prod(values))


def _count(cond: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return rolling count of true values."""
    return cond.rolling(window).apply(lambda values: values.sum())


def _broadcast_series(series: pd.Series, template: pd.DataFrame) -> pd.DataFrame:
    """Return a benchmark series aligned to every column in ``template``."""
    aligned = series.reindex(template.index)
    values = np.repeat(aligned.to_numpy()[:, None], len(template.columns), axis=1)
    return pd.DataFrame(values, index=template.index, columns=template.columns)


def _sumif(frame: pd.DataFrame, window: int, cond: pd.DataFrame) -> pd.DataFrame:
    """Return rolling sum over values selected by ``cond``."""
    values = frame.copy(deep=True)
    values[~cond] = 0
    return values.rolling(window).sum()


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


def _row_max(frame: pd.DataFrame) -> pd.Series:
    """Return row-wise maximum."""
    return frame.max(axis=1)


def _row_min(frame: pd.DataFrame) -> pd.Series:
    """Return row-wise minimum."""
    return frame.min(axis=1)


def _lowday(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return days since the rolling minimum, matching legacy Lowday."""
    return frame.rolling(window).apply(lambda values: len(values) - values.argmin())


def _highday(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return days since the rolling maximum, matching legacy Highday."""
    return frame.rolling(window).apply(lambda values: len(values) - values.argmax())


def _elementwise_max(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise maximum."""
    return np.maximum(left, right)


def _elementwise_min(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return element-wise minimum."""
    return np.minimum(left, right)


def _directional_range_parts(
    template: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    cond: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the paired range components used by Alpha#49-51."""
    movement = _elementwise_max(
        (high - _delay(high, 1)).abs(),
        (low - _delay(low, 1)).abs(),
    )
    part1 = template.copy(deep=True)
    part1.loc[:, :] = np.nan
    part1[cond] = 0
    part1[~cond] = movement
    part2 = template.copy(deep=True)
    part2.loc[:, :] = np.nan
    part2[~cond] = 0
    part2[cond] = movement
    return part1, part2
