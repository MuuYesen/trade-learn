"""Tests for v2 Alpha191 factor entry points."""

import warnings

import pandas as pd

from tradelearn.factor.alpha import alpha191
from tradelearn.query import Query
from tradelearn.query.alpha.alphas191 import Alphas191 as LegacyAlphas191


def test_alpha191_exports_migrated_formulas_like_legacy_query() -> None:
    """The v2 Alpha191 facade returns legacy-compatible long-form columns."""
    stock_data = _stock_data()
    bench_data = _bench_data()
    names = [
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
    ]
    expected = _legacy_alpha191(stock_data, bench_data, names)

    result = alpha191(stock_data, bench_data, names=names)

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_query_alphas191_delegates_supported_formulas_to_v2_facade() -> None:
    """Query.alphas191 keeps its output contract while using the v2 facade."""
    stock_data = _stock_data()
    bench_data = _bench_data()
    names = [
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
    ]

    result = Query.alphas191(stock_data, bench_data, names)
    expected = alpha191(stock_data, bench_data, names=names)

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
    )


def test_alpha191_v2_facade_avoids_future_warning_for_missing_values() -> None:
    """The v2 Alpha191 path should not inherit legacy None-to-float warnings."""
    stock_data = _stock_data()
    bench_data = _bench_data()
    names = ["alpha003", "alpha004", "alpha010", "alpha019", "alpha023"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        alpha191(stock_data, bench_data, names=names)
        Query.alphas191(stock_data, bench_data, names)

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "incompatible dtype" in str(warning.message)
    ]


def _legacy_alpha191(
    stock_data: pd.DataFrame, bench_data: pd.DataFrame, names: list[str]
) -> pd.DataFrame:
    pivoted = stock_data.pivot(index="date", columns="code")
    legacy = LegacyAlphas191(pivoted, bench_data)
    result = pd.DataFrame({"date": [], "code": []})
    for name in names:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            frame = getattr(legacy, name)().copy()
        frame["date"] = frame.index
        frame = frame.melt(
            id_vars="date",
            value_vars=frame.columns.drop("date"),
            value_name=name,
        )
        frame.rename(columns={name: f"{name}_191"}, inplace=True)
        result = pd.merge(result, frame, how="outer", on=["date", "code"])
    for column in result.columns.difference(["date", "code"]):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _stock_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80)
    rows = []
    for symbol_index, code in enumerate(["AAA", "BBB", "CCC"], start=1):
        for day_index, date in enumerate(dates, start=1):
            base = 10.0 * symbol_index + day_index
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": base,
                    "high": base + 1.0,
                    "low": base - 1.0,
                    "close": base + ((-1) ** day_index) * 0.5,
                    "volume": 1_000.0 + symbol_index * 100.0 + day_index * 10.0,
                    "vwap": base + 0.2,
                    "amount": (base + 0.2)
                    * (1_000.0 + symbol_index * 100.0 + day_index * 10.0),
                }
            )
    return pd.DataFrame(rows)


def _bench_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80)
    return pd.DataFrame(
        {
            "open": [3000.0 + day_index for day_index in range(80)],
            "close": [3000.5 + day_index for day_index in range(80)],
        },
        index=dates,
    )
