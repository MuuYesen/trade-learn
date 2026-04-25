"""Tests for v2 Alpha101 factor entry points."""

import pandas as pd

from tradelearn.factor.alpha import alpha101
from tradelearn.query import Query
from tradelearn.query.alpha.alphas101 import Alphas101 as LegacyAlphas101


def test_alpha101_exports_first_three_formulas_like_legacy_query() -> None:
    """The v2 Alpha101 facade returns legacy-compatible long-form columns."""
    data = _stock_data()
    expected = _legacy_alpha101(data, ["alpha001", "alpha002", "alpha003"])

    result = alpha101(data, names=["alpha001", "alpha002", "alpha003"])

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
    )


def test_query_alphas101_delegates_supported_formulas_to_v2_facade() -> None:
    """Query.alphas101 keeps its output contract while using the v2 facade."""
    data = _stock_data()

    result = Query.alphas101(data, ["alpha001", "alpha002", "alpha003"])
    expected = alpha101(data, names=["alpha001", "alpha002", "alpha003"])

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
    )


def _legacy_alpha101(data: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    pivoted = data.pivot(index="date", columns="code")
    legacy = LegacyAlphas101(pivoted)
    result = pd.DataFrame({"date": [], "code": []})
    for name in names:
        frame = getattr(legacy, name)().copy()
        frame["date"] = frame.index
        frame = frame.melt(
            id_vars="date",
            value_vars=frame.columns.drop("date"),
            value_name=name,
        )
        frame.rename(columns={name: f"{name}_101"}, inplace=True)
        result = pd.merge(result, frame, how="outer", on=["date", "code"])
    return result


def _stock_data() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=25)
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
                }
            )
    return pd.DataFrame(rows)
