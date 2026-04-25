"""Tests for v2 Alpha101 factor entry points."""

import importlib

import pandas as pd
import pytest

from tradelearn.factor.alpha import alpha101
from tradelearn.query import Query
from tradelearn.query.alpha.alphas101 import Alphas101 as LegacyAlphas101


def test_alpha101_exports_migrated_formulas_like_legacy_query() -> None:
    """The v2 Alpha101 facade returns legacy-compatible long-form columns."""
    data = _stock_data()
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
        "alpha060",
        "alpha061",
        "alpha062",
        "alpha064",
        "alpha065",
        "alpha066",
        "alpha068",
        "alpha071",
        "alpha072",
        "alpha073",
        "alpha074",
        "alpha075",
        "alpha077",
        "alpha078",
        "alpha081",
        "alpha083",
        "alpha084",
        "alpha085",
        "alpha086",
        "alpha088",
        "alpha092",
        "alpha094",
        "alpha095",
        "alpha096",
        "alpha098",
        "alpha099",
        "alpha101",
    ]
    expected = _legacy_alpha101(data, names)

    result = alpha101(data, names=names)

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
    )


def test_query_alphas101_delegates_supported_formulas_to_v2_facade() -> None:
    """Query.alphas101 keeps its output contract while using the v2 facade."""
    data = _stock_data()
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
        "alpha060",
        "alpha061",
        "alpha062",
        "alpha064",
        "alpha065",
        "alpha066",
        "alpha068",
        "alpha071",
        "alpha072",
        "alpha073",
        "alpha074",
        "alpha075",
        "alpha077",
        "alpha078",
        "alpha081",
        "alpha083",
        "alpha084",
        "alpha085",
        "alpha086",
        "alpha088",
        "alpha092",
        "alpha094",
        "alpha095",
        "alpha096",
        "alpha098",
        "alpha099",
        "alpha101",
    ]

    result = Query.alphas101(data, names)
    expected = alpha101(data, names=names)

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
    )


def test_alpha101_documents_skipped_legacy_formulas() -> None:
    """Intentionally skipped formulas are visible and fail with reasons."""
    alpha101_module = importlib.import_module("tradelearn.factor.alpha.alpha101")
    expected_skipped = {
        "alpha048": "requires industry neutralization input",
        "alpha056": "requires cap input",
        "alpha058": "requires industry neutralization input",
        "alpha059": "requires industry neutralization input",
        "alpha063": "requires industry neutralization input",
        "alpha067": "requires industry neutralization input",
        "alpha069": "requires industry neutralization input",
        "alpha070": "requires industry neutralization input",
        "alpha076": "requires industry neutralization input",
        "alpha079": "requires industry neutralization input",
        "alpha080": "requires industry neutralization input",
        "alpha082": "requires industry neutralization input",
        "alpha087": "requires industry neutralization input",
        "alpha089": "requires industry neutralization input",
        "alpha090": "requires industry neutralization input",
        "alpha091": "requires industry neutralization input",
        "alpha093": "requires industry neutralization input",
        "alpha097": "requires industry neutralization input",
        "alpha100": "requires subindustry neutralization input",
    }

    assert alpha101_module.ALPHA101_SKIPPED == expected_skipped
    assert set(expected_skipped).isdisjoint(alpha101_module.ALPHA101_SUPPORTED)

    with pytest.raises(ValueError) as exc_info:
        alpha101(_stock_data(), names=sorted(expected_skipped))

    message = str(exc_info.value)
    for name, reason in expected_skipped.items():
        assert name in message
        assert reason in message


def test_alpha101_skipped_formulas_are_exported_from_package() -> None:
    """The package facade exposes skipped formulas for callers and docs."""
    alpha_package = importlib.import_module("tradelearn.factor.alpha")
    alpha101_module = importlib.import_module("tradelearn.factor.alpha.alpha101")

    assert alpha101_module.ALPHA101_SKIPPED == alpha_package.ALPHA101_SKIPPED
    assert "ALPHA101_SKIPPED" in alpha_package.__all__


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
    dates = pd.date_range("2024-01-01", periods=280)
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
