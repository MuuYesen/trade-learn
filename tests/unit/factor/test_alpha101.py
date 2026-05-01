"""Tests for v2 Alpha101 factor entry points."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from tradelearn.factor.alpha import alpha101

_LEGACY_ALPHA101_PATH = (
    Path(__file__).resolve().parents[3] / "reference" / "tradelearn_1x" / "query" / "alpha" / "alphas101.py"
)
_legacy_spec = importlib.util.spec_from_file_location("legacy_alpha101", _LEGACY_ALPHA101_PATH)
if _legacy_spec is not None and _legacy_spec.loader is not None:
    _legacy_module = importlib.util.module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)
    LegacyAlphas101 = _legacy_module.Alphas101
else:
    LegacyAlphas101 = None


def test_alpha101_exports_migrated_formulas_like_legacy_oracle() -> None:
    """The v2 Alpha101 facade returns legacy-compatible long-form columns."""
    if LegacyAlphas101 is None:
        pytest.skip("legacy Alpha101 oracle is not available")
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


def test_alpha101_documents_skipped_legacy_formulas() -> None:
    """All Alpha101 formulas are executable in v2."""
    formerly_skipped = {
        "alpha048",
        "alpha056",
        "alpha058",
        "alpha059",
        "alpha063",
        "alpha067",
        "alpha069",
        "alpha070",
        "alpha076",
        "alpha079",
        "alpha080",
        "alpha082",
        "alpha087",
        "alpha089",
        "alpha090",
        "alpha091",
        "alpha093",
        "alpha097",
        "alpha100",
    }

    result = alpha101(_stock_data(), names=sorted(formerly_skipped))

    assert set(result.columns) == {
        "date",
        "code",
        *(f"{name}_101" for name in formerly_skipped),
    }


def test_alpha101_rejects_unknown_formula_names() -> None:
    """Formula selection treats Alpha101 as complete and only rejects unknown names."""
    with pytest.raises(ValueError, match="unknown Alpha101 formulas"):
        alpha101(_stock_data(), names=["alpha999"])


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
