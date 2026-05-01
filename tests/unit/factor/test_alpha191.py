"""Tests for v2 Alpha191 factor entry points."""

import importlib
import warnings
from pathlib import Path

import pandas as pd
import pytest

from tradelearn.factor.alpha import alpha191

_LEGACY_ALPHA191_PATH = (
    Path(__file__).resolve().parents[3] / "reference" / "tradelearn_1x" / "query" / "alpha" / "alphas191.py"
)
_legacy_spec = importlib.util.spec_from_file_location("legacy_alpha191", _LEGACY_ALPHA191_PATH)
if _legacy_spec is not None and _legacy_spec.loader is not None:
    _legacy_module = importlib.util.module_from_spec(_legacy_spec)
    _legacy_spec.loader.exec_module(_legacy_module)
    LegacyAlphas191 = _legacy_module.Alphas191
else:
    LegacyAlphas191 = None


def test_alpha191_exports_migrated_formulas_like_legacy_oracle() -> None:
    """The v2 Alpha191 facade returns legacy-compatible long-form columns."""
    if LegacyAlphas191 is None:
        pytest.skip("legacy Alpha191 oracle is not available")
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
        "alpha183",
        "alpha184",
        "alpha185",
        "alpha186",
        "alpha187",
        "alpha188",
        "alpha189",
        "alpha191",
    ]
    expected = _legacy_alpha191(stock_data, bench_data, names)

    result = alpha191(stock_data, bench_data, names=names)

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_alpha191_supports_reference_benchmark_formulas() -> None:
    """Benchmark-backed formulas migrated from commented 1.x reference formulas."""
    stock_data = _stock_data()
    bench_data = _bench_data()
    result = alpha191(stock_data, bench_data, names=["alpha075", "alpha181", "alpha182"])

    pivoted = stock_data.pivot(index="date", columns="code")
    close = pivoted["close"]
    open_ = pivoted["open"]
    benchmark_down = _broadcast_test_series(
        bench_data["close"] < bench_data["open"],
        close,
    )
    benchmark_up = _broadcast_test_series(
        bench_data["close"] > bench_data["open"],
        close,
    )
    alpha075 = _rolling_count((close > open_) & benchmark_down, 50) / _rolling_count(
        benchmark_down, 50
    )
    returns = close / close.shift(1) - 1
    benchmark_deviation = bench_data["close"] - bench_data["close"].rolling(20).mean()
    alpha181 = (
        (returns - returns.rolling(20).mean())
        .sub(benchmark_deviation.pow(2), axis=0)
        .rolling(20)
        .sum()
        .div(benchmark_deviation.pow(3).rolling(20).sum(), axis=0)
    )
    same_direction = ((close > open_) & benchmark_up) | ((close < open_) & benchmark_down)
    alpha182 = _rolling_count(same_direction, 20) / 20
    expected = _long_alpha191(
        {
            "alpha075": alpha075,
            "alpha181": alpha181,
            "alpha182": alpha182,
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(["date", "code"]).reset_index(drop=True),
        expected.sort_values(["date", "code"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_alpha191_v2_facade_avoids_future_warning_for_missing_values() -> None:
    """The v2 Alpha191 path should not inherit legacy None-to-float warnings."""
    stock_data = _stock_data()
    bench_data = _bench_data()
    names = [
        "alpha003",
        "alpha004",
        "alpha010",
        "alpha019",
        "alpha023",
        "alpha038",
        "alpha040",
        "alpha043",
        "alpha049",
        "alpha050",
        "alpha051",
        "alpha056",
        "alpha059",
        "alpha069",
        "alpha084",
        "alpha086",
        "alpha093",
        "alpha094",
        "alpha098",
        "alpha101",
        "alpha112",
        "alpha123",
        "alpha128",
        "alpha129",
        "alpha137",
        "alpha148",
        "alpha154",
        "alpha160",
        "alpha164",
        "alpha167",
        "alpha172",
        "alpha174",
        "alpha180",
        "alpha186",
        "alpha187",
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        alpha191(stock_data, bench_data, names=names)

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "incompatible dtype" in str(warning.message)
    ]


def test_alpha191_documents_skipped_legacy_formulas() -> None:
    """All Alpha191 formulas are executable in v2."""
    alpha191_module = importlib.import_module("tradelearn.factor.alpha.alpha191")
    formerly_skipped = {"alpha030", "alpha143", "alpha149", "alpha190"}

    assert alpha191_module.ALPHA191_SKIPPED == {}
    assert formerly_skipped.issubset(alpha191_module.ALPHA191_SUPPORTED)

    result = alpha191(_stock_data(), _bench_data(), names=sorted(formerly_skipped))

    assert set(result.columns) == {
        "date",
        "code",
        *(f"{name}_191" for name in formerly_skipped),
    }


def test_alpha191_skipped_formulas_are_exported_from_package() -> None:
    """The package facade exposes skipped formulas for callers and docs."""
    alpha_package = importlib.import_module("tradelearn.factor.alpha")
    alpha191_module = importlib.import_module("tradelearn.factor.alpha.alpha191")

    assert alpha191_module.ALPHA191_SKIPPED == alpha_package.ALPHA191_SKIPPED
    assert "ALPHA191_SKIPPED" in alpha_package.__all__


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


def _rolling_count(cond: pd.DataFrame, window: int) -> pd.DataFrame:
    return cond.rolling(window).apply(lambda values: values.sum())


def _broadcast_test_series(series: pd.Series, template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: series.reindex(template.index)
            for column in template.columns
        },
        index=template.index,
    )


def _long_alpha191(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = pd.DataFrame({"date": [], "code": []})
    for name, frame in frames.items():
        melted = frame.copy()
        melted["date"] = melted.index
        melted = melted.melt(
            id_vars="date",
            value_vars=melted.columns.drop("date"),
            value_name=name,
        )
        melted.rename(columns={name: f"{name}_191"}, inplace=True)
        result = pd.merge(result, melted, how="outer", on=["date", "code"])
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
