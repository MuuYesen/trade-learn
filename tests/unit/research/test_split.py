from __future__ import annotations

import pandas as pd
import pytest

import tradelearn.research as research
from tradelearn.research import ResearchRun


def test_time_split_splits_datetime_index_and_records_step() -> None:
    dataset = pd.DataFrame(
        {"alpha": [1.0, 2.0, 3.0], "label": [0, 1, 1]},
        index=pd.to_datetime(["2023-08-31", "2023-09-01", "2023-09-02"], utc=True),
    )

    with ResearchRun("split-demo") as run:
        train, test = research.time_split(dataset, split="2023-09-01")
        result = run.finish(features=test)

    assert train.index.tolist() == [pd.Timestamp("2023-08-31", tz="UTC")]
    assert test.index.tolist() == [
        pd.Timestamp("2023-09-01", tz="UTC"),
        pd.Timestamp("2023-09-02", tz="UTC"),
    ]
    assert result.steps[0].name == "time_split"
    assert result.params["time_split.split"] == "2023-09-01"


def test_time_split_splits_multiindex_panel_by_datetime_level() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2023-08-31", "2023-09-01"], utc=True),
            ["AAA", "BBB"],
        ],
        names=["timestamp", "symbol"],
    )
    dataset = pd.DataFrame({"alpha": [1.0, 2.0, 3.0, 4.0]}, index=index)

    train, test = research.time_split(dataset, split="2023-09-01")

    assert train.index.get_level_values("symbol").tolist() == ["AAA", "BBB"]
    assert test.index.get_level_values("symbol").tolist() == ["AAA", "BBB"]
    assert train.index.get_level_values("timestamp").unique().tolist() == [
        pd.Timestamp("2023-08-31", tz="UTC")
    ]
    assert test.index.get_level_values("timestamp").unique().tolist() == [
        pd.Timestamp("2023-09-01", tz="UTC")
    ]


def test_split_bars_keeps_test_period_for_datetime_index() -> None:
    bars = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2023-08-31", "2023-09-01", "2023-09-02"], utc=True),
    )

    test_bars = research.split_bars(bars, split="2023-09-01")

    assert test_bars.index.tolist() == [
        pd.Timestamp("2023-09-01", tz="UTC"),
        pd.Timestamp("2023-09-02", tz="UTC"),
    ]
    assert test_bars["close"].tolist() == [11.0, 12.0]


def test_split_bars_keeps_test_period_for_multiindex_panel() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2023-08-31", "2023-09-01"], utc=True),
            ["AAA", "BBB"],
        ],
        names=["timestamp", "symbol"],
    )
    bars = pd.DataFrame({"close": [10.0, 20.0, 11.0, 21.0]}, index=index)

    test_bars = research.split_bars(bars, split="2023-09-01")

    assert test_bars.index.get_level_values("timestamp").unique().tolist() == [
        pd.Timestamp("2023-09-01", tz="UTC")
    ]
    assert test_bars.index.get_level_values("symbol").tolist() == ["AAA", "BBB"]
    assert test_bars["close"].tolist() == [11.0, 21.0]


def test_time_split_raises_for_non_datetime_index() -> None:
    dataset = pd.DataFrame({"alpha": [1.0, 2.0]}, index=["AAA", "BBB"])

    with pytest.raises(ValueError, match="datetime"):
        research.time_split(dataset, split="2023-09-01")
