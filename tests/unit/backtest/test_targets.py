from __future__ import annotations

import math

import pytest

from tradelearn.backtest.targets import TargetWeightSnapshot, build_target_weight_intents


def test_target_weight_intents_validate_cash_and_unknown_label() -> None:
    snapshots = {"AAA": TargetWeightSnapshot(price=10.0, size=0.0)}

    with pytest.raises(ValueError, match=r"Unknown ticker\(s\): \['BBB'\]"):
        build_target_weight_intents(
            {"BBB": 0.5},
            data_by_symbol={"AAA": object()},
            snapshots=snapshots,
            equity=1000.0,
            unknown_label="ticker(s)",
        )

    with pytest.raises(ValueError, match="cash target weight must be non-negative"):
        build_target_weight_intents(
            {"AAA": 0.5, "cash": -0.1},
            data_by_symbol={"AAA": object()},
            snapshots=snapshots,
            equity=1000.0,
        )


def test_target_weight_intents_sell_before_buy_and_close_missing() -> None:
    data = {"AAA": object(), "BBB": object(), "CCC": object()}
    snapshots = {
        "AAA": TargetWeightSnapshot(price=10.0, size=50.0),
        "BBB": TargetWeightSnapshot(price=20.0, size=0.0),
        "CCC": TargetWeightSnapshot(price=30.0, size=3.0),
    }

    intents = build_target_weight_intents(
        {"AAA": 0.2, "BBB": 0.2},
        data_by_symbol=data,
        snapshots=snapshots,
        equity=1000.0,
        close_missing=True,
    )

    assert [(intent.symbol, intent.side) for intent in intents] == [
        ("AAA", "sell"),
        ("CCC", "sell"),
        ("BBB", "buy"),
    ]
    assert intents[0].qty == 30.0
    assert intents[1].qty == 3.0
    assert intents[2].qty == 10.0


def test_target_weight_intents_order_by_delta_weight_like_backtrader() -> None:
    data = {"AAA": object(), "BBB": object(), "CCC": object()}
    snapshots = {
        "AAA": TargetWeightSnapshot(price=10.0, size=0.0),
        "BBB": TargetWeightSnapshot(price=10.0, size=0.0),
        "CCC": TargetWeightSnapshot(price=10.0, size=0.0),
    }

    intents = build_target_weight_intents(
        {"AAA": 0.3, "BBB": 0.1, "CCC": 0.2},
        data_by_symbol=data,
        snapshots=snapshots,
        equity=1000.0,
        close_missing=False,
    )

    assert [intent.symbol for intent in intents] == ["BBB", "CCC", "AAA"]


def test_target_weight_intents_zero_target_closes_full_position_without_rounding_tail() -> None:
    data = {"AAA": object()}
    snapshots = {
        "AAA": TargetWeightSnapshot(price=98.59351217434276, size=1004.0),
    }

    intents = build_target_weight_intents(
        {"AAA": 0.0},
        data_by_symbol=data,
        snapshots=snapshots,
        equity=1_060_000.0,
        close_missing=True,
    )

    assert [(intent.symbol, intent.side, intent.qty) for intent in intents] == [
        ("AAA", "sell", 1004.0)
    ]


def test_target_weight_intents_skip_tiny_or_zero_qty_orders() -> None:
    data = {"AAA": object(), "BBB": object()}
    snapshots = {
        "AAA": TargetWeightSnapshot(price=10.0, size=20.0),
        "BBB": TargetWeightSnapshot(price=10_000.0, size=0.0),
    }

    intents = build_target_weight_intents(
        {"AAA": 0.2, "BBB": 0.001},
        data_by_symbol=data,
        snapshots=snapshots,
        equity=1000.0,
        close_missing=False,
    )

    assert intents == []


def test_target_weight_intents_skip_nan_price_snapshots() -> None:
    data = {"AAA": object(), "BBB": object()}
    snapshots = {
        "AAA": TargetWeightSnapshot(price=math.nan, size=10.0),
        "BBB": TargetWeightSnapshot(price=10.0, size=0.0),
    }

    intents = build_target_weight_intents(
        {"AAA": 0.5, "BBB": 0.5},
        data_by_symbol=data,
        snapshots=snapshots,
        equity=1000.0,
        close_missing=True,
    )

    assert [(intent.symbol, intent.side, intent.qty) for intent in intents] == [
        ("BBB", "buy", 50.0)
    ]
