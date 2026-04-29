from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from tradelearn.core import (
    AccountSnapshot,
    Fill,
    OrderAck,
    OrderRequest,
    OrderStatusUpdate,
    PositionSnapshot,
)


def test_core_broker_contracts_are_broker_neutral_data_shells() -> None:
    ts = pd.Timestamp("2024-01-01T00:00:00Z")

    req = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        qty=2.0,
        order_type="limit",
        limit_price=10.0,
        tif="gtc",
        client_oid="client-1",
    )
    ack = OrderAck(client_oid="client-1", broker_oid="broker-1", accepted_ts=ts)
    fill = Fill(
        broker_oid="broker-1",
        symbol="BTCUSDT",
        qty=2.0,
        price=10.0,
        commission=0.1,
        ts=ts,
    )
    position = PositionSnapshot(symbol="BTCUSDT", qty=2.0, avg_price=10.0, ts=ts)
    account = AccountSnapshot(cash=100.0, equity=120.0, ts=ts)
    status = OrderStatusUpdate(
        broker_oid="broker-1",
        status_str="accepted",
        ts=ts,
        replay=False,
    )

    assert req.limit_price == 10.0
    assert ack.broker_oid == "broker-1"
    assert fill.qty == 2.0
    assert position.avg_price == 10.0
    assert account.equity == 120.0
    assert status.status_str == "accepted"

    with pytest.raises(dataclasses.FrozenInstanceError):
        req.qty = 3.0  # type: ignore[misc]
