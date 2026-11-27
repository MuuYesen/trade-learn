"""Helper views, bar-vector packing, and fill-to-trade projections for RustBroker."""

from __future__ import annotations

from copy import copy
from typing import Any

from tradelearn.backtest.models import Order, Trade

OrderPayload = tuple[int, str, str, str, float, float | None, float | None]
_EMPTY_ORDER_BUFFER: tuple[()] = ()
CompactFillBatch = tuple[
    list[int], list[float], list[float], list[float], list[float], list[float]
]
_ORDER_TYPE_TO_RUST = {
    Order.Market: "market",
    Order.Close: "market",
    Order.Limit: "limit",
    Order.Stop: "stop",
    Order.StopLimit: "stop_limit",
    Order.StopTrail: "stop_trail",
    Order.StopTrailLimit: "stop_trail_limit",
}


class _CommissionInfoView:
    """Facade-only fallback for Backtrader-style commission info."""

    def __init__(self, ratio: float):
        self.p = self.params = type("Params", (), {"commission": ratio})()

    def getcommission(self, size: float, price: float) -> float:
        return abs(size) * price * self.p.commission

    def getsize(self, price, cash):
        return int(cash // price)


def current_bar_vectors(
    datas: list[Any], *, unit: str = "s"
) -> tuple[
    list[str],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Pack active feed bars at each feed's current cursor for Rust multi-bar stepping."""
    from tradelearn.backtest.clock import native_timestamps
    from tradelearn.backtest.data import native_data_symbols

    symbols: list[str] = []
    timestamps: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    execution_symbols = native_data_symbols(datas)
    for index, data in enumerate(datas):
        cursor = getattr(data, "_cursor", -1)
        if cursor < 0:
            continue
        symbols.append(execution_symbols[index])
        timestamps.append(int(native_timestamps(data, unit)[cursor]))
        opens.append(float(data._open[cursor]))
        highs.append(float(data._high[cursor]))
        lows.append(float(data._low[cursor]))
        closes.append(float(data._close[cursor]))
        volumes.append(float(data._volume[cursor]))
    return symbols, timestamps, opens, highs, lows, closes, volumes


def event_bar_vectors(
    datas: list[Any], *, clock_unit: str, event_clock_timestamp: int | None
) -> tuple[
    list[str],
    list[int],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Build current bar vectors with the active event clock feed ordered first."""
    vectors = (
        current_bar_vectors(datas)
        if clock_unit == "s"
        else current_bar_vectors(datas, unit=clock_unit)
    )
    if event_clock_timestamp is None or not vectors[1] or vectors[1][0] == event_clock_timestamp:
        return vectors
    if event_clock_timestamp not in vectors[1]:
        return vectors  # Replay's BT view is distinct from original quote timestamps.
    active = vectors[1].index(event_clock_timestamp)
    order = [active, *range(active), *range(active + 1, len(vectors[1]))]
    return tuple([[column[i] for i in order] for column in vectors])  # type: ignore[return-value]


def is_compact_fill_batch(fills: Any) -> bool:
    """Return whether ``fills`` is a 6-tuple columnar compact fill batch."""
    return isinstance(fills, tuple) and len(fills) == 6


def fill_batch_len(fills: Any) -> int:
    """Return the number of fill records in a compact or row-based batch."""
    if fills is None:
        return 0
    if is_compact_fill_batch(fills):
        return len(fills[0])
    return len(fills)


def iter_rust_fills(fills: Any, resolve_ref: Any):
    """Yield ``(public_ref, signed_size, price, commission, pnl)`` tuples."""
    if is_compact_fill_batch(fills):
        order_ids, sizes, prices, comms, _slippages, pnls = fills
        return (
            (resolve_ref(ref), size, price, comm, pnl)
            for ref, size, price, comm, pnl in zip(
                order_ids, sizes, prices, comms, pnls, strict=True
            )
        )
    return (
        (resolve_ref(fill[0]), fill[2], fill[3], fill[4], fill[6])
        for fill in fills
    )


def project_fill_trade(
    broker: Any,
    owner: Any,
    order: Order,
    old_size: float,
    old_price: float,
    signed_size: float,
    price: float,
    commission: float,
    pnl: float,
    notify: bool,
) -> None:
    """Accumulate views of authoritative fills without mutating account cash.

    A reversal closes the previous Trade and opens a separate Trade. Fees
    from its one native fill are apportioned by the actual closed/open qty.
    """
    data = order.data
    trade = broker._trade_states.get(data)
    total = abs(signed_size)
    closing = min(abs(old_size), total) if old_size * signed_size < 0 else 0.0
    opening = total - closing
    direction = 1 if signed_size > 0 else -1
    bar = len(data) if hasattr(data, "__len__") else 0
    info = broker.getcommissioninfo(data)
    value = getattr(info, "getvaluesize", lambda size, cost: size * cost)
    close_fee = commission * closing / total if closing else 0.0
    if closing:
        if trade is None:
            raise RuntimeError("native fill closes a position without its trade projection")
        trade.justopened = False
        trade.size = old_size + direction * closing
        trade.commission += close_fee
        trade.pnl += pnl
        trade.pnlcomm = trade.pnl - trade.commission
        trade.value = value(trade.size, trade.price)
        trade.barlen = bar - trade.baropen
        trade.isopen = bool(trade.size)
        trade.isclosed = not trade.isopen
        if trade.isclosed:
            trade.status = Trade.Closed
            trade.dtclose = order.executed.dt
            trade.barclose = bar
            broker._trade_states.pop(data)
            if notify:
                broker._notify_trade_event(owner, copy(trade))
            trade = None
    if opening:
        justopened = trade is None
        if justopened:
            trade = Trade(
                ref=broker._next_trade_ref,
                data=data,
                status=Trade.Open,
                isopen=True,
                dtopen=order.executed.dt,
                long=direction > 0,
                baropen=bar,
            )
            broker._next_trade_ref += 1
            broker._trade_states[data] = trade
        old_trade_size = trade.size
        trade.size += direction * opening
        trade.price = (old_trade_size * trade.price + direction * opening * price) / trade.size
        trade.value = value(trade.size, trade.price)
        trade.commission += commission - close_fee
        trade.pnlcomm = trade.pnl - trade.commission
        trade.justopened = justopened
        trade.barlen = bar - trade.baropen
        if justopened and notify:
            broker._notify_trade_event(owner, copy(trade))
