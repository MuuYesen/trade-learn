from __future__ import annotations

from typing import Any

from tradelearn.backtest.models import Order


class PositionProxy:
    """Tradelearn Lite position view returned by ``strategy.position()``."""

    __slots__ = ("_strategy", "_ticker", "_data", "_size_getter_broker", "_size_getter")

    def __init__(self, strategy: Any, ticker: str | None = None, data: Any = None):
        self._strategy = strategy
        self._ticker = ticker
        self._data = data
        self._size_getter_broker = None
        self._size_getter = None

    def _bind_broker_size_getters(self, broker):
        self._size_getter_broker = broker
        self._size_getter = getattr(
            broker,
            "current_position_size",
            getattr(broker, "get_position_size", None),
        )

    def __call__(self, ticker: str | None = None) -> PositionProxy:
        return self._strategy.position(ticker)

    def __bool__(self) -> bool:
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        primary_data = getattr(self._strategy, "_bt_primary_data", None)
        if size_getter is not None and self._data is primary_data:
            return size_getter() != 0
        return self.size != 0

    @property
    def size(self) -> float:
        primary_data = getattr(self._strategy, "_bt_primary_data", None)
        data = self._data or primary_data
        broker = self._strategy.broker
        if broker is not self._size_getter_broker:
            self._bind_broker_size_getters(broker)
        size_getter = self._size_getter
        if size_getter is not None and data is primary_data:
            return size_getter()
        return self._strategy.getposition(data).size

    def close(self, portion: float = 1.0):
        strategy = self._strategy
        data = self._data or strategy.datas[0]
        effective_size = self.size + strategy._pending_size.get(data, 0.0)
        if effective_size > 0:
            return strategy._submit_1x_order(
                Order.Sell,
                data,
                abs(effective_size) * float(portion),
                None,
                None,
                None,
            )
        if effective_size < 0:
            return strategy._submit_1x_order(
                Order.Buy,
                data,
                abs(effective_size) * float(portion),
                None,
                None,
                None,
            )
        return None

    @property
    def pl(self) -> float:
        data = self._data or self._strategy.datas[0]
        pos = self._strategy.getposition(data)
        if pos.size == 0:
            return 0.0
        price = data.get_array("close")[data._cursor]
        return (
            (float(price) - float(pos.price))
            * float(pos.size)
            * getattr(self._strategy.broker, "_mult", 1.0)
        )

    @property
    def pl_pct(self) -> float:
        data = self._data or self._strategy.datas[0]
        pos = self._strategy.getposition(data)
        if pos.size == 0 or pos.price == 0:
            return 0.0
        return (self.pl / (abs(pos.size) * float(pos.price))) * 100.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0


__all__ = ["PositionProxy"]
