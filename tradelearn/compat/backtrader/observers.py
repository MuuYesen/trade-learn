from __future__ import annotations

from typing import Any

from tradelearn.compat.backtrader.base import Params


class ObserverCollection(dict):
    """Dict with Backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Observer:
    """Minimal Backtrader observer base class."""

    params: tuple = ()

    def __init__(self, **kwargs: Any) -> None:
        self.params = self.p = Params(self.params, **kwargs)
        self.strategy = None
        self.values: list[Any] = []

    def _set(self, strategy: Any) -> None:
        self.strategy = strategy

    def start(self) -> None:
        pass

    def next(self) -> None:
        self.values.append(self._observe())

    def stop(self) -> None:
        pass

    def _observe(self) -> Any:
        return None

    def get_analysis(self) -> dict[str, Any]:
        return {self.__class__.__name__.lower(): self.values}


class Value(Observer):
    def _observe(self) -> float:
        return float(self.strategy.broker.getvalue())

    def get_analysis(self) -> dict[str, list[float]]:
        return {"value": self.values}


class Broker(Value):
    pass


class BuySell(Observer):
    def _observe(self) -> int:
        broker = self.strategy.broker
        fills = getattr(broker, "_fills", ())
        return len(fills)

    def get_analysis(self) -> dict[str, list[int]]:
        return {"fills": self.values}


class Trades(Observer):
    def _observe(self) -> int:
        trades = getattr(self.strategy, "_trades", ())
        return len(trades)

    def get_analysis(self) -> dict[str, list[int]]:
        return {"trades": self.values}


class DrawDown(Observer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._peak = 0.0

    def _observe(self) -> float:
        value = float(self.strategy.broker.getvalue())
        self._peak = max(self._peak, value)
        if self._peak <= 0:
            return 0.0
        return (self._peak - value) / self._peak

    def get_analysis(self) -> dict[str, list[float]]:
        return {"drawdown": self.values}


class TimeReturn(Observer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_value: float | None = None

    def _observe(self) -> float:
        value = float(self.strategy.broker.getvalue())
        previous_value = self._last_value
        self._last_value = value
        if previous_value in (None, 0.0):
            return 0.0
        return (value / previous_value) - 1.0

    def get_analysis(self) -> dict[str, list[float]]:
        return {"timereturn": self.values}


__all__ = [
    "Broker",
    "BuySell",
    "DrawDown",
    "Observer",
    "ObserverCollection",
    "TimeReturn",
    "Trades",
    "Value",
]
