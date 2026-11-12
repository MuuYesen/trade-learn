from __future__ import annotations

from typing import Any

from tradelearn.engine.base import (BaseObserver, MetaParams, Params,
                                    collect_param_defaults, _bind_component_data_aliases)


class ObserverCollection(dict):
    """Dict with Backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Observer(BaseObserver, metaclass=MetaParams):
    """Minimal Backtrader observer base class."""

    params: tuple = ()

    def _base_init(self, **kwargs: Any) -> None:
        self.params = self.p = Params(collect_param_defaults(type(self))[1], **kwargs)
        self.strategy = None
        self.values: list[Any] = []

    def __init__(self) -> None:
        pass

    def _initialize_component_context(self, strategy: Any) -> None:
        self._owner = self._clock = self.strategy = strategy
        _bind_component_data_aliases(self)

    def __len__(self) -> int:
        return len(self.strategy)

    def _set(self, strategy: Any) -> None:
        self.strategy = strategy

    def start(self) -> None:
        pass

    def next(self) -> None:
        self.values.append(self._observe())

    def _stop(self) -> None:
        pass

    def _observe(self) -> Any:
        return None

    def _get_analysis(self) -> dict[str, Any]:
        return {self.__class__.__name__.lower(): self.values}


class Value(Observer):
    def _observe(self) -> float:
        return float(self.strategy.broker.getvalue())

    def _get_analysis(self) -> dict[str, list[float]]:
        return {"value": self.values}


class Broker(Value):
    pass


class BuySell(Observer):
    def _observe(self) -> int:
        broker = self.strategy.broker
        fills = getattr(broker, "_fills", ())
        return len(fills)

    def _get_analysis(self) -> dict[str, list[int]]:
        return {"fills": self.values}


class Trades(Observer):
    def _observe(self) -> int:
        trades = getattr(self.strategy, "_trades", ())
        return len(trades)

    def _get_analysis(self) -> dict[str, list[int]]:
        return {"trades": self.values}


class DrawDown(Observer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._peak = None

    def _observe(self) -> float:
        from tradelearn.metrics.risk import _drawdown_observation
        self._peak, ratio, _ = _drawdown_observation(self.strategy.broker.getvalue(), self._peak)
        return -ratio

    def _get_analysis(self) -> dict[str, list[float]]:
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

    def _get_analysis(self) -> dict[str, list[float]]:
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
