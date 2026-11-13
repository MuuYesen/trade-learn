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
        from tradelearn.backtest.lines import Lines
        self.lines = self.l = Lines(self)
        self._line_clock_feed = None
        self._line_last_count = 0

    def __init__(self) -> None:
        pass

    def _initialize_component_context(self, strategy: Any) -> None:
        self._owner = self._clock = self.strategy = strategy
        _bind_component_data_aliases(self)
        from tradelearn.backtest.lines import line_names
        runtime = getattr(strategy, "_runtime", None)
        if runtime is not None:
            clock = getattr(runtime, "_clock_feed", None)
            self._line_clock_feed = clock if clock is not None else runtime._feed_for(self.data)
            self._line_last_count = self._line_clock_feed.received
            for name in line_names(type(self)):
                setattr(self.lines, name, runtime.output_line(self._line_clock_feed))

    def __getattr__(self, name):
        lines = vars(self).get("lines")
        if lines is not None and not name.startswith("_"):
            line = vars(lines).get(name)
            if line is not None:
                return line
        raise AttributeError(name)

    def __getitem__(self, ago):
        return self.lines[0][ago]

    def _next_step(self):
        # Observe on the existing market/clock; never register a new indicator
        # or make observer outputs change strategy eligibility.
        feed = self._line_clock_feed
        if feed is not None:
            for _ in range(feed.received - self._line_last_count):
                for line in self.lines:
                    line._append_output()
            self._line_last_count = feed.received
        runtime = getattr(self.strategy, "_runtime", None)
        phase = (runtime._next_callback(not runtime._warming_up())
                 if runtime is not None and getattr(runtime, "_bt_line_views", False) else "next")
        getattr(self, phase)()

    def __len__(self) -> int:
        return len(self.strategy)

    def _set(self, strategy: Any) -> None:
        self.strategy = strategy

    def start(self) -> None:
        pass

    def prenext(self) -> None:
        self.next()

    def nextstart(self) -> None:
        self.next()

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
