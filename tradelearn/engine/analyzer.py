from __future__ import annotations

from typing import Any

from tradelearn.backtest.models import Stats

from .base import BaseAnalyzer, MetaParams, Params


class Analyzer(BaseAnalyzer, metaclass=MetaParams):
    """Base analyzer class attached through Cerebro.addanalyzer."""

    _metric_key = ""  # Subclasses should define this
    _is_streaming = False  # Set to True for real-time analyzers

    def _base_init(self, **kwargs: Any) -> None:
        params = []
        for base_cls in self.__class__.mro():
            cls_params = getattr(base_cls, "params", ())
            if isinstance(cls_params, dict):
                params.extend(cls_params.items())
            elif isinstance(cls_params, (list, tuple)):
                params.extend(cls_params)
        self.params = self.p = Params(params, **kwargs)
        self.create_analysis()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._strategy = None
        self._instance_id = f"{self.__class__.__name__.lower()}_{id(self)}"

    @property
    def strategy(self) -> Any:
        return self._strategy

    @strategy.setter
    def strategy(self, strat: Any) -> None:
        self._strategy = strat
        if strat is not None:
            # Ensure strategy has a metrics engine
            from tradelearn.metrics.engine import MetricsEngine

            if not hasattr(strat, "metrics_engine"):
                strat.metrics_engine = MetricsEngine()

            # Register this metric with the engine if it has a key
            if self._metric_key and not self._is_streaming:
                # Use self.p (params) which is now handled by MetaParams
                strat.metrics_engine.request(
                    self._instance_id,
                    self._metric_key,
                    getattr(self, "p", None),
                )

    def create_analysis(self):
        from collections import OrderedDict
        self.rets = OrderedDict()

    def start(self): pass
    def next(self): pass
    def prenext(self): self.next()
    def nextstart(self): self.next()
    def notify_order(self, order): pass
    def notify_trade(self, trade): pass
    def notify_cashvalue(self, cash, value): pass

    def get_analysis(self) -> dict[str, Any]:
        if (
            not self._is_streaming
            and self._metric_key
            and self.strategy is not None
            and hasattr(self.strategy, "metrics_engine")
        ):
            return self.strategy.metrics_engine.results.get(self._instance_id, {})
        return self.rets


class AnalyzerCollection(dict):
    """Dict with Backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def getbyname(self, name: str) -> Any:
        return self[name]
