from __future__ import annotations

from typing import Any

from tradelearn.backtest.models import Stats

from .base import BaseAnalyzer, MetaParams, Params


class Analyzer(BaseAnalyzer, metaclass=MetaParams):
    """Base analyzer class attached through Cerebro.addanalyzer."""

    metric_key = ""  # Subclasses should define this
    is_streaming = False  # Set to True for real-time analyzers

    def _base_init(self, **kwargs: Any) -> None:
        params = []
        for base_cls in self.__class__.mro():
            cls_params = getattr(base_cls, "params", ())
            if isinstance(cls_params, dict):
                params.extend(cls_params.items())
            elif isinstance(cls_params, (list, tuple)):
                params.extend(cls_params)
        self.params = self.p = Params(params, **kwargs)

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
            if self.metric_key and not self.is_streaming:
                # Use self.p (params) which is now handled by MetaParams
                strat.metrics_engine.request(
                    self._instance_id,
                    self.metric_key,
                    getattr(self, "p", None),
                )

    def on_bar(self, bar: Any) -> None:
        """Deprecated for Post Analyzers. Kept for Streaming Analyzers."""
        pass

    def on_start(self) -> None:
        pass

    def on_end(self, stats: Stats) -> None:
        pass

    def get_analysis(self) -> dict[str, Any]:
        if (
            not self.is_streaming
            and self.metric_key
            and self.strategy is not None
            and hasattr(self.strategy, "metrics_engine")
        ):
            return self.strategy.metrics_engine.results.get(self._instance_id, {})
        return {}


class AnalyzerCollection(dict):
    """Dict with Backtrader-style attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def getbyname(self, name: str) -> Any:
        return self[name]
