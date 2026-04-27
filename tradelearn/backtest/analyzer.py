from __future__ import annotations
from typing import Any, Dict
from tradelearn.backtest.base import BaseAnalyzer

class Analyzer(BaseAnalyzer):
    """Base analyzer class attached through Cerebro.addanalyzer."""
    metric_key = "" # Subclasses should define this
    is_streaming = False # Set to True for real-time analyzers

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
            from tradelearn.backtest.metrics_engine import MetricsEngine
            if not hasattr(strat, 'metrics_engine'):
                strat.metrics_engine = MetricsEngine()
            
            # Register this metric with the engine if it has a key
            if self.metric_key and not self.is_streaming:
                strat.metrics_engine.request(self._instance_id, self.metric_key, self.p)

    def on_bar(self, bar: Any) -> None:
        """Deprecated for Post Analyzers. Kept for Streaming Analyzers."""
        pass

    def get_analysis(self) -> Dict[str, Any]:
        if not self.is_streaming and self.metric_key and self.strategy is not None and hasattr(self.strategy, 'metrics_engine'):
            return self.strategy.metrics_engine.results.get(self._instance_id, {})
        return {}



