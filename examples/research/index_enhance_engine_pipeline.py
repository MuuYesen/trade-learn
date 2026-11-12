"""Run: python -m examples.research.index_enhance_engine_pipeline --data BARS --split DATE."""

from __future__ import annotations

import tradelearn.engine as bt
from examples.research._workflow import finish, infer, inputs
from tradelearn.backtest.history import window


class EngineResearchIndexEnhance(bt.Strategy):
    params = (("bundle", None),)

    def next(self):
        snapshot = window(self, data=self.datas, size=self.p.bundle["window_size"])
        weights = infer(self.p.bundle, snapshot)
        if weights is None or weights.empty:
            return
        for feed in self.datas:
            self.order_target_percent(data=feed, target=float(weights.get(feed._name, 0.0)))


def main():
    bundle, research_result, test_bars, output, options = inputs()
    cerebro = bt.Cerebro()
    for symbol in test_bars.index.get_level_values("symbol").unique():
        cerebro.adddata(
            bt.feeds.PandasData(dataname=test_bars.xs(symbol, level="symbol")), name=symbol
        )
    cerebro.broker.setcash(100_000.0)
    cerebro.addstrategy(EngineResearchIndexEnhance, bundle=bundle)
    strategies = cerebro.run()
    finish(strategies, research_result, output, options)
    return strategies, research_result


if __name__ == "__main__":
    main()
