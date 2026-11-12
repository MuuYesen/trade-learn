"""Run: python -m examples.research.index_enhance_lite_pipeline --data BARS --split DATE."""

from __future__ import annotations

import tradelearn.lite as tl
from examples.research._workflow import finish, infer, inputs


class LiteResearchIndexEnhance(tl.Strategy):
    bundle = {}

    def next(self):
        snapshot = self.window(size=self.bundle["window_size"], data=self.datas)
        weights = infer(self.bundle, snapshot)
        if weights is None or weights.empty:
            return
        self.target_weights(weights, close_missing=True)


def main():
    bundle, research_result, test_bars, output, options = inputs()
    backtest = tl.Backtest(test_bars, LiteResearchIndexEnhance, cash=100_000.0)
    result = backtest.run(bundle=bundle)
    finish(result, research_result, output, options)
    return result, research_result


if __name__ == "__main__":
    main()
