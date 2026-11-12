"""BT actual subclass Analyzer parameter defaults, including explicit overrides."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl


def actual_params(api, overrides):
    class Parent(api.Analyzer):
        params = (("period", 2), ("enabled", True))

        def get_analysis(self):
            return dict(period=self.p.period, enabled=self.p.enabled)

    class Child(Parent):
        params = (("period", 3),)

    frame = pd.DataFrame(
        {name: [10.0, 10.0] for name in ["open", "high", "low", "close", "volume"]},
        index=pd.date_range("2026-01-01", periods=2),
    )
    engine = api.Cerebro(stdstats=False)
    engine.addstrategy(api.Strategy)
    engine.adddata(api.feeds.PandasData(dataname=frame))
    engine.addanalyzer(Child, _name="custom", **overrides)
    return engine.run()[0].analyzers.custom.get_analysis()


@pytest.mark.parametrize("overrides", [{}, {"period": 4}, {"enabled": False}])
def test_child_analyzer_defaults_and_explicit_params_match_actual_bt(overrides):
    assert actual_params(tl, overrides) == actual_params(official, overrides)
