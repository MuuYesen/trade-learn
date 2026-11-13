"""Custom Observer outputs use the existing Runtime line buffer and BT aliases."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.components import LiveComponents


def observations(api, stream=False, warmup=False, multi=False):
    rows = []
    frame = pd.DataFrame({name: [10., 20., 30.] for name in
                          ('open', 'high', 'low', 'close', 'volume')},
                         index=pd.date_range('2026-01-01', periods=3))
    frames = [frame]
    if multi:
        frames = [frame.iloc[[0, 2]].set_axis(pd.to_datetime(['2026-01-01', '2026-01-04'])),
                  frame.set_axis(pd.to_datetime(['2026-01-02', '2026-01-03', '2026-01-04']))]

    class Sample(api.Observer):
        lines = ('measured',)
        params = (('factor', 2),)
        def __init__(self):
            assert self.l.measured is self.lines[0]
            assert self.lines.getlinealiases() == ('measured',)
        def record(self, phase):
            self.lines.measured[0] = self.data.close[0] * self.p.factor
            rows.append((phase, self.measured[0], list(self.lines.measured.get(size=min(2, len(self))))))
        def prenext(self):
            self.record("prenext")
        def nextstart(self):
            self.record("nextstart")
        def next(self):
            self.record("next")

    class Business(api.Strategy):
        def __init__(self):
            if warmup:
                if api is official:
                    self.average = api.indicators.SMA(self.data.close, period=2)
                else:
                    from tradelearn.indicators import talib
                    self.average = talib.SMA(self.data.close, timeperiod=2)

    engine = api.Cerebro(stdstats=False)
    engine.addobserver(Sample)
    if stream:
        runtime = StrategyRuntime(Business, feeds={str(i): 'bar' for i in range(len(frames))}, capacity=2)
        components = LiveComponents(engine, runtime)
        try:
            runtime.start()
            stamps = frames[0].index
            for data in frames[1:]:
                stamps = stamps.union(data.index)
            for stamp in stamps:
                runtime.step_market_batch([
                    (str(i), dict(ts=stamp.tz_localize('UTC'), **data.loc[stamp].to_dict()))
                    for i, data in enumerate(frames) if stamp in data.index])
                components.after_market()
            runtime.stop()
            components.finish()
            assert len(runtime.strategy.observers.sample.lines.measured._output) == 2
        finally:
            runtime.cleanup()
    else:
        for data in frames:
            engine.adddata(api.feeds.PandasData(dataname=data))
        engine.addstrategy(Business)
        engine.run()
    return rows


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('warmup', [False, True])
def test_custom_observer_line_aliases_write_and_trailing_windows_match_bt(stream, warmup):
    assert observations(tl, stream=stream, warmup=warmup) == observations(official, warmup=warmup)


@pytest.mark.parametrize('stream', [False, True])
def test_multidata_observer_output_clock_retains_every_strategy_step_like_bt(stream):
    assert observations(tl, stream=stream, multi=True) == observations(official, multi=True)
