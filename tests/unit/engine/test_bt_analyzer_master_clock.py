"""BT calendar analyzers use the Strategy clock, never a late source timestamp."""

from types import SimpleNamespace

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.components import LiveComponents


def reference(analyzer):
    frame = pd.DataFrame(
        {key: [100.0, 110.0, 120.0, 90.0, 130.0, 140.0]
         for key in ('open', 'high', 'low', 'close', 'volume')},
        index=pd.to_datetime(['2026-01-01', '2026-01-01', '2026-01-05',
                              '2026-01-05', '2026-01-05', '2026-01-06']),
    )

    class Business(official.Strategy):
        def __init__(self):
            self.broker.getvalue = lambda: self.data.close[0] if len(self.data) else 100.0

    engine = official.Cerebro(stdstats=False)
    engine.adddata(official.feeds.PandasData(dataname=frame))
    engine.addstrategy(Business)
    engine.addanalyzer(getattr(official.analyzers, analyzer),
                       timeframe=official.TimeFrame.Days, _name='sample')
    return dict(engine.run()[0].analyzers.sample.get_analysis())


@pytest.mark.parametrize('analyzer', ['Returns', 'SharpeRatio'])
def test_late_secondary_source_uses_monotonic_master_clock_for_periods(analyzer):
    state = {'value': 100.0}
    broker = SimpleNamespace(getvalue=lambda: state['value'], fundmode=False)
    runtime = StrategyRuntime(tl.Strategy, feeds={'A': 'bar', 'B': 'bar'}, broker=broker)
    engine = tl.Cerebro(stdstats=False)
    engine.addanalyzer(getattr(tl.analyzers, analyzer),
                       timeframe=tl.TimeFrame.Days, _name='sample')
    components = LiveComponents(engine, runtime, account_provider=lambda: state)
    try:
        runtime.start()
        for symbol, day, value in [('A', '01', 100), ('B', '01', 110),
                                   ('A', '05', 120), ('B', '03', 90), ('B', '04', 130), ('A', '06', 140)]:
            state['value'] = value
            runtime.step_market(symbol, dict(ts=f'2026-01-{day}T00:00:00Z',
                                            open=value, high=value, low=value,
                                            close=value, volume=1))
            components.after_market()
            if symbol == 'B' and day == '03':
                assert runtime.context.ts_event_ns == pd.Timestamp('2026-01-03', tz='UTC').value
                assert runtime.strategy.datetime.datetime() == pd.Timestamp('2026-01-05').to_pydatetime()
        runtime.stop()
        components.finish()
        assert engine.analyzer_results['sample'] == pytest.approx(reference(analyzer))
        sample = runtime.strategy.analyzers.sample
        assert getattr(sample, '_period_count', getattr(sample, '_count', None)) == 3
    finally:
        runtime.cleanup()
