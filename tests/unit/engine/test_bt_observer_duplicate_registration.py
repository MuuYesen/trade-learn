"""Independent IA-C6 duplicate registration: Observer differs from Analyzer naming."""
import pytest
import json
from pathlib import Path
import sys
import pandas as pd
import backtrader as official
from tradelearn import engine as tl
def probe(api,kind):
    rows=[];result={}
    if kind=='observer':
        class Sample(api.Observer):
            params=(('marker',0),)
            lines=('measured',)
            def next(self):self.lines.measured[0]=self.p.marker;rows.append(self.p.marker)
    else:
        class Sample(api.Analyzer):
            params=(('marker',0),)
            def next(self):rows.append(self.p.marker)
    class Business(api.Strategy):
        def stop(self):
            collection=self.observers if kind=='observer' else self.analyzers
            result.update(names=list(collection.getnames()),markers=[v.p.marker for v in collection],
                          named=collection.getbyname('sample').p.marker,
                          attribute=collection.sample.p.marker)
    frame=pd.DataFrame({k:[10.,20.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=2))
    c=api.Cerebro(stdstats=False,runonce=False);c.adddata(api.feeds.PandasData(dataname=frame));c.addstrategy(Business)
    add=c.addobserver if kind=='observer' else c.addanalyzer
    add(Sample,marker=11);add(Sample,marker=22);c.run();result['callbacks']=rows;return result

@pytest.mark.parametrize('kind',['observer','analyzer'])
def test_duplicate_component_registration_callbacks_and_name_lookup_match_fixed_bt(kind):
    assert probe(tl,kind)==probe(official,kind)


def test_duplicate_observer_registration_uses_both_owners_on_bounded_stream():
    from tradelearn.backtest.event_runtime import StrategyRuntime
    from tradelearn_runtime.components import LiveComponents
    rows=[]
    class Sample(tl.Observer):
        params=(('marker',0),)
        lines=('measured',)
        def next(self):self.measured[0]=self.p.marker;rows.append(self.p.marker)
    engine=tl.Cerebro(stdstats=False,runonce=False)
    engine.addobserver(Sample,marker=11);engine.addobserver(Sample,marker=22)
    runtime=StrategyRuntime(tl.Strategy,feeds={'A':'bar'},capacity=2)
    components=LiveComponents(engine,runtime)
    try:
        runtime.start()
        for day in [1,2]:
            runtime.step_market('A',dict(ts=f'2026-01-{day:02}T00:00:00Z',open=10.,high=10.,low=10.,close=10.,volume=1.))
            components.after_market()
        runtime.stop();components.finish()
        collection=runtime.strategy.observers
        assert dict(names=collection.getnames(),markers=[v.p.marker for v in collection],named=collection.getbyname('sample').p.marker,attribute=collection.sample.p.marker,callbacks=rows)==probe(official,'observer')
        assert collection[0] is not collection[1]
    finally:runtime.cleanup()
