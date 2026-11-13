"""An Observer's calculation must not change a BT Strategy's trading warmup."""

import backtrader as official
import pandas as pd
import pytest

from tradelearn import engine as tl
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.components import LiveComponents


def observations(api, stream=False, business_period=None):
    business=[];observed=[]
    frame=pd.DataFrame({name:[10.,20.,30.,40.] for name in
                        ('open','high','low','close','volume')},
                       index=pd.date_range('2026-01-01',periods=4))
    class Business(api.Strategy):
        def __init__(self):
            if business_period is not None:
                if api is official:
                    self.mean=api.indicators.SMA(self.data.close,period=business_period)
                else:
                    from tradelearn.indicators import talib
                    self.mean=talib.SMA(self.data.close,timeperiod=business_period)
        def next(self):
            business.append(len(self))
    class Sample(api.Observer):
        lines=('measured',)
        def __init__(self):
            if api is official:
                self.mean=api.indicators.SMA(self.data.close,period=3)
            else:
                from tradelearn.indicators import talib
                self.mean=talib.SMA(self.data.close,timeperiod=3)
        def next(self):
            self.lines.measured[0]=self.mean[0]
            observed.append(self.measured[0])
    engine=api.Cerebro(stdstats=False,**({"runonce":False} if api is official else {}));engine.addobserver(Sample)
    if stream:
        runtime=StrategyRuntime(Business,feeds={'A':'bar'})
        components=LiveComponents(engine,runtime)
        try:
            runtime.start()
            for stamp,bar in frame.iterrows():
                runtime.step_market('A',dict(ts=stamp.tz_localize('UTC'),**bar.to_dict()))
                components.after_market()
            runtime.stop();components.finish()
        finally:runtime.cleanup()
    else:
        engine.adddata(api.feeds.PandasData(dataname=frame));engine.addstrategy(Business)
        engine.run()
    return business,observed


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('business_period',[None,2])
def test_observer_sma_does_not_delay_business_next_and_still_advances_its_own_state(stream,business_period):
    expected_business,expected_observed=observations(official,business_period=business_period)
    business,observed=observations(tl,stream=stream,business_period=business_period)
    assert business==expected_business==([1,2,3,4] if business_period is None else [2,3,4])
    assert observed==pytest.approx(expected_observed,nan_ok=True)


def test_component_construction_context_restores_after_nested_failure():
    from tradelearn.core.strategy_context import _G,engine_context
    owner=object();nested=object()
    assert _G.current_component is None
    with pytest.raises(ValueError,match='context fixture'):
        with engine_context(component=owner):
            assert _G.current_component is owner
            with engine_context(component=nested):
                assert _G.current_component is nested
                raise ValueError('context fixture')
    assert _G.current_component is None


def observer_phases(api, own_indicator, stream=False, manual_period=0):
    phases=[]
    frame=pd.DataFrame({name:[10.,20.,30.,40.] for name in ('open','high','low','close','volume')}, index=pd.date_range('2026-01-01',periods=4))
    class Business(api.Strategy):
        def __init__(self):
            if api is official:self.mean=api.indicators.SMA(self.data.close,period=2)
            else:
                from tradelearn.indicators import talib
                self.mean=talib.SMA(self.data.close,timeperiod=2)
    class Sample(api.Observer):
        lines=('measured',)
        def __init__(self):
            if manual_period:
                self.addminperiod(manual_period)
            if own_indicator:
                if api is official:self.mean=api.indicators.SMA(self.data.close,period=3)
                else:
                    from tradelearn.indicators import talib
                    self.mean=talib.SMA(self.data.close,timeperiod=3)
        def prenext(self):phases.append(('prenext',len(self)))
        def nextstart(self):phases.append(('nextstart',len(self)))
        def next(self):phases.append(('next',len(self)))
    engine=api.Cerebro(stdstats=False,runonce=False);engine.addobserver(Sample)
    if stream:
        runtime=StrategyRuntime(Business,feeds={'A':'bar'})
        components=LiveComponents(engine,runtime)
        try:
            runtime.start()
            for stamp,bar in frame.iterrows():
                runtime.step_market('A',dict(ts=stamp.tz_localize('UTC'),**bar.to_dict()))
                components.after_market()
            runtime.stop();components.finish()
        finally:runtime.cleanup()
    else:
        engine.adddata(api.feeds.PandasData(dataname=frame));engine.addstrategy(Business);engine.run()
    return phases


@pytest.mark.parametrize('own_indicator',[False,True])
@pytest.mark.parametrize('stream',[False,True])
def test_explicit_event_observer_has_its_own_warmup_not_business_warmup(own_indicator,stream):
    assert observer_phases(tl,own_indicator,stream)==observer_phases(official,own_indicator)


def observer_event_clock(api, strategy_clock, stream=False):
    rows=[]
    frames=[pd.DataFrame({n:[10.,40.] for n in ('open','high','low','close','volume')},index=pd.to_datetime(['2026-01-01','2026-01-04'])),
            pd.DataFrame({n:[20.,30.,40.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-02',periods=3))]
    class Sample(api.Observer):
        lines=('measured',)
        _stclock=strategy_clock
        def record(self,phase):
            self.lines.measured[0]=len(self._owner)
            rows.append((phase,len(self),len(self._owner),list(self.lines.measured.get(size=min(2,len(self))))))
        def prenext(self):self.record('prenext')
        def nextstart(self):self.record('nextstart')
        def next(self):self.record('next')
    engine=api.Cerebro(stdstats=False,runonce=False);engine.addobserver(Sample)
    if stream:
        runtime=StrategyRuntime(tl.Strategy,feeds={'0':'bar','1':'bar'},capacity=2)
        components=LiveComponents(engine,runtime)
        try:
            runtime.start()
            for stamp in frames[0].index.union(frames[1].index):
                runtime.step_market_batch([(str(i),dict(ts=stamp.tz_localize('UTC'),**frame.loc[stamp].to_dict())) for i,frame in enumerate(frames) if stamp in frame.index])
                components.after_market()
            runtime.stop();components.finish()
        finally:runtime.cleanup()
    else:
        for frame in frames:engine.adddata(api.feeds.PandasData(dataname=frame))
        engine.addstrategy(api.Strategy);engine.run()
    return rows


@pytest.mark.parametrize('strategy_clock',[False,True])
@pytest.mark.parametrize('stream',[False,True])
def test_event_observer_default_data_clock_and_stclock_use_existing_distinct_clocks(strategy_clock,stream):
    assert observer_event_clock(tl,strategy_clock,stream)==observer_event_clock(official,strategy_clock)


@pytest.mark.parametrize('stream',[False,True])
def test_event_observer_public_addminperiod_does_not_delay_business(stream):
    assert observer_phases(tl,False,stream,manual_period=3)==observer_phases(official,False,manual_period=3)
