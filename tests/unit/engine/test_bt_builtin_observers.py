"""Current six public Observer types use fixed BT lines, units and phase."""
import math
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def builtin_observer_profile(api,fund=None,mode=False,runonce=False,timeframe=None,*,barplot=False,pnlcomm=True,warmup=False,intraday=False,compression=None,analyzer_fund=False):
    observed={};funds=[];business=[]
    frame=pd.DataFrame({n:[100.,110.,90.,120.,80.,140.] for n in ('open','high','low','close','volume')},index=pd.date_range('2026-01-01',periods=6))
    if intraday:
        frame.index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:06','2026-01-02 09:07','2026-01-02 09:08','2026-01-02 09:14'])
    class Minimum(api.Indicator):
        lines=('sample',)
        def __init__(self):self.addminperiod(2)
        def next(self):self.lines.sample[0]=self.data.close[0]
    class Business(api.Strategy):
        def __init__(self):
            if warmup:self.minimum=Minimum(self.data)
        def next(self):
            business.append(('next',len(self)))
            if len(self) in (1,4):self.buy(size=2 if len(self)==1 else 1)
            if len(self) in (3,5):self.sell(size=2 if len(self)==3 else 1)
        def notify_fund(self,cash,value,fundvalue,shares):
            funds.append((len(self),cash,value,fundvalue,shares))
            business.append(('fund',len(self)))
    c=api.Cerebro(stdstats=False,runonce=runonce);c.broker.setcash(10000.)
    c.broker.setcommission(commission=.001);c.broker.set_fundmode(mode,fundstartval=250.)
    c.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes if intraday else api.TimeFrame.Days));c.addstrategy(Business)
    if analyzer_fund:
        class FundProbe(api.Analyzer):
            def notify_fund(self,*facts):business.append(('analyzerfund',len(self)))
        c.addanalyzer(FundProbe)
    for name in ('Value','Broker','BuySell','Trades','DrawDown','TimeReturn'):
        base=getattr(api.observers,name);observed[name]=[]
        def next(self,_base=base,_name=name):
            _base.next(self)
            row=[]
            for line in self.lines:
                value=float(line[0]);row.append(None if math.isnan(value) else value)
            observed[_name].append((len(self),self.lines.getlinealiases(),row))
        cls=type('Observed'+name,(base,),{'next':next})
        options=dict(fund=fund) if name in ('Value','Broker','DrawDown','TimeReturn') else {}
        if name=='TimeReturn' and timeframe is not None:options['timeframe']=timeframe
        if name=='TimeReturn' and compression is not None:options['compression']=compression
        if name=='BuySell':options.update(barplot=barplot,bardist=.1)
        if name=='Trades':options['pnlcomm']=pnlcomm
        c.addobserver(cls,**options)
    c.run()
    return observed,funds,business


@pytest.mark.parametrize('fund',[None,False,True])
@pytest.mark.parametrize('mode',[False,True])
@pytest.mark.parametrize('runonce',[False,True])
def test_current_builtin_observer_lines_and_fund_notifications_match_bt(fund,mode,runonce):
    actual,funds,phase=builtin_observer_profile(tl,fund,mode,runonce)
    golden,expected,expected_phase=builtin_observer_profile(official,fund,mode,runonce)
    assert phase==expected_phase
    assert len(funds)==len(expected)
    for got,want in zip(funds,expected):assert got==pytest.approx(want)
    for name in golden:
        assert len(actual[name])==len(golden[name])
        for got,want in zip(actual[name],golden[name]):
            assert got[:2]==want[:2],name
            assert got[2]==pytest.approx(want[2]),name


@pytest.mark.parametrize('timeframe',[official.TimeFrame.Weeks,official.TimeFrame.Months,official.TimeFrame.NoTimeFrame])
def test_existing_timereturn_observer_uses_calendar_bucket_not_per_bar(timeframe):
    actual,_,_=builtin_observer_profile(tl,timeframe=timeframe)
    golden,_,_=builtin_observer_profile(official,timeframe=timeframe)
    for got,want in zip(actual['TimeReturn'],golden['TimeReturn']):
        assert got[:2]==want[:2]
        assert got[2]==pytest.approx(want[2])


@pytest.mark.parametrize('options',[
    {'barplot':True,'pnlcomm':False},
    {'warmup':True,'analyzer_fund':True},
    {'intraday':True,'timeframe':official.TimeFrame.Minutes,'compression':7},
    {'intraday':True,'timeframe':official.TimeFrame.Minutes,'compression':2},
])
def test_builtin_observer_applicable_parameters_and_prebusiness_fund_phase(options):
    actual,funds,phase=builtin_observer_profile(tl,**options)
    golden,expected,expected_phase=builtin_observer_profile(official,**options)
    assert phase==expected_phase
    for got,want in zip(funds,expected):assert got==pytest.approx(want)
    for name in golden:
        assert len(actual[name])==len(golden[name])
        for got,want in zip(actual[name],golden[name]):
            assert got[:2]==want[:2],name
            assert got[2]==pytest.approx(want[2]),name


def test_builtin_observer_stream_projection_is_bounded_and_uses_supplied_account_facts():
    from types import SimpleNamespace as N
    from tradelearn.backtest.event_runtime import StrategyRuntime
    from tradelearn_runtime.components import LiveComponents
    account=N(fundmode=True,fundvalue=250.,fundshares=40.,getcash=lambda:10000.,getvalue=lambda:account.fundvalue*40)
    class Business(tl.Strategy):pass
    engine=tl.Cerebro(stdstats=False,runonce=False)
    for name in ('Value','Broker','BuySell','Trades','DrawDown','TimeReturn'):
        engine.addobserver(getattr(tl.observers,name))
    runtime=StrategyRuntime(Business,feeds={'A':'bar'},broker=account,capacity=2,
                            feed_metadata={'A':{'timeframe':tl.TimeFrame.Days,'compression':1}})
    components=LiveComponents(engine,runtime,account_provider=object())
    runtime.start()
    try:
        for i,value in enumerate([250.,260.,240.,270.,230.,280.]):
            account.fundvalue=value
            order=N(data=runtime.strategy.data,executed=N(size=1,price=100.+i),isbuy=lambda:i%2==0)
            components.order(order)
            trade=N(data=runtime.strategy.data,isclosed=True,pnl=10.+i,pnlcomm=9.+i)
            components.trade(trade)
            runtime.step_market('A',dict(ts=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(days=i),open=100.,high=100.,low=100.,close=100.,volume=1))
            components.after_market()
        observers=runtime.strategy.observers
        assert observers.value.value[0]==280.
        assert observers.broker.value[0]==280.
        assert math.isnan(observers.broker.cash[0])
        assert observers.buysell.sell[0]==105.
        assert observers.trades.pnlplus[0]==14.
        assert observers.drawdown.drawdown[0]==0.
        assert observers.timereturn.timereturn[0]==pytest.approx(280./230.-1)
        for observer in observers:
            assert len(observer.values)==2
            assert all(len(line._output)==2 for line in observer.lines)
    finally:
        runtime.stop();components.finish();runtime.cleanup()
