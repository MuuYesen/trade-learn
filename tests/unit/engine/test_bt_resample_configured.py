"""P03 already-public resample flags use real BT bars and delivery phases."""
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def profile(api,kind,clone,options,*,index=None,timeframe=None,compression=2,feed_options=None):
    if index is not None:pass
    elif kind=='dense':index=pd.date_range('2026-01-02 09:00',periods=10,freq='min')
    elif kind=='sparse':index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:09','2026-01-02 09:17'])
    else:index=pd.date_range('2026-01-01 23:56',periods=10,freq='min')
    f=pd.DataFrame({n:[float(10*(i+1)) for i in range(len(index))] for n in ('open','high','low','close','volume')},index=index)
    rows=[]
    class Business(api.Strategy):
        def prenext(self):self.next()
        def next(self):
            rows.append((self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.open[0] if len(d) else None,d.close[0] if len(d) else None,d.volume[0] if len(d) else None) for d in self.datas)))
    c=api.Cerebro(stdstats=False,runonce=False)
    d=api.feeds.PandasData(dataname=f,timeframe=timeframe or api.TimeFrame.Minutes,**(feed_options or {}))
    if clone:c.adddata(d,name='raw')
    c.resampledata(d,timeframe=timeframe or api.TimeFrame.Minutes,compression=compression,**options)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['dense','sparse','midnight'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('options',[{}, {'bar2edge':False},{'adjbartime':False},{'rightedge':False},{'boundoff':1},{'boundoff':-1},{'takelate':False}])
def test_resample_existing_flags_and_source_delivery_match_fixed_bt(kind,clone,options):
    assert profile(tl,kind,clone,options)==profile(official,kind,clone,options)


@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('case',['micro','seconds','tz','session','dst','late-true','late-false','duplicates'])
def test_configured_subday_clock_units_sessions_timezone_and_late_source_rows(case,clone):
    feed={};options={};timeframe=official.TimeFrame.Minutes;compression=7
    if case=='micro':
        base=pd.Timestamp('2026-01-02 09:00')
        index=pd.DatetimeIndex([base+pd.Timedelta(v,unit='us') for v in (1,2,5,12,25,40,80)])
        timeframe=official.TimeFrame.MicroSeconds
    elif case=='seconds':
        index=pd.date_range('2026-01-01 23:59:56',periods=10,freq='s');timeframe=official.TimeFrame.Seconds
        options={'adjbartime':False,'rightedge':False,'boundoff':1}
    elif case=='tz':index=pd.date_range('2026-01-02 14:01',periods=10,freq='min');feed={'tz':'America/New_York'}
    elif case=='session':index=pd.to_datetime(['2026-01-01 20:58','2026-01-01 20:59','2026-01-01 21:01']);feed={'tz':'America/New_York','sessionend':__import__('datetime').time(16)}
    elif case=='dst':index=pd.to_datetime(['2026-03-06 20:58','2026-03-06 20:59','2026-03-09 20:01']);feed={'tz':'America/New_York','sessionend':__import__('datetime').time(16)}
    elif case.startswith('late'):
        index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:02','2026-01-02 09:01','2026-01-02 09:03','2026-01-02 09:04'])
        compression=2;options={'takelate':case=='late-true'}
    else:index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:01','2026-01-02 09:02','2026-01-02 09:03','2026-01-02 09:04']);compression=2
    kwargs=dict(index=index,timeframe=timeframe,compression=compression,feed_options=feed)
    assert profile(tl,case,clone,options,**kwargs)==profile(official,case,clone,options,**kwargs)


@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('timeframe',[official.TimeFrame.Days,official.TimeFrame.Weeks,official.TimeFrame.Months,official.TimeFrame.Years])
@pytest.mark.parametrize('options',[{'bar2edge':False},{'adjbartime':False},{'rightedge':False},{'boundoff':1},{'boundoff':-1},{'takelate':False}])
def test_existing_resample_flags_on_calendar_timeframes_follow_bt_actual_effect(timeframe,options,clone):
    rows=[]
    def run(api):
        out=[]
        class Business(api.Strategy):
            def prenext(self):self.next()
            def next(self):out.append((self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.close[0] if len(d) else None) for d in self.datas)))
        index=pd.date_range('2025-12-20',periods=40,freq='D')
        frame=pd.DataFrame({n:list(range(10,50)) for n in ('open','high','low','close','volume')},index=index)
        c=api.Cerebro(stdstats=False,runonce=False);d=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Days)
        if clone:c.adddata(d)
        c.resampledata(d,timeframe=timeframe,compression=2,**options);c.addstrategy(Business);c.run();return out
    assert run(tl)==run(official)


@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('options',[{}, {'bar2edge':False},{'adjbartime':False},{'rightedge':False},
                                    {'boundoff':1},{'boundoff':-1},{'takelate':False}])
def test_configured_resample_market_eligibility_execution_quote_and_fees(clone,options):
    def execute(api):
        index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:03',
            '2026-01-02 09:04','2026-01-02 09:07','2026-01-02 09:08','2026-01-02 09:10','2026-01-02 09:12'])
        prices=[10.,20.,30.,40.,50.,60.,70.,80.]
        frame=pd.DataFrame({'open':prices,'high':[v+2 for v in prices],
                            'low':[v-2 for v in prices],'close':[v+1 for v in prices],
                            'volume':[100.]*8},index=index)
        c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(10000);c.broker.setcommission(commission=.001)
        data=api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Minutes)
        if clone:c.adddata(data)
        target=c.resampledata(data,timeframe=api.TimeFrame.Minutes,compression=2,**options)
        notices=[]
        class Business(api.Strategy):
            def __init__(self):self.sent=False
            def prenext(self):self.next()
            def next(self):
                if len(target) and not self.sent:
                    self.sent=True;self.buy(data=target,size=.25)
            def notify_order(self,order):
                notices.append((order.status,order.created.dt,order.executed.dt,
                    order.executed.size,order.executed.price,order.executed.comm))
        c.addstrategy(Business);c.run()
        return notices,c.broker.getcash(),c.broker.getvalue()
    assert execute(tl)==execute(official)
