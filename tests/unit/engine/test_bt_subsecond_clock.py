"""Subsecond bar clocks must stay exact through the existing integer native ABI."""
import datetime as dt
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def clock_calls(api,validity=False):
    rows=[];notices=[]
    index=pd.to_datetime(['2026-01-02 09:00:00.100000','2026-01-02 09:00:00.500000','2026-01-02 09:00:00.500006','2026-01-02 09:00:00.900000'])
    frame=pd.DataFrame({n:[10.,20.,30.,40.] for n in ['open','high','low','close','volume']},index=index)
    class Business(api.Strategy):
        def next(self):
            rows.append((len(self),self.datetime.datetime().isoformat(),self.data.datetime[0],tuple(self.data.datetime.get(size=min(2,len(self))))))
            if len(self)==1:
                if validity:self.buy(size=1,price=1,exectype=api.Order.Limit,valid=dt.datetime(2026,1,2,9,0,0,500000))
                else:self.buy(size=1)
        def notify_order(self,o):
            if o.status in (api.Order.Completed,api.Order.Expired):notices.append((len(self),o.getstatusname(),o.executed.size,o.executed.price))
    engine=api.Cerebro(stdstats=False,runonce=False);engine.broker.setcash(1000.)
    engine.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.Seconds))
    engine.addstrategy(Business);engine.run()
    return rows,notices,engine.broker.getcash()


@pytest.mark.parametrize('validity',[False,True])
def test_exact_subsecond_bt_dates_market_matching_and_deadline_boundary(validity):
    assert clock_calls(tl,validity)==clock_calls(official,validity)


def microsecond_views(api):
    rows=[]
    index=pd.DatetimeIndex([pd.Timestamp('2026-01-02 09:00:00')+pd.Timedelta(microseconds=n) for n in range(31)])
    frame=pd.DataFrame({n:[10.]*31 for n in ('open','high','low','close','volume')},index=index)
    class Business(api.Strategy):
        def next(self):
            rows.append((self.data.datetime[0],self.data.datetime.datetime().isoformat(),tuple(self.data.datetime.get(size=min(3,len(self)))),self.datetime[0]))
    engine=api.Cerebro(stdstats=False);engine.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.MicroSeconds))
    engine.addstrategy(Business);engine.run()
    return rows


def test_microsecond_date_view_avoids_float_unix_rounding_before_bt_ordinal_conversion():
    assert microsecond_views(tl)==microsecond_views(official)


def mixed_precision_calls(api,primary_fractional,named):
    rows=[];notices=[]
    whole=pd.to_datetime(['2026-01-02 09:00:00','2026-01-02 09:00:01','2026-01-02 09:00:02'])
    fraction=pd.to_datetime(['2026-01-02 09:00:00.500001','2026-01-02 09:00:01.500001','2026-01-02 09:00:02.500001'])
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def prenext(self):self.capture()
        def next(self):
            self.capture()
            if not self.sent:self.buy(data=self.datas[1],size=1);self.sent=True
        def capture(self):rows.append((len(self),self.datetime[0],tuple((len(d),d.datetime[0] if len(d) else None,d.close[0] if len(d) else None) for d in self.datas)))
        def notify_order(self,o):
            if o.status==api.Order.Completed:notices.append((o.executed.dt,o.executed.price))
    c=api.Cerebro(stdstats=False,runonce=False)
    c.broker.setcash(10000.)
    for number,index in enumerate([fraction,whole] if primary_fractional else [whole,fraction]):
        frame=pd.DataFrame({n:[10.,20.,30.] for n in ('open','high','low','close','volume')},index=index)
        c.adddata(api.feeds.PandasData(dataname=frame),**({"name":("duplicate" if named == "duplicate" else f"asset{number}")} if named else {}))
    c.addstrategy(Business);c.run();return rows,notices,c.broker.getcash()

@pytest.mark.parametrize('primary_fractional',[False,True])
@pytest.mark.parametrize('named',[False,True,'duplicate'])
def test_mixed_precision_union_market_order_on_secondary_actual_clock(primary_fractional,named):
    assert mixed_precision_calls(tl,primary_fractional,named)==mixed_precision_calls(official,primary_fractional,named)


def micro_validity_calls(api,microsecond):
    seen=[]
    index=pd.DatetimeIndex([pd.Timestamp('2026-01-02 09:00:00')+pd.Timedelta(microseconds=n) for n in range(31)])
    frame=pd.DataFrame({n:[10.]*31 for n in ('open','high','low','close','volume')},index=index)
    class Business(api.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=1,price=1,exectype=api.Order.Limit,valid=dt.datetime(2026,1,2,9,0,0,microsecond))
        def notify_order(self,o):
            if o.status==api.Order.Expired:seen.append((len(self),self.data.datetime[0],o.executed.size))
    c=api.Cerebro(stdstats=False,runonce=False)
    c.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.MicroSeconds));c.addstrategy(Business);c.run()
    return seen

@pytest.mark.parametrize('microsecond',[0,2,8,13,17,26])
def test_bt_float_day_validity_quantization_at_microsecond_boundaries(microsecond):
    assert micro_validity_calls(tl,microsecond)==micro_validity_calls(official,microsecond)


def market_ordinal_eligibility(api,case):
    offsets = [125000,500000,500000,875000,1000000] if case=='duplicate' else [1,2,5,12,25]
    index=pd.DatetimeIndex([pd.Timestamp('2026-01-02 09:00:00')+pd.Timedelta(microseconds=n) for n in offsets])
    frame=pd.DataFrame({n:[10.,20.,30.,40.,50.] for n in ('open','high','low','close','volume')},index=index)
    rows=[];fills=[]
    class Business(api.Strategy):
        def next(self):
            rows.append((len(self),self.data.datetime[0],self.data.close[0]))
            if len(self)==(2 if case=='duplicate' else 1):self.buy(size=.25)
        def notify_order(self,o):
            if o.status==api.Order.Completed:fills.append((o.executed.dt,o.executed.size,o.executed.price))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(1000.)
    c.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.MicroSeconds));c.addstrategy(Business);c.run()
    return rows,fills,c.broker.getcash()

@pytest.mark.parametrize('case',['duplicate','quantized'])
def test_market_waits_for_strictly_greater_bt_date_without_collapsing_raw_events(case):
    assert market_ordinal_eligibility(tl,case)==market_ordinal_eligibility(official,case)


def ordered_feed_expiry(api,last_target_before_deadline):
    rows=[];expired=[]
    offsets=[[0,500000,1000000,1500000],[250000,750000] if last_target_before_deadline else [250000,750000,1250000]]
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def prenext(self):self.capture()
        def next(self):
            self.capture()
            if not self.sent:
                self.buy(data=self.datas[-1],size=.25,price=1,exectype=api.Order.Limit,valid=dt.datetime(2026,1,2,9,0,0,800000));self.sent=True
        def capture(self):rows.append((self.datetime[0],tuple(d.datetime[0] if len(d) else None for d in self.datas)))
        def notify_order(self,o):
            if o.status==api.Order.Expired:expired.append((o.data.datetime[0],o.executed.size))
    c=api.Cerebro(stdstats=False,runonce=False);c.broker.setcash(1000.)
    for number,times in enumerate(offsets):
        index=pd.DatetimeIndex([pd.Timestamp('2026-01-02 09:00:00')+pd.Timedelta(microseconds=t) for t in times])
        frame=pd.DataFrame({n:[10.+number]*len(times) for n in ('open','high','low','close','volume')},index=index)
        c.adddata(api.feeds.PandasData(dataname=frame,timeframe=api.TimeFrame.MicroSeconds),name=str(number))
    c.addstrategy(Business);c.run();return rows,expired,c.broker.getcash()

@pytest.mark.parametrize('last_target_before_deadline',[False,True])
def test_bt_expiry_uses_ordered_feed_not_an_unrelated_primary_bar(last_target_before_deadline):
    assert ordered_feed_expiry(tl,last_target_before_deadline)==ordered_feed_expiry(official,last_target_before_deadline)
