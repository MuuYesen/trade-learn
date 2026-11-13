"""P03 existing feed timezone fields alter presentation/input interpretation only."""
from datetime import date,datetime
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def profile(api,index,**options):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            d=self.data
            rows.append((len(d),d.datetime[0],d.datetime.datetime().isoformat(),
                         d.datetime.datetime(naive=False).isoformat(),str(d.datetime.date()),
                         str(d.datetime.time()),self.datetime.datetime().isoformat(),d.close[0]))
    f=pd.DataFrame({n:list(range(10,10+len(index))) for n in ('open','high','low','close','volume')},index=index)
    c=api.Cerebro(stdstats=False,runonce=False)
    d=api.feeds.PandasData(dataname=f,**options);c.adddata(d);c.addstrategy(Business);c.run()
    return rows,d


@pytest.mark.parametrize('input_aware',[False,True])
@pytest.mark.parametrize('tz,tzinput',[('America/New_York',None),('Asia/Tokyo',None),(None,'America/New_York'),('Asia/Tokyo','America/New_York')])
def test_feed_tz_and_tzinput_match_actual_bt_while_public_dates_remain_utc_ordinals(input_aware,tz,tzinput):
    index=pd.date_range('2026-01-01 09:00',periods=3,freq='h',tz='America/New_York' if input_aware else None)
    actual,d=profile(tl,index,tz=tz,tzinput=tzinput)
    expected,_=profile(official,index,tz=tz,tzinput=tzinput)
    assert actual==expected
    if tzinput is None:
        from tradelearn.backtest.clock import native_timestamps
        assert list(native_timestamps(d,'ns'))==[value.value for value in index]


@pytest.mark.parametrize('kind',['spring-input','fall-input','spring-output','fall-output'])
def test_feed_daylight_saving_interpretation_and_display_match_fixed_bt(kind):
    if kind=='spring-input':index=pd.to_datetime(['2026-03-08 01:30','2026-03-08 02:30','2026-03-08 03:30']);kw={'tzinput':'America/New_York','tz':'America/New_York'}
    elif kind=='fall-input':index=pd.to_datetime(['2026-11-01 00:30','2026-11-01 01:30','2026-11-01 02:30']);kw={'tzinput':'America/New_York','tz':'America/New_York'}
    elif kind=='spring-output':index=pd.to_datetime(['2026-03-08 06:30','2026-03-08 07:30']);kw={'tz':'America/New_York'}
    else:index=pd.to_datetime(['2026-11-01 05:30','2026-11-01 06:30']);kw={'tz':'America/New_York'}
    assert profile(tl,index,**kw)[0]==profile(official,index,**kw)[0]


@pytest.mark.parametrize('bounds',[(date(2026,1,1),date(2026,1,1)),(datetime(2026,1,1,9),datetime(2026,1,1,10))])
def test_naive_feed_bounds_use_declared_output_timezone_as_in_bt(bounds):
    index=pd.date_range('2026-01-01 00:00',periods=36,freq='h',tz='UTC')
    kw={'tz':'America/New_York','fromdate':bounds[0],'todate':bounds[1]}
    assert profile(tl,index,**kw)[0]==profile(official,index,**kw)[0]
