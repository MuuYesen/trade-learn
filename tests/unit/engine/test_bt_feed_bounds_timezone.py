"""Existing BT fromdate/todate use UTC-naive instants even with aware source index."""
import datetime as dt
import backtrader as official
import pandas as pd
import pytest
from tradelearn import engine as tl


def bounds_call(api,aware,bounds):
    rows=[]
    index=pd.date_range('2026-01-01 09:00',periods=6,freq='min',tz='UTC' if aware else None)
    frame=pd.DataFrame({n:[10.,20.,30.,40.,50.,60.] for n in ['open','high','low','close','volume']},index=index)
    class Business(api.Strategy):
        def next(self):rows.append((self.data.datetime.datetime().isoformat(),self.data.close[0]))
    feed=api.feeds.PandasData(dataname=frame,fromdate=bounds[0],todate=bounds[1])
    engine=api.Cerebro(stdstats=False);engine.adddata(feed);engine.addstrategy(Business);engine.run()
    return rows


@pytest.mark.parametrize('aware',[False,True])
@pytest.mark.parametrize('bounds',[
 (dt.datetime(2026,1,1,9,2),dt.datetime(2026,1,1,9,4)),
 (dt.datetime(2026,1,1,9,2,tzinfo=dt.timezone.utc),dt.datetime(2026,1,1,9,4,tzinfo=dt.timezone.utc)),
 (dt.date(2026,1,1),dt.date(2026,1,1)),
 (dt.datetime(2026,1,1,18,2,tzinfo=dt.timezone(dt.timedelta(hours=9))),dt.datetime(2026,1,1,18,4,tzinfo=dt.timezone(dt.timedelta(hours=9)))),
])
def test_feed_bounds_match_fixed_bt_without_enabling_tz_session_options(aware,bounds):
    assert bounds_call(tl,aware,bounds)==bounds_call(official,aware,bounds)
