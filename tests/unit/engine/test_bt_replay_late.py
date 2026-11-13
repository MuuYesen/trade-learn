"""Existing replay takelate controls actual late/duplicate rows, not sort order."""
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as tl


def frame_for(kind):
    offsets={'duplicate':[0,60,60,180,540],
             'late':[0,60,30,180,120,540],
             'late_edge':[0,120,120,60,540],
             'late_eof':[0,60,30], 'late_edge_eof':[0,120,120], 'quantized_eof':[0,.000001,.000002],
             'quantized':[0,.000001,.000002,.000012,60]}[kind]
    index=pd.DatetimeIndex([pd.Timestamp('2026-01-02 09:00')+pd.Timedelta(seconds=x) for x in offsets])
    values=[float(10*(i+1)) for i in range(len(index))]
    return pd.DataFrame(dict(open=values,high=[v+3 for v in values],low=[v-2 for v in values],
                             close=[v+1 for v in values],volume=values),index=index)


def profile(api,kind,clone,options):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            d=self.datas[-1]
            rows.append((len(self),self.datetime[0],len(d),d.datetime[0],d.open[0],d.high[0],d.low[0],d.close[0],d.volume[0],
                         d.close[-1] if len(d)>1 else None,list(d.close.get(size=min(2,len(d))))))
    c=api.Cerebro(stdstats=False,runonce=False)
    source=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(source)
    c.replaydata(source,timeframe=api.TimeFrame.Minutes,compression=2,**options)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['duplicate','late','late_edge','quantized','late_eof','late_edge_eof','quantized_eof'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('options',[{},dict(takelate=False),dict(adjbartime=True),dict(bar2edge=False)])
def test_late_duplicate_raw_order_and_public_revisions_match_fixed_bt(kind,clone,options):
    assert profile(tl,kind,clone,options)==profile(official,kind,clone,options)


@pytest.mark.parametrize('kind',['duplicate','late','late_edge','quantized','late_eof','late_edge_eof','quantized_eof'])
@pytest.mark.parametrize('options',[{},dict(takelate=False),dict(adjbartime=True),dict(bar2edge=False)])
def test_clone_late_preparation_keeps_one_pending_view_and_actual_bt_cursor_withdrawal(kind,options):
    from tradelearn.engine.replaying import replay_feed
    data=tl.feeds.PandasData(dataname=frame_for(kind),timeframe=tl.TimeFrame.Minutes)
    feed=replay_feed(data,clone=True,timeframe=tl.TimeFrame.Minutes,compression=2,**options)
    actual=[]
    for cursor in range(len(feed._frame)):
        feed._advance(cursor)
        row=(len(feed),feed.datetime[0],feed.open[0],feed.high[0],feed.low[0],feed.close[0],feed.volume[0],
             feed.close[-1] if len(feed)>1 else None,list(feed.close.get(size=min(2,len(feed)))))
        if not actual or actual[-1]!=row:actual.append(row)
    expected=[]
    for row in profile(official,kind,True,options):
        view=row[2:]
        if not expected or expected[-1]!=view:expected.append(view)
    assert actual==expected
