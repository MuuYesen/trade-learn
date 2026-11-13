"""P02 replay aggregation is not completed-bar resampling or a new clock loop."""
import backtrader as bt
import pandas as pd
import pytest
from tradelearn.engine.replaying import _ReplayBarBuilder


def frame_for(kind):
    if kind=='dense':index=pd.date_range('2026-01-02 09:00',periods=10,freq='min')
    elif kind=='sparse':index=pd.to_datetime(['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:09','2026-01-02 09:17'])
    elif kind=='midnight':index=pd.date_range('2026-01-01 23:56',periods=10,freq='min')
    else:index=pd.to_datetime(['2026-01-01 09:00','2026-01-01 10:00','2026-01-02 09:00','2026-01-03 10:00','2026-01-05 09:00'])
    return pd.DataFrame(dict(open=[10.*(i+1) for i in range(len(index))],high=[10.*(i+1)+3 for i in range(len(index))],low=[10.*(i+1)-2 for i in range(len(index))],close=[10.*(i+1)+1 for i in range(len(index))],volume=[float(i+1) for i in range(len(index))]),index=index)


def actual_bt_replay(frame,timeframe,compression):
    rows=[]
    class Business(bt.Strategy):
        def next(self):rows.append((len(self.data),self.data.datetime.datetime().isoformat(),tuple(getattr(self.data,n)[0] for n in ('open','high','low','close','volume'))))
    c=bt.Cerebro(stdstats=False,runonce=False)
    c.replaydata(bt.feeds.PandasData(dataname=frame,timeframe=bt.TimeFrame.Minutes),timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['dense','sparse','midnight','days'])
@pytest.mark.parametrize('timeframe,compression',[(bt.TimeFrame.Minutes,2),(bt.TimeFrame.Minutes,7),(bt.TimeFrame.Days,1),(bt.TimeFrame.Days,2),(bt.TimeFrame.Weeks,1)])
def test_bounded_replay_builder_current_bar_and_logical_length_match_fixed_bt(kind,timeframe,compression):
    frame=frame_for(kind);builder=_ReplayBarBuilder(timeframe,compression,bt.TimeFrame.Minutes);rows=[]
    for stamp,row in frame.iterrows():
        count,ns,bar=builder.push(stamp.value,row.to_dict())
        rows.append((count,pd.Timestamp(ns,unit='ns').isoformat(),tuple(bar[n] for n in ('open','high','low','close','volume'))))
    builder.finish()
    assert rows==actual_bt_replay(frame,timeframe,compression)
    assert builder.builder._bar is None
    with pytest.raises(RuntimeError,match='closed'):builder.push(frame.index[-1].value,frame.iloc[-1].to_dict())


def fixed_bt_feed_views(frame,timeframe,compression):
    rows=[]
    class Business(bt.Strategy):
        def next(self):
            d=self.data
            rows.append((len(d),len(d.close),d.datetime.datetime().isoformat(),d.close[0],
                         d.close[-1] if len(d)>1 else None,list(d.close.get(size=min(2,len(d))))))
    c=bt.Cerebro(stdstats=False,runonce=False)
    c.replaydata(bt.feeds.PandasData(dataname=frame,timeframe=bt.TimeFrame.Minutes),timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['dense','sparse','midnight','days'])
@pytest.mark.parametrize('timeframe,compression',[(bt.TimeFrame.Minutes,2),(bt.TimeFrame.Minutes,7),(bt.TimeFrame.Days,1),(bt.TimeFrame.Weeks,1)])
def test_replay_feed_replaces_visible_current_bar_and_preserves_raw_execution_input(kind,timeframe,compression):
    from tradelearn import engine as tl
    from tradelearn.engine.replaying import replay_feed
    frame=frame_for(kind)
    feed=replay_feed(tl.feeds.PandasData(dataname=frame,timeframe=tl.TimeFrame.Minutes),timeframe=timeframe,compression=compression)
    assert len(feed)==0
    rows=[]
    for i in range(len(frame)):
        feed._advance(i)
        rows.append((len(feed),len(feed.close),feed.datetime.datetime().isoformat(),feed.close[0],
                     feed.close[-1] if len(feed)>1 else None,list(feed.close.get(size=min(2,len(feed))))))
    assert rows==fixed_bt_feed_views(frame,timeframe,compression)
    assert list(feed._execution_quote_values[:,0])==list(frame.open)
    from tradelearn.backtest.clock import native_timestamps
    assert list(native_timestamps(feed,'ns'))==[stamp.value for stamp in frame.index]  # No raw time/observation removed.


@pytest.mark.parametrize('clock',['2025-01-01','2030-01-01'])
@pytest.mark.parametrize('timeframe,compression',[(bt.TimeFrame.Seconds,7),(bt.TimeFrame.Minutes,7),(bt.TimeFrame.Minutes,2)])
def test_clone_private_check_clock_matches_fixed_bt_without_changing_source_business_time(monkeypatch,clock,timeframe,compression):
    from datetime import datetime
    import backtrader.resamplerfilter as filters
    stamp=pd.Timestamp(clock)
    class FrozenDateTime(datetime):
        @classmethod
        def utcnow(cls):return stamp.to_pydatetime()
    monkeypatch.setattr(filters,'datetime',FrozenDateTime)
    frame=frame_for('days');expected=[]
    class Business(bt.Strategy):
        def next(self):
            d=self.datas[-1]
            expected.append((len(d),d.datetime.datetime().isoformat(),tuple(getattr(d,n)[0] for n in ('open','high','low','close','volume'))))
    c=bt.Cerebro(stdstats=False,runonce=False)
    raw=bt.feeds.PandasData(dataname=frame,timeframe=bt.TimeFrame.Minutes)
    c.adddata(raw);c.replaydata(raw,timeframe=timeframe,compression=compression)
    c.addstrategy(Business);c.run()
    check_ns=stamp.value if (timeframe,compression)<(bt.TimeFrame.Minutes,1) else None
    builder=_ReplayBarBuilder(timeframe,compression,bt.TimeFrame.Minutes,check_ns=check_ns)
    actual=[]
    for source,row in frame.iterrows():
        count,ns,bar=builder.push(source.value,row.to_dict())
        assert ns==source.value
        actual.append((count,pd.Timestamp(ns,unit='ns').isoformat(),tuple(bar[n] for n in ('open','high','low','close','volume'))))
    builder.finish()
    assert actual==expected
