"""P02 nondefault replay uses the shared logical revision runtime, not resample."""
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def profile(api,kind,options,clone):
    rows=[]
    class Business(api.Strategy):
        def next(self):
            d=self.datas[-1]
            rows.append((len(self),self.datetime[0],len(d),d.datetime[0],d.open[0],d.high[0],d.low[0],d.close[0],d.volume[0],
                         d.close[-1] if len(d)>1 else None,list(d.close.get(size=min(2,len(d))))))
    c=api.Cerebro(stdstats=False,runonce=False)
    data=api.feeds.PandasData(dataname=frame_for(kind),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(data)
    c.replaydata(data,timeframe=api.TimeFrame.Minutes,compression=2,**options)
    c.addstrategy(Business);c.run();return rows


@pytest.mark.parametrize('kind',['dense','sparse','midnight','days'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('options',[{'bar2edge':False},{'rightedge':False},{'boundoff':1},{'boundoff':-1},{'takelate':False}])
def test_nondefault_replay_boundary_flags_logical_length_current_and_lag_match_bt(kind,clone,options):
    assert profile(tl,kind,options,clone)==profile(official,kind,options,clone)


@pytest.mark.parametrize('kind',['dense','sparse'])
@pytest.mark.parametrize('clone',[False,True])
def test_adjusted_replay_delivers_closed_revision_before_next_actual_row(kind,clone):
    assert profile(tl,kind,{'adjbartime':True},clone)==profile(official,kind,{'adjbartime':True},clone)


def test_adjusted_replay_bounded_builder_emits_current_and_closed_view_in_order():
    from tradelearn.engine.replaying import _ConfiguredReplaySubday
    from tradelearn.backtest.clock import date2num
    import pandas as pd
    source=tl.feeds.PandasData(dataname=frame_for('sparse'),timeframe=tl.TimeFrame.Minutes)
    builder=_ConfiguredReplaySubday(source,tl.TimeFrame.Minutes,2,
        dict(bar2edge=True,adjbartime=True,rightedge=True,boundoff=0,takelate=True))
    rows=[]
    for stamp,row in frame_for('sparse').iterrows():
        for count,label,bar in builder.push(stamp.value,row.to_dict()):
            rows.append((count,date2num(pd.Timestamp(label,unit='ns')),bar['close'],bar['volume']))
    expected=profile(official,'sparse',{'adjbartime':True},False)
    assert rows==[(row[2],row[3],row[7],row[8]) for row in expected]
    builder.finish();assert builder.state._bar is None
