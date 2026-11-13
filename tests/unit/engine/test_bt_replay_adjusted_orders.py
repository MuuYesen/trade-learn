"""BT adjusted replay delivers logical revisions over the actual raw quote turn."""
import backtrader as official
import pytest
from tradelearn import engine as tl
from tests.unit.engine.test_bt_replay_builder import frame_for


def profile(api,when,kind,clone=False):
    trace=[]
    class Business(api.Strategy):
        def __init__(self):self.sent=False
        def notify_order(self,o):
            trace.append(('order',o.getstatusname(),o.created.dt,o.executed.dt,
                          o.executed.size,o.executed.price,o.executed.comm,
                          self.datetime[0],self.datas[-1].datetime[0]))
        def next(self):
            d=self.datas[-1]
            trace.append(('next',len(self),len(d),self.datetime[0],d.datetime[0],d.close[0],
                          self.broker.getcash(),self.broker.getvalue()))
            if not self.sent and d.datetime.datetime().strftime('%H:%M')==when:
                self.sent=True
                kw=dict(data=d,size=.25)
                if kind=='limit':kw.update(exectype=api.Order.Limit,price=35)
                elif kind=='stop':kw.update(exectype=api.Order.Stop,price=35)
                elif kind=='trail':kw.update(exectype=api.Order.StopTrail,trailamount=3)
                self.buy(**kw)
    c=api.Cerebro(stdstats=False,runonce=False)
    c.broker.setcash(1000);c.broker.setcommission(commission=.001)
    if kind=='coc':c.broker.set_coc(True)
    raw=api.feeds.PandasData(dataname=frame_for('sparse'),timeframe=api.TimeFrame.Minutes)
    if clone:c.adddata(raw,name='raw')
    c.replaydata(raw,timeframe=api.TimeFrame.Minutes,compression=2,adjbartime=True)
    c.addstrategy(Business);c.run();return trace


@pytest.mark.parametrize('when',['09:00','09:01','09:02','09:09'])
@pytest.mark.parametrize('kind',['market','limit','stop','trail','coc'])
@pytest.mark.parametrize('clone',[False,True])
def test_adjusted_replay_order_dates_prices_fees_cash_and_equity_match_bt(when,kind,clone):
    assert profile(tl,when,kind,clone)==profile(official,when,kind,clone)


def test_adjusted_replay_keeps_source_asof_and_logical_dates_and_reports_native_equity(monkeypatch):
    import pandas as pd
    import tradelearn.backtest.engine as implementation
    def no_reconstruction(*args,**kwargs):
        raise AssertionError('delivery equity must consume native observations')
    monkeypatch.setattr(implementation,'_build_equity_from_fills',no_reconstruction)
    clocks=[];equity=[];completed=[]
    class Business(tl.Strategy):
        def next(self):
            label=self.data.datetime.datetime().strftime('%H:%M')
            clocks.append((label,self._runtime.context.ts_event_ns))
            equity.append(self.broker.getvalue())
            if label in ('09:01','09:02'):self.buy(size=.25)
        def notify_order(self,o):
            if o.status==o.Completed:completed.append((o.created.dt,o.executed.dt,o.executed.price))
    c=tl.Cerebro(stdstats=False,runonce=False)
    c.replaydata(tl.feeds.PandasData(dataname=frame_for('sparse'),timeframe=tl.TimeFrame.Minutes),
                 timeframe=tl.TimeFrame.Minutes,compression=2,adjbartime=True)
    c.addstrategy(Business);s=c.run()[0]
    assert [x[0] for x in clocks]==['09:00','09:01','09:02','09:09','09:10','09:17']
    assert [x[1] for x in clocks]==[pd.Timestamp(x).value for x in
        ['2026-01-02 09:00','2026-01-02 09:01','2026-01-02 09:09','2026-01-02 09:09','2026-01-02 09:17','2026-01-02 09:17']]
    assert [x[2] for x in completed]==[30.,30.]
    assert completed[0][1]!=completed[1][1]
    assert list(s.stats.equity)==equity
    assert list(s.stats.fills['source_quote_datetime'])==[pd.Timestamp('2026-01-02 09:09',tz='UTC')]*2
