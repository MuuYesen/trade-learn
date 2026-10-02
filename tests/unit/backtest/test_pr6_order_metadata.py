"""Regressions reproduced while reviewing PR #6 against current master."""
import pandas as pd
import pytest
from tradelearn.backtest.engine import _orders_frame
from tradelearn.backtest.models import Order
from tradelearn.engine import Cerebro, Strategy


def bars():
    return pd.DataFrame({k: [10.] * 6 for k in ('open','high','low','close')},
                        index=pd.date_range('2026-01-01', periods=6, tz='UTC')).assign(volume=1000.)


def run(strategy, mode, feeds=1, on_close=False):
    engine = Cerebro(match_mode=mode, trade_on_close=on_close)
    for i in range(feeds):
        engine.adddata(bars(), name=f'asset{i}')
    engine.addstrategy(strategy)
    return engine.run()[0]


@pytest.mark.parametrize('mode', ['exact','smart'])
@pytest.mark.parametrize('feeds', [1,2])
@pytest.mark.parametrize('on_close', [False,True])
def test_order_history_keeps_creation_and_execution_times(mode, feeds, on_close):
    class Timed(Strategy):
        def init(self): self.bar=0
        def next(self):
            self.bar+=1
            if self.bar==1: self.entry=self.buy(size=1, tag='entry')
            if self.bar==3: self.exit=self.sell(size=1)
            if self.bar==4: self.unfilled=self.buy(size=1,price=1,exectype=Order.Limit)
            if self.bar==5: self.cancel(self.unfilled)
    s=run(Timed,mode,feeds,on_close)
    frame=_orders_frame(s.broker).set_index('ref')
    expected=[(s.entry,1,1 if on_close else 2),(s.exit,3,3 if on_close else 4),(s.unfilled,4,None)]
    for order,created,executed in expected:
        row=frame.loc[order.ref]
        assert row['datetime']==pd.Timestamp(f'2026-01-{created:02}',tz='UTC')
        if executed is None: assert pd.isna(row['executed_dt'])
        else: assert row['executed_dt']==pd.Timestamp(f'2026-01-{executed:02}',tz='UTC')
    assert frame.loc[s.entry.ref,'tag']=='entry'
    assert frame.loc[s.entry.ref,'info']=={'tag':'entry'}
    assert frame.loc[s.unfilled.ref,'price']==1
    assert s.unfilled.status==Order.Canceled


@pytest.mark.parametrize('mode',['exact','smart'])
@pytest.mark.parametrize('side',['buy','sell'])
@pytest.mark.parametrize('stop_enabled,limit_enabled',[(False,False),(False,True),(True,False),(True,True)])
def test_bracket_suppressed_sides_create_no_orders(mode,side,stop_enabled,limit_enabled):
    class Bracket(Strategy):
        def init(self): self.bar=0
        def next(self):
            self.bar+=1
            if self.bar==1:
                self.group=getattr(self,side+'_bracket')(
                    size=1,price=10,exectype=Order.Market,
                    stopprice=5 if side=='buy' else 15,
                    limitprice=15 if side=='buy' else 5,
                    stopexec=Order.Stop if stop_enabled else None,
                    limitexec=Order.Limit if limit_enabled else None)
    s=run(Bracket,mode)
    main,stop,limit=s.group
    assert (stop is not None)==stop_enabled
    assert (limit is not None)==limit_enabled
    assert len(s.broker._orders)==1+stop_enabled+limit_enabled
    assert main.status==Order.Completed
    assert s.position.size==(1 if side=='buy' else -1)
    frame=_orders_frame(s.broker).set_index('ref')
    for child in (stop,limit):
        if child is not None: assert frame.loc[child.ref,'parent_ref']==main.ref


@pytest.mark.parametrize('mode',['exact','smart'])
@pytest.mark.parametrize('runner',['native','python'])
def test_sparse_target_uses_submission_clock_and_actual_fill_bar(monkeypatch,mode,runner):
    from tradelearn.backtest import engine as engine_module
    if runner=='python':
        monkeypatch.setattr(engine_module,'_build_clocked_multi_data_runner',lambda datas: None)
    class Sparse(Strategy):
        def init(self): self.bar=0
        def next(self):
            self.bar+=1
            if self.bar==3: self.order=self.buy(data=self.datas[1],size=1)
    primary=pd.concat([bars(),bars().iloc[:2].set_axis(pd.date_range('2026-01-07',periods=2,tz='UTC'))])
    secondary=primary.loc[pd.to_datetime(['2026-01-01','2026-01-02','2026-01-04','2026-01-06','2026-01-07','2026-01-08'],utc=True)]
    engine=Cerebro(match_mode=mode)
    engine.adddata(primary,name='clock');engine.adddata(secondary,name='target');engine.addstrategy(Sparse)
    s=engine.run()[0]
    row=_orders_frame(s.broker).iloc[0]
    assert row['datetime']==pd.Timestamp('2026-01-03',tz='UTC')
    assert row['executed_dt']==pd.Timestamp('2026-01-04',tz='UTC')
    assert pd.Timestamp(s.broker._fills[0]['datetime'],unit='s',tz='UTC')==pd.Timestamp('2026-01-04',tz='UTC')
