"""Existing getvalue(datas) projects qualified native cash-account facts only."""
from types import SimpleNamespace as N
import pytest
from tradelearn_runtime.broker import NautilusBroker


def setup():
    data=N(_name='asset',_symbol='route',close=[12.5]);seen=[]
    def source(requests):
        seen.append(requests)
        return 80.,([25.] if requests is None else [25. for _ in requests])
    b=NautilusBroker(N(account=lambda:dict(cash=80.,equity=105.),cash_valuation=source))
    b.bind(N(datas=[data]));return b,data,seen


@pytest.mark.parametrize('alias',['getvalue','get_value'])
def test_single_raw_multi_cash_duplicate_tuple_and_empty_keep_bt_views(alias):
    b,d,seen=setup();f=getattr(b,alias)
    assert f([d])==25.
    assert f([d,d])==130.
    assert f((d,d))==130.
    assert f([])==f()==105.
    assert seen==[[('route',12.5)],[('route',12.5),('route',12.5)],[('route',12.5),('route',12.5)]]


def test_foreign_data_rejected_before_native_observation():
    b,d,seen=setup()
    with pytest.raises(ValueError,match='does not belong'):b.getvalue([d,N(_name='foreign',close=[1.])])
    assert seen==[]


@pytest.mark.parametrize('price',[float('nan'),float('inf'),-1.,0.])
def test_invalid_selected_view_cannot_enter_native_price_override(price):
    b,d,seen=setup();d.close=[price]
    with pytest.raises(ValueError,match='price'):b.getvalue([d])
    assert seen==[]


def test_futures_margin_sizing_policy_is_not_relabelled_as_cash_asset_value():
    b,d,seen=setup();b.setcommission(margin=20.)
    with pytest.raises(ValueError,match='stocklike'):b.getvalue([d])
    assert seen==[]


@pytest.mark.parametrize('account_kind',['CASH','MARGIN'])
def test_actual_na_cash_position_subset_matches_fixed_bt_snapshot_views(tmp_path,monkeypatch,account_kind):
    import asyncio
    import pandas as pd
    import backtrader as official
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import Equity,InstrumentId,Symbol,Price,AccountType,OmsType,Money,Currency,Bar,BarType
    from tradelearn import engine as bt
    from tests.unit.engine.test_nautilus_store import store
    ins=Equity(InstrumentId.from_str('TLTEST.BINANCE'),Symbol('TLTEST'),Currency.from_str('USD'),2,Price.from_str('0.01'),0,0)
    typ=BarType.from_str(str(ins.id)+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(ins.id.venue,OmsType.NETTING,getattr(AccountType,account_kind),[Money(1000000,Currency.from_str('USD'))],fee_model=FixedFeeModel(Money(1,Currency.from_str('USD'))));native.add_instrument(ins)
    bars=[Bar(typ,*[ins.make_price(price)]*4,ins.make_qty(10),i*60000000000,i*60000000000) for i,price in enumerate([1000.,1010.,1020.,1030.],1)]
    observed=[];errors=[];snapshot=[]
    def provider(cache,portfolio):
        account=cache.account_for_venue(ins.id.venue)
        snapshot[:]=[cache,account]
        return dict(cash=float(account.balances_total()[Currency.from_str('USD')].as_double()),equity=float(portfolio.equity(venue=ins.id.venue,account_id=account.id)[Currency.from_str('USD')].as_double()))
    class Business(bt.Strategy):
        def next(self):
            cash=self.broker.getcash();total=self.broker.getvalue()
            positions=snapshot[0].positions_open(instrument_id=ins.id,account_id=snapshot[1].id)
            qty=float(positions[0].signed_qty) if positions else 0.
            entry=float(positions[0].avg_px_open) if positions else 0.
            # Fixed BT broker view, seeded from this same native snapshot. This
            # verifies valuation semantics; it is not a second matching run.
            reference=official.brokers.BackBroker(cash=cash)
            f=pd.DataFrame({k:[self.data.close[0]] for k in ['open','high','low','close','volume']},index=pd.to_datetime(['2026-01-01']))
            data=official.feeds.PandasData(dataname=f);environment=official.Cerebro();environment.adddata(data);data._start();data.load()
            reference.positions[data].update(qty,entry)
            reference._get_value()
            controls=[(m,l) for m in [False,True] for l in [False,True]]
            if account_kind=='CASH':
                assert [self.broker.getvalue(mkt=m,lever=l) for m,l in controls]==pytest.approx(
                    [reference.getvalue(mkt=m,lever=l) for m,l in controls])
            expected=[reference.getvalue([data]),reference.getvalue([data,data]),reference.getvalue((data,data))]
            if account_kind=='CASH':
                got=[self.broker.getvalue([self.data]),self.broker.getvalue([self.data,self.data]),self.broker.get_value((self.data,self.data))]
                assert got==pytest.approx(expected)
            else:
                got=[self.broker.getvalue([self.data])]
                assert got[0]==pytest.approx(expected[0])
                with pytest.raises(ValueError,match='capital source'):self.broker.getvalue([self.data,self.data])
            assert self.broker.getvalue([])==total
            observed.append((len(self),qty,cash,total,*got))
            if len(self)==1:self.buy(size=2.)
    value=store(tmp_path,account_provider=provider);engine=bt.Cerebro(stdstats=False);engine.setbroker(value.getbroker());engine.adddata(value.getdata(typ));engine.addstrategy(Business)
    class Node:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return N(stop=lambda:None)
        async def run_async(self):
            for bar in bars:
                native.clear_data();native.add_data([bar]);native.run(streaming=True)
                while self.actor._tasks:await asyncio.gather(*tuple(self.actor._tasks))
            native.end()
        def dispose(self):
            nonlocal native
            errors.append(self.actor._worker_error)
            try:native.dispose()
            finally:self.actor=None;native=None
    monkeypatch.setattr(value,'_build_node',Node)
    engine.run()
    assert errors==[None] and len(observed)==4
    assert observed[0][1]==0. and observed[-1][1]==2.

@pytest.mark.parametrize('reply',[(80.,[]),(float('nan'),[25.]),(True,[25.]),(80.,[float('inf')]),(80.,[True])])
def test_incomplete_or_untyped_native_values_do_not_become_subset_money(reply):
    b,d,_=setup();b.gateway.cash_valuation=lambda requests:reply
    with pytest.raises(ValueError,match='complete finite'):b.getvalue([d])


@pytest.mark.parametrize('alias',['getvalue','get_value'])
def test_unleveraged_native_cash_standard_market_lever_views(alias):
    b,d,seen=setup();f=getattr(b,alias)
    assert f(lever=True)==105.
    assert f(mkt=True)==25.
    assert f(mkt=True,lever=True)==25.
    assert f([d],mkt=True,lever=True)==25.
    assert f([d,d],mkt=True,lever=True)==130.


@pytest.mark.parametrize('alias',['getvalue','get_value'])
@pytest.mark.parametrize('selection',[None,[],()])
@pytest.mark.parametrize('mkt',[False,True])
@pytest.mark.parametrize('lever',[False,True])
def test_explicit_empty_selection_ignores_mkt_like_fixed_bt(alias,selection,mkt,lever):
    import backtrader as official
    reference=official.brokers.BackBroker(cash=50000.)
    reference.start();reference._get_value()
    broker=NautilusBroker(N(account=lambda:dict(cash=50000.,equity=50000.),cash_valuation=lambda requests:(50000.,[0.])))
    assert getattr(broker,alias)(selection,mkt=mkt,lever=lever)==getattr(reference,alias)(selection,mkt=mkt,lever=lever)
