"""C5 Trade views use actual fixed-NA position economics, including flip snapshots."""
import math
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar,BarType,Money,Currency,OmsType,AccountType,StrategyId,AccountId
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.testkit.providers import TestInstrumentProvider
from tradelearn import engine as bt
from tradelearn_runtime.execution import create_execution_host


def run_trade_cycle(flip):
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    bars=BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    currency=Currency.from_str('USDT')
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,currency)],fee_model=FixedFeeModel(Money(1,currency)))
    native.add_instrument(instrument)
    native.add_data([Bar(bars,*[instrument.make_price(p)]*4,instrument.make_qty(10),(i+1)*60_000_000_000,(i+1)*60_000_000_000) for i,p in enumerate([1000,1010,1020,1030,1040,1050,1060,1070])])
    rows=[]
    class Business(bt.Strategy):
        def next(self):
            actions=({1:('buy',.25),3:('sell',.375),5:('buy',.125)} if flip else {1:('buy',.25),2:('buy',.125),3:('sell',.125),5:('sell',.25)})
            if len(self) in actions:
                side,size=actions[len(self)];getattr(self,side)(size=size)
        def notify_trade(self,t):
            rows.append({k:getattr(t,k,None) for k in ['ref','size','price','value','commission','pnl','pnlcomm','justopened','long','isopen','isclosed','baropen','barclose','barlen']})
    host=create_execution_host(StrategyConfig(strategy_id=StrategyId('ECON-001'),oms_type=OmsType.NETTING),business_cls=Business,feeds={symbol:'bar'},instruments={symbol:instrument.id},subscriptions=[bars],account_id='BINANCE-001',offline_test=True)
    try:
        native.add_strategy(host);native.run()
        assert all(o.status.name=='FILLED' for o in native.cache.orders())
        account=native.cache.account(AccountId('BINANCE-001'))
        return rows,float(account.balance_total(currency).as_double())
    finally:native.dispose();host=None;native=None


def test_native_partial_cycle_projects_gross_net_fees_value_and_only_bt_trade_notices():
    rows,cash=run_trade_cycle(False)
    assert cash==1000007.25
    assert len(rows)==2
    opened,closed=rows
    assert opened['ref']==closed['ref'] and opened['ref']>0
    assert opened['justopened'] and opened['long'] and opened['isopen']
    assert (opened['size'],opened['price'],opened['value'])==(.25,1000.,250.)
    assert (opened['commission'],opened['pnl'],opened['pnlcomm'])==(1.,0.,-1.)
    assert closed['size']==closed['value']==0
    assert closed['isclosed'] and not closed['justopened']
    assert (closed['commission'],closed['pnl'],closed['pnlcomm'])==(4.,11.25,7.25)
    assert closed['barlen']==closed['barclose']-closed['baropen']


def test_native_flip_uses_closed_snapshot_instead_of_new_cycle_already_in_cache():
    rows,cash=run_trade_cycle(True)
    assert cash==999999.5
    assert len(rows)==4
    first,first_close,second,second_close=rows
    assert first['ref']==first_close['ref'] and second['ref']==second_close['ref']
    assert first['ref']!=second['ref']
    assert first['long'] and not second['long']
    assert first_close['size']==second_close['size']==0
    assert first_close['commission']==1.66666667
    assert first_close['pnl']==pytest.approx(5.)
    assert first_close['pnlcomm']==3.33333333
    assert second['justopened'] and second['size']==-.125
    assert second['commission']==.33333333 and second['pnl']==0.
    assert second_close['commission']==1.33333333
    assert second_close['pnl']==pytest.approx(-2.5)
    assert second_close['pnlcomm']==-3.83333333
    assert all(math.isfinite(row['commission']) for row in rows)


def native_fixture(*,fee=1.,closed=True):
    from types import SimpleNamespace as NS
    currency=Currency.from_str('USDT')
    event=NS(position_id='P',account_id='A',strategy_id='S',instrument_id='I',opening_order_id='OPEN',closing_order_id='CLOSE',ts_event=20,ts_closed=20,signed_qty=0. if closed else .25,avg_px_open=1000.,realized_pnl=Money(3.,currency),currency=currency)
    position=NS(id='P-archive' if closed else 'P',account_id='A',strategy_id='S',instrument_id='I',opening_order_id='OPEN',closing_order_id='CLOSE',ts_last=20,ts_closed=20,signed_qty=event.signed_qty,avg_px_open=1000.,realized_pnl=Money(3.,currency),is_closed=closed,settlement_currency=currency,commissions=lambda:[Money(fee,currency)],notional_value=lambda price:Money(250.,currency))
    cache=NS(position=lambda key:None if closed else position,position_snapshots=lambda *args:[position])
    return event,position,cache


@pytest.mark.parametrize('field,value',[('id','OTHER'),('account_id','FOREIGN'),('strategy_id','OTHER'),('instrument_id','OTHER'),('opening_order_id','OTHER'),('closing_order_id','OTHER'),('ts_last',21),('ts_closed',21),('signed_qty',.125),('avg_px_open',999.)])
def test_foreign_future_or_other_cycle_snapshot_cannot_supply_trade_fees(field,value):
    from tradelearn_runtime.position_facts import native_trade_economics
    event,position,cache=native_fixture();setattr(position,field,value)
    facts=native_trade_economics(cache,event,True)
    assert math.isnan(facts['commission']) and math.isnan(facts['pnl'])
    assert facts['pnlcomm']==3. and facts['value']==0.


@pytest.mark.parametrize('kind',['missing','mixed-currency','nonfinite','conflicting'])
def test_unqualified_native_commission_is_unknown_while_net_pnl_fact_remains(kind):
    from tradelearn_runtime.position_facts import native_trade_economics
    from copy import copy
    event,position,cache=native_fixture()
    if kind=='missing':position.commissions=lambda:[]
    elif kind=='mixed-currency':position.commissions=lambda:[Money(1,Currency.from_str('USD'))]
    elif kind=='nonfinite':
        from types import SimpleNamespace as NS
        position.commissions=lambda:[NS(currency=event.currency,as_double=lambda:float('nan'))]
    else:
        other=copy(position);other.commissions=lambda:[Money(2,event.currency)]
        cache.position_snapshots=lambda *args:[position,other]
    facts=native_trade_economics(cache,event,True)
    assert math.isnan(facts['commission']) and math.isnan(facts['pnl'])
    assert facts['pnlcomm']==3.


@pytest.mark.parametrize('fee',[0.,-1.])
def test_real_native_zero_and_rebate_commissions_are_preserved(fee):
    from tradelearn_runtime.position_facts import native_trade_economics
    event,position,cache=native_fixture(fee=fee)
    facts=native_trade_economics(cache,event,True)
    assert facts['commission']==fee and facts['pnl']==3.+fee and facts['pnlcomm']==3.


def test_newer_current_position_cannot_project_an_earlier_open_event():
    from tradelearn_runtime.position_facts import native_trade_economics
    event,position,cache=native_fixture(closed=False);position.ts_last=21
    facts=native_trade_economics(cache,event,False)
    assert math.isnan(facts['commission']) and math.isnan(facts['value'])


def projection_host():
    from types import SimpleNamespace as NS
    from nautilus_trader.model import OrderSide
    from tradelearn_runtime.execution import ExecutionHost
    event,position,cache=native_fixture()
    event.entry=OrderSide.BUY;event.event_id='POSITION-EVENT';event.ts_opened=10
    class Feed:
        def __len__(self):return 5
    seen=[]
    host=NS(_account_id='A',_callback_strategy_id='S',_instrument_ids={'X':'I'},_position_events={},_position_trade_views={},_next_position_trade_ref=1,_callback_cache=cache,_symbol=lambda key:'X',frontend_broker=NS(_orders={}),frontend=NS(feeds={'X':Feed()},step_broker=lambda *a,**kw:seen.append((a,kw))))
    return host,event,seen,ExecutionHost._project_position


def test_position_event_exact_replay_does_not_emit_a_second_closed_trade():
    host,event,seen,project=projection_host()
    project(host,event,True);project(host,event,True)
    assert len(seen)==1


@pytest.mark.parametrize('field,value',[('signed_qty',.125),('ts_event',21),('realized_pnl',None)])
def test_conflicting_position_event_identity_is_rejected_before_new_projection(field,value):
    host,event,seen,project=projection_host()
    project(host,event,True);setattr(event,field,value)
    with pytest.raises(ValueError,match='conflicting native position event'):
        project(host,event,True)
    assert len(seen)==1


@pytest.mark.parametrize('field',['account_id','strategy_id','instrument_id'])
def test_foreign_position_event_is_ignored_without_cache_or_frontend_access(field):
    host,event,seen,project=projection_host();setattr(event,field,'FOREIGN')
    host._callback_cache.position=lambda key:pytest.fail('foreign event reached cache lookup')
    project(host,event,True)
    assert not seen and not host._position_events


def test_conflicting_native_pnl_currency_is_unknown_before_cache_lookup():
    from tradelearn_runtime.position_facts import native_trade_economics
    event,position,cache=native_fixture();event.currency=Currency.from_str('USD')
    cache.position=lambda key:pytest.fail('currency conflict reached cache')
    facts=native_trade_economics(cache,event,True)
    assert math.isnan(facts['pnlcomm']) and math.isnan(facts['pnl'])


def test_clone_position_without_durable_opening_route_stays_unknown():
    host,event,seen,project=projection_host()
    host._instrument_ids['Y']='I'
    with pytest.raises(ValueError,match='admitted opening feed route'):
        project(host,event,True)
    assert seen==[]


def test_position_cannot_borrow_another_instruments_opening_route():
    from types import SimpleNamespace as NS
    host,event,seen,project=projection_host()
    host._instrument_ids['Y']='OTHER'
    host.frontend_broker._orders['OPEN']=NS(data=NS(_name='same-public',_symbol='Y'))
    with pytest.raises(ValueError,match='opening order scope'):
        project(host,event,True)
    assert seen==[]
