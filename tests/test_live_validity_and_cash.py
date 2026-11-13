from tradelearn.backtest.event_runtime import StrategyRuntime
"""Actual pinned NA order deadlines and authoritative account notifications."""
from datetime import datetime,date,timedelta,timezone
from types import SimpleNamespace as NS,MethodType
from dataclasses import replace
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.common import Clock,OrderFactory
from nautilus_trader.model import InstrumentId,TraderId,StrategyId,Price,Quantity
from tradelearn import engine as bt
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import ExecutionHost


def host():
    instrument=NS(make_qty=lambda value:Quantity.from_str(str(value)),
                  make_price=lambda value:Price.from_str(str(value)))
    value=NS(_assert_account_binding=lambda:None,_account_id=None,strategy_id=StrategyId('TL-001'),
        _instrument_ids={'EURUSD':InstrumentId.from_str('EURUSD.SIM')},
        cache=NS(instrument=lambda iid:instrument,order=lambda oid:None),
        order_factory=OrderFactory(TraderId('TEST-001'),StrategyId('TL-001'),Clock.new_test()))
    value._make_order=MethodType(ExecutionHost._make_order,value)
    return value


@pytest.mark.parametrize('valid,deadline',[
    (datetime(2026,1,1,0,0,10,tzinfo=timezone.utc),1767225610),
    (datetime(2026,1,1,0,0,10),1767225610),(date(2026,1,2),1767312000),
    (timedelta(seconds=10.9),1767225611),(10.9,1767225611),
])
def test_valid_keeps_tl_inclusive_utc_seconds_and_relative_primary_clock(valid,deadline):
    from tradelearn_runtime.broker import _valid_expiry_ns
    from tradelearn.core.broker_contracts import OrderRequest
    primary=NS(datetime=[1767225600.25])
    expiry=_valid_expiry_ns(valid,NS(data=primary))
    assert expiry==(deadline+1)*1_000_000_000
    request=OrderRequest('EURUSD','buy',1,'limit',limit_price=10,tif='gtd',
        client_oid='TEST-VALID',expire_time_ns=expiry)
    native=host();order=native._make_order(request,{})[0]
    assert order.expire_time_ns==expiry and order.time_in_force.name=='GTD'
    ExecutionHost.validate_recovery(native,request,order)
    with pytest.raises(ValueError,match='expir'):
        ExecutionHost.validate_recovery(native,replace(request,expire_time_ns=expiry+1_000_000_000),order)


@pytest.mark.parametrize('valid',[float('nan'),float('inf'),True,object()])
def test_invalid_deadlines_rejected_before_submission(valid):
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,submit=lambda *a,**k:pytest.fail('invalid deadline sent')))
    broker.runtime=NS(step_broker=lambda *a,**k:pytest.fail('invalid deadline published'))
    with pytest.raises((TypeError,ValueError)):
        broker._submit(NS(data=NS(datetime=[1767225600.])),NS(_name='EURUSD'),bt.Order.Buy,
            1,price=10,exectype=bt.Order.Limit,valid=valid)


def test_market_gtd_is_not_silently_reinterpreted():
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,submit=lambda *a,**k:pytest.fail('unsupported order sent')))
    broker.runtime=NS(step_broker=lambda *a,**k:None)
    owner=NS(data=NS(datetime=[1767225600.]))
    with pytest.raises(ValueError,match='Market.*GTD|GTD.*Market'):
        broker._submit(owner,NS(_name='EURUSD'),bt.Order.Buy,1,valid=timedelta(seconds=10))


def test_cash_notification_reads_one_authoritative_snapshot_through_existing_runtime():
    calls=[]
    class Business(bt.Strategy):
        def notify_cashvalue(self,cash,value):calls.append(('cash',cash,value))
    rt=StrategyRuntime(Business,feeds={'EURUSD':'bar'});rt.start()
    def account():calls.append('snapshot');return {'cash':10,'equity':12}
    native=NS(frontend=rt,account=account)
    ExecutionHost._notify_cashvalue(native)
    assert calls==['snapshot',('cash',10.,12.)]
    rt.stop()


def test_cash_notification_does_not_read_provider_for_an_unoverridden_callback():
    rt=StrategyRuntime(bt.Strategy,feeds={'EURUSD':'bar'});rt.start()
    native=NS(frontend=rt,account=lambda:pytest.fail('unrequested provider access'))
    ExecutionHost._notify_cashvalue(native)
    rt.stop()


@pytest.mark.parametrize('values',[{'cash':1},{'cash':float('nan'),'equity':2},{'cash':True,'equity':2}])
def test_cash_notification_never_fills_missing_or_invalid_money(values):
    seen=[]
    class Business(bt.Strategy):
        def notify_cashvalue(self,*args):seen.append(args)
    rt=StrategyRuntime(Business,feeds={'EURUSD':'bar'});rt.start()
    with pytest.raises(ValueError):ExecutionHost._notify_cashvalue(NS(frontend=rt,account=lambda:values))
    assert seen==[]
    rt.stop()


def test_unspecified_validity_keeps_tl_unbounded_intent():
    seen=[]
    gateway=NS(lookup_order=lambda cid:None,submit=lambda request,**controls:seen.append(request))
    broker=NautilusBroker(gateway);broker.runtime=NS(step_broker=lambda *a,**k:None)
    broker._submit(NS(),NS(_name='EURUSD'),bt.Order.Buy,1,price=10,exectype=bt.Order.Limit)
    assert seen[0].tif=='gtc' and seen[0].expire_time_ns is None


@pytest.mark.parametrize('fill_ns,completed',[
    pytest.param(None,False,marks=pytest.mark.xfail(strict=True,raises=AssertionError,
        reason='Pinned NA2rc7 simulator did not expire at the required boundary without changed market prices; not qualified')),
    (2_900_000_000,True),
    pytest.param(3_000_000_000,False,marks=pytest.mark.xfail(strict=True,raises=AssertionError,
        reason='Pinned NA2rc7 simulator matched at the expiry boundary; not equivalent to TL pre-match expiry')),
])
def test_native_gtd_expiry_owns_boundary_without_a_tl_timer(fill_ns,completed):
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model import Bar,BarType,AccountType,OmsType,Money,Currency
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from nautilus_trader.trading import Strategy as NativeStrategy
    from nautilus_trader.model import OrderSide,TimeInForce
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    bar_type=BarType.from_str(symbol+'-1-SECOND-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    start=1767225600*1_000_000_000
    rows=[(0,1000),(1_000_000_000,1000),(2_000_000_000,1000),(4_000_000_000,1000)]
    if fill_ns is not None:rows.append((fill_ns,900))
    native.add_data([Bar(bar_type,*[instrument.make_price(price)]*4,instrument.make_qty(10),
        start+stamp,start+stamp) for stamp,price in sorted(rows)])
    notices=[]
    class Business(NativeStrategy):
        def on_start(self):self.subscribe_bars(bar_type)
        def on_bar(self,bar):
            if hasattr(self,'order'):return
            self.order=self.order_factory.limit(instrument.id,OrderSide.BUY,
                instrument.make_qty(1),instrument.make_price(900),
                time_in_force=TimeInForce.GTD,expire_time=start+3_000_000_000)
            self.submit_order(self.order)
        def on_order_expired(self,event):notices.append(int(event.ts_event))
    bridge=Business(StrategyConfig(strategy_id=StrategyId('TL-001')))
    try:
        native.add_strategy(bridge);native.run()
        assert native.cache.order(bridge.order.client_order_id).status.name==('FILLED' if completed else 'EXPIRED')
        if not completed:assert start+3_000_000_000 in notices
    finally:
        native.dispose()
        # The strict xfail traceback must not retain an unsendable native actor
        # for a later IO worker's GC. Keep both original expiry assertions.
        bridge=None


def test_unqualified_gtd_is_rejected_before_any_native_intent_or_submission():
    gateway=NS(lookup_order=lambda oid:pytest.fail('unqualified GTD admission reached native'),
               submit=lambda *args,**kwargs:pytest.fail('unqualified GTD submitted'))
    broker=NautilusBroker(gateway)
    broker.runtime=NS(step_broker=lambda *args,**kwargs:pytest.fail('unqualified GTD published'))
    with pytest.raises(NotImplementedError,match='GTD.*boundary'):
        broker._submit(NS(data=NS(datetime=[1767225600.])),NS(_name='EURUSD'),
            bt.Order.Buy,1,price=10,exectype=bt.Order.Limit,valid=timedelta(seconds=2))
    assert broker._orders=={} and broker._unresolved==set()
