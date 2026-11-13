"""Real fixed NA maker modification rejection, derived from IA-C4 frozen probe."""
import pytest
import json
from pathlib import Path
import sys
from tradelearn import engine as bt
from tradelearn_runtime.execution import create_execution_host
from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar,BarType,Money,Currency,OmsType,AccountType,StrategyId
from nautilus_trader.testkit.providers import TestInstrumentProvider
from nautilus_trader.execution import FixedFeeModel
def probe(read_position=False):
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    bars=BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    native.add_data([Bar(bars,*[instrument.make_price(1000)]*4,instrument.make_qty(10),
        (i+1)*60_000_000_000,(i+1)*60_000_000_000) for i in range(4)])
    notices=[];native_events=[];position_reads=[]
    class Business(bt.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=.125,price=900,exectype=bt.Order.Limit,post_only=True)
            if len(self)==2:
                [order]=host.cache.orders();host.modify_order(order,price=instrument.make_price(1000))
            if len(self)==3:
                [order]=host.cache.orders();host.cancel_order(order)
        def notify_order(self,o):
            if read_position:position_reads.append(self.position.size)
            notices.append({'status':o.getstatusname(),'native':o.info.get('native_status'),
                            'reason':o.info.get('reason'),'price':o.price})
    host=create_execution_host(StrategyConfig(strategy_id=StrategyId('IAM-001'),oms_type=OmsType.NETTING),
        business_cls=Business,feeds={symbol:'bar'},instruments={symbol:instrument.id},
        subscriptions=[bars],account_id='BINANCE-001',offline_test=True)
    for name in ['on_order_pending_update','on_order_pending_cancel','on_order_modify_rejected','on_order_cancel_rejected']:
        original=getattr(host,name)
        def capture(event,callback=original,label=name):
            native_events.append({'callback':label,'type':type(event).__name__,'reason':getattr(event,'reason',None),
                                  'TL_notice_count_before':len(notices)})
            result=callback(event)
            native_events[-1]['TL_notice_count_after']=len(notices)
            return result
        setattr(host,name,capture)
    try:
        native.add_strategy(host);native.run()
        [order]=native.cache.orders()
        return {'native_events':native_events,'TL_notices':notices,'position_reads':position_reads,'cache_status':order.status.name,
                'cache_events':[type(e).__name__ for e in (order.events() if callable(order.events) else order.events)]}
    finally:native.dispose()


def test_native_control_events_reach_bt_notices_with_reason_without_rejecting_order():
    result=probe()
    assert result['cache_status']=='CANCELED'
    assert [v['type'] for v in result['native_events']]==['OrderPendingUpdate','OrderModifyRejected','OrderPendingCancel']
    for event in result['native_events']:
        assert event['TL_notice_count_after']==event['TL_notice_count_before']+1
        notice=result['TL_notices'][event['TL_notice_count_before']]
        assert notice['status']=='Accepted'
        if event['type']=='OrderModifyRejected':
            assert notice['reason']==event['reason'] and notice['reason']
    assert all(v['status']!='Rejected' for v in result['TL_notices'])


def pending_fixture():
    from types import SimpleNamespace as NS
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn_runtime.execution import ExecutionHost
    class GuardedHost:
        @property
        def cache(self):raise AssertionError('pending callback must not borrow native cache')
        @property
        def strategy_id(self):raise AssertionError('pending callback must not borrow native Strategy')
    events=[]
    gateway=NS(lookup_order=lambda _:None,submit=lambda *a,**kw:None)
    broker=NautilusBroker(gateway);broker.bind_runtime(NS(step_broker=lambda o,**kw:events.append(o.clone())))
    data=type('Data',(),{'_name':'A'})();owner=NS(_pending_size={})
    order=broker._submit(owner,data,bt.Order.Buy,1,price=1.2,exectype=bt.Order.Limit,client_oid='CONTROL-1')
    broker.apply('CONTROL-1',status='PARTIALLY_FILLED',filled_qty=.25,avg_price=1.2,commission=.01,event_id='FILL-1')
    host=GuardedHost();host._account_id='SIM-001';host._callback_strategy_id='STRAT-001';host._instrument_ids={'A':'A.SIM'}
    host.frontend_broker=broker;host.frontend=NS();host._symbol=lambda _: 'A'
    event=NS(account_id='SIM-001',strategy_id='STRAT-001',instrument_id='A.SIM',client_order_id='CONTROL-1',event_id='PENDING-1',ts_event=123)
    return ExecutionHost,host,event,order,events


def test_pending_control_is_owned_duplicate_safe_and_preserves_last_actual_fill_without_cache_reentry():
    Host,host,event,order,events=pending_fixture();before=len(events)
    Host._project_pending_order(host,event,'PENDING_UPDATE')
    assert len(events)==before+1
    assert order.status==bt.Order.Partial
    assert order.executed.size==.25 and order.executed.comm==.01
    assert order.info['native_status']=='PENDING_UPDATE'
    assert order.info['native_event']['event_id']=='PENDING-1'
    Host._project_pending_order(host,event,'PENDING_UPDATE')
    assert len(events)==before+1
    event.ts_event=124
    with pytest.raises(ValueError,match='conflicting native event replay'):
        Host._project_pending_order(host,event,'PENDING_UPDATE')


@pytest.mark.parametrize('field,value',[('account_id','OTHER'),('strategy_id','OTHER'),('instrument_id','OTHER')])
def test_pending_controls_cannot_project_foreign_account_strategy_or_instrument(field,value):
    Host,host,event,order,events=pending_fixture();before=len(events)
    setattr(event,field,value)
    Host._project_pending_order(host,event,'PENDING_CANCEL')
    assert len(events)==before and order.info['native_status']=='PARTIALLY_FILLED'


def test_standard_position_query_is_transparent_during_pending_control_notifications():
    result=probe(read_position=True)
    assert len(result['TL_notices'])==6
    assert result['position_reads']==[0.]*6
    assert [e['TL_notice_count_after']-e['TL_notice_count_before'] for e in result['native_events']]==[1,1,1]
