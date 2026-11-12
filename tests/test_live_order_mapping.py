"""Real pinned NA order objects; no Node, network, broker credentials or emulator."""
from types import SimpleNamespace as NS,MethodType
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.common import Clock,OrderFactory
from nautilus_trader.model import InstrumentId,TraderId,StrategyId,Price,Quantity
from tradelearn.backtest.models import Order
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import ExecutionHost


def native_host():
    instrument=NS(make_qty=lambda value:Quantity.from_str(str(value)),
                  make_price=lambda value:Price.from_str(str(value)))
    host=NS(_assert_account_binding=lambda:None,
        _account_id=None,strategy_id=StrategyId('TL-001'),
        _instrument_ids={'EURUSD':InstrumentId.from_str('EURUSD.SIM')},
        cache=NS(instrument=lambda iid:instrument,order=lambda oid:None),
        order_factory=OrderFactory(TraderId('TEST-001'),StrategyId('TL-001'),Clock.new_test()))
    host._make_order=MethodType(ExecutionHost._make_order,host)
    return host


@pytest.mark.parametrize('exectype,controls,kind',[
    (Order.Stop,{},'STOP_MARKET'),
    (Order.StopLimit,{'pricelimit':1.25},'STOP_LIMIT'),
])
def test_stop_request_native_mapping_and_projection(exectype,controls,kind):
    host=native_host();sent=[];requests=[]
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def submit(self,request,**controls):
            requests.append(request);sent.append(host._make_order(request,controls)[0])
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:None)
    order=broker._submit(NS(),NS(_name='EURUSD'),Order.Buy,1.0,price=1.2,
        exectype=exectype,client_oid='TEST-STOP-1',tif='gtc',**controls)
    assert len(sent)==1 and sent[0].order_type.name==kind
    assert str(sent[0].trigger_price)=='1.2'
    assert order.exectype==exectype and order.price==1.2
    ExecutionHost.validate_recovery(host,requests[0],sent[0])
    from dataclasses import replace
    with pytest.raises(ValueError,match='trigger price'):
        ExecutionHost.validate_recovery(host,replace(requests[0],stop_price=1.3),sent[0])
    if exectype==Order.StopLimit:
        assert str(sent[0].price)=='1.25' and order.pricelimit==1.25


@pytest.mark.parametrize('exectype,price,controls',[
    (Order.Stop,None,{}),(Order.Stop,0,{}),
    (Order.StopLimit,1.2,{}),(Order.StopLimit,1.2,{'pricelimit':float('nan')}),
    (Order.Market,None,{'pricelimit':1.2}),
])
def test_invalid_price_combination_never_reaches_submit(exectype,price,controls):
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,
        submit=lambda *a,**k:pytest.fail('invalid request reached submit')))
    broker.runtime=NS(step_broker=lambda *a,**k:pytest.fail('invalid request published'))
    with pytest.raises(ValueError):
        broker._submit(NS(),NS(_name='EURUSD'),Order.Buy,1,price=price,
                       exectype=exectype,**controls)


@pytest.mark.parametrize('side,trail,trigger,offset,offset_type',[
    (Order.Buy,{'trailamount':0.1},'1.3','0.1','PRICE'),
    (Order.Sell,{'trailamount':0.1},'1.1','0.1','PRICE'),
    (Order.Sell,{'trailpercent':0.1},'1.08','1000.0','BASIS_POINTS'),
])
def test_trailing_stop_market_compiles_once_to_native(side,trail,trigger,offset,offset_type):
    from decimal import Decimal
    host=native_host();sent=[];requests=[]
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def submit(self,request,**controls):
            requests.append(request);sent.append(host._make_order(request,controls)[0])
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:None)
    order=broker._submit(NS(),NS(_name='EURUSD'),side,1.0,price=1.2,
        exectype=Order.StopTrail,client_oid='TEST-TRAIL-1',tif='gtc',**trail)
    native=sent[0]
    assert native.order_type.name=='TRAILING_STOP_MARKET'
    assert Decimal(str(native.trigger_price))==Decimal(trigger)
    assert native.trailing_offset==Decimal(offset)
    assert native.trailing_offset_type.name==offset_type
    from nautilus_trader.model import TriggerType
    assert native.emulation_trigger in (None,TriggerType.NO_TRIGGER)
    assert order.price==1.2 and order.trailamount==trail.get('trailamount')
    assert order.trailpercent==trail.get('trailpercent')
    ExecutionHost.validate_recovery(host,requests[0],native)
    # A trailed current trigger is not the original immutable admission price.
    current={n:getattr(native,n) for n in ('strategy_id','client_order_id','instrument_id',
        'side','order_type','quantity','time_in_force','account_id','init_event',
        'trailing_offset','trailing_offset_type')}
    current['trigger_price']=Price.from_str('1.4')
    ExecutionHost.validate_recovery(host,requests[0],NS(**current))


@pytest.mark.parametrize('trail',[{}, {'trailamount':0}, {'trailpercent':float('nan')},
    {'trailamount':0.1,'trailpercent':float('nan')}])
def test_invalid_trailing_offsets_never_publish_or_submit(trail):
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,
        submit=lambda *a,**k:pytest.fail('invalid trailing request submitted')))
    broker.runtime=NS(step_broker=lambda *a,**k:pytest.fail('invalid trailing request published'))
    with pytest.raises(ValueError):
        broker._submit(NS(),NS(_name='EURUSD'),Order.Sell,1,price=1.2,
            exectype=Order.StopTrail,**trail)


def test_parent_transmit_builds_one_native_linked_group_without_single_submits():
    host=native_host();host._native_strategy_config=NS(manage_contingent_orders=False)
    calls=[]
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def validate_group(self,entries):return ExecutionHost._make_group(host,entries)
        def submit_group(self,entries):calls.append(ExecutionHost._make_group(host,entries))
        def submit(self,*a,**k):pytest.fail('bracket sent as unrelated single orders')
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:None)
    owner=NS();data=NS(_name='EURUSD')
    parent=broker._submit(owner,data,Order.Buy,1,price=1.2,exectype=Order.Limit,
        tif='gtc',client_oid='GROUP-ENTRY',transmit=False)
    stop=broker._submit(owner,data,Order.Sell,1,price=1.1,exectype=Order.Stop,
        tif='gtc',client_oid='GROUP-STOP',parent=parent,transmit=False)
    assert calls==[]
    take=broker._submit(owner,data,Order.Sell,1,price=1.3,exectype=Order.Limit,
        tif='gtc',client_oid='GROUP-TAKE',parent=parent,oco=stop,transmit=True)
    assert len(calls)==1
    native=calls[0];assert len(native)==3
    assert [str(o.client_order_id) for o in native]==['GROUP-ENTRY','GROUP-STOP','GROUP-TAKE']
    assert native[0].contingency_type.name=='OTO'
    assert str(native[1].parent_order_id)=='GROUP-ENTRY'
    assert native[1].contingency_type.name==native[2].contingency_type.name=='OUO'
    assert str(native[1].linked_order_ids[0])=='GROUP-TAKE'
    assert native[0].order_list_id==native[1].order_list_id==native[2].order_list_id
    assert take.parent is parent and take.oco is stop


def test_known_adapter_or_missing_na_group_policy_rejects_before_staging():
    host=native_host();host._native_strategy_config=NS(manage_contingent_orders=False)
    host._adapter_supports_groups=False
    with pytest.raises(ValueError,match='adapter'):
        ExecutionHost._make_group(host,[])
    host._adapter_supports_groups=None
    host._native_strategy_config=NS(manage_contingent_orders=True)
    with pytest.raises(ValueError,match='overlay'):
        ExecutionHost._make_group(host,[])


def test_conflicting_tif_valid_is_rejected_before_projection():
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,
        submit=lambda *a,**k:pytest.fail('conflicting request submitted')))
    with pytest.raises(ValueError,match='mutually exclusive'):
        broker._submit(NS(),NS(_name='EURUSD'),Order.Buy,1,tif='gtc',valid='day')


def test_callback_cancelling_untransmitted_group_cannot_resurrect_or_send_children():
    host=native_host();host._native_strategy_config=NS(manage_contingent_orders=False)
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def validate_group(self,entries):return ExecutionHost._make_group(host,entries)
        def submit_group(self,entries):pytest.fail('cancelled group sent')
        def submit(self,*a,**k):pytest.fail('single send')
    broker=NautilusBroker(Gateway())
    class Data: _name='EURUSD'
    data=Data();owner=NS(_pending_size={data:0})
    def notice(value,**kwargs):
        if value.status==Order.Submitted:broker.cancel(value)
    broker.runtime=NS(step_broker=notice)
    parent=broker._submit(owner,data,Order.Buy,1,exectype=Order.Market,
        tif='gtc',client_oid='CANCEL-ENTRY',transmit=False)
    child=broker._submit(owner,data,Order.Sell,1,price=1.1,exectype=Order.Stop,
        tif='gtc',client_oid='CANCEL-STOP',parent=parent,transmit=True)
    assert parent.status==child.status==Order.Canceled


def test_native_group_metadata_mismatch_blocks_recovery():
    host=native_host();host._native_strategy_config=NS(manage_contingent_orders=False)
    from tradelearn.core.broker_contracts import OrderRequest
    entries=[dict(request=OrderRequest('EURUSD','buy',1,client_oid='PARENT'),controls={},parent=None,oco=None),
        dict(request=OrderRequest('EURUSD','sell',1,'stop',stop_price=1.1,client_oid='CHILD'),
             controls={},parent='PARENT',oco=None)]
    orders=ExecutionHost._make_group(host,entries)
    native=orders[1];row=type(native).to_dict(native)
    fields=('order_list_id','parent_order_id','linked_order_ids','contingency_type')
    expected={name:row[name] for name in fields}
    ExecutionHost._validate_group_recovery(expected,native)
    expected['parent_order_id']='FOREIGN'
    with pytest.raises(ValueError,match='recovery differs'):
        ExecutionHost._validate_group_recovery(expected,native)


def test_native_quantity_update_revises_only_projection_and_duplicate_conflicts_fail():
    class Data:_name='EURUSD'
    data=Data();owner=NS(_pending_size={data:10.0})
    broker=NautilusBroker(NS(lookup_order=lambda oid:None,submit=lambda *a,**k:None))
    broker.runtime=NS(step_broker=lambda *a,**k:None)
    order=broker._submit(owner,data,Order.Buy,10,client_oid='UPDATE-1')
    broker.apply('UPDATE-1',status='ACCEPTED',filled_qty=0,event_id='ACCEPT-1')
    assert broker.apply('UPDATE-1',status='ACCEPTED',filled_qty=0,total_qty=5,event_id='UPDATE-EVENT')
    assert order.size==5 and owner._pending_size[data]==5
    assert not broker.apply('UPDATE-1',status='ACCEPTED',filled_qty=0,total_qty=5,event_id='UPDATE-EVENT')
    with pytest.raises(ValueError,match='conflicting native event'):
        broker.apply('UPDATE-1',status='ACCEPTED',filled_qty=0,total_qty=4,event_id='UPDATE-EVENT')
    broker.apply('UPDATE-1',status='PARTIALLY_FILLED',filled_qty=3,avg_price=1.2,event_id='FILL-1')
    with pytest.raises(ValueError):
        broker.apply('UPDATE-1',status='ACCEPTED',filled_qty=3,total_qty=2,avg_price=1.2,event_id='BAD-UPDATE')
    assert order.size==5 and owner._pending_size[data]==2


@pytest.mark.parametrize('fail_at',[None,0,1])
def test_group_durable_admission_prepares_every_member_before_single_native_command(tmp_path,fail_at):
    import asyncio
    from tradelearn.core.broker_contracts import OrderRequest
    from tradelearn_runtime.intents import IntentJournal
    entries=[dict(request=OrderRequest('EURUSD','buy',1,client_oid='DURABLE-PARENT'),
                  controls={},parent=None,oco=None),
             dict(request=OrderRequest('EURUSD','sell',1,'stop',stop_price=1.1,client_oid='DURABLE-CHILD'),
                  controls={},parent='DURABLE-PARENT',oco=None)]
    host=native_host();host._native_strategy_config=NS(manage_contingent_orders=False)
    orders=ExecutionHost._make_group(host,entries)
    journal=IntentJournal(tmp_path/'intents.sqlite',identity='synthetic-group')
    host.intent_journal=journal;host._stopping=False;host.frontend=NS(failed=False)
    host.frontend_broker=NS(_unresolved=set())
    original=journal.prepare;prepared=[];sent=[]
    def prepare(oid,payload):
        if len(prepared)==fail_at:raise OSError('synthetic durable failure')
        original(oid,payload);prepared.append(oid)
    journal.prepare=prepare
    def send(orders,**controls):
        assert set(journal.unfinished())=={'DURABLE-PARENT','DURABLE-CHILD'}
        sent.append(orders)
    host.submit_order_list=send
    async def run():
        host._io_lock=asyncio.Lock()
        await ExecutionHost._admit_group(host,entries,orders,{})
    try:
        if fail_at is None:
            asyncio.run(run());assert len(sent)==1
        else:
            with pytest.raises(OSError,match='durable failure'):asyncio.run(run())
            assert not sent and host.frontend_broker._unresolved=={'DURABLE-PARENT','DURABLE-CHILD'}
    finally:journal.close()
    reopened=IntentJournal(tmp_path/'intents.sqlite',identity='synthetic-group')
    try:
        pending=reopened.unfinished()
        assert set(pending)==set(prepared)
        for oid,payload in pending.items():
            native=next(order for order in orders if str(order.client_order_id)==oid)
            ExecutionHost._validate_group_recovery(payload['group'],native)
    finally:reopened.close()


@pytest.mark.parametrize('exectype,price,extra', [
    (Order.Limit, 1.2, {}),
    (Order.StopLimit, 1.2, {'pricelimit': 1.25}),
    (Order.StopTrailLimit, 1.2, {'pricelimit': 1.25, 'trailamount': 0.1}),
])
def test_post_only_existing_broker_kwarg_reaches_real_native_limit_factory(exectype, price, extra):
    host = native_host()
    sent = []
    requests = []
    class Gateway:
        def lookup_order(self, cid):return None
        def validate(self, request, **controls):host._make_order(request, dict(controls))
        def submit(self, request, **controls):
            requests.append(request)
            sent.append(host._make_order(request, dict(controls))[0])
    broker = NautilusBroker(Gateway())
    broker.runtime = NS(step_broker=lambda *a, **k:None)
    broker._submit(NS(), NS(_name='EURUSD'), Order.Buy, 1., price=price,
                   exectype=exectype, post_only=True, client_oid='POST-ONLY-1', **extra)
    assert len(sent) == 1 and sent[0].is_post_only is True
    proof = dict(reduce_only=False, position_id=None, post_only=True)
    ExecutionHost.validate_recovery(host, requests[0], sent[0], native_controls=proof)
    with pytest.raises(ValueError, match='post-only'):
        ExecutionHost.validate_recovery(host, requests[0], sent[0],
                                       native_controls={**proof, 'post_only':False})
    with pytest.raises(ValueError, match='post-only'):
        ExecutionHost.validate_recovery(host, requests[0], sent[0],
                                       native_controls=dict(reduce_only=False, position_id=None))


@pytest.mark.parametrize('exectype,post_only', [
    (Order.Limit, 'true'), (Order.Limit, 1), (Order.Market, True), (Order.Stop, True),
])
def test_post_only_wrong_type_or_order_kind_never_submits(exectype, post_only):
    host = native_host()
    class Gateway:
        def lookup_order(self, cid):return None
        def validate(self, request, **controls):host._make_order(request, dict(controls))
        def submit(self, *a, **k):pytest.fail('invalid post-only reached submit')
    broker = NautilusBroker(Gateway())
    broker.runtime = NS(step_broker=lambda *a, **k:pytest.fail('invalid request published'))
    with pytest.raises((TypeError, ValueError), match='post_only|post-only'):
        broker._submit(NS(), NS(_name='EURUSD'), Order.Buy, 1.,
                       price=None if exectype == Order.Market else 1.2,
                       exectype=exectype, post_only=post_only)


def test_admitted_post_only_flag_survives_actual_intent_journal_json(tmp_path):
    from tradelearn_runtime.execution import _native_route_proof
    from tradelearn_runtime.intents import IntentJournal
    from tradelearn.core.broker_contracts import OrderRequest
    host = native_host()
    request = OrderRequest('EURUSD', 'buy', 1., 'limit', limit_price=1.2, client_oid='POST-JOURNAL')
    native, route = host._make_order(request, {'post_only':True})
    assert route == {}
    proof = _native_route_proof(native, route)
    with_journal = IntentJournal(tmp_path / 'intent.sqlite', identity='offline-test')
    try:
        with_journal.prepare(request.client_oid, {'native_controls':proof})
        restored = with_journal.unfinished()[request.client_oid]['native_controls']
        assert restored['post_only'] is True
        ExecutionHost.validate_recovery(host, request, native, native_controls=restored)
    finally:
        with_journal.close()
