"""Typed existing BT broker kwargs reach fixed native fields and durable recovery."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn import engine as bt
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import ExecutionHost,_native_route_proof
from tests.test_live_order_mapping import native_host


def submit_fields(kind,fields,host=None):
    host=host or native_host();sent=[];requests=[]
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def submit(self,request,**controls):
            requests.append(request);sent.append(host._make_order(request,controls)[0])
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:None)
    kwargs=dict(price=1.2,exectype=kind,tif='gtc',client_oid='FIELD-001',**fields)
    if kind==bt.Order.StopLimit:kwargs['pricelimit']=1.3
    broker._submit(NS(),NS(_name='EURUSD'),bt.Order.Buy,.125,**kwargs)
    return host,requests[0],sent[0]


@pytest.mark.parametrize('kind,fields,attribute,expected',[
    (bt.Order.Limit,{'display_qty':.05},'display_qty','0.05'),
    (bt.Order.Stop,{'display_qty':.05},'display_qty','0.05'),
    (bt.Order.Stop,{'trigger_type':'LAST_PRICE'},'trigger_type','LAST_PRICE'),
    (bt.Order.StopLimit,{'trigger_type':'BID_ASK'},'trigger_type','BID_ASK'),
    (bt.Order.Limit,{'emulation_trigger':'BID_ASK'},'emulation_trigger','BID_ASK'),
    (bt.Order.Limit,{'exec_algorithm_id':'TWAP','exec_algorithm_params':{'horizon_secs':'60','interval_secs':'5'}},'exec_algorithm_id','TWAP'),
])
def test_typed_field_compiles_to_actual_native_factory_and_recovery(kind,fields,attribute,expected):
    host,request,native=submit_fields(kind,fields)
    assert str(getattr(native,attribute))==expected
    proof=_native_route_proof(native,{'position_id':None})
    ExecutionHost.validate_recovery(host,request,native,native_controls=proof)
    if 'exec_algorithm_params' in fields:
        assert native.exec_algorithm_params==fields['exec_algorithm_params']
        assert proof['exec_algorithm_params']==fields['exec_algorithm_params']


@pytest.mark.parametrize('fields,match',[
    ({'display_qty':0},'display'),({'display_qty':.126},'display'),
    ({'display_qty':True},'display'),({'display_qty':float('nan')},'display'),
    ({'trigger_type':'NOT_A_TRIGGER'},'trigger'),({'trigger_type':True},'trigger'),
    ({'emulation_trigger':'NOT_A_TRIGGER'},'emulation'),
    ({'exec_algorithm_id':True},'algorithm'),
    ({'exec_algorithm_params':{'interval_secs':'5'}},'algorithm'),
    ({'exec_algorithm_id':'TWAP','exec_algorithm_params':{'interval_secs':5}},'algorithm'),
])
def test_invalid_typed_control_is_rejected_before_submitted_or_submit(fields,match):
    host=native_host()
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def submit(self,*a,**k):pytest.fail('invalid native controls reached submit')
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:pytest.fail('invalid native controls published'))
    with pytest.raises((ValueError,TypeError),match=match):
        broker._submit(NS(),NS(_name='EURUSD'),bt.Order.Buy,.125,price=1.2,exectype=bt.Order.Stop,**fields)
    assert not broker.get_orders_history()


def test_display_precision_is_checked_without_rounding_or_quantity_change():
    from nautilus_trader.model import Quantity
    host=native_host();host.cache.instrument=lambda iid:NS(make_qty=lambda v:Quantity.from_str(f'{float(v):.3f}'),make_price=lambda v:None)
    with pytest.raises(ValueError,match='display.*precision'):
        submit_fields(bt.Order.Stop,{'display_qty':.0005},host)


@pytest.mark.parametrize('field,bad',[
    ('display_qty','0.060'),('trigger_type','BID_ASK'),
    ('emulation_trigger','LAST_PRICE'),('exec_algorithm_id','OTHER'),
    ('exec_algorithm_params',{'horizon_secs':'60','interval_secs':'9'}),
])
def test_native_recovery_rejects_each_changed_admission_field(field,bad):
    fields=dict(display_qty=.05,trigger_type='LAST_PRICE',emulation_trigger='BID_ASK',exec_algorithm_id='TWAP',exec_algorithm_params={'horizon_secs':'60','interval_secs':'5'})
    host,request,native=submit_fields(bt.Order.StopLimit,fields)
    proof=_native_route_proof(native,{'position_id':None})
    assert field in proof
    with pytest.raises(ValueError,match='recovery'):
        ExecutionHost.validate_recovery(host,request,native,native_controls={**proof,field:bad})


def test_native_control_proof_roundtrips_in_existing_intent_journal(tmp_path):
    from tradelearn_runtime.intents import IntentJournal
    host,request,native=submit_fields(bt.Order.Limit,dict(display_qty=.05,emulation_trigger='BID_ASK',exec_algorithm_id='TWAP',exec_algorithm_params={'horizon_secs':'60','interval_secs':'5'}))
    proof=_native_route_proof(native,{'position_id':None})
    journal=IntentJournal(tmp_path/'intents.sqlite',identity='FIELDS-TEST')
    try:
        journal.prepare(request.client_oid,{'native_controls':proof})
        loaded=journal.unfinished()[request.client_oid]['native_controls']
        assert loaded==proof
        ExecutionHost.validate_recovery(host,request,native,native_controls=loaded)
    finally:journal.close()


def actual_native_control_replay(case,*,restart=False,recovery_view=None):
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model import Bar,BarType,Money,Currency,OmsType,AccountType,StrategyId,QuoteTick,ExecAlgorithmId
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from nautilus_trader.trading import ExecutionAlgorithm,ExecutionAlgorithmConfig
    from tradelearn_runtime.execution import create_execution_host
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(1,Currency.from_str('USDT'))))
    native.add_instrument(instrument);cache=native.cache;notices=[];algorithm_calls=[];algorithm=None
    quotes=case.startswith('emulated')
    bars=BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL')
    if quotes:
        native.add_data([QuoteTick(instrument.id,instrument.make_price(bid),instrument.make_price(ask),instrument.make_qty(10),instrument.make_qty(10),(i+1)*60_000_000_000,(i+1)*60_000_000_000) for i,(bid,ask) in enumerate([(1005,1006),(1006,1007),(1007,1008),(1008,1009)] if case=='emulated-pending' else [(1005,1006),(999,1000),(1000,1001),(1001,1002)])])
    else:
        native.add_data([Bar(bars,*[instrument.make_price(p)]*4,instrument.make_qty(10),(i+1)*60_000_000_000,(i+1)*60_000_000_000) for i,p in enumerate([1000,1000,1010,1020,1030])])
    if case.startswith('algorithm'):
        class RouteOnce(ExecutionAlgorithm):
            def on_order(self,order):
                algorithm_calls.append(dict(order.exec_algorithm_params))
                if case.startswith('algorithm-spawn'):
                    for qty in (.05,.075):
                        child=(self.spawn_limit(order,instrument.make_qty(qty),instrument.make_price(900))
                               if 'pending' in case or 'cancel' in case else self.spawn_market(order,instrument.make_qty(qty)))
                        self.submit_order(child)
                else:
                    self.submit_order(order)
            def on_order_canceled(self,event):
                if case!='algorithm-spawn-cancel-children':return
                # Native plugin owns its child cancellation policy. TL does
                # not issue a second cancellation loop or forge terminal ACKs.
                for child in cache.orders_for_exec_spawn(event.client_order_id):
                    if child.client_order_id!=event.client_order_id and not child.is_closed:
                        self.cancel_order(child)
        algorithm=RouteOnce(ExecutionAlgorithmConfig(exec_algorithm_id=ExecAlgorithmId('ROUTE-ONCE')))
        native.add_exec_algorithm(algorithm)
    class Business(bt.Strategy):
        def next(self):
            if len(self)==2 and case.startswith('algorithm-spawn-cancel'):self.cancel(self.order)
            if len(self)!=1:return
            controls=dict(exectype=bt.Order.Limit,price=1000.)
            if case=='display':controls['display_qty']=.05
            elif case=='trigger':controls.update(exectype=bt.Order.Stop,price=1010.,trigger_type='LAST_PRICE')
            elif case.startswith('emulated'):controls['emulation_trigger']='BID_ASK'
            else:controls.update(exec_algorithm_id='ROUTE-ONCE',exec_algorithm_params={'marker':'actual-native-algorithm'})
            self.order=self.buy(size=.125,**controls)
        def notify_order(self,o):notices.append((o.getstatusname(),o.executed.size,o.executed.comm))
    host=create_execution_host(StrategyConfig(strategy_id=StrategyId('FIELDS-001'),oms_type=OmsType.NETTING),business_cls=Business,feeds={symbol:'tick' if quotes else 'bar'},instruments={symbol:instrument.id},subscriptions=[] if quotes else [bars],account_id='BINANCE-001',offline_test=True)
    admitted=[];admitted_requests=[]
    from types import MethodType
    original_make=host._make_order
    def capture(self,*args,**kwargs):
        made=original_make(*args,**kwargs)
        admitted.append(_native_route_proof(made[0],{}));admitted_requests.append(args[0])
        return made
    host._make_order=MethodType(capture,host)
    try:
        native.add_strategy(host);native.run()
        rows=[]
        for order in native.cache.orders():
            rows.append(dict(status=order.status.name,qty=str(order.quantity),filled=str(order.filled_qty),events=[type(e).__name__ for e in order.events()],controls=_native_route_proof(order,{'position_id':None})))
        if case in ('emulated','emulated-pending') or case.startswith('algorithm-spawn'):
            from tradelearn.core.broker_contracts import OrderRequest
            order=next(o for o in native.cache.orders() if o.exec_spawn_id is None or o.exec_spawn_id==o.client_order_id)
            request=admitted_requests[-1]
            proof=admitted[-1]
            if case=='emulated':
                with pytest.raises(ValueError,match='durable admission controls'):
                    host.validate_recovery(request,order)
                with pytest.raises(ValueError,match='emulation_trigger'):
                    host.validate_recovery(request,order,native_controls={k:v for k,v in proof.items() if k!='emulation_trigger'})
            host.validate_recovery(request,recovery_view(order) if recovery_view else order,native_controls=proof)
            broker=NautilusBroker(host);broker.bind(NS(_pending_size={}))
            broker.bind_runtime(NS(step_broker=lambda *a,**kw:None))
            projected=broker.restore(request,host.frontend.feeds[symbol],order)
            rows[0]['restored_status']=projected.getstatusname()
            rows[0]['restored_native_status']=projected.info['native_status']
            rows[0]['restored_execution']=(projected.executed.size,projected.executed.comm)
            rows[0]['restored_pending']=broker.owner._pending_size[projected.data]
            if restart:
                rows[0]['pre_restart_notices']=list(notices)
                import asyncio,tempfile
                from dataclasses import asdict
                from pathlib import Path
                from tradelearn_runtime.intents import IntentJournal
                with tempfile.TemporaryDirectory() as directory:
                    journal=IntentJournal(Path(directory)/'restart.sqlite',identity='ALG-RESTART')
                    journal.prepare(request.client_oid,dict(request=asdict(request),native_controls=proof))
                    host.intent_journal=journal;host._recovering=journal.unfinished()
                    host._stopping=False;scheduled=[]
                    host._schedule=scheduled.append
                    host.on_strategy_start()
                    async def finish():
                        for task in scheduled:await task
                    asyncio.run(finish())
                    rows[0]['startup_status']=host.frontend_broker._orders[request.client_oid].getstatusname()
                    rows[0]['startup_unfinished']=list(journal.unfinished())
                    host.on_strategy_stop();journal.close();host.intent_journal=None
        return rows,notices,algorithm_calls
    finally:native.dispose();host=None;algorithm=None;native=None


@pytest.mark.parametrize('case',['display','trigger','emulated','algorithm'])
def test_native_control_is_consumed_by_actual_cache_and_execution_callbacks(case):
    rows,notices,algorithms=actual_native_control_replay(case)
    assert len(rows)==1 and rows[0]['status']=='FILLED'
    assert rows[0]['qty']==rows[0]['filled']=='0.125'
    assert notices[-1]==('Completed',.125,1.)
    if case=='display':assert rows[0]['controls']['display_qty']=='0.050'
    if case=='trigger':assert rows[0]['controls']['trigger_type']=='LAST_PRICE'
    if case=='emulated':assert 'OrderEmulated' in rows[0]['events'] and 'OrderReleased' in rows[0]['events']
    if case=='algorithm':assert algorithms==[{'marker':'actual-native-algorithm'}]


def test_actual_emulated_inflight_order_recovers_as_pending_not_a_venue_ack():
    rows,notices,algorithms=actual_native_control_replay('emulated-pending')
    assert rows[0]['status']=='EMULATED'
    assert rows[0]['restored_status']=='Submitted'
    assert rows[0]['restored_native_status']=='EMULATED'
    assert all(status!='Completed' for status,_,_ in notices)


def test_actual_algorithm_spawned_fills_project_once_to_admitted_bt_order():
    rows,notices,algorithms=actual_native_control_replay('algorithm-spawn')
    assert [row['status'] for row in rows].count('FILLED')==2
    assert ('Partial',.05,1.) in notices
    assert notices[-1]==('Completed',.125,2.)
    assert sum(status=='Completed' for status,_,_ in notices)==1


def test_actual_algorithm_group_recovers_from_native_reduced_primary_and_children():
    rows,notices,algorithms=actual_native_control_replay('algorithm-spawn')
    assert rows[0]['restored_status']=='Completed'
    assert rows[0]['restored_execution']==(.125,2.)
    assert rows[0]['restored_pending']==0


def test_actual_algorithm_pending_children_recover_and_keep_parent_outstanding():
    rows,notices,algorithms=actual_native_control_replay('algorithm-spawn-pending')
    assert rows[0]['restored_status']=='Accepted'
    assert rows[0]['restored_pending']==.125


def test_actual_algorithm_default_cancel_does_not_hide_still_live_native_children():
    rows,notices,algorithms=actual_native_control_replay('algorithm-spawn-cancel')
    assert [row['status'] for row in rows].count('CANCELED')==1
    assert [row['status'] for row in rows].count('ACCEPTED')==2
    assert notices[-1][0]=='Accepted'
    assert rows[0]['restored_status']=='Accepted'
    assert rows[0]['restored_pending']==.125

@pytest.fixture(autouse=True)
def native_callback_errors_are_not_test_success(capfd):
    yield
    import re
    out,err=capfd.readouterr()
    assert not re.search(r'Python on_\w+ failed:|Exception ignored in:|unraisable',out+err)


def group_fixture():
    from decimal import Decimal as D
    from tradelearn_runtime.algorithm_facts import spawn_projection
    def member(cid,qty,filled,status,account='ACCOUNT-1',fee=1,currency='USD'):
        money=NS(currency=currency,as_double=lambda:fee)
        return NS(client_order_id=cid,exec_spawn_id='ROOT',exec_algorithm_id='ALGO',
                  strategy_id='STRATEGY-1',instrument_id='SYMBOL',account_id=account,
                  side='BUY',status=NS(name=status),quantity=D(qty),filled_qty=D(filled),
                  avg_px=1000.,is_closed=status in ('FILLED','CANCELED'),
                  init_event=NS(quantity=D('.125')),last_event=NS(event_id='EVENT-'+cid),commissions=lambda:{currency:money})
    root=member('ROOT','0','0','INITIALIZED',None)
    child=member('CHILD','.125','.125','FILLED')
    members=[root,child]
    cache=NS(orders_for_exec_spawn=lambda _:members,order=lambda _:root,
             exec_spawn_total_quantity=lambda *a:sum(o.quantity for o in members),
             exec_spawn_total_filled_qty=lambda *a:sum(o.filled_qty for o in members))
    def project():return spawn_projection(cache,child,strategy_id='STRATEGY-1',account_id='ACCOUNT-1',instruments=['SYMBOL'])
    return root,child,members,cache,project


@pytest.mark.parametrize('field,value',[
    ('strategy_id','FOREIGN'),('instrument_id','FOREIGN'),('account_id','FOREIGN'),
    ('exec_algorithm_id','OTHER'),('exec_spawn_id','OTHER'),('side','SELL'),
])
def test_spawned_group_rejects_foreign_or_changed_native_scope(field,value):
    root,child,members,cache,project=group_fixture()
    # A root foreign account is independently rejected, too.
    setattr(child,field,value)
    with pytest.raises(ValueError,match='spawn'):project()


def test_spawned_group_rejects_missing_primary_duplicate_and_quantity_conflict():
    root,child,members,cache,project=group_fixture()
    members.append(child)
    with pytest.raises(ValueError,match='scope'):project()
    members.pop();cache.exec_spawn_total_filled_qty=lambda *a:0
    with pytest.raises(ValueError,match='quantities'):project()
    cache.order=lambda _:None
    with pytest.raises(ValueError,match='primary'):project()


def test_spawned_group_preserves_unknown_and_mixed_currency_fees():
    from decimal import Decimal as D
    root,child,members,cache,project=group_fixture()
    assert project()['commission']==1
    root.quantity=D('.05');root.filled_qty=D('.05');root.account_id='ACCOUNT-1'
    root.avg_px=900
    root.init_event.quantity=D('.175')
    root.commissions=lambda:{'EUR':NS(currency='EUR',as_double=lambda:2)}
    assert project()['commission'] is None
    root.commissions=lambda:{}
    assert project()['commission'] is None
    assert project()['filled_qty']==.175


def test_local_recovery_requires_durable_proof_and_verified_account_even_without_ack():
    host,request,native=submit_fields(bt.Order.Limit,{'emulation_trigger':'BID_ASK'})
    host._account_id='ACCOUNT-1'
    native_account=NS()
    # A real initialized factory object has no native account assignment.
    assert native.account_id is None
    with pytest.raises(ValueError,match='account'):
        ExecutionHost.validate_recovery(host,request,native)
    host._assert_account_binding=lambda:(_ for _ in ()).throw(ValueError('verified account absent'))
    with pytest.raises(ValueError,match='verified account absent'):
        ExecutionHost.validate_recovery(host,request,native,native_controls=_native_route_proof(native,{}))


@pytest.mark.parametrize('case,status,unfinished',[
    ('algorithm-spawn','Completed',False),
    ('algorithm-spawn-pending','Accepted',True),
    ('algorithm-spawn-cancel','Accepted',True),
])
def test_actual_native_cache_startup_reconciliation_keeps_only_true_unfinished_group(case,status,unfinished):
    rows,_,_=actual_native_control_replay(case,restart=True)
    assert rows[0]['startup_status']==status
    assert bool(rows[0]['startup_unfinished'])==unfinished


def test_actual_native_algorithm_own_child_cancel_policy_finishes_parent_only_after_all_ack():
    rows,notices,_=actual_native_control_replay('algorithm-spawn-cancel-children',restart=True)
    assert [row['status'] for row in rows].count('CANCELED')==3
    assert notices[-1][0]=='Canceled'
    assert sum(status=='Canceled' for status,_,_ in rows[0]['pre_restart_notices'])==1
    assert rows[0]['restored_status']==rows[0]['startup_status']=='Canceled'
    assert rows[0]['restored_pending']==0
    assert not rows[0]['startup_unfinished']


def test_native_algorithm_cannot_grow_admitted_quantity_by_duplicating_spawn_allocation():
    from decimal import Decimal as D
    root,child,members,cache,project=group_fixture()
    root.quantity=D('.125')  # Child did not reduce its primary allocation.
    with pytest.raises(ValueError,match='admitted primary quantity'):project()


def test_actual_emulator_release_preserves_original_admission_during_recovery_and_startup():
    rows,notices,_=actual_native_control_replay('emulated',restart=True)
    assert rows[0]['status']=='FILLED'
    assert rows[0]['restored_status']==rows[0]['startup_status']=='Completed'
    assert rows[0]['restored_execution']==(.125,1.)
    assert not rows[0]['startup_unfinished']


@pytest.mark.parametrize('mutation',['missing_release','missing_emulated','wrong_scope','wrong_quantity','wrong_type','released_control'])
def test_emulation_type_conversion_needs_full_original_native_release_evidence(mutation):
    from nautilus_trader.model import OrderInitialized,Quantity,ExecAlgorithmId,OrderType
    class View:
        def __init__(self,native):
            self.native=native;self.history=list(native.events())
            if mutation=='missing_release':self.history=[e for e in self.history if type(e).__name__!='OrderReleased']
            if mutation=='missing_emulated':self.history=[e for e in self.history if type(e).__name__!='OrderEmulated']
            if mutation in ('wrong_scope','wrong_quantity'):
                index=[i for i,e in enumerate(self.history) if isinstance(e,OrderInitialized)][1]
                values=self.history[index].to_dict()
                if mutation=='wrong_scope':values['client_order_id']='FOREIGN-CID'
                else:values['quantity']='0.126'
                self.history[index]=OrderInitialized.from_dict(values)
            if mutation=='wrong_type':self.order_type=OrderType.LIMIT
            if mutation=='released_control':self.exec_algorithm_id=ExecAlgorithmId('FOREIGN')
        def events(self):return self.history
        def __getattr__(self,name):return getattr(self.native,name)
    with pytest.raises(ValueError,match='recovery|emulation'):
        actual_native_control_replay('emulated',recovery_view=View)
