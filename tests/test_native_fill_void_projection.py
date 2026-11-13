"""Existing native corrections must project without replaying fills or account cash."""
from types import SimpleNamespace as N
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader import model as m
from nautilus_trader.core import UUID4
from nautilus_trader.testkit.providers import TestInstrumentProvider
from tradelearn_runtime.order_fees import cumulative_native_commission


def filled_order(kind='market'):
    ins=TestInstrumentProvider.btcusdt_perp_binance();tid=m.TraderId('TEST-001');sid=m.StrategyId('S-001');cid=m.ClientOrderId('O-001');vid=m.VenueOrderId('V-001');aid=m.AccountId('BINANCE-001');currency=m.Currency.from_str('USDT')
    if kind=='market':
        order=m.MarketOrder(tid,sid,ins.id,cid,m.OrderSide.BUY,ins.make_qty(.125),UUID4(),1,m.TimeInForce.GTC,False,False)
    else:
        order=m.StopLimitOrder(tid,sid,ins.id,cid,m.OrderSide.BUY,ins.make_qty(.125),ins.make_price(1001),ins.make_price(1000),m.TriggerType.DEFAULT,m.TimeInForce.GTC,False,False,False,UUID4(),1)
    order.apply(m.OrderSubmitted(tid,sid,ins.id,cid,aid,UUID4(),2,2));order.apply(m.OrderAccepted(tid,sid,ins.id,cid,vid,aid,UUID4(),3,3,False))
    order.apply(m.OrderFilled(tid,sid,ins.id,cid,vid,aid,m.TradeId('T-001'),m.OrderSide.BUY,order.order_type,ins.make_qty(.125),ins.make_price(1000),currency,m.LiquiditySide.TAKER,UUID4(),4,4,False,position_id=m.PositionId('P-001'),commission=m.Money(1,currency)))
    return ins,order


def void_event(ins,order,**kwargs):
    fields=dict(trader_id=order.trader_id,strategy_id=order.strategy_id,instrument_id=ins.id,client_order_id=order.client_order_id,venue_order_id=order.venue_order_id,account_id=order.account_id,correction_id='CORR-1',trade_id=m.TradeId('T-001'),voided_qty=ins.make_qty(.125),order_side=m.OrderSide.BUY,order_type=order.order_type,last_px=ins.make_price(1000),currency=m.Currency.from_str('USDT'),liquidity_side=m.LiquiditySide.TAKER,event_id=UUID4(),ts_event=5,ts_init=5,reconciliation=True,is_reopened=True,commission_voided=m.Money(1,m.Currency.from_str('USDT')),position_id=m.PositionId('P-001'))
    fields.update(kwargs);return m.OrderFillVoided(**fields)


@pytest.mark.parametrize('kind',['market','conditional'])
def test_native_fully_voided_fees_are_known_zero_not_unavailable(kind):
    ins,order=filled_order(kind);assert cumulative_native_commission(order)==1.
    order.apply(void_event(ins,order))
    assert str(order.filled_qty)=='0.000'
    if callable(getattr(order,'commissions',None)):assert order.commissions()=={}
    assert cumulative_native_commission(order)==0.


def test_zero_filled_without_any_native_void_evidence_is_not_zero_fee():
    empty=N(commissions=lambda:{},filled_qty='0',events=lambda:[])
    assert cumulative_native_commission(empty) is None


def test_real_native_void_reopens_same_bt_projection_and_pending_size_without_cash_mutation():
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn.core.broker_contracts import OrderRequest
    from tradelearn import engine as bt
    ins,native=filled_order();owner=N(_pending_size={});seen=[]
    broker=NautilusBroker(N());broker.bind(owner)
    broker.bind_runtime(N(step_broker=lambda order,**kwargs:seen.append(order.clone())))
    data=type('Data',(),{'_name':str(ins.id)})()
    request=OrderRequest(str(ins.id),'buy',.125,client_oid=str(native.client_order_id))
    order=broker.restore(request,data,native)
    assert order.status==bt.Order.Completed and owner._pending_size[data]==0.
    event=void_event(ins,native);native.apply(event)
    proof=dict(type='OrderFillVoided',event_id=str(event.event_id),reason=None,ts_event=int(event.ts_event))
    assert broker.apply(request.client_oid,status=native.status.name,filled_qty=float(native.filled_qty.as_double()),avg_price=native.avg_px,event_id=str(event.event_id),commission=cumulative_native_commission(native),native_ts_ns=int(event.ts_event),native_event=proof,native_correction=(native,event))
    assert broker._orders[request.client_oid] is order
    assert order.status==bt.Order.Accepted and order.executed.size==0. and order.executed.comm==0.
    assert owner._pending_size[data]==.125
    assert not broker.apply(request.client_oid,status=native.status.name,filled_qty=0.,avg_price=None,event_id=str(event.event_id),commission=0.,native_ts_ns=int(event.ts_event),native_event=proof,native_correction=(native,event))


def test_void_reopen_keeps_original_durable_intent_and_terminal_proof(tmp_path):
    from tradelearn_runtime.intents import IntentJournal
    from tradelearn_runtime.order_corrections import native_void_evidence
    import sqlite3,json
    ins,native=filled_order();event=void_event(ins,native);native.apply(event)
    evidence=native_void_evidence((native,event));oid=evidence['client_oid']
    original={'request':dict(client_oid=oid,symbol=str(ins.id),quantity=.125)}
    journal=IntentJournal(tmp_path/'intent.sqlite',identity='test-correction-owner')
    try:
        journal.prepare(oid,original)
        terminal=dict(status='FILLED',filled_qty='.125',native_event_id='original-native-fill')
        journal.terminal(oid,terminal);assert journal.unfinished()=={}
        journal.corrected(oid,evidence)
        assert journal.unfinished()=={oid:original}
        journal.corrected(oid,evidence)
        with sqlite3.connect(journal.path) as db:
            rows=db.execute('SELECT previous_terminal FROM corrections WHERE client_oid=?',(oid,)).fetchall()
        assert len(rows)==1 and json.loads(rows[0][0])==terminal
    finally:journal.close()


@pytest.mark.parametrize('spawn,reopened,void_both,refill',[(False,False,False,False),(False,True,False,False),(True,False,False,False),(True,True,False,False),(True,False,True,False),(True,True,True,False),(False,True,False,True),(True,True,False,True)])
def test_actual_official_node_fill_void_projects_and_physically_reopens_same_intent(tmp_path,spawn,reopened,void_both,refill):
    import asyncio
    from tradelearn import engine as bt
    from tradelearn_runtime.execution import create_execution_host
    from tradelearn_runtime.intents import IntentJournal
    from nautilus_trader.common import Environment,LoggerConfig
    from nautilus_trader.config import LiveNodeConfig,DataClientConfig,ExecutionClientConfig,StrategyConfig,LiveExecutionEngineConfig
    from nautilus_trader.live import LiveNode
    from nautilus_trader.live.clients import MarketDataClient,DataClientFactory,ExecutionClient,ExecutionClientFactory
    ins=TestInstrumentProvider.btcusdt_perp_binance();currency=m.Currency.from_str('USDT');aid=m.AccountId('BINANCE-001');symbol=str(ins.id)
    bars=m.BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL');notices=[];errors=[];correction_tasks=[]
    class Data(MarketDataClient):
        async def _connect(self):
            self._handle_instrument(ins)
            now=self.clock.timestamp_ns()
            self._handle_data(m.QuoteTick(ins.id,ins.make_price(1000),ins.make_price(1001),ins.make_qty(1),ins.make_qty(1),now,now))
        async def _disconnect(self):pass
        async def _subscribe_bars(self,command):
            await asyncio.sleep(.01)
            now=self.clock.timestamp_ns()
            self._handle_data(m.Bar(bars,*[ins.make_price(1000)]*4,ins.make_qty(10),now,now))
    class DF(DataClientFactory):
        def create(self,**kwargs):return Data(**kwargs,venue=ins.id.venue)
    class Exec(ExecutionClient):
        async def _connect(self):
            cash=m.Money(10000,currency)
            self.generate_account_state([m.AccountBalance(cash,m.Money(0,currency),cash)],[],True,self.clock.timestamp_ns())
        async def _disconnect(self):pass
        async def _submit_order(self,command):
            order=command.order;index=getattr(self,'_fill_index',0)+1;self._fill_index=index;vid=m.VenueOrderId('V-'+str(index))
            self.generate_order_submitted(order);await asyncio.sleep(.005)
            self.generate_order_accepted(order,vid,self.clock.timestamp_ns());await asyncio.sleep(.005)
            self.generate_order_filled(order,vid,None,m.TradeId('T-'+str(index)),order.quantity,ins.make_price(1000),currency,m.Money(1,currency),m.LiquiditySide.TAKER,self.clock.timestamp_ns())
            async def deliver():
                await asyncio.sleep(.03)
                if spawn and not void_both and str(order.quantity)!='0.050':return
                native=self.cache.order(order.client_order_id)
                event=void_event(ins,native,trade_id=native.last_event.trade_id,voided_qty=native.quantity,is_reopened=reopened,position_id=native.last_event.position_id,ts_event=self.clock.timestamp_ns(),ts_init=self.clock.timestamp_ns())
                self._handle_event(event)
                await asyncio.sleep(.01)  # Let the official asynchronous execution bus apply the event.
                assert cumulative_native_commission(self.cache.order(order.client_order_id))==0.
                if refill:
                    corrected=self.cache.order(order.client_order_id)
                    self.generate_order_filled(corrected,corrected.venue_order_id,None,m.TradeId('REFILL-'+str(index)),corrected.leaves_qty,ins.make_price(1100),currency,m.Money(.5,currency),m.LiquiditySide.TAKER,self.clock.timestamp_ns())
                    await asyncio.sleep(.01)
                    self._handle_event(event)  # Exact old void after a real new native fill.
                    await asyncio.sleep(.01)
            correction_tasks.append(asyncio.create_task(deliver()))
    class EF(ExecutionClientFactory):
        def create(self,**kwargs):return Exec(**kwargs,venue=ins.id.venue,account_id=aid,account_type=m.AccountType.MARGIN,oms_type=m.OmsType.NETTING,base_currency=currency)
    class Business(bt.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=.125,**(dict(exec_algorithm_id='VOID-ROUTE') if spawn else {}))
        def notify_order(self,order):
            notices.append((order.status,order.executed.size,order.executed.comm,dict(order.info)))
    async def run():
        node=LiveNode.build('VOID-OFFLINE',LiveNodeConfig(environment=Environment.SANDBOX,trader_id=m.TraderId('TEST-001'),logging=LoggerConfig(bypass_logging=True),exec_engine=LiveExecutionEngineConfig(reconciliation=False),timeout_connection_secs=2,timeout_disconnection_secs=1,delay_post_stop_secs=0,data_clients={'BINANCE':DataClientConfig()},exec_clients={'BINANCE':ExecutionClientConfig()}),data_factories={'BINANCE':DF()},exec_factories={'BINANCE':EF()})
        host=create_execution_host(StrategyConfig(strategy_id=m.StrategyId('S-001'),oms_type=m.OmsType.NETTING),business_cls=Business,feeds={symbol:'bar'},instruments={symbol:ins.id},subscriptions=[bars],account_id=str(aid),intent_path=tmp_path/'intent.sqlite',intent_identity='void-offline')
        if spawn:
            from nautilus_trader.trading import ExecutionAlgorithm,ExecutionAlgorithmConfig
            class RouteOnce(ExecutionAlgorithm):
                def on_order(self,order):
                    for qty in (.05,.075):self.submit_order(self.spawn_market(order,ins.make_qty(qty)))
            node.add_exec_algorithm(RouteOnce(ExecutionAlgorithmConfig(exec_algorithm_id=m.ExecAlgorithmId('VOID-ROUTE'))))
        node.add_strategy(host)
        handle=node.handle();cache=node.cache
        async def stop():await asyncio.sleep(.25);handle.stop()
        stopper=asyncio.create_task(stop())
        try:
            async with asyncio.timeout(4):await node.run_async()
            await asyncio.gather(*correction_tasks)
            await host.drain();errors.append(host._worker_error)
            [order]=host.frontend_broker._orders.values()
            expected_qty=.125 if refill else (.075 if spawn and not void_both else 0.)
            expected_fee=(1.5 if spawn else .5) if refill else (1. if spawn and not void_both else 0.)
            assert order.status==(bt.Order.Completed if refill else ((bt.Order.Partial if spawn and not void_both else bt.Order.Accepted) if reopened else bt.Order.Canceled)), (order.status,order.info,[(x.status.name,str(x.quantity),str(x.filled_qty),x.is_closed) for x in cache.orders()])
            assert order.executed.size==pytest.approx(expected_qty) and order.executed.comm==expected_fee, [(x.status.name,[type(e).__name__ for e in x.events()]) for x in cache.orders()]
            assert host.frontend.strategy._pending_size[order.data]==pytest.approx((.05 if spawn and not void_both else .125) if reopened and not refill else 0.)
            oid=order.info['client_oid']
            assert set(host.intent_journal.unfinished())==({oid} if reopened and not refill else set())
            # Rebuild a frontend from the same authoritative native cache and
            # original physical SQLite request. No private financial overlay.
            from tradelearn_runtime.broker import NautilusBroker
            from tradelearn.core.broker_contracts import OrderRequest
            import sqlite3,json
            with sqlite3.connect(host.intent_journal.path) as db:
                payload=json.loads(db.execute('SELECT payload FROM intents WHERE client_oid=?',(oid,)).fetchone()[0])
                assert db.execute('SELECT COUNT(*) FROM intents').fetchone()[0]==1
                assert db.execute('SELECT COUNT(*) FROM corrections').fetchone()[0]==(2 if void_both else 1)
            recovery=NautilusBroker(host);recovery.bind(N(_pending_size={}))
            recovery.bind_runtime(N(step_broker=lambda *a,**k:None))
            restored=recovery.restore(OrderRequest(**payload['request']),order.data,cache.order(m.ClientOrderId(oid)))
            assert (restored.status,restored.executed.size,restored.executed.comm)==(order.status,order.executed.size,order.executed.comm)
            for child in cache.orders():
                for historical in child.events():
                    if type(historical) is m.OrderFillVoided:
                        from tradelearn_runtime.execution import ExecutionHost
                        duplicate=N(frontend_broker=recovery,_project_order=lambda event:(_ for _ in ()).throw(AssertionError('restored correction replay must not project')))
                        ExecutionHost.on_strategy_order_fill_voided(duplicate,historical)
        finally:
            await stopper
            host.intent_journal.close();node.dispose()
        with_journal=IntentJournal(tmp_path/'intent.sqlite',identity='void-offline')
        try:assert (oid in with_journal.unfinished()) is (reopened and not refill)
        finally:with_journal.close()
    asyncio.run(run())
    assert errors==[None]
    assert any(status==bt.Order.Completed and qty==.125 and fee==(2. if spawn else 1.) for status,qty,fee,_ in notices), [(a,b,c) for a,b,c,_ in notices]
    assert notices[-1][1:3]==((.125,(1.5 if spawn else .5)) if refill else ((.075,1.) if spawn and not void_both else (0.,0.)))
    assert notices[-1][3]['native_correction']['is_reopened'] is reopened


@pytest.mark.parametrize('reopened',[False,True])
@pytest.mark.parametrize('quantity,void_fee',[(.125,1.),(.062,.5)])
def test_partial_or_terminal_void_preserves_native_remainder_and_pending(reopened,quantity,void_fee):
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn.core.broker_contracts import OrderRequest
    from tradelearn import engine as bt
    ins,native=filled_order();owner=N(_pending_size={});broker=NautilusBroker(N());broker.bind(owner)
    broker.bind_runtime(N(step_broker=lambda *a,**k:None));data=type('Data',(),{'_name':str(ins.id)})()
    request=OrderRequest(str(ins.id),'buy',.125,client_oid=str(native.client_order_id));order=broker.restore(request,data,native)
    event=void_event(ins,native,is_reopened=reopened,voided_qty=ins.make_qty(quantity),commission_voided=m.Money(void_fee,m.Currency.from_str('USDT')));native.apply(event)
    meta=dict(type='OrderFillVoided',event_id=str(event.event_id),reason=None,ts_event=int(event.ts_event))
    broker.apply(request.client_oid,status=native.status.name,filled_qty=float(native.filled_qty.as_double()),avg_price=native.avg_px,event_id=str(event.event_id),commission=cumulative_native_commission(native),native_ts_ns=int(event.ts_event),native_event=meta,native_correction=(native,event))
    assert order.executed.size==pytest.approx(.125-quantity)
    assert order.executed.comm==pytest.approx(1.-void_fee)
    assert owner._pending_size[data]==pytest.approx(quantity if reopened else 0.)
    assert order.alive() is reopened
    if not reopened:assert order.status==bt.Order.Canceled and order.info['native_status']=='VOIDED'


@pytest.mark.parametrize('bad',['commission','price','total','event_kind','timestamp'])
def test_native_correction_cannot_rewrite_unrelated_projection_fields(bad):
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn.core.broker_contracts import OrderRequest
    ins,native=filled_order();broker=NautilusBroker(N());owner=N(_pending_size={});broker.bind(owner)
    broker.bind_runtime(N(step_broker=lambda *a,**k:None));data=type('Data',(),{'_name':str(ins.id)})()
    request=OrderRequest(str(ins.id),'buy',.125,client_oid=str(native.client_order_id));order=broker.restore(request,data,native)
    event=void_event(ins,native,voided_qty=ins.make_qty(.062),commission_voided=m.Money(.5,m.Currency.from_str('USDT')));native.apply(event)
    fields=dict(status=native.status.name,filled_qty=float(native.filled_qty.as_double()),avg_price=native.avg_px,event_id=str(event.event_id),commission=.5,native_ts_ns=int(event.ts_event),native_event=dict(type='OrderFillVoided',event_id=str(event.event_id),reason=None,ts_event=int(event.ts_event)),native_correction=(native,event))
    if bad=='commission':fields['commission']=9.
    elif bad=='price':fields['avg_price']=2000.
    elif bad=='total':fields['total_qty']=1.
    elif bad=='event_kind':fields['native_event']['type']='OrderAccepted'
    else:fields['native_ts_ns']=99
    with pytest.raises(ValueError):broker.apply(request.client_oid,**fields)
    assert order.executed.size==.125 and owner._pending_size[data]==0.


@pytest.mark.parametrize('field,value',[
    ('account_id',m.AccountId('FOREIGN-001')),('strategy_id',m.StrategyId('FOREIGN-001')),
    ('venue_order_id',m.VenueOrderId('FOREIGN-001')),('trade_id',m.TradeId('FOREIGN-001')),
    ('order_side',m.OrderSide.SELL),('order_type',m.OrderType.LIMIT),
    ('currency',m.Currency.from_str('USD')),('last_px',m.Price.from_str('2000')),
])
def test_native_void_proof_rejects_foreign_or_changed_retained_event(field,value):
    from tradelearn_runtime.order_corrections import native_void_evidence
    ins,native=filled_order();event=void_event(ins,native);native.apply(event)
    changed=void_event(ins,native,event_id=event.event_id,**{field:value})
    with pytest.raises(ValueError):native_void_evidence((native,changed))


def test_native_void_journal_rejects_conflict_without_reopening_later_terminal(tmp_path):
    from tradelearn_runtime.intents import IntentJournal
    from tradelearn_runtime.order_corrections import native_void_evidence
    ins,native=filled_order();event=void_event(ins,native);native.apply(event)
    proof=native_void_evidence((native,event));oid=proof['client_oid']
    journal=IntentJournal(tmp_path/'journal.sqlite',identity='test-owner')
    try:
        journal.prepare(oid,{'request':{'client_oid':oid}})
        journal.terminal(oid,dict(status='FILLED'))
        journal.corrected(oid,proof)
        journal.terminal(oid,dict(status='FILLED',native_event_id='later-real-fill'))
        journal.corrected(oid,proof)
        assert journal.unfinished()=={}  # Replayed old correction cannot undo a later fill.
        with pytest.raises(ValueError,match='conflicting'):
            journal.corrected(oid,dict(proof,voided_qty='.062'))
        assert journal.unfinished()=={}
        with pytest.raises(ValueError,match='original'):
            journal.corrected('NO-INTENT',dict(proof,client_oid='NO-INTENT'))
    finally:journal.close()


def test_old_exact_void_replay_after_later_real_fill_does_not_reopen_again():
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn.core.broker_contracts import OrderRequest
    from tradelearn import engine as bt
    ins,native=filled_order();broker=NautilusBroker(N());owner=N(_pending_size={});broker.bind(owner)
    broker.bind_runtime(N(step_broker=lambda *a,**k:None));data=type('Data',(),{'_name':str(ins.id)})()
    request=OrderRequest(str(ins.id),'buy',.125,client_oid=str(native.client_order_id));order=broker.restore(request,data,native)
    event=void_event(ins,native);native.apply(event)
    fields=dict(status='ACCEPTED',filled_qty=0.,avg_price=None,event_id=str(event.event_id),commission=0.,native_ts_ns=int(event.ts_event),native_event=dict(type='OrderFillVoided',event_id=str(event.event_id),reason=None,ts_event=int(event.ts_event)),native_correction=(native,event))
    broker.apply(request.client_oid,**fields)
    fill=m.OrderFilled(native.trader_id,native.strategy_id,ins.id,native.client_order_id,native.venue_order_id,native.account_id,m.TradeId('T-002'),m.OrderSide.BUY,m.OrderType.MARKET,ins.make_qty(.125),ins.make_price(1100),m.Currency.from_str('USDT'),m.LiquiditySide.TAKER,UUID4(),6,6,False,position_id=m.PositionId('P-001'),commission=m.Money(1,m.Currency.from_str('USDT')))
    native.apply(fill)
    broker.apply(request.client_oid,status='FILLED',filled_qty=.125,avg_price=1100.,event_id=str(fill.event_id),commission=1.,native_ts_ns=6)
    assert not broker.apply(request.client_oid,**fields)
    assert order.status==bt.Order.Completed and order.executed.price==1100. and owner._pending_size[data]==0.


def test_exact_callback_replay_after_newer_cache_state_does_not_read_or_overwrite_cache():
    from tradelearn_runtime.execution import ExecutionHost
    from tradelearn_runtime.order_corrections import native_void_payload
    ins,native=filled_order();event=void_event(ins,native);native.apply(event)
    class Owner:
        @property
        def cache(self):raise AssertionError('exact replay must not read newer cache')
        def _project_order(self,event):raise AssertionError('exact owned replay must not project again')
    owner=Owner();owner.frontend_broker=N(_void_event_payloads={(str(event.client_order_id),str(event.event_id)):native_void_payload(event)})
    ExecutionHost.on_strategy_order_fill_voided(owner,event)

@pytest.mark.parametrize('reopened',[False,True])
@pytest.mark.parametrize('kind',['market','conditional'])
def test_partial_native_void_proof_is_json_safe_in_physical_journal(tmp_path,reopened,kind):
    from tradelearn_runtime.order_corrections import native_void_evidence
    from tradelearn_runtime.intents import IntentJournal
    ins,native=filled_order(kind)
    event=void_event(ins,native,is_reopened=reopened,voided_qty=ins.make_qty(.062),commission_voided=m.Money(.5,m.Currency.from_str('USDT')))
    native.apply(event);proof=native_void_evidence((native,event));oid=str(native.client_order_id)
    journal=IntentJournal(tmp_path/'partial.sqlite',identity='partial-correction')
    try:
        journal.prepare(oid,{'request':{'client_oid':oid}});journal.terminal(oid,{'status':'FILLED'})
        journal.corrected(oid,proof)
        assert (oid in journal.unfinished()) is reopened
    finally:journal.close()


def test_restored_void_metadata_is_present_in_first_business_notice_and_old_replay_is_safe():
    from tradelearn_runtime.broker import NautilusBroker
    from tradelearn_runtime.execution import ExecutionHost
    from tradelearn.core.broker_contracts import OrderRequest
    ins,native=filled_order();event=void_event(ins,native);native.apply(event)
    owner=N(_pending_size={});notices=[];broker=NautilusBroker(N());broker.bind(owner)
    broker.bind_runtime(N(step_broker=lambda order,**kwargs:notices.append(dict(order.info))))
    data=type('Data',(),{'_name':str(ins.id)})();request=OrderRequest(str(ins.id),'buy',.125,client_oid=str(native.client_order_id))
    broker.restore(request,data,native)
    assert notices[0]['native_correction']['event_id']==str(event.event_id)
    replay=N(frontend_broker=broker,_project_order=lambda event:(_ for _ in ()).throw(AssertionError('old restored correction must not read newer cache')))
    ExecutionHost.on_strategy_order_fill_voided(replay,event)
