"""Typed close routes never turn a close into an unqualified reverse position."""
from types import SimpleNamespace as NS,MethodType
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import ExecutionHost
from nautilus_trader.model import InstrumentId,StrategyId,PositionId


def native(positions):
    iid=InstrumentId.from_str('A.SIM');strategy=StrategyId('TL-001')
    host=NS(_assert_account_binding=lambda:None,_account_id='SIM-001',strategy_id=strategy,
        _instrument_ids={'A':iid},cache=NS(positions_open=lambda **kwargs:positions,
            position=lambda pid:next((p for p in positions if p.id==pid),None)))
    host._bound_position=MethodType(ExecutionHost._bound_position,host)
    return host


def position(pid='P1',account='SIM-001',strategy='TL-001',size=2):
    return NS(id=PositionId(pid),instrument_id=InstrumentId.from_str('A.SIM'),
        account_id=account,strategy_id=StrategyId(strategy),signed_qty=size)


def test_close_requires_exact_owned_position_and_hedged_route():
    one,two=position(),position('P2',size=-3)
    with pytest.raises(ValueError,match='explicit position_id'):
        ExecutionHost.close_controls(native([one,two]),'A')
    route=ExecutionHost.close_controls(native([one,two]),'A',position_id='P2')
    assert route==dict(position_id=two.id,size=3,side='buy')
    for other in (position(account='SIM-002'),position(strategy='OTHER-001')):
        with pytest.raises(ValueError,match='position'):
            ExecutionHost.close_controls(native([other]),'A',position_id='P1')


def test_close_size_is_bounded_and_controls_preserve_native_identity():
    requests=[]
    gateway=NS(close_controls=lambda *a,**k:dict(size=2,side='sell',position_id=PositionId('P1')),
        lookup_order=lambda cid:None,submit=lambda request,**controls:requests.append((request,controls)))
    broker=NautilusBroker(gateway);broker.runtime=NS(step_broker=lambda *a,**k:None)
    broker._close_position(NS(),NS(_name='A'),size=.5)
    request,controls=requests[0]
    assert request.side=='sell' and request.qty==.5
    assert controls=={'position_id':PositionId('P1'),'reduce_only':True}
    for changes in ({'size':3},{'size':True},{'size':float('nan')},{'reduce_only':False}):
        with pytest.raises(ValueError):broker._close_position(NS(),NS(_name='A'),**changes)
    assert len(requests)==1


def test_existing_private_typed_gateway_can_keep_own_close_policy():
    broker=NautilusBroker(NS())
    assert broker._close_position(NS(),NS(_name='A')) is NotImplemented


def test_close_control_proof_is_persisted_in_existing_intent_journal(tmp_path):
    import asyncio
    from tradelearn_runtime.intents import IntentJournal
    from tradelearn.core.broker_contracts import OrderRequest
    calls=[]
    journal=IntentJournal(tmp_path/'intents.sqlite',identity='synthetic-close-binding')
    request=OrderRequest('A','sell',1,client_oid='CLOSE-1')
    async def check():
        host=NS(_io_lock=asyncio.Lock(),intent_journal=journal,_stopping=False,
            frontend=NS(failed=False),_assert_account_binding=lambda:None,
            submit_order=lambda *a,**k:calls.append(k),
            frontend_broker=NS(_unresolved=set(),_orders={}))
        await ExecutionHost._admit_and_submit(host,request,NS(is_reduce_only=True),{'position_id':PositionId('P1')})
    try:
        asyncio.run(check())
        proof=journal.unfinished()['CLOSE-1']['native_controls']
        assert proof=={'position_id':'P1','reduce_only':True,'post_only':False} and len(calls)==1
    finally:journal.close()


def test_recovery_refuses_changed_reduce_only_or_position_route():
    from nautilus_trader.common import Clock,OrderFactory
    from nautilus_trader.model import TraderId,Quantity,OrderSide,ClientOrderId
    from tradelearn.core.broker_contracts import OrderRequest
    factory=OrderFactory(TraderId('TEST-001'),StrategyId('TL-001'),Clock.new_test())
    order=factory.market(instrument_id=InstrumentId.from_str('A.SIM'),order_side=OrderSide.SELL,
        quantity=Quantity.from_str('1'),client_order_id=ClientOrderId('CLOSE-1'),reduce_only=True)
    row=type(order).to_dict(order);row['position_id']='P1';order=type(order).from_dict(row)
    request=OrderRequest('A','sell',1,client_oid='CLOSE-1')
    host=native([]);host._account_id=None
    # An initialized/inflight order can carry its route in the NA cache index.
    host.cache.position_id=lambda cid:PositionId('P1') if str(cid)=='CLOSE-1' else None
    proof={'position_id':'P1','reduce_only':True}
    ExecutionHost.validate_recovery(host,request,order,native_controls=proof)
    for changed in ({'position_id':'P2','reduce_only':True},{'position_id':'P1','reduce_only':False}):
        with pytest.raises(ValueError,match='route|reduce'):
            ExecutionHost.validate_recovery(host,request,order,native_controls=changed)


def test_typed_native_close_uses_standard_sell_and_bt_validity_without_private_user_hooks():
    from datetime import date
    from tradelearn import engine as bt
    from tradelearn.backtest.event_runtime import StrategyRuntime
    from tradelearn_runtime.broker import NautilusBroker
    from types import SimpleNamespace
    calls=[];sent=[]
    gateway=SimpleNamespace(
        close_controls=lambda symbol,position_id=None:dict(size=2,side='sell',position_id='P1'),
        lookup_order=lambda cid:None,
        submit=lambda request,**controls:sent.append((request,controls)),
    )
    broker=NautilusBroker(gateway)
    class Business(bt.Strategy):
        def sell(self,*args,**kwargs):
            calls.append('sell')
            return super().sell(*args,**kwargs)
        def next(self):
            self.order=self.close(size=1,valid=date(2026,1,2),tif='ioc')
    runtime=StrategyRuntime(Business,feeds={'A':'bar'},broker=broker)
    try:
        runtime.start()
        runtime.step_market('A',dict(ts='2026-01-01T00:00:00Z',open=10,high=10,low=10,close=10,volume=1))
        assert calls==['sell']
        assert runtime.strategy.order.valid==bt.date2num(date(2026,1,2))
        request,controls=sent[0]
        assert request.side=='sell' and request.qty==1 and request.tif=='ioc'
        assert controls=={'position_id':'P1','reduce_only':True}
    finally:runtime.cleanup()
