"""BT original price/plimit offset compiled to the real fixed NA order factory."""
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

pytest.importorskip('nautilus_trader')
from tradelearn.backtest.models import Order
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import ExecutionHost
from tests.test_live_order_mapping import native_host


@pytest.mark.parametrize('side,plimit,trigger,limit,offset', [
    (Order.Sell, 98, '99', '97', '3'),
    (Order.Buy, 102, '101', '103', '3'),
    (Order.Sell, 102, '99', '101', '-1'),
    (Order.Buy, 98, '101', '99', '-1'),
    (Order.Sell, None, '99', '99', '1'),
    (Order.Buy, None, '101', '101', '1'),
])
def test_original_bt_offset_is_preserved_in_native_admission_and_recovery(side,plimit,trigger,limit,offset):
    host=native_host();sent=[];requests=[]
    class Gateway:
        def lookup_order(self,cid):return None
        def validate(self,request,**controls):host._make_order(request,controls)
        def submit(self,request,**controls):
            requests.append(request);sent.append(host._make_order(request,controls)[0])
    broker=NautilusBroker(Gateway());broker.runtime=NS(step_broker=lambda *a,**k:None)
    controls={} if plimit is None else {'pricelimit':plimit}
    projected=broker._submit(NS(),NS(_name='EURUSD'),side,1,price=100,
        exectype=Order.StopTrailLimit,trailamount=1,**controls)
    native=sent[0]
    assert native.order_type.name=='TRAILING_STOP_LIMIT'
    assert Decimal(str(native.trigger_price))==Decimal(trigger)
    assert Decimal(str(native.price))==Decimal(limit)
    assert native.limit_offset==Decimal(offset)
    assert native.trailing_offset==Decimal('1')
    assert projected.pricelimit==(100 if plimit is None else plimit)
    ExecutionHost.validate_recovery(host,requests[0],native)
    with pytest.raises(ValueError,match='trailing'):
        ExecutionHost.validate_recovery(host,replace(requests[0],limit_price=90),native)


def test_percentage_with_fixed_nonzero_bt_limit_difference_is_not_reinterpreted_as_percentage():
    host=native_host()
    from tradelearn.core.broker_contracts import OrderRequest
    request=OrderRequest('EURUSD','sell',1,'stop_trail_limit',stop_price=100,
        limit_price=98,trail_percent=.1,client_oid='OFFSET-PERCENT')
    with pytest.raises(NotImplementedError,match='percent.*fixed|fixed.*percent'):
        host._make_order(request,{})


@pytest.mark.parametrize('side,trigger',[('buy','110'),('sell','90')])
def test_percentage_with_zero_original_offset_maps_exactly(side,trigger):
    host=native_host()
    from tradelearn.core.broker_contracts import OrderRequest
    request=OrderRequest('EURUSD',side,1,'stop_trail_limit',stop_price=100,
        limit_price=100,trail_percent=.1,client_oid='OFFSET-ZERO')
    native=host._make_order(request,{})[0]
    assert Decimal(str(native.price))==Decimal(trigger)
    assert Decimal(str(native.trigger_price))==Decimal(trigger)
    assert native.limit_offset==native.trailing_offset==Decimal('1000')
    ExecutionHost.validate_recovery(host,request,native)


@pytest.mark.parametrize('side,basis',[('sell',110),('buy',90)])
@pytest.mark.parametrize('kind',['amount','inverse_limit','default_limit','percent_zero_delta'])
def test_real_native_matching_updates_preserve_bt_original_limit_delta(side,basis,kind):
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import QuoteTick,AccountType,OmsType,Money,Currency,StrategyId
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from nautilus_trader.trading import Strategy as NativeStrategy
    from tradelearn.core.broker_contracts import OrderRequest
    instrument=TestInstrumentProvider.btcusdt_perp_binance()
    standard=98 if side=='sell' else 102
    limit=(102 if side=='sell' else 98) if kind=='inverse_limit' else standard
    if kind in ('default_limit','percent_zero_delta'):limit=100
    percent=.1 if kind=='percent_zero_delta' else None
    amount=None if percent is not None else 1
    request=OrderRequest(str(instrument.id),side,1,'stop_trail_limit',stop_price=100,
        limit_price=limit,trail_amount=amount,trail_percent=percent,client_oid='BT-OFFSET-NATIVE')
    engine=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    engine.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    engine.add_instrument(instrument)
    start=1767225600*1_000_000_000
    engine.add_data([QuoteTick(instrument.id,instrument.make_price(price),instrument.make_price(price),
        instrument.make_qty(10),instrument.make_qty(10),start+index*1_000_000_000,start+index*1_000_000_000)
        for index,price in enumerate([100,basis])])
    class Business(NativeStrategy):
        def on_start(self):self.subscribe_quotes(instrument.id)
        def on_quote(self,tick):
            if hasattr(self,'order'):return
            self.host=NS(_assert_account_binding=lambda:None,_account_id=None,
                strategy_id=self.strategy_id,_instrument_ids={str(instrument.id):instrument.id},
                cache=self.cache,order_factory=self.order_factory)
            self.order=ExecutionHost._make_order(self.host,request,{})[0]
            self.submit_order(self.order)
    business=Business(StrategyConfig(strategy_id=StrategyId('TL-001')))
    try:
        engine.add_strategy(business);engine.run()
        native=engine.cache.order(business.order.client_order_id)
        trigger=basis*(1+.1 if side=='buy' else 1-.1) if percent is not None else basis+(1 if side=='buy' else -1)
        assert native.status.name=='ACCEPTED'
        assert float(str(native.trigger_price))==pytest.approx(trigger)
        assert float(str(native.price))==pytest.approx(trigger-(100-limit))
        ExecutionHost.validate_recovery(business.host,request,native)
    finally:engine.dispose()
