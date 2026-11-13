"""Actual fixed NA conditional orders must project fills and complete fee evidence."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import Bar,BarType,Money,Currency,OmsType,AccountType,StrategyId
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.testkit.providers import TestInstrumentProvider
from tradelearn import engine as bt
from tradelearn.core.broker_contracts import OrderRequest
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.execution import create_execution_host


def test_stoplimit_cache_filled_reaches_bt_callbacks_and_recovery_with_actual_fee():
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    bars=BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(1,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    native.add_data([Bar(bars,*[instrument.make_price(p)]*4,instrument.make_qty(10),(i+1)*60_000_000_000,(i+1)*60_000_000_000) for i,p in enumerate([1000,1000,1010,1020,1030])])
    notices=[]
    class Business(bt.Strategy):
        def next(self):
            if len(self)==1:self.order=self.buy(size=.125,exectype=bt.Order.StopLimit,price=1010,plimit=1020)
        def notify_order(self,o):notices.append((o.status,o.executed.size,o.executed.comm))
    host=create_execution_host(StrategyConfig(strategy_id=StrategyId('FEE-001'),oms_type=OmsType.NETTING),business_cls=Business,feeds={symbol:'bar'},instruments={symbol:instrument.id},subscriptions=[bars],account_id='BINANCE-001',offline_test=True)
    try:
        native.add_strategy(host);native.run()
        [order]=native.cache.orders()
        assert order.status.name=='FILLED' and str(order.filled_qty)=='0.125'
        assert any(status==bt.Order.Accepted for status,_,_ in notices)
        assert notices[-1]==(bt.Order.Completed,.125,1.)
        restored=NautilusBroker(NS());restored.bind(NS(_pending_size={}))
        restored.bind_runtime(NS(step_broker=lambda *a,**k:None))
        request=OrderRequest(symbol,'buy',.125,order_type='stop_limit',limit_price=1020,stop_price=1010,client_oid=str(order.client_order_id))
        projected=restored.restore(request,host.frontend.feeds[symbol],order)
        assert projected.status==bt.Order.Completed and projected.executed.comm==1.
    finally:
        native.dispose();host=None;native=None


def fill_event(identity, quantity, amount, currency='USDT'):
    from nautilus_trader.model import Quantity
    return type('OrderFilled',(),{} )() if amount is None else type('OrderFilled',(),dict(event_id=identity,last_qty=Quantity.from_str(quantity),commission=Money(amount,Currency.from_str(currency))))()


@pytest.mark.parametrize('kind', ['missing','partial','mixed','voided','conflict','nonfinite'])
def test_incomplete_fee_evidence_is_unknown_not_zero(kind):
    from tradelearn_runtime.order_fees import cumulative_native_commission
    events=[fill_event('a','0.125',1)]
    quantity='0.125'
    if kind=='missing':events=[fill_event('a','0.125',None)]
    elif kind=='partial':quantity='0.250'
    elif kind=='mixed':events.append(fill_event('b','0.125',1,'USD'));quantity='0.250'
    elif kind=='voided':events.append(type('OrderFillVoided',(),{})())
    elif kind=='conflict':events.append(fill_event('a','0.125',2))
    else:events=[type('OrderFilled',(),dict(event_id='a',last_qty='0.125',commission=NS(currency='USDT',as_double=lambda:float('nan'))))()]
    assert cumulative_native_commission(NS(events=events,filled_qty=quantity)) is None


@pytest.mark.parametrize('fee',[0,1,-1])
def test_complete_partial_fee_events_sum_once_and_preserve_real_zero_or_rebate(fee):
    from tradelearn_runtime.order_fees import cumulative_native_commission
    first=fill_event('a','0.125',fee);second=fill_event('b','0.125',fee)
    assert cumulative_native_commission(NS(events=[first,first,second],filled_qty='0.250'))==fee*2
