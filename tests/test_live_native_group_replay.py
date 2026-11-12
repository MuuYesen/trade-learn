"""Linked orders in the fixed official NA simulator, never a venue or credentials."""
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn import engine as bt
from tradelearn_runtime.execution import create_execution_host
from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import (AccountType,OmsType,Money,Currency,Bar,BarType,
                                  Price,Quantity,StrategyId)
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.testkit.providers import TestInstrumentProvider


@pytest.mark.parametrize('cash,partial',[(False,False),(True,False),(False,True)])
def test_public_bracket_uses_one_na_contingent_group_and_cancels_sibling(cash,partial):
    instrument=(TestInstrumentProvider.btcusdt_binance() if cash
                else TestInstrumentProvider.btcusdt_perp_binance())
    symbol=str(instrument.id)
    bar_type=BarType.from_str(symbol+'-1-MINUTE-LAST-EXTERNAL')
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    from nautilus_trader.execution import LimitOrderPartialFillModel
    fill_options={'fill_model':LimitOrderPartialFillModel(random_seed=1)} if partial else {}
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.CASH if cash else AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],
        fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))),**fill_options)
    native.add_instrument(instrument)
    rows=[('1000.00','1000.00','1000.00','1000.00'),('1000.00','1000.00','1000.00','1000.00'),
          ('1000.00','1200.00','1000.00','1100.00'),('1100.00','1100.00','1100.00','1100.00')]
    if partial:
        rows=[('1000.00','1000.00','1000.00','1000.00')]*25+[('1100.00','1200.00','1100.00','1100.00')]*35
    native.add_data([Bar(bar_type,*[instrument.make_price(float(v)) for v in values],
        instrument.make_qty(10),(i+1)*60_000_000_000,(i+1)*60_000_000_000)
        for i,values in enumerate(rows)])
    if partial:
        from nautilus_trader.model import QuoteTick
        # Matching liquidity only; the TL Runtime still subscribes solely to bars.
        native.add_data([QuoteTick(instrument.id,
            instrument.make_price(999.9 if i<25 else 1100),
            instrument.make_price(1000 if i<25 else 1100.1),
            instrument.make_qty(5),instrument.make_qty(5),
            (i+1)*60_000_000_000-1,(i+1)*60_000_000_000-1) for i in range(len(rows))])
    notices=[];groups=[]
    class Business(bt.Strategy):
        def next(self):
            if len(self.data)==1:
                self.orders=self.buy_bracket(size=10 if partial else 2,price=1000,stopprice=900,limitprice=1100,
                    oargs={'tif':'gtc'},stopargs={'tif':'gtc'},limitargs={'tif':'gtc'})
        def notify_order(self,order):
            notices.append((order.info['client_oid'],order.status))
    host=create_execution_host(StrategyConfig(strategy_id=StrategyId('TL-001'),
        oms_type=OmsType.NETTING,manage_contingent_orders=False),business_cls=Business,
        feeds={symbol:'bar'},instruments={symbol:instrument.id},
        subscriptions=[bar_type],account_id='BINANCE-001',offline_test=True)
    # Record the existing public NA group boundary, preserving its implementation.
    submit=host.submit_order_list
    def capture(orders,**controls):
        groups.append(tuple(orders));return submit(orders,**controls)
    host.submit_order_list=capture
    try:
        native.add_strategy(host);native.run()
        orders=host.frontend.strategy.orders
        assert len(groups)==1 and len(groups[0])==3
        if cash:
            # The unchanged NA cash-risk gate rejects the whole list: no BTC inventory.
            assert [order.status for order in orders]==[bt.Order.Rejected]*3
            assert sum(status==bt.Order.Completed for _,status in notices)==0
        else:
            assert [order.status for order in orders]==[bt.Order.Completed,bt.Order.Canceled,bt.Order.Completed]
            assert sum(status==bt.Order.Completed for _,status in notices)==2
        if partial:
            assert any(status==bt.Order.Partial for _,status in notices)
        assert host.frontend.strategy.broker is host.frontend_broker
    finally:native.dispose()


def test_native_projection_dependency_is_present_in_installed_distribution():
    # NA can log Python callback failures rather than propagate them to run().
    # Check the shipped callback dependency explicitly, even without a fill event.
    from tradelearn_runtime.notification_evidence import notification_evidence
    assert notification_evidence(object()) is None
