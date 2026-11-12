"""Existing BT buy kwargs through the real pinned NA simulator, never a venue."""

import pytest

pytest.importorskip("nautilus_trader")
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.model import (
    AccountType,
    Bar,
    BarType,
    Currency,
    Money,
    OmsType,
    QuoteTick,
    StrategyId,
)
from nautilus_trader.testkit.providers import TestInstrumentProvider

from tradelearn import engine as bt
from tradelearn_runtime.execution import create_execution_host


@pytest.mark.parametrize(
    "price,expected", [(1000.0, bt.Order.Rejected), (900.0, bt.Order.Accepted)]
)
def test_post_only_native_crossing_rejects_and_non_crossing_rests(price, expected):
    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    symbol = str(instrument.id)
    bars = BarType.from_str(symbol + "-1-MINUTE-LAST-EXTERNAL")
    engine = BacktestEngine(BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    engine.add_venue(
        instrument.id.venue,
        OmsType.NETTING,
        AccountType.MARGIN,
        [Money(1000000, Currency.from_str("USDT"))],
        fee_model=FixedFeeModel(Money(0, Currency.from_str("USDT"))),
    )
    engine.add_instrument(instrument)
    for index in range(2):
        ts = (index + 1) * 60_000_000_000
        engine.add_data(
            [
                QuoteTick(
                    instrument.id,
                    instrument.make_price(999.0),
                    instrument.make_price(1000.0),
                    instrument.make_qty(10),
                    instrument.make_qty(10),
                    ts - 1,
                    ts - 1,
                )
            ]
        )
        engine.add_data(
            [Bar(bars, *[instrument.make_price(1000.0)] * 4, instrument.make_qty(10), ts, ts)]
        )
    notices = []

    class Business(bt.Strategy):
        def __init__(self):
            assert self.data._timeframe == bt.TimeFrame.Minutes
            assert self.data._compression == self.data.p.compression == 1

        def next(self):
            if len(self) == 1:
                self.order = self.buy(size=1, price=price, exectype=bt.Order.Limit, post_only=True)

        def notify_order(self, order):
            notices.append(order.status)

    host = create_execution_host(
        StrategyConfig(strategy_id=StrategyId("TL-001"), oms_type=OmsType.NETTING),
        business_cls=Business,
        feeds={symbol: "bar"},
        instruments={symbol: instrument.id},
        subscriptions=[bars],
        account_id="BINANCE-001",
        offline_test=True,
    )
    try:
        engine.add_strategy(host)
        engine.run()
        assert host.frontend.strategy.order.status == expected
        assert expected in notices
        assert bt.Order.Completed not in notices
        native = host.cache.orders()[0]
        assert native.is_post_only is True
    finally:
        engine.dispose()
