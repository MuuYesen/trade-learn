"""Plain BT user code through Store logging, warmup and actual NA order events."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip('nautilus_trader')
from nautilus_trader.backtest import BacktestEngine, BacktestEngineConfig
from nautilus_trader.execution import FixedFeeModel, LimitOrderPartialFillModel
from nautilus_trader.model import AccountType, OmsType, Money, Currency, Bar, BarType, QuoteTick
from nautilus_trader.testkit.providers import TestInstrumentProvider

from tradelearn import engine as bt
from tradelearn.indicators import talib
from tests.unit.backtest._engine_config import feed as as_feed
from tests.unit.engine.test_nautilus_store import store


@pytest.mark.parametrize('logging', [False, True])
@pytest.mark.parametrize('reject', [False, True])
def test_plain_strategy_warmup_position_partial_and_rejection_remain_native(
    tmp_path, monkeypatch, logging, reject,
):
    instrument = TestInstrumentProvider.btcusdt_perp_binance()
    symbol = str(instrument.id)
    bars = BarType.from_str(symbol + '-1-MINUTE-LAST-EXTERNAL')
    native = BacktestEngine(BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    native.add_venue(
        instrument.id.venue, OmsType.NETTING, AccountType.MARGIN,
        [Money(1000000, Currency.from_str('USDT'))],
        fee_model=FixedFeeModel(Money(0, Currency.from_str('USDT'))),
        fill_model=LimitOrderPartialFillModel(random_seed=1),
    )
    native.add_instrument(instrument)
    for index in range(25):
        stamp = (index + 1) * 60_000_000_000
        native.add_data([QuoteTick(instrument.id, instrument.make_price(999.9),
                                   instrument.make_price(1000), instrument.make_qty(5),
                                   instrument.make_qty(5), stamp - 1, stamp - 1)])
        native.add_data([Bar(bars, *[instrument.make_price(1000)] * 4,
                              instrument.make_qty(10), stamp, stamp)])
    phases, notices = [], []

    class Business(bt.Strategy):
        params = (('period', 2),)
        def __init__(self):
            self.average = bt.talib.SMA(self.data.close, timeperiod=self.p.period)
        def prenext(self):
            phases.append(('warmup', len(self)))
        def next(self):
            if len(self) == self.p.period:
                assert not self.position
                phases.append(('submit', len(self)))
                self.order = self.buy(price=1000, exectype=bt.Order.Limit, post_only=reject)
        def notify_order(self, order):
            notices.append((order.status, order.executed.size))
        def stop(self):
            # Query while NA/cache still owns the account; after dispose fails closed.
            self.final_position_size = self.position.size

    value = store(tmp_path)
    engine = bt.Cerebro(stdstats=False)
    engine.setbroker(value.getbroker())
    engine.adddata(as_feed(value.getdata(bars)))
    engine.addstrategy(Business, period=2)
    engine.addsizer(bt.sizers.FixedSize, stake=10)

    class ReplayNode:
        def add_strategy(self, actor):
            self.actor = actor
            native.add_strategy(actor)
        def handle(self):
            return SimpleNamespace(stop=lambda: None)
        async def run_async(self):
            # The synchronous simulator must yield between native batches so
            # production admission workers can fsync before sending orders.
            # This harness never enables offline_test or bypasses the journal.
            for index in range(25):
                native.run(end=(index + 1) * 60_000_000_000, streaming=True)
                while self.actor._tasks:
                    await asyncio.gather(*tuple(self.actor._tasks))
            native.end()
        def dispose(self):
            nonlocal native
            try:
                native.dispose()
            finally:
                # This replay fixture owns a Python class/closure/native cycle.
                # Release its simulator and registered actor on the owner thread.
                self.actor = None
                native = None

    monkeypatch.setattr(value, '_build_node', ReplayNode)
    [business] = engine.run(monitor=True) if logging else engine.run()
    assert native is None, 'fixture retained its unsendable replay engine'
    assert phases == [('warmup', 1), ('submit', 2)]
    assert isinstance(business, Business) and business.broker is value.getbroker()
    assert business._runtime is value._host.frontend
    if reject:
        assert business.order.status == bt.Order.Rejected
        assert business.final_position_size == 0
        assert not any(status in (bt.Order.Partial, bt.Order.Completed) for status, _ in notices)
    else:
        assert business.order.status == bt.Order.Completed
        assert business.final_position_size == business.order.executed.size == 10
        assert any(status == bt.Order.Partial and 0 < size < 10 for status, size in notices)
    if logging:
        assert business.monitor_logger is value._host.logger
        assert business.monitor_config.context['run_id'] == value._host.logger.run_id
    else:
        assert not hasattr(business, 'monitor_logger')
    assert 'monitor_config' not in vars(Business)
