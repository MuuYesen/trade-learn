"""Actual fixed NA bar delivery uses the existing normal timer/Runtime seam."""
import asyncio
from types import SimpleNamespace
from datetime import time,timedelta
import pandas as pd
import pytest
pytest.importorskip('nautilus_trader')
from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
from nautilus_trader.execution import FixedFeeModel
from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
from nautilus_trader.testkit.providers import TestInstrumentProvider
from tradelearn import engine as bt
from tradelearn.indicators import talib
from tests.unit.backtest._engine_config import feed as as_feed
from tests.unit.engine.test_nautilus_store import store


def test_actual_na_bar_timer_retains_previous_indicator_phase_without_new_clock_or_orders(tmp_path,monkeypatch):
    instrument=TestInstrumentProvider.btcusdt_perp_binance()
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    bars=BarType.from_str(str(instrument.id)+'-1-MINUTE-LAST-EXTERNAL')
    stamps=[pd.Timestamp(v,tz='UTC').value for v in ['2026-01-01 08:59','2026-01-01 09:00','2026-01-01 09:06','2026-01-01 09:09','2026-01-02 09:20']]
    native.add_data([Bar(bars,*[instrument.make_price(1000)]*4,instrument.make_qty(10),stamp,stamp) for stamp in stamps])
    notices=[];business=[];worker_errors=[]
    class Business(bt.Strategy):
        def __init__(self):self.average=talib.SMA(self.data.close,timeperiod=3)
        def notify_timer(self,timer,when,*args,**kwargs):
            import math
            value=float(self.average[0]) if len(self) else None
            if value is not None and math.isnan(value):value=None
            notices.append((when.isoformat(),len(self),len(self.data),value,kwargs))
        def next(self):business.append(len(self))
    value=store(tmp_path);engine=bt.Cerebro(stdstats=False)
    engine.setbroker(value.getbroker());engine.adddata(as_feed(value.getdata(bars)));engine.addstrategy(Business)
    engine.add_timer(time(9),repeat=timedelta(minutes=5),strats=True,marker=42)
    class ReplayNode:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return SimpleNamespace(stop=lambda:None)
        async def run_async(self):native.run()
        def dispose(self):
            nonlocal native
            worker_errors.append(self.actor._worker_error)
            native.dispose();self.actor=None;native=None
    monkeypatch.setattr(value,'_build_node',ReplayNode)
    [strategy]=engine.run()
    assert notices==[('2026-01-01T09:00:00',1,2,None,{'marker':42}),('2026-01-01T09:05:00',2,3,None,{'marker':42}),('2026-01-02T09:00:00',4,5,1000.,{'marker':42})]
    assert business==[3,4,5]
    assert strategy._runtime is value._host.frontend and worker_errors==[None]
    assert native is None
