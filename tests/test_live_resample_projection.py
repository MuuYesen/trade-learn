"""Native source bars drive one bounded completed-bar view, no second engine."""
from types import SimpleNamespace as NS
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as bt


def golden(frame,compression,replay=False,clone=False,unit="minutes"):
    result=[]
    class Business(official.Strategy):
        def __init__(self):self.means=[official.indicators.SimpleMovingAverage(d.close,period=2) for d in self.datas]
        def next(self):result.append((len(self),tuple((len(d),d.datetime.datetime(0).isoformat(),d.open[0],d.close[0],d.volume[0],self.means[i][0]) for i,d in enumerate(self.datas))))
    c=official.Cerebro(stdstats=False);source=official.feeds.PandasData(dataname=frame,timeframe=official.TimeFrame.Minutes if unit=='minutes' else official.TimeFrame.Seconds)
    if clone:c.adddata(source)
    (c.replaydata if replay else c.resampledata)(source,timeframe=official.TimeFrame.Minutes if unit=='minutes' else official.TimeFrame.Seconds,compression=compression);c.addstrategy(Business);c.run();return result


@pytest.mark.parametrize('unit',['minutes','seconds'])
@pytest.mark.parametrize('clone',[False,True])
@pytest.mark.parametrize('replay',[False,True])
@pytest.mark.parametrize('compression,offsets,start',[(2,[0,1,2,3,4,5,6],'2026-01-02T09:00:00Z'),(7,[0,1,6,7,8,13],'2026-01-02T09:00:00Z'),(2,[0,1,2,3],'2026-01-02T09:00:00Z'),(7,[0,1,6,7,8],'2026-01-02T09:00:00Z'),(2,[0,1,9,10],'2026-01-02T09:00:00Z'),(2,[0,1,2,3,4,5,6],'2026-01-02T23:58:00Z'),(7,[0,1,2,3,4,5,6],'2026-01-02T23:58:00Z')])
def test_actual_native_source_uses_completed_live_resample_without_source_eof(tmp_path,monkeypatch,compression,offsets,start,replay,clone,unit):
    pytest.importorskip('nautilus_trader')  # This real-native case requires the optional ABI.
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.model import AccountType,OmsType,Money,Currency,Bar,BarType
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tests.unit.engine.test_nautilus_store import store
    instrument=TestInstrumentProvider.btcusdt_perp_binance()
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,[Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    bar_type=BarType.from_str(str(instrument.id)+('-1-MINUTE-LAST-EXTERNAL' if unit=='minutes' else '-1-SECOND-LAST-EXTERNAL'))
    stamps=[pd.Timestamp(start)+pd.Timedelta(v,unit='min' if unit=='minutes' else 's') for v in offsets]
    prices=[1000.+10*i for i in range(len(stamps))]
    native_bars=[Bar(bar_type,*[instrument.make_price(price)]*4,instrument.make_qty(10),stamp.value,stamp.value) for price,stamp in zip(prices,stamps)]
    rows=[];errors=[];fills=[];trades=[]
    class Business(bt.Strategy):
        def __init__(self):
            self.means=[bt.talib.SMA(d.close,timeperiod=2) for d in self.datas]
            self.sent=False
        def next(self):
            rows.append((len(self),tuple((len(d),d.datetime.datetime(0).isoformat(),d.open[0],d.close[0],d.volume[0],self.means[i][0]) for i,d in enumerate(self.datas))))
            if compression==2 and len(offsets)==7 and not self.sent:
                self.sent=True;self.buy(data=self.datas[-1],size=.125)
        def notify_trade(self,trade):
            trades.append(trade.data is self.datas[-1])
        def notify_order(self,order):
            if order.status==order.Completed:fills.append((order.data is self.datas[-1],order.executed.size,order.executed.price,order.executed.comm))
    value=store(tmp_path,buffer_capacity=3);engine=bt.Cerebro(stdstats=False)
    engine.setbroker(value.getbroker());source=value.getdata(bar_type)
    if clone:engine.adddata(source)
    target=(engine.replaydata if replay else engine.resampledata)(source,timeframe=bt.TimeFrame.Minutes if unit=='minutes' else bt.TimeFrame.Seconds,compression=compression)
    engine.addstrategy(Business)
    class ReplayNode:
        def add_strategy(self,actor):self.actor=actor;native.add_strategy(actor)
        def handle(self):return NS(stop=lambda:None)
        async def run_async(self):
            import asyncio
            for bar in native_bars:
                native.clear_data();native.add_data([bar]);native.run(streaming=True)
                # Let the existing durable-admission worker run, as a real
                # asynchronous Node does; do not bypass intent persistence.
                while self.actor._tasks:
                    await asyncio.gather(*tuple(self.actor._tasks))
            native.end()
        def dispose(self):
            nonlocal native
            errors.append(self.actor._worker_error)
            try:native.dispose()
            finally:self.actor=None;native=None
    monkeypatch.setattr(value,'_build_node',ReplayNode)
    [business]=engine.run()
    frame=pd.DataFrame({k:prices for k in ('open','high','low','close')},index=pd.DatetimeIndex(stamps));frame['volume']=10.
    expected=golden(frame,compression,replay,clone,unit)
    if not replay:
        # Historical EOF may materialize a future-labelled incomplete bar. A
        # native stream stop is not source EOF and must not create that bar.
        last=stamps[-1].tz_localize(None).isoformat()
        expected=[row for row in expected if row[1][-1][1]<=last]
    assert rows==expected
    if compression==2 and len(offsets)==7:
        assert fills==[(True,.125,1020. if replay else 1030.,0.)]
        assert trades and all(trades)
    assert errors==[None] and business._runtime is value._host.frontend
    assert all(d.retained<=3 for d in business.datas)
    assert [d._name for d in business.datas]==([source._name,''] if clone else [source._name])
