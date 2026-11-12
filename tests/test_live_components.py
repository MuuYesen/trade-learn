from tradelearn.backtest.event_runtime import StrategyRuntime
"""Cerebro components share the existing Runtime; no execution or scheduler loop."""
from types import SimpleNamespace as NS
import pytest
pytest.importorskip('nautilus_trader')
from tradelearn_runtime import NautilusStore
from tradelearn import engine as bt


def test_registered_streaming_analyzer_and_observer_use_existing_callbacks():
    from tradelearn_runtime.components import LiveComponents
    calls=[]
    class Business(bt.Strategy):
        def next(self):calls.append(('next',float(self.data.close[0])))
    class Analyzer(bt.Analyzer):
        is_streaming=True
        def on_start(self):calls.append('analyzer-start')
        def on_bar(self,bar):calls.append(('bar',bar.close))
        def on_order(self,order):calls.append(('order',order))
        def on_end(self,stats):calls.append(('end',stats['bars']))
        def stop(self):calls.append('analyzer-stop')
        def get_analysis(self):return {'seen':1}
    class Observer(bt.Observer):
        def start(self):calls.append('observer-start')
        def next(self):calls.append(('observer',float(self.strategy.data.close[0])))
        def stop(self):calls.append('observer-stop')
    engine=bt.Cerebro(stdstats=False);engine.addanalyzer(Analyzer,_name='stream');engine.addobserver(Observer)
    runtime=StrategyRuntime(Business,feeds={'A':'bar'});components=LiveComponents(engine,runtime)
    runtime.start();components.start()
    runtime.step_market('A',dict(ts='2026-01-01T00:00:00Z',open=10,high=10,low=10,close=10,volume=1))
    components.after_market();components.order('projection')
    runtime.stop();components.finish();components.finish()
    assert calls==['analyzer-start','observer-start',('next',10.),('bar',10.),('observer',10.),
        ('order','projection'),('end',1),'analyzer-stop','observer-stop']
    assert engine.analyzer_results=={'stream':{'seen':1}}
    assert runtime.strategy.analyzers.stream.strategy is runtime.strategy


def test_post_backtest_stats_analyzer_is_not_a_live_streaming_analyzer():
    from tradelearn_runtime.components import validate_components
    engine=bt.Cerebro();engine.addanalyzer(bt.analyzers.SharpeRatio)
    with pytest.raises(NotImplementedError,match='backtest Stats'):
        validate_components(engine,account_provider=None)


def test_explicit_value_observer_requires_an_authoritative_provider():
    from tradelearn_runtime.components import validate_components
    engine=bt.Cerebro();engine.addobserver(bt.observers.Value)
    with pytest.raises(ValueError,match='authoritative'):
        validate_components(engine,account_provider=None)


def test_signal_specs_are_normalized_without_mutating_cerebro_or_reimplementing_signal_logic():
    from tradelearn_runtime.components import strategy_spec
    from tradelearn.engine.signal import SignalStrategy,SIGNAL_LONG
    engine=bt.Cerebro();engine.add_signal(SIGNAL_LONG,object)
    before=list(engine.signals)
    cls,args,parameters=strategy_spec(engine)
    assert cls is SignalStrategy and args==()
    assert parameters['signals']==before and engine.signals==before and engine.strats==[]
    engine.addstrategy(bt.Strategy)
    with pytest.raises(ValueError,match='one.*strategy|single'):
        strategy_spec(engine)


def test_components_stop_every_started_component_when_one_stop_raises():
    from tradelearn_runtime.components import LiveComponents
    calls=[]
    class Analyzer(bt.Analyzer):
        def stop(self):calls.append('analyzer');raise ValueError('component stop failed')
    class Observer(bt.Observer):
        def stop(self):calls.append('observer')
    engine=bt.Cerebro(stdstats=False);engine.addanalyzer(Analyzer);engine.addobserver(Observer)
    runtime=StrategyRuntime(bt.Strategy,feeds={'A':'bar'});components=LiveComponents(engine,runtime)
    runtime.start();components.start();runtime.stop()
    with pytest.raises(ValueError,match='component stop failed'):components.finish()
    assert calls==['analyzer','observer']
    components.finish()


def test_native_stop_preserves_business_exception_when_components_also_fail():
    from tradelearn_runtime.execution import ExecutionHost
    calls=[]
    def primary():raise ValueError('primary business stop')
    def secondary():calls.append('finish');raise OSError('secondary component stop')
    native=NS(frontend=NS(stop=primary,_cerebro_components=NS(finish=secondary)))
    with pytest.raises(ValueError,match='primary business stop'):ExecutionHost.on_strategy_stop(native)
    assert calls==['finish']


def test_native_store_components_and_signal_strategy_replay_use_one_runtime(tmp_path):
    from tradelearn_runtime.execution import create_execution_host
    from nautilus_trader.backtest import BacktestEngine,BacktestEngineConfig
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.common import Environment
    from nautilus_trader.live import LiveNodeConfig
    from nautilus_trader.adapters.binance import (BinanceDataClientConfig,BinanceDataClientFactory,
        BinanceExecutionClientConfig,BinanceExecutionClientFactory,BinanceEnvironment)
    from nautilus_trader.model import (AccountId,TraderId,StrategyId,Bar,BarType,OmsType,AccountType,Money,Currency)
    from nautilus_trader.execution import FixedFeeModel
    from nautilus_trader.testkit.providers import TestInstrumentProvider
    from tradelearn.engine.signal import SignalStrategy,SIGNAL_LONG
    instrument=TestInstrumentProvider.btcusdt_perp_binance();symbol=str(instrument.id)
    node=LiveNodeConfig(trader_id=TraderId('TL-001'),environment=Environment.SANDBOX,
        data_clients={'BINANCE':BinanceDataClientConfig(environment=BinanceEnvironment.TESTNET)},
        exec_clients={'BINANCE':BinanceExecutionClientConfig(environment=BinanceEnvironment.TESTNET,account_id=AccountId('BINANCE-001'))})
    store=NautilusStore(node,data_factories={'BINANCE':BinanceDataClientFactory()},
        exec_factories={'BINANCE':BinanceExecutionClientFactory()},state_dir=tmp_path)
    engine=bt.Cerebro();engine.setbroker(store.getbroker())
    bar_type=BarType.from_str(symbol+'-1-SECOND-LAST-EXTERNAL');engine.adddata(store.getdata(bar_type))
    class Sign:
        def __init__(self):self.data=__import__('tradelearn.engine.base',fromlist=['_G'])._G.current_data
        def __getitem__(self,index):return 1 if len(self.data)==1 else -1
    engine.add_signal(SIGNAL_LONG,Sign)
    class Stream(bt.Analyzer):
        is_streaming=True
        def __init__(self):super().__init__();self.kinds=[]
        def on_broker_event(self,event):self.kinds.append(event.kind)
        def get_analysis(self):return {'kinds':self.kinds}
    engine.addanalyzer(Stream,_name='stream');engine.addobserver(bt.observers.BuySell,_name='fills')
    engine.addobserver(bt.observers.Trades,_name='trades')
    store._validate(engine,{});store._engine=engine
    cls,args,parameters=store.strategy_spec
    bridge=create_execution_host(StrategyConfig(strategy_id=StrategyId('TL-001')),
        business_cls=cls,strategy_args=args,parameters=parameters,feeds={symbol:'bar'},instruments={symbol:instrument.id},
        subscriptions=[bar_type],account_id='BINANCE-001',offline_test=True,
        frontend_broker=store.getbroker(),runtime_observer_factory=store.prepare_runtime)
    native=BacktestEngine(BacktestEngineConfig(bypass_logging=True,run_analysis=False))
    native.add_venue(instrument.id.venue,OmsType.NETTING,AccountType.MARGIN,
        [Money(1000000,Currency.from_str('USDT'))],fee_model=FixedFeeModel(Money(0,Currency.from_str('USDT'))))
    native.add_instrument(instrument)
    native.add_data([Bar(bar_type,*[instrument.make_price(1000)]*4,instrument.make_qty(10),
        (i+1)*1_000_000_000,(i+1)*1_000_000_000) for i in range(3)])
    try:
        native.add_strategy(bridge);native.run()
        business=bridge.frontend.strategy
        assert isinstance(business,SignalStrategy) and business.broker is store.getbroker()
        assert business.observers.fills.values[-1]==2 and business.observers.trades.values[-1]==1
        assert engine.analyzer_results['stream']['kinds'].count('fill')==2
        orders=sorted(native.cache.orders(),key=lambda order:order.ts_init)
        assert [order.is_reduce_only for order in orders]==[False,True]
        assert orders[1].position_id is not None
        assert len(engine.signals)==1 and engine.strats==[]
    finally:native.dispose()


def test_native_fixed_size_maps_quantity_without_requesting_fake_economics():
    from tradelearn_runtime.components import native_sizer,validate_components
    engine=bt.Cerebro(stdstats=False)
    engine.addsizer(bt.sizers.FixedSize,stake=11,tranches=3)
    assert native_sizer(engine).getsizing(None,True)==3
    engine.addsizer(bt.sizers.PercentSizer)
    with pytest.raises(NotImplementedError,match='FixedSize only'):
        validate_components(engine,account_provider=None)
