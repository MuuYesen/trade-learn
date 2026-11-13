"""Real common Runtime on explicitly declared periods and authoritative account values."""

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("nautilus_trader")
from tradelearn import engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.components import LiveComponents, validate_components


@pytest.mark.parametrize("explicit", [False, True])
def test_live_returns_counts_real_periods_without_stats_or_recalculating_history(explicit):
    state = {"cash": 100.0, "equity": 100.0}
    reads = []

    def account():
        reads.append(dict(state))
        return dict(state)

    broker = NautilusBroker(SimpleNamespace(account=account))
    from nautilus_trader.model import BarType, InstrumentId

    from tradelearn_runtime.execution import _feed_period_metadata

    declaration = BarType.from_str("EURUSD.SIM-1-DAY-MID-INTERNAL")
    metadata = _feed_period_metadata({"A": InstrumentId.from_str("EURUSD.SIM")}, [declaration])
    runtime = StrategyRuntime(
        bt.Strategy, feeds={"A": "bar"}, broker=broker, feed_metadata=metadata
    )
    e = bt.Cerebro(stdstats=False)
    options = (
        dict(timeframe=bt.TimeFrame.Days, compression=1, tann=365, fund=False) if explicit else {}
    )
    e.addanalyzer(bt.analyzers.Returns, _name="returns", **options)
    components = LiveComponents(e, runtime, account_provider=account)
    runtime.start()
    components.start()
    for day, value in enumerate([100.0, 110.0, 90.0, 120.0], 1):
        state["equity"] = value
        runtime.step_market(
            "A",
            dict(ts=f"2026-01-{day:02}T00:00:00Z", open=10, high=10, low=10, close=10, volume=1),
        )
        components.after_market()
    runtime.stop()
    components.finish()
    components.finish()
    result = e.analyzer_results["returns"]
    annual_periods = 365 if explicit else 252
    assert result == pytest.approx(
        dict(
            rtot=math.log(1.2),
            ravg=math.log(1.2) / 4,
            rnorm=math.expm1(math.log(1.2) / 4 * annual_periods),
            rnorm100=math.expm1(math.log(1.2) / 4 * annual_periods) * 100,
        )
    )
    assert len(reads) == 2
    assert getattr(runtime.strategy, "stats", None) is None
    assert runtime.strategy.analyzers.returns._period_count == 4


def test_live_returns_requires_an_authoritative_account_provider():
    e = bt.Cerebro(stdstats=False)
    e.addanalyzer(bt.analyzers.Returns, timeframe=bt.TimeFrame.Days, compression=1)
    with pytest.raises(ValueError, match="authoritative account"):
        validate_components(e, account_provider=None)


@pytest.mark.parametrize('options',[{}, {'riskfreerate':.03}, {'factor':365,'convertrate':False,'stddev_sample':True,'annualize':True}, {'fund':True}])
def test_live_legacyannual_sharpe_uses_bounded_annual_state_and_same_bt_observed_values(options):
    import backtrader as official
    from tests.unit.engine.test_bt_sharpe_actual import legacy_annual_call
    state={'cash':100.,'equity':100.}
    class Business(bt.Strategy):
        def next(self):state['equity']=float(self.data.close[0])
    account=lambda:dict(state)
    broker=NautilusBroker(SimpleNamespace(account=account))
    runtime=StrategyRuntime(Business,feeds={'A':'bar'},broker=broker,capacity=2)
    engine=bt.Cerebro(stdstats=False)
    engine.addanalyzer(bt.analyzers.SharpeRatio,_name='ratio',legacyannual=True,**options)
    components=LiveComponents(engine,runtime,account_provider=account)
    try:
        runtime.start()
        for day,value in zip(['2023-01-01','2023-12-31','2024-01-01','2024-12-31','2025-01-01','2025-12-31'],[110.,120.,140.,150.,170.,160.]):
            runtime.step_market('A',dict(ts=day+'T00:00:00Z',open=value,high=value,low=value,close=value,volume=1))
            components.after_market()
        runtime.stop();components.finish()
        analyzer=runtime.strategy.analyzers.ratio
        assert dict(analyzer.get_analysis())==pytest.approx(legacy_annual_call(official,options))
        assert analyzer._count==3
        assert all(len(values)==2 for values in runtime.feeds['A']._buffer.arrays.values())
        assert not hasattr(analyzer,'anret')
    finally:runtime.cleanup()
