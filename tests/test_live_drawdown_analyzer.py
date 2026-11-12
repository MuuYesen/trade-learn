"""BT-compatible incremental observed-value DrawDown; no investment-return claim."""
from types import SimpleNamespace as NS
import pandas as pd
import pytest
import backtrader as official
from tradelearn import engine as bt
from tradelearn.backtest.event_runtime import StrategyRuntime
from tradelearn_runtime.broker import NautilusBroker
from tradelearn_runtime.components import LiveComponents,validate_components


def run_backtest(api):
    frame=pd.DataFrame(dict(open=[10.,12.,9.,8.,11.],high=[10.,12.,9.,8.,11.],
        low=[10.,12.,9.,8.,11.],close=[10.,12.,9.,8.,11.],volume=[100.]*5),
        index=pd.date_range('2026-01-01',periods=5))
    class Business(api.Strategy):
        def next(self):
            if len(self)==1:self.buy(size=1)
    engine=api.Cerebro(stdstats=False);engine.broker.setcash(100)
    engine.adddata(api.feeds.PandasData(dataname=frame));engine.addstrategy(Business)
    engine.addanalyzer(api.analyzers.DrawDown,_name='draw')
    return engine.run()[0].analyzers.draw.get_analysis()


def test_real_bt_backtest_result_fields_units_and_attribute_access_match():
    expected=run_backtest(official);actual=run_backtest(bt)
    assert set(actual)==set(expected)
    for key in ['len','drawdown','moneydown']:assert actual[key]==pytest.approx(expected[key])
    assert dict(actual['max'])==pytest.approx(dict(expected['max']))
    assert actual.max.drawdown==pytest.approx(expected.max.drawdown)


def test_live_incremental_observed_equity_has_no_backtest_stats_or_recalculation():
    current={'cash':7.,'equity':100.};calls=[]
    def account():calls.append(dict(current));return dict(current)
    broker=NautilusBroker(NS(account=account))
    runtime=StrategyRuntime(bt.Strategy,feeds={'A':'bar'},broker=broker)
    engine=bt.Cerebro(stdstats=False);engine.addanalyzer(bt.analyzers.DrawDown,_name='draw')
    components=LiveComponents(engine,runtime,account_provider=account)
    runtime.start();components.start()
    for day,value in enumerate([100.,120.,90.,110.],1):
        current['equity']=value
        runtime.step_market('A',dict(ts=f'2026-01-{day:02}T00:00:00Z',open=10,high=10,low=10,close=10,volume=1))
        components.after_market()
    runtime.stop();components.finish();components.finish()
    result=engine.analyzer_results['draw']
    assert result.drawdown==pytest.approx(100/12)
    assert result.moneydown==10 and result.len==2
    assert dict(result.max)==dict(drawdown=25.,moneydown=30.,len=2)
    assert len(calls)==4
    assert getattr(runtime.strategy,'stats',None) is None


def test_missing_live_value_provider_is_not_replaced_with_fake_money():
    engine=bt.Cerebro(stdstats=False);engine.addanalyzer(bt.analyzers.DrawDown)
    with pytest.raises(ValueError,match='authoritative.*provider'):
        validate_components(engine,account_provider=None)


def test_boolean_equity_is_not_a_real_valuation():
    broker=NautilusBroker(NS(account=lambda:{'cash':7.,'equity':True}))
    with pytest.raises(ValueError,match='equity'):
        broker.getvalue()
