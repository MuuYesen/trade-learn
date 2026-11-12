"""Backtest producers preserve canonical values rather than presentation scaling."""
import numpy as np
import pandas as pd
import pytest
from tradelearn import metrics
from tradelearn.backtest import engine


@pytest.mark.parametrize('name,public', [('annual_return','annual_return'),('volatility','volatility'),('sharpe','sharpe'),('sortino','sortino'),('calmar','calmar')])
@pytest.mark.parametrize('values',[[],[0.,0.,0.],[.01,0.,0.],[.1,-.2,.05,-.1]])
def test_summary_metric_matches_shared_formula_including_degenerate_inputs(name,public,values):
    returns=pd.Series(values,dtype=float)
    expected=getattr(metrics,public)(returns,periods=252)
    actual=getattr(engine,'_summary_'+name)(returns,periods=252)
    if np.isnan(expected):assert np.isnan(actual)
    else:assert actual==pytest.approx(expected)


def test_summary_does_not_hide_metric_backend_errors(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('invalid input')
    monkeypatch.setattr(metrics.risk,'sharpe',fail)
    with pytest.raises(ValueError,match='invalid input'):
        engine._summary_sharpe(pd.Series([.1,-.1]))


def test_stats_and_report_share_values_units_and_missing_reasons():
    from tradelearn.lite import Backtest,Strategy
    from tradelearn.report import Reporter
    class S(Strategy):
        def next(self):pass
    frame=pd.DataFrame({k:[1.,2.,3.] for k in ['open','high','low','close','volume']},index=pd.date_range('2026-01-01',periods=3))
    stats=Backtest(frame,S).run()
    facts=Reporter(stats).metric_facts().set_index('name')
    for name in ('annual_return','max_drawdown'):
        assert stats.summary[name]==facts.loc[name,'value']
        assert stats.metric_metadata[name]['unit']==facts.loc[name,'unit']=='ratio'
    assert np.isnan(stats.summary['sharpe'])
    assert facts.loc['sharpe_ratio','reason']=='not_calculable'
    assert stats.metric_metadata['sharpe']['reason']=='not_calculable'


def test_summary_presentation_scales_canonical_ratios_once():
    from tradelearn.backtest.models import SummaryDict
    text=str(SummaryDict(annual_return=.12,volatility=.08,max_drawdown=-.05))
    assert '12.00' in text and '8.00' in text and '5.00' in text


@pytest.mark.parametrize('values',[[],[100.],[0.],[100.,-50.,0.,25.]])
def test_trade_summary_uses_shared_net_pnl_metrics_and_missing_rules(values):
    frame=pd.DataFrame({'pnlcomm':values,'isclosed':[True]*len(values)})
    result=engine._summary_trade_metrics(frame)
    for field in ('profit_factor','expectancy','avg_win','avg_loss'):
        expected=getattr(metrics,field)(pd.Series(values,dtype=float))
        if np.isnan(expected):assert np.isnan(result[field])
        else:assert result[field]==pytest.approx(expected)


def test_report_consumes_explicit_net_trade_basis_from_stats():
    from tradelearn.backtest.models import Stats
    from tradelearn.report import Reporter
    trades=pd.DataFrame({'pnl':[100.,-50.],'pnlcomm':[98.,-52.],'isclosed':[True,True]})
    stats=Stats(returns=pd.Series([0.,.01]),equity=pd.Series([100.,101.]),trades=trades,
        positions=pd.DataFrame(),orders=pd.DataFrame(),fills=pd.DataFrame(),config={},analyzers={},
        summary={'profit_factor':98/52})
    assert stats.metric_metadata['profit_factor']['pnl_basis']=='pnlcomm'
    assert Reporter(stats).summary()['profit_factor']==pytest.approx(stats.summary['profit_factor'])
    assert trades.pnl.tolist()==[100.,-50.]
