"""Actual pinned BT Sharpe defaults and period/risk-free profiles."""

import backtrader as official
import pytest

from tests.unit.engine.test_bt_returns_actual import returns_calls
from tradelearn import engine as tl


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"timeframe": "Days"},
        {"timeframe": "Days", "factor": 365, "annualize": True},
        {"timeframe": "Months", "stddev_sample": True},
        {"timeframe": "Weeks", "convertrate": False, "factor": 4},
        {"timeframe": "Seconds", "compression": 60, "riskfreerate": 0.02, "annualize": True},
        {"timeframe": "NoTimeFrame"},
        {"fund": False},
        {"timeframe": "Days", "daysfactor": 120, "factor": 365},
    ],
)
def test_sharpe_defaults_periods_rate_conversion_and_sample_variance_match_bt(options):
    expected = returns_calls(official, options, analyzer="SharpeRatio")
    actual = returns_calls(tl, options, analyzer="SharpeRatio")
    if expected["sharperatio"] is None:
        assert actual == expected
    else:
        assert actual == pytest.approx(expected)


def legacy_annual_call(api, options, flat=False):
    import pandas as pd
    frame=pd.DataFrame({name:([100.]*6 if flat else [110.,120.,140.,150.,170.,160.]) for name in ('open','high','low','close','volume')},index=pd.to_datetime(['2023-01-01','2023-12-31','2024-01-01','2024-12-31','2025-01-01','2025-12-31']))
    state={'value':100.}
    class Business(api.Strategy):
        def __init__(self):self.broker.getvalue=lambda:state['value']
        def next(self):state['value']=float(self.data.close[0])
    engine=api.Cerebro();engine.broker.setcash(100.)
    engine.adddata(api.feeds.PandasData(dataname=frame));engine.addstrategy(Business)
    engine.addanalyzer(api.analyzers.SharpeRatio,_name='ratio',legacyannual=True,**options)
    return dict(engine.run()[0].analyzers.ratio.get_analysis())


@pytest.mark.parametrize('options',[{}, {'riskfreerate':.03}, {'factor':365,'convertrate':False,'stddev_sample':True,'annualize':True}, {'fund':True}])
def test_legacyannual_uses_postbusiness_broker_annual_values_and_ignores_nonlegacy_controls(options):
    assert legacy_annual_call(tl,options)==pytest.approx(legacy_annual_call(official,options))


def test_legacyannual_zero_variance_has_same_bt_failure_instead_of_silent_none():
    for api in (official,tl):
        with pytest.raises(ZeroDivisionError):legacy_annual_call(api,{},flat=True)
