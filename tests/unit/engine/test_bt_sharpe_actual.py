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
