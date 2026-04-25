"""Consistency checks for metrics against the 1.x vendored empyrical code."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from tradelearn.metrics import (
    annual_return,
    calmar,
    cum_returns,
    cvar,
    max_drawdown,
    omega,
    sharpe,
    simple_returns,
    tail_ratio,
    var,
    volatility,
)

ROOT = Path(__file__).resolve().parents[2]
sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
sys.modules.setdefault("pandas_datareader", types.ModuleType("pandas_datareader"))
sys.modules.setdefault("pandas_datareader.data", types.ModuleType("pandas_datareader.data"))
sys.path.insert(0, str(ROOT / "tradelearn" / "strategy" / "evaluate"))
from empyrical import stats as empyrical  # noqa: E402


def test_return_metrics_match_vendored_empyrical() -> None:
    """Implemented return metrics match the 1.x oracle."""
    prices = pd.Series([100.0, 95.0, 101.0, 99.0, 105.0])
    returns = simple_returns(prices)

    pd.testing.assert_series_equal(simple_returns(prices), empyrical.simple_returns(prices))
    pd.testing.assert_series_equal(cum_returns(returns), empyrical.cum_returns(returns))
    assert np.isclose(
        annual_return(returns, periods=252),
        empyrical.annual_return(returns, annualization=252),
        rtol=1e-10,
    )


def test_risk_metrics_match_vendored_empyrical() -> None:
    """Implemented risk metrics match the 1.x oracle on a drawdown-heavy fixture."""
    returns = pd.Series([-0.10, 0.05, -0.03, 0.08, -0.02, 0.04])

    assert np.isclose(max_drawdown(returns), empyrical.max_drawdown(returns), rtol=1e-10)
    assert np.isclose(volatility(returns, periods=252), empyrical.annual_volatility(
        returns,
        annualization=252,
    ), rtol=1e-10)
    assert np.isclose(sharpe(returns, periods=252), empyrical.sharpe_ratio(
        returns,
        annualization=252,
    ), rtol=1e-10)
    assert np.isclose(calmar(returns, periods=252), empyrical.calmar_ratio(
        returns,
        annualization=252,
    ), rtol=1e-10)
    assert np.isclose(omega(returns, threshold=0.0, periods=252), empyrical.omega_ratio(
        returns,
        annualization=252,
    ), rtol=1e-10)
    assert np.isclose(tail_ratio(returns), empyrical.tail_ratio(returns), rtol=1e-10)
    assert np.isclose(var(returns), empyrical.value_at_risk(returns), rtol=1e-10)
    assert np.isclose(
        cvar(returns),
        empyrical.conditional_value_at_risk(returns),
        rtol=1e-10,
    )
