"""Consistency checks for metrics against the 1.x vendored empyrical code."""

import importlib.util
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
    rank_ic,
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


def test_factor_rank_ic_matches_vendored_alphalens() -> None:
    """rank_ic matches the 1.x alphalens Spearman IC implementation."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB", "CCC"]], names=["date", "asset"])
    factor_data = pd.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "1D": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        },
        index=index,
    )

    alphalens = _load_legacy_alphalens_performance()
    expected = alphalens.factor_information_coefficient(factor_data)["1D"]
    expected.name = "rank_ic"

    result = rank_ic(factor_data["factor"], factor_data["1D"])

    pd.testing.assert_series_equal(result, expected, check_freq=False)


def _load_legacy_alphalens_performance() -> types.ModuleType:
    """Load vendored alphalens performance without importing old package initializers."""
    package_name = "legacy_alphalens"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "tradelearn" / "strategy" / "examine" / "alphalens")]
    sys.modules[package_name] = package

    previous_modules = {
        name: sys.modules.get(name)
        for name in [
            "tradelearn.strategy",
            "tradelearn.strategy.evaluate",
            "tradelearn.strategy.evaluate.empyrical",
        ]
    }
    sys.modules["tradelearn.strategy"] = types.ModuleType("tradelearn.strategy")
    sys.modules["tradelearn.strategy.evaluate"] = types.ModuleType("tradelearn.strategy.evaluate")
    sys.modules["tradelearn.strategy.evaluate.empyrical"] = types.ModuleType(
        "tradelearn.strategy.evaluate.empyrical"
    )
    try:
        for module_name in ["utils", "performance"]:
            module_path = (
                ROOT / "tradelearn" / "strategy" / "examine" / "alphalens" / f"{module_name}.py"
            )
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.{module_name}",
                module_path,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load legacy alphalens module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"{package_name}.{module_name}"] = module
            spec.loader.exec_module(module)
        return sys.modules[f"{package_name}.performance"]
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
