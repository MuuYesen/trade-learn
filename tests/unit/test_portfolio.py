from __future__ import annotations

import math
import sys
import types

import pandas as pd
import pytest

from tradelearn.portfolio import (
    EqualWeightOptimizer,
    PortfolioConstraints,
    RiskfolioOptimizer,
    RiskPolicy,
    TopKSelector,
    select_top,
)


def test_select_top_returns_highest_score_keys() -> None:
    scores = {"AAPL": 0.10, "MSFT": 0.25, "GOOG": 0.18}

    assert select_top(scores, k=2) == ["MSFT", "GOOG"]


def test_select_top_filters_nan_and_min_score() -> None:
    scores = {"AAPL": math.nan, "MSFT": -0.01, "GOOG": 0.03}

    assert select_top(scores, k=3, min_score=0.0) == ["GOOG"]


def test_select_top_can_select_lowest_when_reverse_is_false() -> None:
    scores = {"AAPL": 0.10, "MSFT": 0.25, "GOOG": 0.18}

    assert select_top(scores, k=2, reverse=False) == ["AAPL", "GOOG"]


def test_select_top_can_filter_by_max_score() -> None:
    scores = {"AAPL": 0.10, "MSFT": 0.25, "GOOG": 0.18}

    assert select_top(scores, k=3, reverse=False, max_score=0.18) == ["AAPL", "GOOG"]


def test_portfolio_selector_optimizer_and_risk_policy_are_public() -> None:
    scores = pd.Series({"AAPL": 0.10, "MSFT": 0.25, "GOOG": 0.18})
    selected = TopKSelector(k=2).select(scores)
    weights = EqualWeightOptimizer(gross=0.8).optimize(selected, scores)
    adjusted = PortfolioConstraints(max_weight=0.4, normalize=True).apply(weights)

    assert selected == ["MSFT", "GOOG"]
    assert adjusted.to_dict() == {"MSFT": 0.5, "GOOG": 0.5}


def test_risk_policy_remains_compatibility_alias() -> None:
    assert RiskPolicy is PortfolioConstraints


def test_riskfolio_optimizer_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "riskfolio", None)
    optimizer = RiskfolioOptimizer()

    with pytest.raises(ImportError, match="trade-learn\\[riskfolio\\]"):
        optimizer.optimize(pd.DataFrame({"A": [0.01], "B": [0.02]}))


def test_riskfolio_optimizer_wraps_weights(monkeypatch) -> None:
    class FakePortfolio:
        def __init__(self, *, returns):
            self.returns = returns
            self.stats_kwargs = None

        def assets_stats(self, **kwargs):
            self.stats_kwargs = kwargs

        def optimization(self, **kwargs):
            assert kwargs["model"] == "Classic"
            assert kwargs["rm"] == "MV"
            assert kwargs["obj"] == "Sharpe"
            return pd.DataFrame({"weights": [0.6, 0.4]}, index=["A", "B"])

    fake = types.SimpleNamespace(Portfolio=FakePortfolio)
    monkeypatch.setitem(sys.modules, "riskfolio", fake)
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.02, 0.01]})

    weights = RiskfolioOptimizer().optimize(returns)

    assert weights.to_dict() == {"A": 0.6, "B": 0.4}
