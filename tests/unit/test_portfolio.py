from __future__ import annotations

import math

import pandas as pd

from tradelearn.portfolio import EqualWeightOptimizer, RiskPolicy, TopKSelector, select_top


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
    adjusted = RiskPolicy(max_weight=0.4, normalize=True).apply(weights)

    assert selected == ["MSFT", "GOOG"]
    assert adjusted.to_dict() == {"MSFT": 0.5, "GOOG": 0.5}
