import pandas as pd
import pytest

from tradelearn.research import portfolio as pf


def test_riskfolio_selection_passes_only_selected_returns(monkeypatch):
    seen = []

    def optimize(self, returns):
        seen.append(returns)
        return pd.Series({"A": 0.4, "B": 0.6})

    monkeypatch.setattr(pf.RiskfolioOptimizer, "optimize", optimize)
    allocator = pf.Allocator(
        select=pf.TopK(2), weight=pf.RiskfolioWeight(objective="min_variance", gross=0.8)
    )
    returns = pd.DataFrame({"A": [0.1, 0.2], "B": [0.2, 0.1], "C": [0.3, 0.4]})
    weights = allocator.build(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}), returns=returns)
    assert seen[0].columns.tolist() == ["A", "B"]
    assert weights.to_dict() == pytest.approx({"A": 0.32, "B": 0.48})


def test_riskfolio_rejects_missing_inputs_and_invalid_solution(monkeypatch):
    allocator = pf.Allocator(select=pf.TopK(2), weight=pf.RiskfolioWeight())
    with pytest.raises(ValueError, match="returns"):
        allocator.build(pd.Series({"A": 1.0}))
    monkeypatch.setattr(pf.RiskfolioOptimizer, "optimize", lambda *_: pd.Series({"A": -1.0}))
    with pytest.raises(ValueError, match="solution"):
        allocator.build(pd.Series({"A": 1.0}), returns=pd.DataFrame({"A": [0.1, 0.2]}))


def test_constraints_reject_normalization_that_breaks_cap():
    with pytest.raises(ValueError, match="max_weight"):
        pf.Constraints(max_weight=0.3, normalize=True).apply(pd.Series({"A": 0.5, "B": 0.5}))


def test_function_constraints_cannot_normalize_past_declared_cap():
    with pytest.raises(ValueError, match="max_weight"):
        pf.apply_constraints(pd.Series({"A": 0.5, "B": 0.5}), max_weight=0.3, normalize=True)


def test_riskfolio_constraints_cannot_silently_reduce_target_gross(monkeypatch):
    monkeypatch.setattr(
        pf.RiskfolioOptimizer, "optimize", lambda *_: pd.Series({"A": 0.7, "B": 0.3})
    )
    allocator = pf.Allocator(
        select=pf.TopK(2), weight=pf.RiskfolioWeight(), constrain=pf.Constraints(max_weight=0.5)
    )
    with pytest.raises(ValueError, match="gross"):
        allocator.build(
            pd.Series({"A": 2.0, "B": 1.0}),
            returns=pd.DataFrame({"A": [0.1, 0.2], "B": [0.2, 0.1]}),
        )
