from __future__ import annotations

import sys
import types

import pytest

import tradelearn as tl
from tradelearn.optimize import OptunaBacktestSearch, OptunaSearch


def test_optimize_namespace_is_lazy_public_alias() -> None:
    assert tl.optimize.OptunaSearch is OptunaSearch


def test_optuna_search_requires_optional_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "optuna", None)
    search = OptunaSearch(lambda trial: 1.0, n_trials=1)

    with pytest.raises(ImportError, match="trade-learn\\[optuna\\]"):
        search.run()


def test_optuna_search_runs_study(monkeypatch) -> None:
    calls = {}

    class FakeStudy:
        def optimize(self, objective, *, n_trials, **kwargs):
            calls["n_trials"] = n_trials
            calls["value"] = objective(types.SimpleNamespace())

    def create_study(**kwargs):
        calls["study_kwargs"] = kwargs
        return FakeStudy()

    monkeypatch.setitem(sys.modules, "optuna", types.SimpleNamespace(create_study=create_study))
    search = OptunaSearch(lambda trial: 2.5, direction="maximize", n_trials=3)

    study = search.run()

    assert isinstance(study, FakeStudy)
    assert calls["n_trials"] == 3
    assert calls["value"] == 2.5
    assert calls["study_kwargs"]["direction"] == "maximize"


def test_optuna_backtest_search_builds_params_and_reads_metric(monkeypatch) -> None:
    class FakeTrial:
        def suggest_int(self, name, low, high, **kwargs):
            assert (name, low, high) == ("lookback", 5, 20)
            return 10

        def suggest_categorical(self, name, choices):
            assert name == "mode"
            return choices[0]

    class FakeStudy:
        def optimize(self, objective, *, n_trials, **kwargs):
            self.value = objective(FakeTrial())

    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(create_study=lambda **kwargs: FakeStudy()),
    )

    def backtest_factory(params):
        assert params == {"lookback": 10, "mode": "fast"}
        return types.SimpleNamespace(run=lambda: {"sharpe": 1.25})

    search = OptunaBacktestSearch(
        backtest_factory=backtest_factory,
        param_space={"lookback": {"type": "int", "low": 5, "high": 20}, "mode": ["fast", "slow"]},
        metric="sharpe",
        n_trials=1,
    )

    study = search.run()

    assert study.value == 1.25
