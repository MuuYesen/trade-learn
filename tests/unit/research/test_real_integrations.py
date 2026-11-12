"""Small deterministic actual-library acceptance, with no market network."""

import pickle

import numpy as np
import pandas as pd
import pytest

from tradelearn.ml.causal import CausalSelector
from tradelearn.research.derive import SymbolicFeatureGenerator
from tradelearn.research.portfolio import Allocator, Constraints, RiskfolioWeight, TopK


def test_symbolic_real_expression_artifact_and_restore():
    pytest.importorskip("gplearn")
    from gplearn.genetic import SymbolicTransformer
    from sklearn.base import BaseEstimator

    original_transformer_validation = getattr(SymbolicTransformer, "_validate_data", None)
    original_base_validation = getattr(BaseEstimator, "_validate_data", None)
    x = np.linspace(1.0, 3.0, 24)
    frame = pd.DataFrame({"x": x, "y": x * x, "label": 2 * x + 1})
    generator = SymbolicFeatureGenerator(
        n_features=2,
        random_state=7,
        generations=2,
        population_size=20,
        hall_of_fame=5,
        stopping_criteria=2.0,
    )
    generator.fit(frame.iloc[:18], columns=["x", "y"], target="label")
    assert getattr(SymbolicTransformer, "_validate_data", None) is original_transformer_validation
    assert getattr(BaseEstimator, "_validate_data", None) is original_base_validation
    output = generator.transform(frame.iloc[18:].drop(columns="label"))
    restored = pickle.loads(pickle.dumps(generator))
    pd.testing.assert_frame_equal(output, restored.transform(frame.iloc[18:].drop(columns="label")))
    artifact = generator.to_dict()
    assert len(artifact["expressions"]) == output.shape[1]
    assert artifact["input_columns"] == ["x", "y"]
    assert artifact["training"]["rows"] == 18
    assert len(artifact["candidates"]) == 40
    assert artifact["candidate_capture"] == "complete"


def test_real_riskfolio_allocator_and_constraints():
    pytest.importorskip("riskfolio")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.01, (40, 3)), columns=list("ABC"))
    allocator = Allocator(
        select=TopK(3), weight=RiskfolioWeight(gross=0.9), constrain=Constraints(max_weight=0.8)
    )
    weights = allocator.build(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}), returns=returns)
    assert weights.sum() == pytest.approx(0.9)
    assert (weights >= 0).all() and (weights <= 0.8).all()


@pytest.mark.parametrize("method", ["pc", "fci"])
def test_real_causal_pc_transform_without_label_and_restore(method):
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(7)
    x = rng.normal(size=80)
    frame = pd.DataFrame(
        {"x": x, "noise": rng.normal(size=80), "label": 3 * x + rng.normal(0, 0.1, 80)}
    )
    selector = CausalSelector(target="label", features=["x", "noise"], method=method)
    selector.fit(frame.iloc[:60])
    result = selector.transform(frame.iloc[60:].drop(columns="label"))
    assert "x" in result
    pd.testing.assert_frame_equal(
        result,
        pickle.loads(pickle.dumps(selector)).transform(frame.iloc[60:].drop(columns="label")),
    )


def test_real_optuna_keeps_failed_pruned_and_complete_trials():
    optuna = pytest.importorskip("optuna")
    from tradelearn.optimize import OptunaSearch
    from tradelearn.research import ResearchRun

    def objective(trial):
        value = trial.suggest_float("regularization", 0.1, 1.0)
        if trial.number == 0:
            raise ValueError("explicit failure")
        if trial.number == 1:
            raise optuna.TrialPruned("explicit prune")
        return value

    search = OptunaSearch(objective, n_trials=3, sampler=optuna.samplers.RandomSampler(seed=7))
    with ResearchRun("optuna-trials") as run:
        run.register("search", search)
        study = run.call("search", "run", n_jobs=1, timeout=10, catch=(ValueError,))
        records = [
            {"number": t.number, "state": t.state.name, "params": t.params, "value": t.value}
            for t in study.trials
        ]
        result = run.finish(
            artifacts={"trials": records, "budget": {"n_jobs": 1, "timeout": 10, "n_trials": 3}}
        )
    assert [t["state"] for t in result.artifacts["trials"]] == ["FAIL", "PRUNED", "COMPLETE"]
    assert study.best_trial.number == 2
