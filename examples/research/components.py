"""Optional components on explicit train/validation data; Test is untouched.

Use only authorized local Bars. This example never publishes or trades.
"""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from tradelearn import research
from tradelearn.ml import CausalSelector
from tradelearn.optimize import OptunaSearch
from tradelearn.research.derive import SymbolicFeatureGenerator
from tradelearn.research.temporal import assessment, forward_return, purged_splits


def returns(panel):
    return panel.close.pct_change(1, fill_method=None)


def momentum(panel):
    return panel.close.pct_change(3, fill_method=None)


def run_components(bars, *, train_end, validation_end):
    """Run bounded symbolic/causal/model selection and one risk allocation.

    The two-observation label determines the common conservative purge; both
    horizons are assessed only within validation, never the reserved Test.
    Small budgets demonstrate wiring, not an economically validated strategy.
    """
    feature_set = research.FeatureSet({"return": returns, "momentum": momentum})
    symbolic = SymbolicFeatureGenerator(
        n_features=1,
        random_state=7,
        generations=2,
        population_size=20,
        hall_of_fame=5,
        stopping_criteria=2.0,
    )
    selector = CausalSelector(
        features=["return", "momentum", "symbolic_1"],
        target="label",
        method="correlation",
        max_features=2,
    )
    pipeline = research.Pipeline(
        [research.preprocess.StandardScaler(columns=["return", "momentum", "symbolic_1"])]
    )
    allocator = research.Allocator(
        select=research.TopK(2), weight=research.RiskfolioWeight(gross=0.8)
    )
    with research.ResearchRun("optional-components") as run:
        for name, component in (
            ("features", feature_set),
            ("symbolic", symbolic),
            ("selector", selector),
            ("pipeline", pipeline),
            ("allocator", allocator),
        ):
            run.register(name, component)
        factors = run.call("features", "transform", bars)
        labels = {horizon: forward_return(bars, horizon=horizon) for horizon in (1, 2)}
        sections = purged_splits(labels[2], train_end=train_end, validation_end=validation_end)
        train = factors.loc[sections["train"]].dropna()
        validation = factors.loc[sections["validation"]].dropna()
        if len(train) < 3 or len(validation) < 3:
            raise ValueError("insufficient mature train/validation observations")
        target_train = labels[1].loc[train.index, "forward_return"]
        target_validation = labels[1].loc[validation.index, "forward_return"]
        run.call(
            "symbolic",
            "fit",
            train.assign(label=target_train),
            columns=["return", "momentum"],
            target="label",
        )
        train = train.join(run.call("symbolic", "transform", train))
        validation = validation.join(run.call("symbolic", "transform", validation))
        train = run.call("pipeline", "fit_transform", train)
        validation = run.call("pipeline", "transform", validation)
        run.call("selector", "fit", train.assign(label=target_train))
        train_selected = run.call("selector", "transform", train)
        validation_selected = run.call("selector", "transform", validation)
        if train_selected.shape[1] == 0:
            raise ValueError("causal selector returned no features")

        def objective(trial):
            model = Ridge(alpha=trial.suggest_float("alpha", 0.1, 2.0))
            name = f"trial_model_{trial.number}"
            run.register(name, model)
            run.call(name, "fit", train_selected, target_train)
            prediction = run.call(name, "predict", validation_selected)
            return float(mean_squared_error(target_validation, prediction))

        import optuna

        search = OptunaSearch(
            objective,
            n_trials=3,
            direction="minimize",
            sampler=optuna.samplers.RandomSampler(seed=7),
        )
        run.register("search", search)
        study = run.call("search", "run", n_jobs=1, timeout=30)
        # Winning model was fit on Train only. No fit on Validation or Test.
        model = run.components[f"trial_model_{study.best_trial.number}"]
        scorer = research.ModelScorer(model, features=selector.selected_features_)
        run.register("scorer", scorer)
        scores = run.call("scorer", "predict", validation_selected)
        if scores.empty:
            raise ValueError("no current scores; no allocation intent")
        decision = validation_selected.index.get_level_values("timestamp").max()
        historical = bars.loc[bars.index.get_level_values("timestamp") <= decision]
        risk_returns = historical.close.unstack("symbol").pct_change(fill_method=None).dropna()
        weights = run.call("allocator", "build", scores, returns=risk_returns)
        reports = {
            str(h): assessment(
                factors.loc[validation.index],
                labels[h].loc[validation.index],
                factors=["return", "momentum"],
                min_pairs=3,
            )
            for h in (1, 2)
        }
        return run.finish(
            features=validation_selected,
            model=model,
            scores=scores,
            weights=weights,
            selected_features=selector.selected_features_,
            artifacts={
                "symbolic": symbolic.to_dict(),
                "causal": selector.to_dict(),
                "assessment": reports,
                "test_consumed": False,
                "split": {"train_end": str(train_end), "validation_end": str(validation_end)},
                "budget": {"n_trials": 3, "n_jobs": 1, "timeout": 30},
                "trials": [
                    {
                        "number": t.number,
                        "params": t.params,
                        "distributions": {
                            name: str(value) for name, value in t.distributions.items()
                        },
                        "state": t.state.name,
                        "value": t.value,
                        "started_at": None
                        if t.datetime_start is None
                        else t.datetime_start.isoformat(),
                        "finished_at": None
                        if t.datetime_complete is None
                        else t.datetime_complete.isoformat(),
                    }
                    for t in study.trials
                ],
            },
        )
