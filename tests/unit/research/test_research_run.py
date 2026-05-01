from __future__ import annotations

import pandas as pd

from tradelearn.research import ResearchRun, ResearchStep
from tradelearn.research.portfolio import apply_constraints, equal_weight, select_top
from tradelearn.research.preprocess import fill_missing, neutralize, standardize, winsorize


def test_research_run_records_preprocess_and_portfolio_function_params() -> None:
    features = pd.DataFrame(
        {
            "alpha": [1.0, 100.0, None, -50.0],
            "size": [10.0, 12.0, 14.0, 16.0],
        },
        index=["AAA", "BBB", "CCC", "DDD"],
    )
    exposures = features[["size"]].fillna(features["size"].median())

    with ResearchRun("index_enhance_v1") as run:
        processed = fill_missing(features, columns=["alpha"], method="median")
        processed = winsorize(processed, columns=["alpha"], limits=(0.25, 0.75))
        processed = neutralize(
            processed,
            exposures=exposures,
            columns=["alpha"],
            method="ols",
        )
        processed = standardize(processed, columns=["alpha"])
        scores = processed["alpha"]
        selected = select_top(scores.to_dict(), k=2)
        weights = equal_weight(selected, gross=0.8)
        weights = apply_constraints(weights, max_weight=0.4, normalize=True)
        result = run.finish(
            features=processed,
            scores=scores,
            selected=selected,
            weights=weights,
        )

    assert [step.name for step in result.steps] == [
        "fill_missing",
        "winsorize",
        "neutralize",
        "standardize",
        "select_top",
        "equal_weight",
        "apply_constraints",
    ]
    assert isinstance(result.steps[0], ResearchStep)
    assert result.params["fill_missing.columns"] == ["alpha"]
    assert result.params["neutralize.method"] == "ols"
    assert result.params["select_top.k"] == 2
    assert result.params["equal_weight.gross"] == 0.8
    assert result.weights.abs().sum() == 1.0
    assert result.as_weight_dict() == {
        str(symbol): float(weight)
        for symbol, weight in result.weights.items()
    }


def test_research_run_supports_static_method_tracking() -> None:
    features = pd.DataFrame({"alpha": [1.0, None]}, index=["AAA", "BBB"])

    with ResearchRun("static-demo") as run:
        processed = ResearchRun.preprocess.fill_missing(
            features,
            columns=["alpha"],
            method="zero",
        )
        result = run.finish(features=processed)

    assert result.steps[0].name == "fill_missing"
    assert result.params["fill_missing.method"] == "zero"
    assert processed["alpha"].tolist() == [1.0, 0.0]
