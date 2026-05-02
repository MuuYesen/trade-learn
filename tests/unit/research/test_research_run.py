from __future__ import annotations

import pandas as pd
import pytest

import tradelearn.research as research
from tradelearn.research import ResearchRun, ResearchStep
from tradelearn.research.portfolio import (
    Constraints,
    EqualWeight,
    TopK,
    WeightBuilder,
    apply_constraints,
    equal_weight,
    select_top,
    topk_equal_weights,
)
from tradelearn.research.preprocess import (
    MADWinsorizer,
    Neutralizer,
    StandardScaler,
    Winsorizer,
    clip_outliers,
    fill_by_group,
    fill_missing,
    label_by_quantile,
    rank,
)


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
        processed = Winsorizer(columns=["alpha"], limits=(0.25, 0.75)).fit_transform(
            processed
        )
        processed = Neutralizer(columns=["alpha"], method="ols").fit_transform(
            processed,
            exposures=exposures,
        )
        processed = StandardScaler(columns=["alpha"]).fit_transform(processed)
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
        "Winsorizer",
        "Neutralizer",
        "StandardScaler",
        "select_top",
        "equal_weight",
        "apply_constraints",
    ]
    assert isinstance(result.steps[0], ResearchStep)
    assert result.params["fill_missing.columns"] == ["alpha"]
    assert result.params["Neutralizer.method"] == "ols"
    assert result.params["select_top.k"] == 2
    assert result.params["equal_weight.gross"] == 0.8
    assert result.weights.abs().sum() == 1.0
    assert result.as_weight_dict() == {
        str(symbol): float(weight)
        for symbol, weight in result.weights.items()
    }


def test_topk_equal_weights_builds_multi_period_weight_panel() -> None:
    scores = pd.Series(
        [0.1, 0.3, 0.4, 0.2],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "BBB"),
            ],
            names=["timestamp", "symbol"],
        ),
        name="score",
    )

    weights = topk_equal_weights(scores, k=1, gross=0.9, max_weight=0.8)

    assert weights.index.names == ["timestamp", "symbol"]
    assert weights.name == "weight"
    assert weights.to_dict() == {
        (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"): 0.8,
        (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"): 0.8,
    }

    with ResearchRun("weights-helper") as run:
        topk_equal_weights(scores, k=1, gross=0.9)
        result = run.finish()

    assert [step.name for step in result.steps] == ["topk_equal_weights"]


def test_portfolio_functions_pipe_multi_period_scores_to_weights() -> None:
    scores = pd.Series(
        [0.1, 0.3, 0.4, 0.2],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "BBB"),
            ],
            names=["timestamp", "symbol"],
        ),
        name="score",
    )

    selected = select_top(scores, k=1)
    weights = equal_weight(selected, gross=0.9)
    weights = apply_constraints(weights, max_weight=0.8)

    assert selected.index.names == ["timestamp", "symbol"]
    assert selected.name == "selected"
    assert weights.index.names == ["timestamp", "symbol"]
    assert weights.name == "weight"
    assert weights.to_dict() == {
        (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"): 0.8,
        (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"): 0.8,
    }

    with ResearchRun("panel-selected") as run:
        result = run.finish(selected=selected, weights=weights)

    assert isinstance(result.selected, pd.Series)
    assert result.selected.index.names == ["timestamp", "symbol"]
    assert result.to_dict()["result"]["selected"] == {
        str((pd.Timestamp("2024-01-01", tz="UTC"), "BBB")): True,
        str((pd.Timestamp("2024-01-02", tz="UTC"), "AAA")): True,
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


def test_research_preprocess_namespace_supports_lightweight_tools() -> None:
    features = pd.DataFrame(
        {"alpha": [1.0, 100.0, None], "quality": [0.1, 0.2, 0.3]},
        index=["AAA", "BBB", "CCC"],
    )

    with ResearchRun("preprocess-tools") as run:
        processed = ResearchRun.preprocess.fill_missing(
            features,
            columns=["alpha"],
            method="median",
        )
        clipped = ResearchRun.preprocess.clip_outliers(
            processed,
            columns=["alpha"],
            limits=(0.0, 0.5),
        )
        ranked = ResearchRun.preprocess.rank(clipped["alpha"])
        result = run.finish(features=clipped, scores=ranked)

    assert processed["alpha"].isna().sum() == 0
    assert clipped["alpha"].max() == 50.5
    assert ranked.name == "alpha"
    assert ranked.loc["AAA"] < ranked.loc["CCC"]
    assert [step.name for step in result.steps] == [
        "fill_missing",
        "clip_outliers",
        "rank",
    ]


def test_preprocess_grouped_by_multiindex_date() -> None:
    index = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            ["AAA", "BBB", "CCC"],
        ],
        names=["date", "symbol"],
    )
    features = pd.DataFrame(
        {
            "alpha": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0],
            "value": [1.0, None, 3.0, 4.0, None, 6.0],
        },
        index=index,
    )

    filled = fill_missing(features, columns=["value"], method="median", by="date")
    clipped = clip_outliers(filled, columns=["alpha"], limits=(0.0, 0.5), by="date")
    ranked = rank(clipped, columns=["alpha"], by="date")
    standardized = StandardScaler(columns=["alpha"], by="date").fit_transform(clipped)

    assert filled.loc[(index[0][0], "BBB"), "value"] == 2.0
    assert clipped.loc[(index[0][0], "CCC"), "alpha"] == 2.0
    assert ranked.groupby(level="date")["alpha"].max().round(12).tolist() == [
        0.833333333333,
        0.833333333333,
    ]
    means = standardized.groupby(level="date")["alpha"].mean().round(12)
    assert means.tolist() == [0.0, 0.0]


def test_preprocess_migrates_1x_group_fill_mad_winsorize_and_quantile_labels() -> None:
    features = pd.DataFrame(
        {
            "code": ["AAA", "BBB", "CCC", "DDD"],
            "alpha": [1.0, None, 100.0, -100.0],
            "ind_code": ["tech", "tech", "finance", "finance"],
            "return": [0.05, -0.01, 0.20, -0.10],
        }
    )

    with ResearchRun("migrated-preprocess") as run:
        filled = fill_by_group(
            features,
            columns=["alpha"],
            group="ind_code",
            fallback="mean",
        )
        clipped = MADWinsorizer(columns=["alpha"], scale=1.0).fit_transform(filled)
        labeled = label_by_quantile(
            clipped,
            target="return",
            positive_quantile=0.75,
            negative_quantile=0.25,
        )
        result = run.finish(features=labeled)

    assert filled.loc[1, "alpha"] == 1.0
    assert clipped["alpha"].max() <= 100.0
    assert clipped["alpha"].min() >= -100.0
    assert labeled["label"].isna().tolist() == [True, True, False, False]
    assert labeled["label"].dropna().tolist() == [1, 0]
    assert [step.name for step in result.steps] == [
        "fill_by_group",
        "MADWinsorizer",
        "label_by_quantile",
    ]
    assert result.params["fill_by_group.group"] == "ind_code"
    assert result.params["MADWinsorizer.scale"] == 1.0
    assert result.params["label_by_quantile.target"] == "return"


def test_winsorizer_uses_train_bounds_on_test_data() -> None:
    train = pd.DataFrame({"alpha": [1.0, 2.0, 100.0]})
    test = pd.DataFrame({"alpha": [-100.0, 3.0, 200.0]})

    transformer = Winsorizer(columns=["alpha"], limits=(0.0, 0.5))
    fitted = transformer.fit(train)
    transformed = transformer.transform(test)

    assert fitted is transformer
    assert transformed["alpha"].tolist() == [1.0, 2.0, 2.0]
    assert transformer.get_params() == {
        "type": "Winsorizer",
        "columns": ["alpha"],
        "limits": (0.0, 0.5),
        "by": None,
    }


def test_mad_winsorizer_uses_train_bounds_on_test_data() -> None:
    train = pd.DataFrame({"alpha": [1.0, 2.0, 100.0]})
    test = pd.DataFrame({"alpha": [-100.0, 3.0, 200.0]})

    transformer = MADWinsorizer(columns=["alpha"], scale=1.0)
    fitted = transformer.fit(train)
    transformed = transformer.transform(test)

    assert fitted is transformer
    assert transformed["alpha"].tolist() == [1.0, 3.0, 3.0]
    assert transformer.get_params() == {
        "type": "MADWinsorizer",
        "columns": ["alpha"],
        "scale": 1.0,
        "by": None,
    }


def test_research_pipeline_fits_train_and_transforms_test() -> None:
    train = pd.DataFrame(
        {"alpha": [1.0, 2.0, 100.0], "size": [1.0, 2.0, 3.0]}
    )
    test = pd.DataFrame(
        {"alpha": [-100.0, 3.0, 200.0], "size": [1.0, 2.0, 3.0]}
    )

    with ResearchRun("pipeline") as run:
        preprocess = research.Pipeline(
            [
                Winsorizer(columns=["alpha"], limits=(0.0, 0.5)),
                Neutralizer(columns=["alpha"], exposures=["size"]),
                StandardScaler(columns=["alpha"]),
            ]
        )
        train_out = preprocess.fit_transform(train)
        test_out = preprocess.transform(test)
        result = run.finish(features=test_out)

    assert list(train_out.columns) == ["alpha", "size"]
    assert list(test_out.columns) == ["alpha", "size"]
    assert [step.name for step in result.steps] == [
        "Pipeline",
        "Winsorizer",
        "Neutralizer",
        "StandardScaler",
    ]
    assert result.params["Pipeline.steps"] == [
        "Winsorizer",
        "Neutralizer",
        "StandardScaler",
    ]
    assert preprocess.get_params()["steps"][0]["type"] == "Winsorizer"


def test_weight_builder_builds_panel_weights_from_scores() -> None:
    scores = pd.Series(
        [0.1, 0.3, 0.4, 0.2],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"),
                (pd.Timestamp("2024-01-02", tz="UTC"), "BBB"),
            ],
            names=["timestamp", "symbol"],
        ),
        name="score",
    )

    with ResearchRun("weights") as run:
        builder = WeightBuilder(
            select=TopK(k=1),
            weight=EqualWeight(gross=0.9),
            constrain=Constraints(max_weight=0.8),
        )
        weights = builder.build(scores)
        result = run.finish(scores=scores, weights=weights)

    assert weights.to_dict() == {
        (pd.Timestamp("2024-01-01", tz="UTC"), "BBB"): 0.8,
        (pd.Timestamp("2024-01-02", tz="UTC"), "AAA"): 0.8,
    }
    assert [step.name for step in result.steps] == ["WeightBuilder"]
    assert result.params["WeightBuilder.select.type"] == "TopK"
    assert result.params["WeightBuilder.weight.type"] == "EqualWeight"
    assert result.params["WeightBuilder.constrain.type"] == "Constraints"


def test_standard_scaler_fit_transform_matches_fit_then_transform() -> None:
    train = pd.DataFrame({"alpha": [1.0, 2.0, 3.0]})

    one_step = StandardScaler(columns=["alpha"]).fit_transform(train)
    transformer = StandardScaler(columns=["alpha"])
    two_step = transformer.fit(train).transform(train)

    assert one_step["alpha"].round(6).tolist() == two_step["alpha"].round(6).tolist()
    assert transformer.mean_ == {"alpha": 2.0}
    assert transformer.scale_ == {"alpha": 0.816496580927726}
    assert transformer.get_params() == {
        "type": "StandardScaler",
        "columns": ["alpha"],
        "ddof": 0,
        "by": None,
    }


def test_neutralizer_fits_train_exposures_and_reuses_coefficients() -> None:
    train = pd.DataFrame({"alpha": [3.0, 5.0, 7.0]})
    train_exposure = pd.DataFrame({"size": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"alpha": [9.0]})
    test_exposure = pd.DataFrame({"size": [4.0]})

    transformer = Neutralizer(columns=["alpha"]).fit(train, exposures=train_exposure)
    transformed = transformer.transform(test, exposures=test_exposure)

    assert abs(transformed.loc[0, "alpha"]) < 1e-12
    assert transformer.exposure_columns_ == ["size"]
    assert transformer.get_params() == {
        "type": "Neutralizer",
        "columns": ["alpha"],
        "exposures": None,
        "method": "ols",
    }


def test_neutralizer_can_use_exposure_columns_from_same_frame() -> None:
    train = pd.DataFrame({"alpha": [3.0, 5.0, 7.0], "size": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"alpha": [9.0], "size": [4.0]})

    transformer = Neutralizer(columns=["alpha"], exposures=["size"]).fit(train)
    transformed = transformer.transform(test)

    assert abs(transformed.loc[0, "alpha"]) < 1e-12
    assert transformer.exposure_columns_ == ["size"]
    assert transformer.get_params() == {
        "type": "Neutralizer",
        "columns": ["alpha"],
        "exposures": ["size"],
        "method": "ols",
    }


def test_preprocess_estimators_require_fit_before_transform() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        Winsorizer(columns=["alpha"]).transform(pd.DataFrame({"alpha": [1.0]}))

    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        StandardScaler(columns=["alpha"]).transform(pd.DataFrame({"alpha": [1.0]}))

    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        Neutralizer(columns=["alpha"]).transform(
            pd.DataFrame({"alpha": [1.0]}),
            exposures=pd.DataFrame({"size": [1.0]}),
        )
