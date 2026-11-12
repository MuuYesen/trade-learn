"""Shared explicit train/inference preparation for the two runtime examples."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from tradelearn import research
from tradelearn.data.bars import bars_fingerprint
from tradelearn.research.preprocess import StandardScaler
from tradelearn.research.temporal import forward_return, purged_splits


def momentum(panel):
    return panel.close.pct_change(3, fill_method=None)


def train(bars: pd.DataFrame, *, split):
    """Fit only mature training labels; preserve all post-split raw bars."""
    feature_set = research.FeatureSet({"momentum": momentum})
    pipeline = research.Pipeline([StandardScaler(columns=["momentum"])])
    model = Ridge(alpha=1.0)
    scorer = research.ModelScorer(model, features=["momentum"])
    allocator = research.Allocator(select=research.TopK(2), weight=research.EqualWeight(gross=0.8))
    boundary = pd.Timestamp(split)
    boundary = (
        boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
    )
    with research.ResearchRun("bounded-research-template") as run:
        for name, component in (
            ("feature_set", feature_set),
            ("pipeline", pipeline),
            ("model", model),
            ("scorer", scorer),
            ("allocator", allocator),
        ):
            run.register(name, component)
        features = run.call("feature_set", "transform", bars)
        labels = forward_return(bars, horizon=2)
        sections = purged_splits(labels, train_end=boundary)
        train_features = features.loc[sections["train"]].dropna()
        if train_features.empty:
            raise ValueError("no mature training features before split")
        train_ready = run.call("pipeline", "fit_transform", train_features)
        run.call(
            "model",
            "fit",
            train_ready[["momentum"]],
            labels.loc[train_ready.index, "forward_return"],
        )
        _, test = research.time_split(features, split=boundary)
        test_ready = run.call("pipeline", "transform", test)
        scores = run.call("scorer", "predict", test_ready)
        weights = run.call("allocator", "build", scores) if not scores.empty else None
        prior = labels.index.get_level_values("timestamp") < boundary
        result = run.finish(
            features=test_ready,
            target=labels.loc[train_ready.index],
            model=model,
            scores=scores,
            weights=weights,
            artifacts={
                "label": {
                    "horizon": 2,
                    "unit": "observations",
                    "split": boundary.isoformat(),
                    "purged_rows": int(prior.sum() - len(sections["train"])),
                    "max_train_label_end": labels.loc[train_ready.index, "label_end"]
                    .max()
                    .isoformat(),
                    "train_rows": len(train_ready),
                },
                "source": dict(bars.attrs),
                "input_fingerprint": bars_fingerprint(bars),
                "window_size": 4,
                "input_columns": list(bars.columns),
            },
        )
    bundle = {
        "feature_set": feature_set,
        "pipeline": pipeline,
        "model": model,
        "scorer": scorer,
        "allocator": allocator,
        "window_size": 4,
        "metadata": result.artifacts,
        "input_schema": {str(name): str(dtype) for name, dtype in bars.dtypes.items()},
    }
    return bundle, result, research.split_bars(bars, split=boundary)


def infer(bundle, bars):
    """Apply fitted state only. Empty input is no decision, never liquidation."""
    features = bundle["pipeline"].transform(bundle["feature_set"].transform(bars))
    scores = bundle["scorer"].predict(features)
    return None if scores.empty else bundle["allocator"].build(scores)


def inputs():
    import argparse
    from pathlib import Path

    from tradelearn.data import open_dataset
    from tradelearn.ml.registry import load_bundle, save_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="authorized CSV/Parquet dataset glob")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", default="examples/output/research")
    parser.add_argument("--mlflow-uri", help="explicit post-run tracking destination")
    parser.add_argument("--experiment-name", default="research-template")
    args = parser.parse_args()
    with open_dataset(args.data) as dataset:
        bars = dataset.read()
    bundle, result, test_bars = train(bars, split=args.split)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_bundle(
        bundle,
        output / "bundle",
        code_paths={
            "examples/research/_workflow.py": __file__,
            "examples/research/__init__.py": Path(__file__).parent / "__init__.py",
            "examples/__init__.py": Path(__file__).parent / "__init__.py",
        },
    )
    result.target.to_parquet(output / "train-labels.parquet")
    (output / "research.json").write_text(
        __import__("json").dumps(result.to_dict(), indent=2, default=str)
    )
    restored = load_bundle(output / "bundle", trusted=True)
    return restored, result, test_bars, output, args


def finish(primary, research_result, output, options):
    """Render and optionally log only after the runtime has finished."""
    from tradelearn.report import Reporter

    stats = primary[0].stats if isinstance(primary, list) else primary
    Reporter(stats).report(output / "backtest.html")
    if options.mlflow_uri:
        from tradelearn.integrations.mlflow import log

        logged = log(
            primary,
            research_result=research_result,
            experiment_name=options.experiment_name,
            uri=options.mlflow_uri,
        )
        print(f"MLflow status={logged.status} run_id={logged.run_id} error={logged.error}")
        if logged.error:
            raise RuntimeError(logged.error)
