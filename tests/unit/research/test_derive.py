from __future__ import annotations

import sys

import pandas as pd
import pytest

import tradelearn.research.derive as dv


def test_derive_exports_only_stateful_feature_generators() -> None:
    assert not hasattr(dv, "ratio_features")
    assert dv.__all__ == ["SymbolicFeatureGenerator"]


def test_symbolic_feature_generator_requires_gplearn_when_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "gplearn", None)

    with pytest.raises(ImportError, match="gplearn"):
        dv.SymbolicFeatureGenerator(n_features=2).fit(
            pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [10.0, 11.0, 12.0]}),
            columns=["volume"],
            target="close",
        )


def test_symbolic_feature_generator_exposes_params_and_uses_gplearn(monkeypatch) -> None:
    class FakeSymbolicTransformer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_calls = 0

        def fit(self, x, y):
            self.fit_calls += 1
            self.fit_shape = x.shape
            return self

        def transform(self, x):
            return [[1.0, 2.0] for _ in range(len(x))]

    monkeypatch.setattr(dv, "_load_symbolic_transformer", lambda: FakeSymbolicTransformer)

    generator = dv.SymbolicFeatureGenerator(
        n_features=2,
        random_state=7,
        generations=4,
        population_size=30,
    )

    train = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0],
            "volume": [10.0, 11.0, 12.0],
            "target": [0.1, 0.2, 0.3],
        }
    )
    test = pd.DataFrame(
        {
            "close": [4.0, 5.0],
            "volume": [13.0, 14.0],
            "target": [0.4, 0.5],
        }
    )

    fitted = generator.fit(
        train,
        columns=["close", "volume"],
        target="target",
    )
    features = generator.transform(test)

    assert fitted is generator
    assert generator.input_columns_ == ["close", "volume"]
    assert generator.feature_names_ == ["symbolic_1", "symbolic_2"]
    assert features.index.equals(test.index)
    assert features.columns.tolist() == ["symbolic_1", "symbolic_2"]
    assert features.iloc[0].tolist() == [1.0, 2.0]
    assert generator.get_params() == {
        "type": "SymbolicFeatureGenerator",
        "n_features": 2,
        "random_state": 7,
        "generations": 4,
        "population_size": 30,
        "metric": "pearson",
    }


def test_symbolic_feature_generator_fit_transform_shortcut(monkeypatch) -> None:
    class FakeSymbolicTransformer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x, y):
            return self

        def transform(self, x):
            return [[1.0, 2.0] for _ in range(len(x))]

    monkeypatch.setattr(dv, "_load_symbolic_transformer", lambda: FakeSymbolicTransformer)

    generator = dv.SymbolicFeatureGenerator(n_features=2)

    features = generator.fit_transform(
        pd.DataFrame(
            {
                "close": [1.0, 2.0, 3.0],
                "volume": [10.0, 11.0, 12.0],
                "target": [0.1, 0.2, 0.3],
            }
        ),
        columns=["close", "volume"],
        target="target",
    )

    assert features.columns.tolist() == ["symbolic_1", "symbolic_2"]
    assert features.iloc[0].tolist() == [1.0, 2.0]


def test_symbolic_feature_generator_transform_requires_fit() -> None:
    generator = dv.SymbolicFeatureGenerator(n_features=2)

    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        generator.transform(pd.DataFrame({"close": [1.0]}))
