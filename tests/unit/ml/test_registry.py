from __future__ import annotations

import pandas as pd

from tradelearn.compat.backtrader import Cerebro
from tradelearn.ml import MLStrategy, ModelRegistry, model_uri


class FakePyfunc:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.model = ConstantModel()

    def load_model(self, uri: str) -> ConstantModel:
        self.loaded.append(uri)
        return self.model


class FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.pyfunc = FakePyfunc()

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)


class ConstantModel:
    def __init__(self) -> None:
        self.predict_X: list[list[float]] = []

    def predict(self, X: list[list[float]]) -> list[float]:
        self.predict_X.extend(X)
        return [0.8]


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100.0, 100.0],
            "feature": [0.1, 0.2],
        },
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )


def test_model_uri_accepts_name_stage_and_alias_references() -> None:
    assert model_uri("alpha_model:production") == "models:/alpha_model/production"
    assert model_uri("alpha_model@champion") == "models:/alpha_model@champion"
    assert model_uri("models:/alpha_model/1") == "models:/alpha_model/1"


def test_model_registry_loads_mlflow_pyfunc_model() -> None:
    mlflow = FakeMlflow()
    registry = ModelRegistry(tracking_uri="https://mlflow.example", mlflow_module=mlflow)

    model = registry.load("alpha_model:production")

    assert model is mlflow.pyfunc.model
    assert mlflow.tracking_uris == ["https://mlflow.example"]
    assert mlflow.pyfunc.loaded == ["models:/alpha_model/production"]


def test_mlstrategy_loads_string_model_reference_from_registry() -> None:
    mlflow = FakeMlflow()

    class DemoMLStrategy(MLStrategy):
        model = "alpha_model:production"
        features = ("feature",)

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(
        DemoMLStrategy,
        threshold=0.5,
        size=1,
        model_registry=ModelRegistry(mlflow_module=mlflow),
    )
    [strategy] = cerebro.run()

    assert strategy.model_ is mlflow.pyfunc.model
    assert mlflow.pyfunc.loaded == ["models:/alpha_model/production"]
    assert strategy.stats is not None
    assert strategy.stats.summary["total_fills"] == 1
