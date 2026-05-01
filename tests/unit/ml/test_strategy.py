from __future__ import annotations

import pandas as pd

import tradelearn.engine as bt
import tradelearn.ml as ml
from tradelearn.engine import Cerebro
from tradelearn.ml import MLStrategy


class RecordingModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = predictions
        self.fit_X: list[list[float]] | None = None
        self.fit_y: list[float] | None = None
        self.predict_X: list[list[float]] = []

    def fit(self, X: list[list[float]], y: list[float]) -> RecordingModel:
        self.fit_X = X
        self.fit_y = y
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        self.predict_X.extend(X)
        index = min(len(self.predict_X) - 1, len(self.predictions) - 1)
        return [self.predictions[index]]


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [100.0, 100.0, 100.0, 100.0],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "target": [0.0, 1.0, 1.0, 0.0],
        },
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )


def test_engine_facade_exports_mlstrategy() -> None:
    """Engine users can import MLStrategy from the Engine facade."""
    assert bt.MLStrategy is MLStrategy
    assert "MLStrategy" in bt.__all__


def test_ml_facade_exports_automl_next_to_causal_selector() -> None:
    """AutoML is exposed directly from the ML facade."""

    assert ml.AutoML.__name__ == "AutoML"
    assert ml.CausalSelector.__name__ == "CausalSelector"


def test_mlstrategy_trains_predicts_and_places_orders() -> None:
    model = RecordingModel([0.8, 0.8, 0.8, 0.8])

    class DemoMLStrategy(MLStrategy):
        features = ("feature",)
        target = "target"

    DemoMLStrategy.model = model

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(DemoMLStrategy, threshold=0.5, size=2)
    [strategy] = cerebro.run()

    fitted_model = strategy.model_
    assert fitted_model.fit_X == [[0.1], [0.2], [0.3], [0.4]]
    assert fitted_model.fit_y == [0.0, 1.0, 1.0, 0.0]
    assert fitted_model.predict_X == [[0.1], [0.2], [0.3], [0.4]]
    assert strategy.stats is not None
    assert strategy.stats.summary["total_fills"] == 1
    assert strategy.stats.fills.iloc[0]["size"] == 2.0


def test_mlstrategy_defaults_to_full_equity_when_size_not_set() -> None:
    model = RecordingModel([0.8, 0.8, 0.8, 0.8])

    class DemoMLStrategy(MLStrategy):
        features = ("feature",)
        target = "target"

    DemoMLStrategy.model = model

    cerebro = Cerebro(trade_on_close=True)
    cerebro.broker.setcash(1000.0)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(DemoMLStrategy, threshold=0.5)
    [strategy] = cerebro.run()

    assert strategy.stats is not None
    assert strategy.stats.fills.iloc[0]["size"] == 95.0


def test_mlstrategy_negative_prediction_closes_long_position() -> None:
    model = RecordingModel([0.8, -0.8, -0.8, -0.8])

    class DemoMLStrategy(MLStrategy):
        features = ("feature",)
        target = "target"

    DemoMLStrategy.model = model

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(DemoMLStrategy, threshold=0.5, size=1)
    [strategy] = cerebro.run()

    assert strategy.stats is not None
    assert strategy.stats.summary["total_fills"] == 2
    assert strategy.stats.positions.iloc[-1]["size"] == 0.0


def test_mlstrategy_feature_vector_override_used_for_training() -> None:
    """feature_vector() override drives both training and inference."""
    model = RecordingModel([0.8, 0.8, 0.8, 0.8])

    class OverrideMLStrategy(MLStrategy):
        target = "target"

        def __init__(self) -> None:
            # Use close line directly as feature
            pass

        def feature_vector(self) -> list[float]:
            return [float(self.data.close[0])]

    OverrideMLStrategy.model = model

    cerebro = Cerebro(trade_on_close=True)
    cerebro.adddata(_bars(), name="demo")
    cerebro.addstrategy(OverrideMLStrategy, threshold=0.5, size=1)
    [strategy] = cerebro.run()

    fitted_model = strategy.model_
    # Training X should contain close prices [10.5, 11.5, 12.5, 13.5]
    assert fitted_model.fit_X is not None
    assert len(fitted_model.fit_X) == 4
    assert fitted_model.fit_X[0] == [10.5]
    assert fitted_model.fit_X[3] == [13.5]
    # Inference X matches training feature
    assert fitted_model.predict_X[0] == [10.5]
