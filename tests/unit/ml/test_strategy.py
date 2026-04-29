from __future__ import annotations

import pandas as pd

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
