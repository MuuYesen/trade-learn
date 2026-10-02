"""Gradient Boosting Machine Strategy using plain bt.Strategy."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import tradelearn.engine as bt


class Alpha101GBMStrategy(bt.Strategy):
    """
    Gradient Boosting strategy over Alpha101 feature columns.

    Predicts next-day returns and trades based on a prediction threshold.
    Fit uses complete feature/target rows only. Missing evaluation features
    suppress the signal for that bar; market bars and positions remain intact.
    """

    params = (
        ("model", GradientBoostingRegressor(random_state=7, n_estimators=50, max_depth=3)),
        ("features", ()),
        ("target", "target"),
        ("threshold", 0.001),
        ("size", 100),
        ("training_data", None),
    )

    def start(self) -> None:
        data = self.p.training_data
        self.model_ = self.p.model
        if data is not None and self.p.features:
            frame = data.dropna(subset=[*self.p.features, self.p.target])
            self.model_.fit(frame.loc[:, self.p.features], frame[self.p.target])

    def next(self) -> None:
        if not self.p.features:
            return
        vector = [float(self.data.get_value(feature)) for feature in self.p.features]
        if not all(math.isfinite(value) for value in vector):
            return
        prediction = float(self.model_.predict(pd.DataFrame([vector], columns=self.p.features))[0])
        if prediction > self.p.threshold and self.position.size <= 0:
            self.buy(size=self.p.size)
        elif prediction < -self.p.threshold and self.position.size > 0:
            self.close()
