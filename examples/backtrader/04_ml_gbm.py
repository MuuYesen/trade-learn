"""Gradient Boosting Machine Strategy using tradelearn.ml."""

from __future__ import annotations
from sklearn.ensemble import GradientBoostingRegressor
import tradelearn.engine as bt

class Alpha101GBMStrategy(bt.MLStrategy):
    """
    Gradient Boosting strategy over Alpha101 feature columns.
    
    Predicts next-day returns and trades based on a prediction threshold.
    """

    model = GradientBoostingRegressor(random_state=7, n_estimators=50, max_depth=3)
    target = "target"
    
    params = (
        ("threshold", 0.001),
        ("size", 100),
    )

