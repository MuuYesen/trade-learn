"""Standard Random Forest Portfolio Rotation Strategy."""

from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tradelearn.engine as bt

class RandomForestRotation(bt.Strategy):
    """
    Buy the top-ranked symbols based on Random Forest predictions.
    
    This strategy demonstrates rotation across multiple assets.
    """

    params = (
        ("threshold", 0.51),
        ("top_n", 2),
        ("size", 10),
    )

    def __init__(self) -> None:
        # Placeholder for model or signals
        self._model = None
        # Ensure we have at least 1 previous bar for momentum calculation
        self.addminperiod(1)

    def next(self) -> None:
        # 1. Collect current data for all assets
        scores = {}
        for data in self.datas:
            # In a real scenario, you might calculate features here or use pre-calculated ones
            # For this example, we assume some logic or signal is available
            # Let's say we just use a simple momentum signal as a proxy for RF output
            scores[data._name] = data.close[0] / data.close[-1] - 1.0
            
        # 2. Rank assets
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_assets = [name for name, score in ranked[:self.p.top_n] if score > 0]
        
        # 3. Rebalance
        for data in self.datas:
            if data._name in top_assets:
                if not self.getposition(data):
                    self.buy(data=data, size=self.p.size)
            else:
                if self.getposition(data):
                    self.close(data=data)
