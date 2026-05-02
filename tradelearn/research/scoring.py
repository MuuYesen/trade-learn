from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


class ModelScorer:
    """Turn model predictions into score Series for portfolio builders."""

    def __init__(
        self,
        model: Any,
        *,
        features: Sequence[str],
        current: bool = True,
    ) -> None:
        self.model = model
        self.features = [str(feature) for feature in features]
        self.current = bool(current)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Predict scores from a feature frame."""
        frame = pd.DataFrame(data)
        if self.current and isinstance(frame.index, pd.MultiIndex):
            time_level = frame.index.names[0] if frame.index.names[0] is not None else 0
            latest = frame.index.get_level_values(time_level).max()
            frame = frame.xs(latest, level=time_level, drop_level=True)
        frame = frame.dropna(subset=self.features)
        if frame.empty:
            return pd.Series(dtype="float64", name="score")
        values = self.model.predict(frame.loc[:, self.features])
        return pd.Series(values, index=frame.index, name="score", dtype="float64")

    def get_params(self) -> dict[str, Any]:
        """Return serializable scorer parameters."""
        return {
            "type": type(self).__name__,
            "model": type(self.model).__name__,
            "features": list(self.features),
            "current": self.current,
        }


__all__ = ["ModelScorer"]
