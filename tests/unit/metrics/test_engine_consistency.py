from types import SimpleNamespace

import pandas as pd
import pytest

from tradelearn.metrics.engine import MetricsEngine
from tradelearn.metrics.risk import drawdown_from_equity


def test_equity_drawdown_keeps_analyzer_positive_magnitude() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 99.0])
    drawdown = drawdown_from_equity(equity)
    assert drawdown.tolist() == pytest.approx([0.0, 0.0, -0.25, -0.175])

    engine = MetricsEngine()
    engine.request("return", "returns")
    engine.request("drawdown", "drawdown")
    engine.compute(SimpleNamespace(returns=pd.Series([0.2, -0.25, 0.1]), equity=equity))

    assert engine.results["return"]["total"] == pytest.approx(-0.01)
    assert engine.results["drawdown"] == pytest.approx(
        {"drawdown": 0.175, "maxdrawdown": 0.25}
    )
