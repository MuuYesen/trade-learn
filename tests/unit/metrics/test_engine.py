"""Contracts for deferred analyzer metric requests and isolated parameter instances."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradelearn.metrics import sharpe
from tradelearn.metrics.engine import MetricsEngine


def test_engine_computes_requested_metrics_and_recomputes_for_new_stats():
    engine = MetricsEngine()
    engine.request("returns", "returns")
    engine.request("risk", "drawdown")
    stats = SimpleNamespace(
        returns=pd.Series([0.1, np.nan, -0.2, 0.05]),
        equity=pd.Series([100.0, 110.0, np.nan, 88.0, 92.4]),
    )
    engine.compute(stats)
    assert engine.results["returns"] == pytest.approx({"total": -0.076, "average": -1 / 60})
    assert engine.results["risk"] == pytest.approx({"drawdown": 0.16, "maxdrawdown": 0.2})
    engine.compute(SimpleNamespace())
    assert engine.results["returns"] == {"total": 0.0, "average": 0.0}
    assert engine.results["risk"] == {"drawdown": 0.0, "maxdrawdown": 0.0}


def test_engine_keeps_sharpe_instances_and_first_registration_independent():
    class Params:
        riskfreerate = 0.05

        def asdict(self):
            return {"riskfreerate": self.riskfreerate}

    engine = MetricsEngine()
    engine.request("base", "sharpe", {"riskfreerate": 0.01})
    engine.request("higher_rf", "sharpe", Params())
    engine.request("default", "sharpe", object())
    engine.request("base", "returns")
    returns = pd.Series([0.01, -0.02, 0.04, np.nan])
    engine.compute(SimpleNamespace(returns=returns))
    for name, rf in [("base", 0.01), ("higher_rf", 0.05), ("default", 0.0)]:
        assert engine.results[name]["sharperatio"] == pytest.approx(
            sharpe(returns, rf=rf, periods=252)
        )
    assert engine.results["higher_rf"]["sharperatio"] < engine.results["base"]["sharperatio"]
