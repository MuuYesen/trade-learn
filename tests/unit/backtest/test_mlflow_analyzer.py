"""TL-specific tracking analyzer retires; ordinary evaluation stays public."""

import importlib.util

import pandas as pd


def test_tracking_analyzer_is_removed_after_callers_migrate():
    from tradelearn.engine import analyzers

    assert not hasattr(analyzers, "MLflowAnalyzer")
    assert importlib.util.find_spec("tradelearn.engine.analyzers.mlflow") is None
    assert importlib.util.find_spec("tradelearn.report.mlflow") is None
    assert {"Returns", "SharpeRatio", "Drawdown"} <= set(analyzers.__all__)


def test_grid_records_completed_results_via_independent_integration(monkeypatch):
    import tradelearn.integrations.mlflow as integration
    from tradelearn.engine import Strategy
    from tradelearn.integrations.search import grid_search

    class ParamStrategy(Strategy):
        params = (("fast", 1),)

        def next(self):
            pass

    calls = []
    monkeypatch.setattr(
        integration,
        "log",
        lambda result, **options: (
            calls.append((result, options)) or integration.LogResult("native", "FINISHED")
        ),
    )
    bars = pd.DataFrame(
        {key: [1.0, 2.0, 3.0] for key in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=3),
    )
    results = grid_search(ParamStrategy, bars, {"fast": [2, 4]}, mlflow={"experiment_name": "grid"})
    assert len(results) == 2 and len(calls) == 1
    call = calls[0]
    assert call[0] == [result.strategy for result in results]
    for result in results:
        assert result.strategy.stats is not None
        assert result.strategy.mlflow_result.run_id == "native"
        assert call[1]["experiment_name"] == "grid"
