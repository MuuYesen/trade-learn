"""Lite forwards completed results to the single integration."""
from types import SimpleNamespace

import pandas as pd
import pytest

from tradelearn.lite import Backtest, Strategy


def backtest():
    return Backtest(pd.DataFrame({"open":[1.,2.], "high":[1.,2.], "low":[1.,2.], "close":[1.,2.], "volume":[1.,1.]}, index=pd.date_range("2026-01-01", periods=2)), Strategy)


def test_lite_log_requires_completed_run():
    with pytest.raises(RuntimeError, match="run\\(\\)"):
        backtest().log_mlflow()


def test_lite_forwards_without_runtime_or_tracking_wrapper(monkeypatch):
    import tradelearn.integrations.mlflow as integration
    bt = backtest()
    bt._last_stats = SimpleNamespace(summary={"value":1.})
    expected = integration.LogResult("real-id", "RUNNING", reused=True)
    calls = []
    monkeypatch.setattr(integration, "log", lambda result, **options: calls.append((result,options)) or expected)
    assert bt.log_mlflow(experiment_name="research", log_report=False) is expected
    assert calls[0][0] is bt
    assert calls[0][1]["experiment_name"] == "research"
    assert calls[0][1]["attachments"] == ()


def test_lite_rejects_removed_nested_run_option():
    with pytest.raises(TypeError, match="nested"):
        backtest().log_mlflow(nested=True)
