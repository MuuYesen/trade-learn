from __future__ import annotations

import math
from pathlib import Path

from demos.rf_fund import run_demo as run_rf_fund_demo
from demos.sma_single import run_demo as run_sma_single_demo


def test_sma_single_demo_runs_on_new_api() -> None:
    result = run_sma_single_demo()

    assert result["strategy"] == "DemoSmaCross"
    assert result["bars"] >= 30
    assert result["fills"] >= 1
    assert math.isfinite(result["final_value"])
    assert math.isfinite(result["return_pct"])


def test_rf_fund_demo_runs_on_new_api() -> None:
    result = run_rf_fund_demo()

    assert result["strategy"] == "RandomForestRotation"
    assert result["symbols"] == ["asset_a", "asset_b", "asset_c"]
    assert result["selected_symbol"] in result["symbols"]
    assert set(result["probabilities"]) == set(result["symbols"])
    assert result["fills"] >= 1
    assert math.isfinite(result["final_value"])
    assert result["train_rows"] > 0


def test_demos_no_longer_use_legacy_query_backtest_api() -> None:
    for path in (Path("demos/sma_single.py"), Path("demos/rf_fund.py")):
        source = path.read_text()
        assert "tradelearn.query" not in source
        assert "tradelearn.strategy.backtest" not in source
        assert "Backtest(" not in source
        assert "def init(" not in source
        assert "self.I" not in source
        assert "engine='tdx'" not in source
