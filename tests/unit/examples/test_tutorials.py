from __future__ import annotations

import math

from examples.tutorials import run_tutorial_smoke


def test_tutorial_smoke_covers_stage9_topics() -> None:
    result = run_tutorial_smoke()

    assert set(result) == {
        "factor_research",
        "strategy_backtest",
        "portfolio",
        "ml_strategy",
        "mlflow",
        "jupyterlab",
        "backtrader_migration",
    }
    assert result["factor_research"]["feature_count"] >= 1
    assert result["strategy_backtest"]["fills"] >= 1
    assert result["portfolio"]["position_rows"] >= 1
    assert result["ml_strategy"]["selected_features"]
    assert math.isfinite(result["ml_strategy"]["final_value"])
    assert result["mlflow"]["status"] == "logged"
    assert result["jupyterlab"]["mcp_command"] == [
        "tradelearn",
        "mcp",
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]
    assert result["backtrader_migration"]["fills"] >= 1
