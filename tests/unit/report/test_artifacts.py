from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from tradelearn import metrics
from tradelearn.research import ResearchResult
from tradelearn.report.artifacts import write_artifact_bundle


def test_write_artifact_bundle_writes_tables_report_plot_and_weights(tmp_path) -> None:
    stats = _stats()
    strategy = SimpleNamespace(
        pipeline_result=SimpleNamespace(
            weights=pd.Series({"AAA": 0.6, "BBB": 0.4}, name="weight")
        )
    )
    market_data = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.8, 11.2, 11.7],
            "low": [9.8, 10.2, 10.8],
            "close": [10.6, 11.0, 11.5],
            "volume": [100.0, 110.0, 120.0],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )

    files = write_artifact_bundle(
        stats,
        tmp_path,
        strategy=strategy,
        market_data=market_data,
        log_report=True,
        log_plot=True,
    )

    names = {path.name for path in files}
    assert {
        "equity.parquet",
        "trades.parquet",
        "weights.parquet",
        "stats.json",
        "report.html",
        "plot.html",
    }.issubset(names)
    weights = pd.read_parquet(tmp_path / "weights.parquet")["weight"]
    assert weights.to_dict() == {"AAA": 0.6, "BBB": 0.4}


def test_write_artifact_bundle_writes_research_result_weights(tmp_path) -> None:
    stats = _stats()
    strategy = SimpleNamespace(
        research_result=ResearchResult(
            name="research",
            weights=pd.Series({"AAA": 0.7, "BBB": 0.3}, name="weight"),
        )
    )

    files = write_artifact_bundle(stats, tmp_path, strategy=strategy)

    names = {path.name for path in files}
    assert "weights.parquet" in names
    weights = pd.read_parquet(tmp_path / "weights.parquet")["weight"]
    assert weights.to_dict() == {"AAA": 0.7, "BBB": 0.3}


def _stats() -> SimpleNamespace:
    returns = pd.Series(
        [0.02, -0.01, 0.015],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        name="returns",
    )
    return SimpleNamespace(
        returns=returns,
        equity=metrics.cum_returns(returns, starting_value=100_000.0),
        trades=pd.DataFrame({"pnl": [100.0, -25.0]}),
        fills=pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=2, tz="UTC"),
                "side": ["buy", "sell"],
                "price": [10.6, 11.0],
            }
        ),
        positions=pd.DataFrame(),
        summary={"strategy_name": "artifact-demo"},
        analyzers={},
        config={"strategy": "artifact-demo"},
    )
