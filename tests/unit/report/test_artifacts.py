from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from tradelearn import metrics
from tradelearn.report.artifacts import write_artifact_bundle
from tradelearn.research import ResearchResult


def test_write_artifact_bundle_writes_tables_report_plot_and_weights(tmp_path) -> None:
    stats = _stats()
    strategy = SimpleNamespace(
        research_result=ResearchResult(
            name="artifact_research",
            weights=pd.Series({"AAA": 0.6, "BBB": 0.4}, name="weight"),
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
        "artifacts.xlsx",
        "weights.csv",
        "trades.csv",
        "report.html",
        "plot.html",
    }.issubset(names)
    assert not (tmp_path / "csv").exists()
    assert (tmp_path / "artifacts.xlsx").is_file()
    metadata = pd.read_csv(tmp_path / "metadata.csv").set_index("key")["value"]
    assert metadata["artifact_schema_version"] == "1.0"
    assert metadata["artifact_kind"] == "backtest"
    weights_csv = pd.read_csv(tmp_path / "weights.csv").set_index("symbol")["weight"]
    assert weights_csv.to_dict() == {"AAA": 0.6, "BBB": 0.4}
    trades_csv = pd.read_csv(tmp_path / "trades.csv")
    assert trades_csv["pnl"].tolist() == [100.0, -25.0]


def test_write_artifact_bundle_writes_research_result_weights(tmp_path) -> None:
    stats = _stats()
    strategy = SimpleNamespace(
        research_result=ResearchResult(
            name="research",
            selected=["AAA"],
            weights=pd.Series({"AAA": 0.7, "BBB": 0.3}, name="weight"),
            artifacts={
                "name": "custom-name",
                "lookback": 20,
                "symbols": ["AAA", "BBB"],
                "profile": {"rows": 3},
            },
        )
    )

    files = write_artifact_bundle(stats, tmp_path, strategy=strategy)

    names = {path.name for path in files}
    assert "artifacts.xlsx" in names
    assert "weights.csv" in names
    assert "research.csv" in names
    assert "report.html" in names
    assert not (tmp_path / "csv").exists()
    assert (tmp_path / "artifacts.xlsx").is_file()
    weights_csv = pd.read_csv(tmp_path / "weights.csv").set_index("symbol")["weight"]
    assert weights_csv.to_dict() == {"AAA": 0.7, "BBB": 0.3}
    research = pd.read_csv(tmp_path / "research.csv").set_index("key")["value"]
    assert research["name"] == "research"
    assert research["selected"] == "AAA"
    assert research["artifacts.name"] == "custom-name"
    assert int(research["artifacts.lookback"]) == 20
    assert research["artifacts.symbols"] == "AAA,BBB"
    assert int(research["artifacts.profile.rows"]) == 3


def test_write_artifact_bundle_writes_benchmark_aware_tables(tmp_path) -> None:
    stats = _stats()
    stats.positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "value": [60.0, 40.0, 25.0, 75.0],
        }
    )
    benchmark = pd.Series(
        [0.01, -0.005, 0.01],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        name="benchmark",
    )

    files = write_artifact_bundle(stats, tmp_path, benchmark=benchmark)

    names = {path.name for path in files}
    assert "active_returns.csv" in names
    assert "active_weights.csv" in names
    assert "performance_attr.csv" in names
    assert "artifacts.xlsx" in names


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
