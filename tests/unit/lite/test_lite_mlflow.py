from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

import tradelearn.engine as bt
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.lite import Backtest, Strategy
from tradelearn.research import ResearchResult


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "volume": [100.0, 110.0, 120.0, 130.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.runs: list[dict[str, object]] = []
        self.params: list[dict[str, object]] = []
        self.metrics: list[dict[str, float]] = []
        self.dicts: list[tuple[dict[str, object], str]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.artifact_dirs: list[tuple[str, str | None]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(self, experiment: str) -> None:
        self.experiments.append(experiment)

    @contextmanager
    def start_run(self, *, run_name: str | None = None, nested: bool = False):
        self.runs.append({"run_name": run_name, "nested": nested})
        yield object()

    def log_params(self, params: dict[str, object]) -> None:
        self.params.append(dict(params))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(dict(metrics))

    def log_dict(self, dictionary: dict[str, object], artifact_file: str) -> None:
        self.dicts.append((dict(dictionary), artifact_file))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((Path(local_path).name, artifact_path))

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        self.artifact_dirs.append((Path(local_dir).name, artifact_path))


def test_lite_log_mlflow_requires_run_first() -> None:
    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    bt = Backtest(_data(), NoopStrategy, cash=1000.0)

    with pytest.raises(RuntimeError, match="run\\(\\) must be called"):
        bt.log_mlflow(mlflow_module=FakeMLflow())


def test_lite_log_mlflow_logs_stats_config_params_and_artifacts() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                self.record(score=self.data.close[0])
                self.buy(size=1)

    fake = FakeMLflow()
    bt = Backtest(_data(), LiteStrategy, cash=1000.0, commission=0.001)
    stats = bt.run()

    status = bt.log_mlflow(
        experiment_name="lite-exp",
        run_name="lite-run",
        params={"fast": 3},
        tags={"mode": "lite"},
        uri="https://mlflow.example",
        log_report=False,
        log_plot=False,
        mlflow_module=fake,
    )

    assert status == "logged"
    assert fake.tracking_uris == ["https://mlflow.example"]
    assert fake.experiments == ["lite-exp"]
    assert fake.runs == [{"run_name": "lite-run", "nested": False}]
    assert fake.params[0]["fast"] == 3
    assert fake.params[0]["broker.initial_cash"] == 1000.0
    assert fake.params[0]["broker.commission"] == 0.001
    assert fake.params[0]["tag.mode"] == "lite"
    assert fake.metrics[0]["final_value"] == stats.summary["final_value"]
    assert fake.metrics[0]["total_orders"] == 1.0
    assert fake.dicts == []
    assert sorted(name for name, _ in fake.artifacts) == [
        "artifacts.xlsx",
        "equity.csv",
        "fills.csv",
        "positions.csv",
        "summary.csv",
        "trades.csv",
    ]
    assert {artifact_path for _, artifact_path in fake.artifacts} == {None}
    assert fake.artifact_dirs == []


def test_lite_log_mlflow_logs_upload_summary(caplog) -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            pass

    fake = FakeMLflow()
    backtest = Backtest(_data(), LiteStrategy, cash=1000.0)
    backtest.run()
    caplog.set_level(logging.INFO, logger="tradelearn.lite.mlflow")

    backtest.log_mlflow(
        experiment_name="lite-exp",
        run_name="lite-run",
        upload_artifacts=False,
        mlflow_module=fake,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "MLflow logging started" in message
        and "experiment=lite-exp" in message
        and "run=lite-run" in message
        for message in messages
    )
    assert any(
        "MLflow logging finished" in message and "artifacts=False" in message
        for message in messages
    )


def test_lite_log_mlflow_can_skip_artifact_uploads() -> None:
    class LiteStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            pass

    fake = FakeMLflow()
    bt = Backtest(_data(), LiteStrategy, cash=1000.0)
    stats = bt.run()

    bt.log_mlflow(
        experiment_name="lite-exp",
        uri="https://mlflow.example",
        upload_artifacts=False,
        mlflow_module=fake,
    )

    assert fake.metrics[0]["final_value"] == stats.summary["final_value"]
    assert fake.dicts == []
    assert fake.artifacts == []


def test_lite_log_mlflow_logs_research_result_params_and_artifacts() -> None:
    result = ResearchResult(
        name="lite_research",
        params={"select_top.k": 1},
        artifacts={
            "lookback": 20,
            "profile": {"rows": 4, "numeric": {"open": {"25%": 10.5}}},
        },
        scores=pd.Series({"primary": 1.0}, name="score"),
        selected=["primary"],
        weights=pd.Series({"primary": 0.5}, name="weight"),
    )

    class LiteResearchStrategy(Strategy):
        research_result = None

        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                self.record_research_result(self.research_result)
                self.buy(size=1)

    fake = FakeMLflow()
    bt = Backtest(_data(), LiteResearchStrategy, cash=1000.0)
    bt.run(research_result=result)

    bt.log_mlflow(
        experiment_name="lite-research",
        log_report=False,
        log_plot=False,
        mlflow_module=fake,
    )

    assert fake.params[0]["research.name"] == "lite_research"
    assert fake.params[0]["research.select.k"] == 1
    assert fake.params[0]["research.artifacts.lookback"] == 20
    assert "research.artifacts.profile.rows" not in fake.params[0]
    assert "research.artifacts.profile.numeric.open.25pct" not in fake.params[0]
    assert fake.dicts == []
    assert ("artifacts.xlsx", None) in fake.artifacts
    assert ("weights.csv", None) in fake.artifacts
    assert ("research.csv", None) in fake.artifacts
    assert fake.artifact_dirs == []


def test_lite_and_engine_mlflow_params_use_shared_schema() -> None:
    result = ResearchResult(
        name="index_enhance_research",
        params={
            "select_top.k": 2,
            "select_top.reverse": True,
            "equal_weight.gross": 0.95,
            "apply_constraints.max_weight": 0.5,
            "apply_constraints.normalize": True,
        },
        artifacts={"lookback": 20},
        weights=pd.Series({"primary": 0.5}, name="weight"),
    )

    class LiteResearchStrategy(Strategy):
        def init(self) -> None:
            self.start_on_bar(2)

        def next(self) -> None:
            if len(self.data) == 3:
                self.record_research_result(self.research_result)

    class EngineResearchStrategy(bt.Strategy):
        params = (("research_result", None), ("lookback", 20))

        def __init__(self) -> None:
            super().__init__()
            self.addminperiod(self.p.lookback + 1)

        def next(self) -> None:
            if len(self.data) == 3:
                self.record_research_result(self.research_result)

    lite_mlflow = FakeMLflow()
    lite = Backtest(_data(), LiteResearchStrategy, cash=1000.0, commission=0.001)
    lite.run(research_result=result)
    lite.log_mlflow(
        experiment_name="shared-schema",
        params={"runtime.mode": "lite", "runtime.pipeline": False},
        upload_artifacts=False,
        mlflow_module=lite_mlflow,
    )

    engine_mlflow = FakeMLflow()
    cerebro = bt.Cerebro()
    cerebro.setcash(1000.0)
    cerebro.setcommission(0.001)
    cerebro.adddata(_data())
    cerebro.addstrategy(EngineResearchStrategy, research_result=result, lookback=20)
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        experiment="shared-schema",
        params={"runtime.mode": "engine", "runtime.pipeline": False},
        upload_artifacts=False,
        mlflow_module=engine_mlflow,
    )
    cerebro.run()

    lite_params = lite_mlflow.params[0]
    engine_params = engine_mlflow.params[0]
    expected = {
        "broker.initial_cash",
        "broker.final_cash",
        "broker.final_value",
        "broker.commission",
        "broker.trade_on_close",
        "research.name",
        "research.select.k",
        "research.select.reverse",
        "research.weight.gross",
        "research.constraints.max_weight",
        "research.constraints.normalize",
        "research.artifacts.lookback",
    }
    assert expected <= lite_params.keys()
    assert expected <= engine_params.keys()
    assert lite_params["broker.initial_cash"] == engine_params["broker.initial_cash"] == 1000.0
    assert lite_params["broker.commission"] == engine_params["broker.commission"] == 0.001
    assert (
        lite_params["research.name"]
        == engine_params["research.name"]
        == "index_enhance_research"
    )
    assert lite_params["research.select.k"] == engine_params["research.select.k"] == 2
    assert lite_params["runtime.mode"] == "lite"
    assert engine_params["runtime.mode"] == "engine"
    assert lite_params["runtime.pipeline"] is False
    assert engine_params["runtime.pipeline"] is False
    assert "broker.cash" not in engine_params
    assert "broker.value" not in engine_params
    assert "config.broker.cash" not in lite_params
    assert "research.select_top.k" not in lite_params
    assert "research.equal_weight.gross" not in lite_params
