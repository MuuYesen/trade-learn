from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from tradelearn.core import BrokerEvent, StreamBar
from tradelearn.engine import Cerebro, IndexEnhanceStrategy, Strategy, grid_search
from tradelearn.engine.analyzer import Analyzer
from tradelearn.engine.analyzers import MLflowAnalyzer
from tradelearn.ml import (
    EqualWeightOptimizer,
    FactorTransformer,
    ModelAdapter,
    PipelineResult,
    StrategyPipeline,
    TopKSelector,
)
from tradelearn.research import ResearchResult, ResearchStep


def bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 110.0, 120.0],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
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


def test_mlflow_analyzer_logs_params_stats_and_artifacts(monkeypatch) -> None:
    class NoopStrategy(Strategy):
        params = (("fast", 3),)

        def next(self) -> None:
            pass

    fake = FakeMLflow()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://env.example")

    cerebro = Cerebro()
    cerebro.broker.setcash(123.0)
    cerebro.broker.setcommission(0.001)
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy, fast=5)
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        experiment="stage4",
        run_name="case-1",
        uri="https://kwarg.example",
        mlflow_module=fake,
    )

    [strategy] = cerebro.run()

    assert fake.tracking_uris == ["https://kwarg.example"]
    assert fake.experiments == ["stage4"]
    assert fake.runs == [{"run_name": "case-1", "nested": False}]
    assert fake.params[0]["strategy.fast"] == 5
    assert fake.params[0]["broker.cash"] == 123.0
    assert fake.params[0]["broker.commission"] == 0.001
    assert fake.metrics == [
        {
            "bars": 3.0,
            "final_cash": 123.0,
            "final_value": 123.0,
            "return_pct": 0.0,
            "final_realized_pnl": 0.0,
            "final_unrealized_pnl": 0.0,
            "final_margin_used": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0.0,
            "win_rate_pct": 0.0,
            "total_orders": 0.0,
            "total_fills": 0.0,
        }
    ]
    assert "sharpe" not in fake.metrics[0]
    [artifact] = fake.dicts
    payload, artifact_file = artifact
    summary = payload["summary"]
    assert summary["max_drawdown"] == 0.0
    assert math.isnan(summary["sharpe"])
    assert fake.dicts == [
        (
            {
                "summary": summary,
                "analyzers": {},
                "config": {
                    "callback_batch": 1,
                    "trade_on_close": False,
                    "exactbars": False,
                    "stdstats": True,
                    "broker": {"cash": 123.0, "commission": 0.001},
                },
            },
            artifact_file,
        )
    ]
    assert artifact_file == "stats.json"
    assert strategy.analyzer_results["mlflow"]["status"] == "logged"


def test_mlflow_analyzer_logs_strategy_pipeline_params_and_artifacts() -> None:
    class ImportanceModel:
        feature_importances_ = [0.25, 0.75]

        def fit(self, X, y):
            return self

        def predict(self, X):
            return [row[0] for row in X]

    class PipelineStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.pipeline = StrategyPipeline(
                [
                    ("features", FactorTransformer(["value", "quality"])),
                    ("model", ModelAdapter(ImportanceModel())),
                    ("selector", TopKSelector(k=1)),
                    ("optimizer", EqualWeightOptimizer(gross=0.9)),
                ]
            )
            self.pipeline.fit(
                pd.DataFrame({"value": [0.8, 0.2], "quality": [0.1, 0.7]}),
                [1.0, 0.0],
            )
            self.pipeline_result = PipelineResult(
                scores=pd.Series({"AAA": 0.8, "BBB": 0.2}, name="score"),
                selected=["AAA"],
                weights=pd.Series({"AAA": 0.9}, name="weight"),
            )

        def next(self) -> None:
            pass

    fake = FakeMLflow()
    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(PipelineStrategy)
    cerebro.addanalyzer(MLflowAnalyzer, name="mlflow", mlflow_module=fake)

    cerebro.run()

    params = fake.params[0]
    assert params["pipeline.features.features"] == "value,quality"
    assert params["pipeline.model.estimator"] == "ImportanceModel"
    assert params["pipeline.selector.k"] == 1
    assert params["pipeline.optimizer.gross"] == 0.9
    artifacts = {artifact_file: payload for payload, artifact_file in fake.dicts}
    assert artifacts["pipeline.json"] == {
        "params": {
            "steps": ["features", "model", "selector", "optimizer"],
            "features": {
                "type": "FactorTransformer",
                "features": ["value", "quality"],
                "feature_store": False,
            },
            "model": {
                "type": "ModelAdapter",
                "estimator": "ImportanceModel",
                "score_column": None,
            },
            "selector": {
                "type": "TopKSelector",
                "k": 1,
                "ascending": False,
                "threshold": None,
            },
            "optimizer": {
                "type": "EqualWeightOptimizer",
                "gross": 0.9,
            },
        },
        "result": {
            "scores": {"AAA": 0.8, "BBB": 0.2},
            "selected": ["AAA"],
            "weights": {"AAA": 0.9},
        },
        "explanation": {"value": 0.25, "quality": 0.75},
    }


def test_mlflow_analyzer_logs_report_plot_and_table_artifacts() -> None:
    class BundleStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.pipeline_result = PipelineResult(
                scores=pd.Series({"AAA": 0.7, "BBB": 0.3}, name="score"),
                selected=["AAA"],
                weights=pd.Series({"AAA": 0.8, "BBB": 0.2}, name="weight"),
            )

        def next(self) -> None:
            if len(self.data) == 1:
                self.buy(size=1)
            elif len(self.data) == 3:
                self.close()

    fake = FakeMLflow()
    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(BundleStrategy)
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        mlflow_module=fake,
        artifact_bundle=True,
        log_report=True,
        log_plot=True,
        artifact_path="bundle",
    )

    cerebro.run()

    artifacts = {name: artifact_path for name, artifact_path in fake.artifacts}
    assert {
        "report.html",
        "plot.html",
        "trades.parquet",
        "equity.parquet",
        "weights.parquet",
    }.issubset(artifacts)
    assert all(artifact_path == "bundle" for artifact_path in artifacts.values())


def test_mlflow_analyzer_logs_pipeline_target_weight_artifacts() -> None:
    class PipelineWeights(IndexEnhanceStrategy):
        rebalance_freq = 1

        def __init__(self) -> None:
            super().__init__()
            self.pipeline = StrategyPipeline(
                [
                    ("model", ModelAdapter(score_column="close")),
                    ("selector", TopKSelector(k=1)),
                    ("optimizer", EqualWeightOptimizer(gross=0.5)),
                ]
            )

        def next(self) -> None:
            if not self.should_rebalance():
                return
            universe = self.current_universe()
            self.pipeline_result = self.pipeline.predict_weights(universe)
            self.target_weights(self.pipeline_result.as_weight_dict())

    fake = FakeMLflow()
    cerebro = Cerebro(trade_on_close=True)
    cerebro.setcash(100_000.0)
    cerebro.adddata(bars(), name="AAA")
    cerebro.adddata(
        bars().assign(
            open=lambda frame: frame["open"] + 10.0,
            high=lambda frame: frame["high"] + 10.0,
            low=lambda frame: frame["low"] + 10.0,
            close=lambda frame: frame["close"] + 10.0,
        ),
        name="BBB",
    )
    cerebro.addstrategy(PipelineWeights)
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        mlflow_module=fake,
        artifact_bundle=True,
        log_report=True,
        log_plot=True,
        artifact_path="pipeline-run",
    )

    [strategy] = cerebro.run()

    assert strategy.getpositionbyname("BBB").size > 0
    payloads = {artifact_file: payload for payload, artifact_file in fake.dicts}
    assert payloads["pipeline.json"]["result"]["selected"] == ["BBB"]
    assert payloads["pipeline.json"]["result"]["weights"] == {"BBB": 0.5}
    artifacts = {name: artifact_path for name, artifact_path in fake.artifacts}
    assert {
        "equity.parquet",
        "trades.parquet",
        "weights.parquet",
        "report.html",
        "plot.html",
    }.issubset(artifacts)
    assert all(artifact_path == "pipeline-run" for artifact_path in artifacts.values())


def test_mlflow_analyzer_logs_research_result_from_strategy_params() -> None:
    result = ResearchResult(
        name="index_enhance_v1",
        steps=[
            ResearchStep("winsorize", "preprocess", {"columns": ["alpha"]}),
            ResearchStep("select_top", "portfolio", {"k": 1}),
        ],
        params={"winsorize.columns": ["alpha"], "select_top.k": 1},
        scores=pd.Series({"AAA": 0.2, "BBB": 0.8}, name="score"),
        selected=["BBB"],
        weights=pd.Series({"BBB": 0.5}, name="weight"),
    )

    class ResearchWeights(IndexEnhanceStrategy):
        params = (("research_results", None),)
        rebalance_freq = 1

        def __init__(self) -> None:
            super().__init__()
            self.research_results = self.p.research_results

        def next(self) -> None:
            if not self.should_rebalance():
                return
            current = self.research_results["2026-01-01"]
            self.record_research_result(current)
            self.target_weights(current.as_weight_dict())

    fake = FakeMLflow()
    cerebro = Cerebro(trade_on_close=True)
    cerebro.setcash(100_000.0)
    cerebro.adddata(bars(), name="AAA")
    cerebro.adddata(
        bars().assign(close=lambda frame: frame["close"] + 10.0),
        name="BBB",
    )
    cerebro.addstrategy(ResearchWeights, research_results={"2026-01-01": result})
    cerebro.addanalyzer(
        MLflowAnalyzer,
        name="mlflow",
        mlflow_module=fake,
        artifact_bundle=True,
        artifact_path="research-run",
    )

    [strategy] = cerebro.run()

    assert strategy.getpositionbyname("BBB").size > 0
    params = fake.params[0]
    assert params["research.name"] == "index_enhance_v1"
    assert params["research.select_top.k"] == 1
    payloads = {artifact_file: payload for payload, artifact_file in fake.dicts}
    assert payloads["research.json"]["result"]["selected"] == ["BBB"]
    assert payloads["research.json"]["result"]["weights"] == {"BBB": 0.5}
    assert payloads["research.json"]["steps"][1]["name"] == "select_top"
    artifacts = {name: artifact_path for name, artifact_path in fake.artifacts}
    assert artifacts["weights.parquet"] == "research-run"


def test_mlflow_analyzer_uses_env_uri_and_warns_without_interrupt(monkeypatch, caplog) -> None:
    class FailingMLflow(FakeMLflow):
        def start_run(self, *, run_name: str | None = None, nested: bool = False):
            raise RuntimeError("mlflow down")

    class NoopStrategy(Strategy):
        def next(self) -> None:
            pass

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://env.example")

    cerebro = Cerebro()
    cerebro.adddata(bars())
    cerebro.addstrategy(NoopStrategy)
    cerebro.addanalyzer(MLflowAnalyzer, name="mlflow", mlflow_module=FailingMLflow())

    [strategy] = cerebro.run()

    assert strategy.analyzer_results["mlflow"]["status"] == "warning"
    assert "mlflow down" in caplog.text


def test_mlflow_analyzer_records_live_broker_event_fields() -> None:
    analyzer = MLflowAnalyzer()
    analyzer.on_broker_event(
        BrokerEvent(
            "status",
            order_id="A1",
            status="pending_confirmation",
            replay=True,
            requires_confirmation=True,
            max_notional=5000.0,
            risk_tags=("daily-limit", "manual-confirm"),
        )
    )
    analyzer.on_broker_event(
        BrokerEvent("partial", order_id="A1", payload={"filled": 40, "remaining": 60})
    )

    payload = analyzer.live_event_summary()

    assert payload["total"] == 2
    assert payload["by_kind"] == {"partial": 1, "status": 1}
    assert payload["replay"] == 1
    assert payload["requires_confirmation"] == 1
    assert payload["max_notional"] == 5000.0
    assert payload["risk_tags"] == ["daily-limit", "manual-confirm"]
    assert payload["last_status"] == "pending_confirmation"
    assert payload["last_order_id"] == "A1"


def test_cerebro_event_mode_dispatches_broker_events_to_analyzers() -> None:
    class CountingStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.closes: list[float] = []

        def next(self) -> None:
            self.closes.append(self.data.close[0])

    class EventRecorder(Analyzer):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[BrokerEvent] = []
            self.stats: dict[str, object] = {}

        def on_broker_event(self, event: BrokerEvent) -> None:
            self.events.append(event)

        def on_end(self, stats) -> None:
            self.stats = dict(stats)

        def get_analysis(self) -> dict[str, object]:
            return {
                "events": len(self.events),
                "broker_events": self.stats.get("broker_events"),
                "mode": self.stats.get("mode"),
            }

    bars_ = [
        StreamBar(
            ts=pd.Timestamp("2026-01-01", tz="UTC"),
            symbol="AAPL",
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.5,
            volume=1.0,
        )
    ]
    events = [BrokerEvent("status", order_id="A1", status="accepted")]

    cerebro = Cerebro(mode="live", live_poller=lambda: bars_, broker_event_poller=lambda: events)
    cerebro.addstrategy(CountingStrategy)
    cerebro.addanalyzer(EventRecorder, name="events")

    [strategy] = cerebro.run()

    assert strategy.closes == [1.5]
    assert strategy.analyzer_results["events"] == {
        "events": 1,
        "broker_events": 1,
        "mode": "live",
    }


def test_grid_search_runs_nested_mlflow_runs(monkeypatch) -> None:
    class ParamStrategy(Strategy):
        params = (("fast", 1),)

        def next(self) -> None:
            pass

    fake = FakeMLflow()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    results = grid_search(
        ParamStrategy,
        bars(),
        {"fast": [2, 4]},
        mlflow={"experiment": "grid", "mlflow_module": fake},
    )

    assert [result.params for result in results] == [{"fast": 2}, {"fast": 4}]
    assert fake.runs == [
        {"run_name": "ParamStrategy[0]", "nested": True},
        {"run_name": "ParamStrategy[1]", "nested": True},
    ]
    assert [params["strategy.fast"] for params in fake.params] == [2, 4]
