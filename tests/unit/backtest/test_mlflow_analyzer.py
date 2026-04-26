from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

from tradelearn.backtest import Cerebro, Strategy, grid_search
from tradelearn.backtest.analyzers import MLflowAnalyzer


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
            "total_trades": 0.0,
            "total_orders": 0.0,
            "total_fills": 0.0,
        }
    ]
    assert fake.dicts == [
        (
            {
                "summary": {
                    "bars": 3,
                    "final_cash": 123.0,
                    "final_value": 123.0,
                    "total_trades": 0,
                    "total_orders": 0,
                    "total_fills": 0,
                },
                "analyzers": {},
                "config": {
                    "callback_batch": 1,
                    "trade_on_close": False,
                    "exactbars": False,
                    "stdstats": True,
                    "broker": {"cash": 123.0, "commission": 0.001},
                },
            },
            "stats.json",
        )
    ]
    assert strategy.analyzer_results["mlflow"]["status"] == "logged"


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
