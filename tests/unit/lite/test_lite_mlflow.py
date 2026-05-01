from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from tradelearn.lite import Backtest, Strategy


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
        artifact_path="lite",
        artifact_bundle=True,
        log_report=False,
        log_plot=False,
        mlflow_module=fake,
    )

    assert status == "logged"
    assert fake.tracking_uris == ["https://mlflow.example"]
    assert fake.experiments == ["lite-exp"]
    assert fake.runs == [{"run_name": "lite-run", "nested": False}]
    assert fake.params[0]["fast"] == 3
    assert fake.params[0]["config.broker.cash"] == 1000.0
    assert fake.params[0]["config.broker.commission"] == 0.001
    assert fake.params[0]["tag.mode"] == "lite"
    assert fake.metrics[0]["final_value"] == stats.summary["final_value"]
    assert fake.metrics[0]["total_orders"] == 1.0
    payload, artifact_file = fake.dicts[0]
    assert artifact_file == "stats.json"
    assert payload["summary"] == stats.summary
    assert payload["config"] == stats.config
    assert sorted(name for name, _ in fake.artifacts) == [
        "equity.parquet",
        "trades.parquet",
    ]
    assert {artifact_path for _, artifact_path in fake.artifacts} == {"lite"}
