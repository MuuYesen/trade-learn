"""Post-run integration tested against MLflow's real local tracking backend."""

import os
from types import SimpleNamespace

import pytest


def stats():
    return SimpleNamespace(summary={"return_pct": 1.234567890123, "missing": float("nan")})


@pytest.fixture
def tracking(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    previous = mlflow.get_tracking_uri()
    previous_env = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    mlflow.set_experiment("Default")
    yield mlflow
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(previous)
    if previous_env is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous_env


def test_owned_run_returns_real_identity_and_only_provided_metrics(tracking):
    from tradelearn.integrations.mlflow import log

    result = log(stats(), experiment_name="test")
    assert result.status == "FINISHED" and result.error is None
    stored = tracking.get_run(result.run_id)
    assert stored.data.metrics == {"return_pct": 1.234567890123}
    assert tracking.active_run() is None


def test_reuses_external_run_and_rejects_conflicting_configuration(tracking):
    from tradelearn.integrations.mlflow import log

    tracking.set_experiment("external")
    with tracking.start_run(run_name="native") as active:
        result = log(stats(), experiment_name="external", run_name="native")
        assert result.run_id == active.info.run_id and result.status == "RUNNING"
        assert tracking.active_run().info.run_id == active.info.run_id
        for options in (
            {"uri": "http://different.invalid"},
            {"experiment_name": "other"},
            {"run_name": "other"},
        ):
            with pytest.raises(ValueError, match="active run"):
                log(stats(), **options)
        assert tracking.active_run().info.run_id == active.info.run_id


def test_engine_list_uses_one_run_and_does_not_call_runtime(tracking):
    from tradelearn.integrations.mlflow import log

    def forbidden():
        raise AssertionError("must not execute runtime")

    strategies = [
        SimpleNamespace(stats=stats(), run=forbidden),
        SimpleNamespace(stats=stats(), fit=forbidden),
    ]
    result = log(strategies)
    assert tracking.get_run(result.run_id).data.metrics == {
        "strategy.0.return_pct": 1.234567890123,
        "strategy.1.return_pct": 1.234567890123,
    }
    assert not tracking.MlflowClient().list_artifacts(result.run_id)


def test_research_only_records_explicit_metadata_without_raw_data(tracking):
    from tradelearn.integrations.mlflow import log

    research = SimpleNamespace(
        name="research",
        steps=[
            SimpleNamespace(
                name="fit", category="model", params={"token": "PRIVATE", "columns": ["x"]}
            )
        ],
        params={"api_key": "PRIVATE", "folds": 3},
        features=object(),
        model=object(),
    )
    result = log(stats(), research_result=research)
    params = tracking.get_run(result.run_id).data.params
    assert params["research.name"] == "research" and params["research.folds"] == "3"
    assert "PRIVATE" not in repr(params)
    assert params["research.step.0.name"] == "fit"
    assert not tracking.MlflowClient().list_artifacts(result.run_id)


def test_empty_or_unrun_input_is_rejected_before_tracking(tracking):
    from tradelearn.integrations.mlflow import log

    for value in ([], object(), SimpleNamespace(_last_stats=None)):
        with pytest.raises((ValueError, TypeError), match="result|run"):
            log(value)
    assert tracking.active_run() is None


def test_tracking_failure_returns_error_and_preserves_external_run(tracking, monkeypatch):
    from tradelearn.integrations.mlflow import log

    def fail(*args, **kwargs):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(tracking, "log_metrics", fail)
    with tracking.start_run() as active:
        result = log(stats())
        assert result.run_id == active.info.run_id
        assert result.status == "FAILED" and result.error == "tracking unavailable"
        assert result.run_status == "RUNNING"
        assert tracking.active_run().info.run_id == active.info.run_id
    result = log(stats())
    assert result.status == "FAILED" and result.run_id
    assert tracking.get_run(result.run_id).info.status == "FAILED"
    assert tracking.active_run() is None


def test_research_result_alone_records_actual_components_and_steps(tracking):
    from tradelearn.integrations.mlflow import log
    from tradelearn.research.run import ResearchResult, ResearchStep

    research = ResearchResult(
        name="standalone",
        steps=[ResearchStep(name="score", category="scorer")],
        artifacts={"components": {"model": {"type": "LinearRegression", "status": "fitted"}}},
    )
    result = log(research)
    stored = tracking.get_run(result.run_id)
    assert stored.data.metrics == {}
    assert stored.data.params["research.components.model.type"] == "LinearRegression"
    assert stored.data.params["research.step.0.category"] == "scorer"


def test_completed_configuration_and_strategy_params_exclude_secret_keys(tracking):
    from tradelearn.integrations.mlflow import log

    result_stats = stats()
    result_stats.config = {"broker": {"cash": 1000}, "api_token": "PRIVATE", "source": "frozen"}
    strategy = SimpleNamespace(
        stats=result_stats, p=SimpleNamespace(asdict=lambda: {"fast": 3, "password": "PRIVATE"})
    )
    recorded = log([strategy])
    params = tracking.get_run(recorded.run_id).data.params
    assert params["config.broker.cash"] == "1000"
    assert params["strategy.fast"] == "3"
    assert "PRIVATE" not in repr(params)


def test_provided_metric_units_and_missing_reasons_reach_native_tracking(tracking):
    from tradelearn.integrations.mlflow import log

    result = stats()
    result.metric_metadata = {
        "return_pct": {
            "unit": "percent",
            "pnl_basis": "pnlcomm",
            "periods": 252,
            "period_source": "default_per_bar_profile",
        },
        "missing": {"unit": "ratio", "reason": "no_baseline"},
    }
    recorded = log(result)
    tags = tracking.get_run(recorded.run_id).data.tags
    assert tags["tradelearn.metric.return_pct.unit"] == "percent"
    assert tags["tradelearn.metric.return_pct.pnl_basis"] == "pnlcomm"
    assert tags["tradelearn.metric.return_pct.periods"] == "252"
    assert tags["tradelearn.metric.return_pct.period_source"] == "default_per_bar_profile"
    assert tags["tradelearn.metric.missing.reason"] == "no_baseline"


def test_long_provided_metadata_is_not_silently_truncated(tracking):
    from tradelearn.integrations.mlflow import log

    research = SimpleNamespace(name="long", steps=[], params={"description": "x" * 1000})
    recorded = log(stats(), research_result=research)
    assert tracking.get_run(recorded.run_id).data.params["research.description"] == "x" * 1000


def test_explicit_complete_bundle_directory_upload_and_name_conflict(tracking, tmp_path):
    from tradelearn.integrations.mlflow import log

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"schema":1}')
    (bundle / "state.bin").write_bytes(b"fitted fixture")
    recorded = log(stats(), attachments=[bundle])
    assert recorded.error is None, recorded
    entries = tracking.MlflowClient().list_artifacts(recorded.run_id, "attachments/bundle")
    assert {entry.path for entry in entries} == {
        "attachments/bundle/manifest.json",
        "attachments/bundle/state.bin",
    }
    other = tmp_path / "other"
    other.mkdir()
    (other / "manifest.json").write_text("other")
    with pytest.raises(ValueError, match="attachment"):
        log(stats(), attachments=[bundle / "manifest.json", other / "manifest.json"])


def test_real_grid_reuses_one_external_native_run_after_all_combinations(tracking):
    import pandas as pd

    from tradelearn.engine import Strategy
    from tradelearn.integrations.search import grid_search

    class Parameterized(Strategy):
        params = (("period", 1),)

        def next(self):
            pass

    bars = pd.DataFrame(
        {name: [10.0, 11.0, 12.0] for name in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=3),
    )
    with tracking.start_run(run_name="native-grid-owner") as active:
        results = grid_search(Parameterized, bars, {"period": [2, 4]}, mlflow={})
        assert tracking.active_run().info.run_id == active.info.run_id
        assert all(result.strategy.mlflow_result.run_id == active.info.run_id for result in results)
        assert all(result.strategy.mlflow_result.status == "RUNNING" for result in results)
        run = tracking.get_run(active.info.run_id)
        assert run.data.params["strategy.0.strategy.period"] == "2"
        assert run.data.params["strategy.1.strategy.period"] == "4"
        assert len(tracking.search_runs([active.info.experiment_id])) == 1


def test_final_cli_default_local_tracking_needs_no_service_and_keeps_active_run(
    tmp_path, monkeypatch
):
    """Use the exact lab configuration with real MLflow, without a server."""
    from tradelearn.core.config import TradelearnConfig
    from tradelearn.integrations.mlflow import log

    mlflow = pytest.importorskip("mlflow")
    previous = mlflow.get_tracking_uri()
    previous_env = os.environ.get("MLFLOW_TRACKING_URI")
    monkeypatch.chdir(tmp_path)
    try:
        uri = TradelearnConfig().mlflow_tracking_uri
        assert uri == "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(uri)
        owned = log(stats(), experiment_name="final-cli-local")
        assert owned.status == "FINISHED", owned.error
        assert (tmp_path / "mlflow.db").is_file()
        with mlflow.start_run(run_name="external") as active:
            result = log(stats())
            assert result.run_id == active.info.run_id and result.reused
            assert mlflow.active_run().info.run_id == active.info.run_id
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)
        if previous_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_env
