from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

from tradelearn.report.artifacts import market_data_from_strategy, write_artifact_bundle

DEFAULT_MLFLOW_URI = "https://mlflow.leafquant.com"


def log_lite_run(
    backtest: Any,
    *,
    experiment_name: str = "tradelearn-lite",
    run_name: str | None = None,
    uri: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
    artifact_file: str = "stats.json",
    artifact_path: str = "tradelearn",
    artifact_bundle: bool = True,
    log_report: bool = True,
    log_plot: bool = True,
    mlflow_module: Any | None = None,
) -> str:
    """Log the latest Lite run to MLflow.

    Lite keeps MLflow as a post-run operation: run the backtest first, then
    upload the shared Stats payload and optional report artifacts.
    """

    stats = getattr(backtest, "_last_stats", None)
    strategy = getattr(backtest, "_last_strategy", None)
    if stats is None or strategy is None:
        raise RuntimeError("run() must be called before log_mlflow()")

    mlflow = mlflow_module or _import_mlflow()
    tracking_uri = uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
    mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=bool(nested)):
        tag_payload = dict(tags or {})
        if tag_payload and hasattr(mlflow, "set_tags"):
            mlflow.set_tags(tag_payload)
        mlflow.log_params(
            _params_payload(stats, strategy=strategy, params=params, tags=tag_payload)
        )
        mlflow.log_metrics(_metrics_payload(stats))
        mlflow.log_dict(_stats_payload(stats), artifact_file)
        research_payload = _research_payload(strategy)
        if research_payload:
            mlflow.log_dict(research_payload, "research.json")
        if artifact_bundle:
            _log_artifact_bundle(
                mlflow,
                stats=stats,
                strategy=strategy,
                artifact_path=artifact_path,
                log_report=log_report,
                log_plot=log_plot,
            )
    return "logged"


def _import_mlflow() -> Any:
    import mlflow

    return mlflow


def _log_artifact_bundle(
    mlflow: Any,
    *,
    stats: Any,
    strategy: Any,
    artifact_path: str | None,
    log_report: bool,
    log_plot: bool,
) -> None:
    if not hasattr(mlflow, "log_artifact"):
        return

    with tempfile.TemporaryDirectory(prefix="tradelearn-lite-mlflow-") as directory:
        artifacts = write_artifact_bundle(
            stats,
            Path(directory),
            strategy=strategy,
            market_data=market_data_from_strategy(strategy),
            log_report=log_report,
            log_plot=log_plot,
        )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact), artifact_path=artifact_path)


def _params_payload(
    stats: Any,
    *,
    strategy: Any,
    params: dict[str, Any] | None,
    tags: dict[str, Any],
) -> dict[str, Any]:
    payload = _flatten_params("config", getattr(stats, "config", {}))
    payload.update(_flatten_params("research", _research_params(strategy)))
    if params:
        payload.update(params)
    if tags:
        payload.update({f"tag.{key}": value for key, value in tags.items()})
    return payload


def _research_payload(strategy: Any) -> dict[str, Any]:
    result = _first_attr(strategy, ("research_result", "research_result_"))
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        return payload if isinstance(payload, dict) else {}
    return {}


def _research_params(strategy: Any) -> dict[str, Any]:
    result = _first_attr(strategy, ("research_result", "research_result_"))
    if result is None:
        return {}
    payload: dict[str, Any] = {}
    name = getattr(result, "name", None)
    if name:
        payload["name"] = str(name)
    params = getattr(result, "params", None)
    if isinstance(params, dict):
        payload.update(params)
    return payload


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _metrics_payload(stats: Any) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in getattr(stats, "summary", {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metric = float(value)
        if math.isfinite(metric):
            payload[key] = metric
    return payload


def _stats_payload(stats: Any) -> dict[str, Any]:
    return {
        "summary": dict(getattr(stats, "summary", {})),
        "analyzers": dict(getattr(stats, "analyzers", {})),
        "config": dict(getattr(stats, "config", {})),
    }


def _flatten_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            payload.update(_flatten_params(name, value))
        else:
            payload[name] = value
    return payload
