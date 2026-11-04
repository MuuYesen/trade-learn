from __future__ import annotations

from typing import Any

from tradelearn.core.logging import get_logger
from tradelearn.report.mlflow import build_run_metrics, build_run_params, log_artifact_bundle

DEFAULT_MLFLOW_URI = "http://127.0.0.1:5050"
LOGGER = get_logger("lite.mlflow")


def log_lite_run(
    backtest: Any,
    *,
    experiment_name: str = "tradelearn-lite",
    run_name: str | None = None,
    uri: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    nested: bool = False,
    upload_artifacts: bool = True,
    log_artifacts: bool | None = None,
    artifact_path: str | None = None,
    log_report: bool = True,
    log_plot: bool = False,
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
    tracking_uri = uri or DEFAULT_MLFLOW_URI
    LOGGER.info(
        "MLflow logging started experiment=%s run=%s uri=%s artifacts=%s",
        experiment_name,
        run_name or "auto",
        tracking_uri,
        upload_artifacts if log_artifacts is None else bool(log_artifacts),
    )
    mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    if log_artifacts is not None:
        upload_artifacts = bool(log_artifacts)

    with mlflow.start_run(run_name=run_name, nested=bool(nested)):
        tag_payload = dict(tags or {})
        if tag_payload and hasattr(mlflow, "set_tags"):
            mlflow.set_tags(tag_payload)
        mlflow.log_params(
            build_run_params(
                stats,
                strategy=strategy,
                params=params,
                tags=tag_payload,
                include_strategy_params=False,
            )
        )
        mlflow.log_metrics(build_run_metrics(stats))
        if upload_artifacts:
            log_artifact_bundle(
                mlflow,
                stats=stats,
                strategy=strategy,
                artifact_path=artifact_path,
                log_report=log_report,
                log_plot=log_plot,
            )
    LOGGER.info(
        "MLflow logging finished experiment=%s run=%s artifacts=%s",
        experiment_name,
        run_name or "auto",
        upload_artifacts,
    )
    return "logged"


def _import_mlflow() -> Any:
    import mlflow

    return mlflow
