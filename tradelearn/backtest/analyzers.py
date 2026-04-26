"""Built-in analyzers for the backtest facade."""

from __future__ import annotations

import logging
import os
from typing import Any

from tradelearn.backtest.engine import Analyzer

LOGGER = logging.getLogger(__name__)
DEFAULT_MLFLOW_URI = "https://mlflow.leafquant.com"


class MLflowAnalyzer(Analyzer):
    """Log strategy params, broker settings, stats, and artifacts to MLflow."""

    params = (
        ("experiment", None),
        ("run_name", None),
        ("uri", None),
        ("nested", False),
        ("mlflow_module", None),
        ("artifact_file", "stats.json"),
    )

    def __init__(self) -> None:
        self._status = "pending"
        self._message = ""

    def on_end(self, stats: dict[str, Any]) -> None:
        try:
            mlflow = self.p.mlflow_module or _import_mlflow()
            uri = self.p.uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
            mlflow.set_tracking_uri(uri)
            if self.p.experiment:
                mlflow.set_experiment(self.p.experiment)
            with mlflow.start_run(run_name=self.p.run_name, nested=bool(self.p.nested)):
                mlflow.log_params(_params_payload(self.strategy))
                mlflow.log_metrics(_metrics_payload(stats))
                mlflow.log_dict(dict(stats), self.p.artifact_file)
            self._status = "logged"
        except Exception as exc:  # pragma: no cover - exercised through fake module
            self._status = "warning"
            self._message = str(exc)
            LOGGER.warning("MLflow logging skipped: %s", exc)

    def get_analysis(self) -> dict[str, Any]:
        analysis = {"status": self._status}
        if self._message:
            analysis["message"] = self._message
        return analysis


def _import_mlflow() -> Any:
    import mlflow

    return mlflow


def _params_payload(strategy: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if strategy is None:
        return payload
    if hasattr(strategy, "p"):
        payload.update(
            {f"strategy.{key}": value for key, value in strategy.p.asdict().items()}
        )
    broker = getattr(strategy, "broker", None)
    if broker is not None:
        payload["broker.cash"] = broker.getcash()
        payload["broker.value"] = broker.getvalue()
        payload["broker.commission"] = getattr(broker, "commission", 0.0)
    return payload


def _metrics_payload(stats: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in stats.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
