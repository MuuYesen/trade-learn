from __future__ import annotations

import logging
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tradelearn.core.broker_events import BrokerEvent
from tradelearn.report.artifacts import market_data_from_strategy, write_artifact_bundle
from tradelearn.report.mlflow import build_run_metrics, build_run_params

from ..analyzer import Analyzer

LOGGER = logging.getLogger(__name__)
DEFAULT_MLFLOW_URI = "http://127.0.0.1:5050"
class MLflowAnalyzer(Analyzer):
    """Log strategy params, broker settings, stats, and artifacts to MLflow."""

    params = (
        ("experiment", None),
        ("run_name", None),
        ("uri", None),
        ("nested", False),
        ("mlflow_module", None),
        ("log_mlflow", True),
        ("upload_artifacts", True),
        ("log_artifacts", None),
        ("artifact_path", None),
        ("log_report", True),
        ("log_plot", False),
        ("params", None),
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = "pending"
        self._message = ""
        self._live_event_counts: Counter[str] = Counter()
        self._live_replay_count = 0
        self._live_confirmation_count = 0
        self._live_max_notional: float | None = None
        self._live_risk_tags: set[str] = set()
        self._last_live_status: str | None = None
        self._last_live_order_id: Any = None

    def on_end(self, stats: Any) -> None:
        if not self.p.log_mlflow:
            self._status = "skipped"
            return
        try:
            mlflow = self.p.mlflow_module or _import_mlflow()
            uri = self.p.uri or DEFAULT_MLFLOW_URI
            LOGGER.info(
                "MLflow logging started experiment=%s run=%s uri=%s artifacts=%s",
                self.p.experiment or "default",
                self.p.run_name or "auto",
                uri,
                self.p.upload_artifacts if self.p.log_artifacts is None else bool(self.p.log_artifacts),
            )
            mlflow.set_tracking_uri(uri)
            if self.p.experiment:
                mlflow.set_experiment(self.p.experiment)
            upload_artifacts = self.p.upload_artifacts
            if self.p.log_artifacts is not None:
                upload_artifacts = bool(self.p.log_artifacts)
            with mlflow.start_run(run_name=self.p.run_name, nested=bool(self.p.nested)):
                mlflow.log_params(
                    build_run_params(
                        stats,
                        strategy=self.strategy,
                        params=self.p.params,
                    )
                )
                mlflow.log_metrics(build_run_metrics(stats))
                if upload_artifacts:
                    _log_artifact_bundle(
                        mlflow,
                        strategy=self.strategy,
                        stats=stats,
                        artifact_path=self.p.artifact_path,
                        log_report=bool(self.p.log_report),
                        log_plot=bool(self.p.log_plot),
                    )
            self._status = "logged"
            LOGGER.info(
                "MLflow logging finished experiment=%s run=%s artifacts=%s",
                self.p.experiment or "default",
                self.p.run_name or "auto",
                upload_artifacts,
            )
        except Exception as exc:  # pragma: no cover - exercised through fake module
            self._status = "warning"
            self._message = str(exc)
            LOGGER.warning("MLflow logging skipped: %s", exc)

    def get_analysis(self) -> dict[str, Any]:
        analysis = {"status": self._status}
        if self._message:
            analysis["message"] = self._message
        live_events = self.live_event_summary()
        if live_events["total"]:
            analysis["live_events"] = live_events
        return analysis

    def on_broker_event(self, event: BrokerEvent | dict[str, Any]) -> None:
        """Record paper/live broker event fields for later MLflow artifacts."""
        normalized = _coerce_broker_event(event)
        self._live_event_counts[normalized.kind] += 1
        if normalized.replay:
            self._live_replay_count += 1
        if normalized.requires_confirmation:
            self._live_confirmation_count += 1
        if normalized.max_notional is not None:
            self._live_max_notional = (
                normalized.max_notional
                if self._live_max_notional is None
                else max(self._live_max_notional, normalized.max_notional)
            )
        self._live_risk_tags.update(normalized.risk_tags)
        if normalized.status:
            self._last_live_status = normalized.status
        self._last_live_order_id = normalized.order_id

    def live_event_summary(self) -> dict[str, Any]:
        """Return accumulated paper/live broker event diagnostics."""
        return {
            "total": int(sum(self._live_event_counts.values())),
            "by_kind": dict(sorted(self._live_event_counts.items())),
            "replay": int(self._live_replay_count),
            "requires_confirmation": int(self._live_confirmation_count),
            "max_notional": self._live_max_notional,
            "risk_tags": sorted(self._live_risk_tags),
            "last_status": self._last_live_status,
            "last_order_id": self._last_live_order_id,
        }


def _import_mlflow() -> Any:
    import mlflow
    return mlflow


def _log_artifact_bundle(
    mlflow: Any,
    *,
    strategy: Any,
    stats: Any,
    artifact_path: str | None,
    log_report: bool,
    log_plot: bool,
) -> None:
    if not hasattr(mlflow, "log_artifact"):
        return

    with tempfile.TemporaryDirectory(prefix="tradelearn-mlflow-") as directory:
        artifacts = write_artifact_bundle(
            stats,
            Path(directory),
            strategy=strategy,
            market_data=market_data_from_strategy(strategy),
            log_report=log_report,
            log_plot=log_plot,
        )
        for artifact in artifacts:
            _log_path(mlflow, artifact, artifact_path)


def _log_path(mlflow: Any, path: Path, artifact_path: str | None) -> None:
    if path.is_dir():
        destination = _join_artifact_path(artifact_path, path.name)
        if hasattr(mlflow, "log_artifacts"):
            mlflow.log_artifacts(str(path), artifact_path=destination)
            return
        for child in sorted(path.iterdir()):
            if child.is_file():
                mlflow.log_artifact(str(child), artifact_path=destination)
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _join_artifact_path(base: str | None, name: str) -> str:
    return f"{base}/{name}" if base else name


def _coerce_broker_event(event: BrokerEvent | dict[str, Any]) -> BrokerEvent:
    return BrokerEvent.coerce(event)
