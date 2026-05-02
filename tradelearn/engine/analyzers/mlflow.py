from __future__ import annotations

import logging
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tradelearn.core import BrokerEvent
from tradelearn.report.artifacts import market_data_from_strategy, write_artifact_bundle

from ..analyzer import Analyzer

LOGGER = logging.getLogger(__name__)
DEFAULT_MLFLOW_URI = "http://127.0.0.1:5050"
_MLFLOW_PARAM_KEY_RE = re.compile(r"[^0-9A-Za-z_\-. :/]+")
_RESEARCH_PARAM_KEYS = {"research_result", "research_result_", "research_results"}

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
            mlflow.set_tracking_uri(uri)
            if self.p.experiment:
                mlflow.set_experiment(self.p.experiment)
            upload_artifacts = self.p.upload_artifacts
            if self.p.log_artifacts is not None:
                upload_artifacts = bool(self.p.log_artifacts)
            with mlflow.start_run(run_name=self.p.run_name, nested=bool(self.p.nested)):
                mlflow.log_params(_params_payload(self.strategy))
                mlflow.log_metrics(_metrics_payload(stats))
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


def _params_payload(strategy: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if strategy is None:
        return payload
    if hasattr(strategy, "p"):
        payload.update(_strategy_params(strategy.p.asdict()))
    broker = getattr(strategy, "broker", None)
    if broker is not None:
        payload["broker.cash"] = broker.getcash()
        payload["broker.value"] = broker.getvalue()
        payload["broker.commission"] = getattr(
            broker, "commission", getattr(broker, "commission_ratio", 0.0)
        )
    research_params = _research_params(strategy)
    if research_params:
        payload.update(_flatten_params("research", research_params))
    return payload


def _strategy_params(values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        if key in _RESEARCH_PARAM_KEYS:
            continue
        payload[f"strategy.{_mlflow_param_key(key)}"] = _json_scalar(value)
    return payload


def _metrics_payload(stats: Any) -> dict[str, float]:
    summary = _stats_summary(stats)
    payload: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metric = float(value)
        if math.isfinite(metric):
            payload[key] = metric
    return payload


def _stats_payload(stats: Any) -> dict[str, Any]:
    return {
        "summary": _stats_summary(stats),
        "analyzers": _stats_field(stats, "analyzers", {}),
        "config": _stats_field(stats, "config", {}),
    }


def _research_payload(strategy: Any) -> dict[str, Any]:
    result = _first_attr(strategy, ("research_result", "research_result_"))
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        return payload if isinstance(payload, dict) else {}
    return _result_payload(result)


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
    artifacts = getattr(result, "artifacts", None)
    if isinstance(artifacts, dict):
        payload["artifacts"] = artifacts
    return payload


def _result_payload(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    scores = getattr(result, "scores", None)
    if scores is not None:
        payload["scores"] = _series_dict(scores)
    selected = getattr(result, "selected", None)
    if selected is not None:
        payload["selected"] = [str(item) for item in selected]
    weights = getattr(result, "weights", None)
    if weights is not None:
        payload["weights"] = _series_dict(weights)
    return payload


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _series_dict(values: Any) -> dict[str, Any]:
    if hasattr(values, "to_dict"):
        return {str(key): _json_scalar(value) for key, value in values.to_dict().items()}
    if isinstance(values, dict):
        return {str(key): _json_scalar(value) for key, value in values.items()}
    return {}


def _flatten_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{_mlflow_param_key(key)}"
        if isinstance(value, dict):
            flattened.update(_flatten_params(name, value))
        elif isinstance(value, list | tuple):
            flattened[name] = ",".join(str(item) for item in value)
        else:
            flattened[name] = _json_scalar(value)
    return flattened


def _mlflow_param_key(value: Any) -> str:
    text = str(value).replace("%", "pct")
    text = _MLFLOW_PARAM_KEY_RE.sub("_", text).strip("._ ")
    return text or "value"


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, list | tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return value


def _coerce_broker_event(event: BrokerEvent | dict[str, Any]) -> BrokerEvent:
    return BrokerEvent.coerce(event)


def _stats_summary(stats: Any) -> dict[str, Any]:
    if isinstance(stats, dict):
        return dict(stats)
    return dict(_stats_field(stats, "summary", {}) or {})


def _stats_field(stats: Any, name: str, default: Any) -> Any:
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)
