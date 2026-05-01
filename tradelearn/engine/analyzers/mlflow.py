from __future__ import annotations

import logging
import math
import os
from collections import Counter
from typing import Any

from tradelearn.core import BrokerEvent

from ..analyzer import Analyzer

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
        try:
            mlflow = self.p.mlflow_module or _import_mlflow()
            uri = self.p.uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
            mlflow.set_tracking_uri(uri)
            if self.p.experiment:
                mlflow.set_experiment(self.p.experiment)
            with mlflow.start_run(run_name=self.p.run_name, nested=bool(self.p.nested)):
                mlflow.log_params(_params_payload(self.strategy))
                mlflow.log_metrics(_metrics_payload(stats))
                mlflow.log_dict(_stats_payload(stats), self.p.artifact_file)
                pipeline_payload = _pipeline_payload(self.strategy)
                if pipeline_payload:
                    mlflow.log_dict(pipeline_payload, "pipeline.json")
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
        payload["broker.commission"] = getattr(
            broker, "commission", getattr(broker, "commission_ratio", 0.0)
        )
    pipeline_params = _pipeline_params(strategy)
    if pipeline_params:
        payload.update(_flatten_params("pipeline", pipeline_params))
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


def _pipeline_payload(strategy: Any) -> dict[str, Any]:
    pipeline = _first_attr(strategy, ("pipeline", "pipeline_"))
    params = _pipeline_params(strategy)
    result = _pipeline_result(strategy)
    explanation = _pipeline_explanation(pipeline)
    payload: dict[str, Any] = {}
    if params:
        payload["params"] = params
    if result:
        payload["result"] = result
    if explanation:
        payload["explanation"] = explanation
    return payload


def _pipeline_params(strategy: Any) -> dict[str, Any]:
    pipeline = _first_attr(strategy, ("pipeline", "pipeline_"))
    if pipeline is None or not hasattr(pipeline, "get_params"):
        return {}
    params = pipeline.get_params()
    return params if isinstance(params, dict) else {}


def _pipeline_result(strategy: Any) -> dict[str, Any]:
    result = _first_attr(strategy, ("pipeline_result", "pipeline_result_"))
    if result is None:
        return {}
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


def _pipeline_explanation(pipeline: Any) -> dict[str, Any]:
    if pipeline is None or not hasattr(pipeline, "explain"):
        return {}
    try:
        explanation = pipeline.explain()
    except Exception:
        return {}
    return _series_dict(explanation)


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
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_params(name, value))
        elif isinstance(value, list | tuple):
            flattened[name] = ",".join(str(item) for item in value)
        else:
            flattened[name] = _json_scalar(value)
    return flattened


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
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
