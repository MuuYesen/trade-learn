"""Shared MLflow payload helpers for Tradelearn frontends."""

from __future__ import annotations

import math
import re
from typing import Any

_MLFLOW_PARAM_KEY_RE = re.compile(r"[^0-9A-Za-z_\-. :/]+")
_RESEARCH_PARAM_KEYS = {"research_result", "research_result_", "research_results"}


def build_run_params(
    stats: Any,
    *,
    strategy: Any = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    include_strategy_params: bool = True,
) -> dict[str, Any]:
    """Return a stable MLflow param schema shared by Engine and Lite."""

    payload: dict[str, Any] = {}
    payload.update(_broker_params(stats, strategy))
    if include_strategy_params and strategy is not None and hasattr(strategy, "p"):
        payload.update(_strategy_params(strategy.p.asdict()))
    payload.update(_flatten_params("research", _research_params(strategy)))
    if params:
        payload.update(params)
    if tags:
        payload.update(
            {
                f"tag.{_mlflow_param_key(key)}": _json_scalar(value)
                for key, value in tags.items()
            }
        )
    return payload


def build_run_metrics(stats: Any) -> dict[str, float]:
    """Return finite numeric stats summary values suitable for MLflow metrics."""

    payload: dict[str, float] = {}
    for key, value in _stats_summary(stats).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metric = float(value)
        if math.isfinite(metric):
            payload[key] = metric
    return payload


def _broker_params(stats: Any, strategy: Any) -> dict[str, Any]:
    summary = _stats_summary(stats)
    config = _stats_field(stats, "config", {}) or {}
    broker_config = config.get("broker", {}) if isinstance(config, dict) else {}
    broker = getattr(strategy, "broker", None)

    initial_cash = _first_not_none(
        broker_config.get("cash") if isinstance(broker_config, dict) else None,
        getattr(broker, "_cash", None),
    )
    commission = _first_not_none(
        broker_config.get("commission") if isinstance(broker_config, dict) else None,
        getattr(broker, "commission", None),
        getattr(broker, "commission_ratio", None),
    )
    trade_on_close = _first_not_none(
        config.get("trade_on_close") if isinstance(config, dict) else None,
        getattr(broker, "_trade_on_close", None),
    )
    final_cash = _first_not_none(summary.get("final_cash"), _call_or_none(broker, "getcash"))
    final_value = _first_not_none(summary.get("final_value"), _call_or_none(broker, "getvalue"))

    payload: dict[str, Any] = {}
    if initial_cash is not None:
        payload["broker.initial_cash"] = _json_scalar(initial_cash)
    if final_cash is not None:
        payload["broker.final_cash"] = _json_scalar(final_cash)
    if final_value is not None:
        payload["broker.final_value"] = _json_scalar(final_value)
    if commission is not None:
        payload["broker.commission"] = _json_scalar(commission)
    if trade_on_close is not None:
        payload["broker.trade_on_close"] = bool(trade_on_close)
    return payload


def _strategy_params(values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        if key in _RESEARCH_PARAM_KEYS:
            continue
        payload[f"strategy.{_mlflow_param_key(key)}"] = _json_scalar(value)
    return payload


def _research_params(strategy: Any) -> dict[str, Any]:
    result = _first_attr(strategy, ("research_result", "research_result_"))
    if result is None:
        return {}

    payload: dict[str, Any] = {}
    name = getattr(result, "name", None)
    if name:
        payload["name"] = str(name)

    steps = getattr(result, "steps", None)
    if steps:
        payload["steps"] = ",".join(str(getattr(step, "name", step)) for step in steps)

    params = getattr(result, "params", None)
    if isinstance(params, dict):
        payload.update(_normalize_research_params(params))

    artifacts = getattr(result, "artifacts", None)
    if isinstance(artifacts, dict):
        compact_artifacts = _compact_artifact_params(artifacts)
        if compact_artifacts:
            payload["artifacts"] = compact_artifacts
    return payload


def _normalize_research_params(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        key_text = str(key)
        if key_text == "Pipeline.steps":
            target = "steps"
        elif key_text.startswith("Allocator.select."):
            target = "select." + key_text.removeprefix("Allocator.select.")
        elif key_text.startswith("Allocator.weight."):
            target = "weight." + key_text.removeprefix("Allocator.weight.")
        elif key_text.startswith("Allocator.constrain."):
            target = "constraints." + key_text.removeprefix("Allocator.constrain.")
        elif key_text.startswith("select_top."):
            target = "select." + key_text.removeprefix("select_top.")
        elif key_text.startswith("equal_weight."):
            target = "weight." + key_text.removeprefix("equal_weight.")
        elif key_text.startswith("apply_constraints."):
            target = "constraints." + key_text.removeprefix("apply_constraints.")
        else:
            target = key_text
        normalized[target] = value
    return normalized


def _compact_artifact_params(values: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            continue
        compact[str(key)] = value
    return compact


def _flatten_params(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{_mlflow_param_key(key)}"
        if isinstance(value, dict):
            payload.update(_flatten_params(name, value))
        elif isinstance(value, list | tuple):
            payload[name] = ",".join(str(_json_scalar(item)) for item in value)
        else:
            payload[name] = _json_scalar(value)
    return payload


def _mlflow_param_key(value: Any) -> str:
    text = str(value).replace("%", "pct")
    text = _MLFLOW_PARAM_KEY_RE.sub("_", text).strip("._ ")
    return text or "value"


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return ",".join(str(_json_scalar(item)) for item in value)
    return value


def _stats_summary(stats: Any) -> dict[str, Any]:
    if isinstance(stats, dict):
        return dict(stats)
    return dict(_stats_field(stats, "summary", {}) or {})


def _stats_field(stats: Any, name: str, default: Any) -> Any:
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    if obj is None:
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _call_or_none(obj: Any, name: str) -> Any:
    func = getattr(obj, name, None)
    if callable(func):
        return func()
    return None
