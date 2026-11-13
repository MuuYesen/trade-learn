"""Record completed TradeLearn results in native MLflow runs.

No training, backtest, market-data retrieval or model publication happens here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

_SECRET = re.compile(r"password|secret|token|credential|api.?key|private.?key", re.I)


@dataclass(frozen=True)
class LogResult:
    """Actual tracking identity and status, including transport failures."""

    run_id: str | None
    status: str
    error: str | None = None
    reused: bool = False
    run_status: str | None = None


def log(
    result_or_engine_strategies: Any,
    *,
    research_result: Any = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    uri: str | None = None,
    log_report: bool = False,
    attachments: Sequence[str | Path] = (),
) -> LogResult:
    """Log supplied results once, reusing a compatible native active run.

    Invalid inputs/configuration raise before writes. Tracking failures return
    FAILED with the actual run id if one was allocated. Only runs created here
    are ended here. Reports and attachments require explicit opt-in.
    """
    primary = result_or_engine_strategies
    if _field(primary, "summary") is None and _field(primary, "steps") is not None:
        if research_result is not None and research_result is not primary:
            raise ValueError("conflicting research results")
        research_result = primary
        results = []
    else:
        results = _results(primary)
    paths = [Path(path) for path in attachments]
    if any(not (path.is_file() or path.is_dir()) for path in paths):
        raise ValueError("attachments must be existing files or directories")
    if len({path.name.casefold() for path in paths}) != len(paths):
        raise ValueError("attachment names must be unique")
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow integration requires trade-learn[research]") from exc

    active = mlflow.active_run()
    if active is not None:
        _check_active(mlflow, active, uri, experiment_name, run_name)
    owned = active is None
    run_id = None if active is None else active.info.run_id
    try:
        if owned:
            if uri is not None:
                mlflow.set_tracking_uri(uri)
            if experiment_name is not None:
                mlflow.set_experiment(experiment_name)
            active = mlflow.start_run(run_name=run_name)
            run_id = active.info.run_id
        from tradelearn import __version__

        mlflow.set_tags({"tradelearn.version": __version__})
        for index, (stats, strategy) in enumerate(results):
            prefix = f"strategy.{index}." if len(results) > 1 else ""
            summary = _field(stats, "summary", {})
            metrics = {
                prefix + str(key): float(value)
                for key, value in summary.items()
                if isinstance(value, Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            mlflow.log_metrics(metrics)
            supplied_metadata = _field(stats, "metric_metadata", {}) or {}
            metric_tags = {}
            for name, value in summary.items():
                if not isinstance(value, Real) or isinstance(value, bool):
                    continue
                metadata = supplied_metadata.get(name, {})
                metric_tags[f"tradelearn.metric.{prefix}{name}.unit"] = str(
                    metadata.get("unit", "unspecified")
                )
                for field in ("pnl_basis", "periods", "period_source"):
                    detail = metadata.get(field)
                    if isinstance(detail, (str, int, float)) and not isinstance(detail, bool):
                        metric_tags[f"tradelearn.metric.{prefix}{name}.{field}"] = str(detail)
                if not math.isfinite(float(value)):
                    metric_tags[f"tradelearn.metric.{prefix}{name}.reason"] = str(
                        metadata.get("reason", "not_calculable")
                    )
            if metric_tags:
                mlflow.set_tags(metric_tags)
            config_params: dict[str, Any] = {}
            _flatten(config_params, prefix + "config", _field(stats, "config", {}) or {})
            parameters = getattr(strategy, "p", None)
            if parameters is not None and callable(getattr(parameters, "asdict", None)):
                _flatten(config_params, prefix + "strategy", parameters.asdict())
            if config_params:
                mlflow.log_params(config_params)
            if log_report:
                _report(mlflow, stats, prefix)
        if research_result is not None:
            mlflow.log_params(_research_params(research_result))
            trials = _research_trials(research_result)
            if trials:
                mlflow.log_dict(_redact_trials(trials), 'search/trials.json')
        for path in paths:
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path="attachments/" + path.name)
            else:
                mlflow.log_artifact(str(path), artifact_path="attachments")
        if owned:
            mlflow.end_run(status="FINISHED")
        status = "FINISHED" if owned else "RUNNING"
        return LogResult(run_id, status, reused=not owned, run_status=status)
    except Exception as exc:
        error = str(exc)
        run_status = "RUNNING" if not owned else None
        if owned and run_id is not None:
            try:
                mlflow.end_run(status="FAILED")
                run_status = "FAILED"
            except Exception as close_error:
                error += f"; ending owned Run failed: {close_error}"
        return LogResult(run_id, "FAILED", error, reused=not owned, run_status=run_status)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a completed result's field without running any computation."""
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _results(value: Any) -> list[tuple[Any, Any]]:
    """Normalize completed Lite/Stats/Engine results, rejecting unrun objects."""
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError("at least one completed result is required")
    results = []
    for item in values:
        stats = getattr(item, "_last_stats", None)
        if stats is None:
            stats = getattr(item, "stats", item)
        summary = _field(stats, "summary")
        if not isinstance(summary, Mapping):
            raise TypeError("expected completed result with summary; run first")
        strategy = getattr(item, "_last_strategy", None)
        if strategy is None:
            strategy = getattr(item, "strategy", item)
        results.append((stats, strategy))
    return results


def _check_active(mlflow: Any, active: Any, uri: Any, experiment: Any, name: Any) -> None:
    """Reject conflicting configuration without changing native run state."""
    if uri is not None and uri != mlflow.get_tracking_uri():
        raise ValueError("uri conflicts with active run")
    if experiment is not None:
        existing = mlflow.get_experiment_by_name(experiment)
        if existing is None or str(existing.experiment_id) != str(active.info.experiment_id):
            raise ValueError("experiment_name conflicts with active run")
    if name is not None:
        current = mlflow.get_run(active.info.run_id).data.tags.get("mlflow.runName")
        if name != current:
            raise ValueError("run_name conflicts with active run")


def _research_trials(result):
    artifacts = _field(result, 'artifacts', {}) or {}
    if 'trials' in artifacts:
        return artifacts['trials']
    return [_field(step, 'params', {}) for step in _field(result, 'steps', ())
            if _field(step, 'category') == 'search.trial']


def _redact_trials(value):
    if isinstance(value, Mapping):
        return {str(key): _redact_trials(item) for key, item in value.items()
                if not _SECRET.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_redact_trials(item) for item in value]
    return value


def _research_params(result: Any) -> dict[str, Any]:
    """Record only supplied metadata, excluding raw arrays/models and secrets."""
    params: dict[str, Any] = {}
    name = _field(result, "name")
    if name is not None:
        params["research.name"] = str(name)
    _flatten(params, "research", _field(result, "params", {}) or {})
    for index, step in enumerate(_field(result, "steps", ())):
        prefix = f"research.step.{index}"
        for key in ("name", "category"):
            value = _field(step, key)
            if value is not None:
                params[f"{prefix}.{key}"] = str(value)
        _flatten(params, prefix + ".params", _field(step, "params", {}) or {})
    artifacts = _field(result, "artifacts", {}) or {}
    components = artifacts.get("components", {})
    if isinstance(components, Mapping):
        _flatten(params, "research.components", components)
    return params


def _flatten(output: dict, prefix: str, values: Mapping) -> None:
    """Keep bounded scalar metadata only, with credential-key filtering."""
    for key, value in values.items():
        if _SECRET.search(str(key)):
            continue
        name = prefix + "." + str(key)
        if isinstance(value, Mapping):
            _flatten(output, name, value)
        elif value is None or isinstance(value, (str, bool, int, float)):
            output[name] = str(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) for x in value
        ):
            output[name] = str(value)


def _report(mlflow: Any, stats: Any, prefix: str) -> None:
    """Render an explicitly requested report without uploading source bars."""
    import tempfile

    from tradelearn.report import Reporter

    with tempfile.TemporaryDirectory(prefix="tradelearn-report-") as directory:
        path = Reporter(stats).report(Path(directory) / "report.html")
        mlflow.log_artifact(str(path), artifact_path=prefix.rstrip(".") or "report")
