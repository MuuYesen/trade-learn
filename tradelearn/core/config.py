"""Configuration loading for CLI and runtime defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tradelearn.core.errors import ConfigurationError

DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.leafquant.com"
DEFAULT_DATA_CACHE_DIR = Path("./data")
DEFAULT_LOG_LEVEL = "INFO"


@dataclass(frozen=True)
class TradelearnConfig:
    """Resolved trade-learn configuration."""

    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    data_cache_dir: Path = DEFAULT_DATA_CACHE_DIR
    log_level: str = DEFAULT_LOG_LEVEL
    data_offline: bool = False
    cache_ttl_seconds: int | None = None


def load_config(
    *,
    project_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    include_user: bool = True,
) -> TradelearnConfig:
    """Load config from defaults, yaml files, and environment variables."""

    values: dict[str, Any] = {
        "mlflow_tracking_uri": DEFAULT_MLFLOW_TRACKING_URI,
        "data_cache_dir": DEFAULT_DATA_CACHE_DIR,
        "log_level": DEFAULT_LOG_LEVEL,
        "data_offline": False,
        "cache_ttl_seconds": None,
    }
    if include_user:
        _merge_yaml(values, Path.home() / ".tradelearn" / "config.yaml")
    if project_dir is not None:
        _merge_yaml(values, Path(project_dir) / ".tradelearn" / "config.yaml")
    if config_path is not None:
        _merge_yaml(values, Path(config_path))
    _merge_env(values, os.environ if env is None else env)
    return TradelearnConfig(
        mlflow_tracking_uri=str(values["mlflow_tracking_uri"]),
        data_cache_dir=Path(values["data_cache_dir"]),
        log_level=str(values["log_level"]),
        data_offline=bool(values["data_offline"]),
        cache_ttl_seconds=_optional_int(values["cache_ttl_seconds"], "cache_ttl_seconds"),
    )


def _merge_yaml(values: dict[str, Any], path: Path) -> None:
    if not path.exists():
        return
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Cannot parse config file {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Config file {path} must contain a mapping.")
    _merge_mapping(values, loaded, path)


def _merge_mapping(values: dict[str, Any], data: Mapping[str, Any], path: Path) -> None:
    allowed_top = {"mlflow", "data", "log_level"}
    for key in data:
        if key not in allowed_top:
            raise ConfigurationError(f"Unknown config key {key!r} in {path}")
    if "log_level" in data:
        values["log_level"] = data["log_level"]
    if "mlflow" in data:
        mlflow = _mapping(data["mlflow"], "mlflow", path)
        _reject_unknown(mlflow, {"tracking_uri"}, "mlflow", path)
        if "tracking_uri" in mlflow:
            values["mlflow_tracking_uri"] = mlflow["tracking_uri"]
    if "data" in data:
        data_config = _mapping(data["data"], "data", path)
        _reject_unknown(
            data_config,
            {"cache_dir", "offline", "cache_ttl_seconds"},
            "data",
            path,
        )
        if "cache_dir" in data_config:
            values["data_cache_dir"] = Path(str(data_config["cache_dir"]))
        if "offline" in data_config:
            values["data_offline"] = _bool(data_config["offline"], "data.offline")
        if "cache_ttl_seconds" in data_config:
            values["cache_ttl_seconds"] = _optional_int(
                data_config["cache_ttl_seconds"],
                "data.cache_ttl_seconds",
            )


def _merge_env(values: dict[str, Any], env: Mapping[str, str]) -> None:
    if "MLFLOW_TRACKING_URI" in env:
        values["mlflow_tracking_uri"] = env["MLFLOW_TRACKING_URI"]
    if "TRADELEARN_DATA_CACHE_DIR" in env:
        values["data_cache_dir"] = Path(env["TRADELEARN_DATA_CACHE_DIR"])
    if "TRADELEARN_LOG_LEVEL" in env:
        values["log_level"] = env["TRADELEARN_LOG_LEVEL"]
    if "TRADELEARN_DATA_OFFLINE" in env:
        values["data_offline"] = _bool(env["TRADELEARN_DATA_OFFLINE"], "TRADELEARN_DATA_OFFLINE")
    if "TRADELEARN_CACHE_TTL_SECONDS" in env:
        values["cache_ttl_seconds"] = _optional_int(
            env["TRADELEARN_CACHE_TTL_SECONDS"],
            "TRADELEARN_CACHE_TTL_SECONDS",
        )


def _mapping(value: Any, key: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"Config key {key!r} in {path} must be a mapping.")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], key: str, path: Path) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown config key {key}.{unknown[0]!r} in {path}")


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"{key} must be a boolean value.")


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be an integer.") from exc
