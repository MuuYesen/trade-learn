from __future__ import annotations

from pathlib import Path

import pytest

from tradelearn.core.config import TradelearnConfig, load_config
from tradelearn.core.errors import ConfigurationError


def test_load_config_merges_user_project_and_environment(tmp_path, monkeypatch) -> None:
    user_home = tmp_path / "home"
    project = tmp_path / "project"
    (user_home / ".tradelearn").mkdir(parents=True)
    (project / ".tradelearn").mkdir(parents=True)
    (user_home / ".tradelearn" / "config.yaml").write_text(
        "data:\n  cache_dir: user-cache\n  offline: true\nlog_level: WARNING\n",
        encoding="utf-8",
    )
    (project / ".tradelearn" / "config.yaml").write_text(
        "data:\n  cache_ttl_seconds: 60\nmlflow:\n  tracking_uri: http://project-mlflow\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("TRADELEARN_DATA_CACHE_DIR", "env-cache")
    monkeypatch.setenv("TRADELEARN_LOG_LEVEL", "DEBUG")

    config = load_config(project_dir=project)

    assert config == TradelearnConfig(
        mlflow_tracking_uri="http://project-mlflow",
        data_cache_dir=Path("env-cache"),
        log_level="DEBUG",
        data_offline=True,
        cache_ttl_seconds=60,
    )


def test_load_config_rejects_unknown_keys(tmp_path) -> None:
    project = tmp_path / "project"
    (project / ".tradelearn").mkdir(parents=True)
    (project / ".tradelearn" / "config.yaml").write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Unknown config key"):
        load_config(project_dir=project, include_user=False)
