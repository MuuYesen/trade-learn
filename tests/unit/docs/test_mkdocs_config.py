from __future__ import annotations

from pathlib import Path

import yaml


def test_mkdocs_material_config_targets_v2_docs() -> None:
    config_path = Path("mkdocs.yml")
    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["docs_dir"] == "docs"
    assert "PROGRESS.md" in config["exclude_docs"]
    assert config["theme"]["name"] == "material"
    assert any(plugin == "mkdocstrings" or "mkdocstrings" in plugin for plugin in config["plugins"])
    assert any(item == {"Quickstart": "README.md"} for item in config["nav"])
    assert any("Specs" in item for item in config["nav"])
