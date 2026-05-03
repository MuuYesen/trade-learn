from __future__ import annotations

from pathlib import Path

import yaml


def test_mkdocs_material_config_targets_docs_dir() -> None:
    config_path = Path("mkdocs.yml")
    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["docs_dir"] == "docs"
    assert config["theme"]["name"] == "material"
    assert any(plugin == "mkdocstrings" or "mkdocstrings" in plugin for plugin in config["plugins"])
    assert any(item == {"Quickstart": "quickstart.md"} for item in config["nav"])
    assert any("API Reference" in item for item in config["nav"])


def test_mkdocs_mobile_drawer_uses_tradelearn_header_color() -> None:
    css = Path("docs/mkdocs/css/backtrader-header.css").read_text(encoding="utf-8")

    assert "--md-primary-fg-color: var(--tl-header)" in css
    assert ".md-nav--primary .md-nav__title[for=\"__drawer\"]" in css
    assert ".md-nav__source" in css
    assert "background-color: var(--tl-header-dark)" in css
