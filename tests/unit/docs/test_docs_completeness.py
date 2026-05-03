from __future__ import annotations

from pathlib import Path

from scripts.check_docs_completeness import check_docs_completeness


def test_docs_completeness_requires_current_public_navigation() -> None:
    report = check_docs_completeness(Path("."))

    assert report.ok
    assert report.missing_guide_pages == ()
    assert report.missing_design_note_pages == ()
    assert report.missing_benchmark_pages == ()
    assert report.missing_api_reference_pages == ()


def test_docs_navigation_keeps_lite_before_engine() -> None:
    report = check_docs_completeness(Path("."))

    assert report.nav_paths.index("guides/lite.md") < report.nav_paths.index("guides/engine.md")
    assert report.nav_paths.index("guides/lite-api.md") < report.nav_paths.index(
        "guides/engine-api.md"
    )
    assert report.nav_paths.index("api/reference/lite.md") < report.nav_paths.index(
        "api/reference/engine.md"
    )


def test_root_readme_is_chinese_default_and_links_to_english() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "./README_en.md" in text
    assert "English version" in text
    assert "Python 写策略与投研流程，Rust 扛事件驱动回测内核" in text
    assert "核心亮点" in text
    assert "Lite 是推荐起点" in text
    assert "Backtrader 风格 Engine" in text
    assert "MLflow / JupyterLab / MCP" in text


def test_english_readme_links_back_to_chinese_default() -> None:
    text = Path("README_en.md").read_text(encoding="utf-8")

    assert "./README.md" in text
    assert "中文主页" in text
    assert "Python for strategy and research, Rust for the event-driven backtest core" in text


def test_quickstart_is_compatible_with_homepage_positioning() -> None:
    text = Path("docs/quickstart.md").read_text(encoding="utf-8")

    assert "## trade-learn 是什么" in text
    assert "**Python** 表达策略、因子、模型、研究流程" in text
    assert "**Rust** 承担撮合、订单推进、bar runner、portfolio" in text
    assert "入口选择" in text
    assert "Lite 和 Engine **共享同一套 backtest runtime + Rust 撮合内核**" in text
    assert "## 核心能力" in text
    assert "## Lite：最短路径" in text
    assert "## Engine：Backtrader 风格" in text
    assert "## 下一步" in text


def test_legacy_chinese_readme_alias_is_removed() -> None:
    assert not Path("README_zh.md").exists()
