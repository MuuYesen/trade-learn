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

    assert report.nav_paths.index("README.md") < report.nav_paths.index("quickstart.md")
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

    assert "## 安装与环境" in text
    assert "## Lite：最短路径" in text
    assert "## Engine：Backtrader 风格" in text
    assert "## 多标的数据" in text
    assert "## 指标写法" in text
    assert "## 下一步" in text


def test_user_docs_include_runtime_boundaries_and_extension_guides() -> None:
    architecture = Path("docs/concepts/architecture.md").read_text(encoding="utf-8")
    data = Path("docs/guides/data.md").read_text(encoding="utf-8")
    extensions = Path("docs/guides/extensions.md").read_text(encoding="utf-8")
    live = Path("docs/guides/live.md").read_text(encoding="utf-8")
    contracts = Path("docs/internals/contracts.md").read_text(encoding="utf-8")

    assert "RustBroker 仅用于回测" in architecture
    assert "Cerebro.adddata(bars)" in data
    assert "NASDAQ:AAPL" in data
    assert "自定义 Engine 指标" in extensions
    assert "broker-neutral 协议" in extensions
    assert "from tradelearn.core" not in extensions
    assert "from tradelearn.backtest" not in extensions
    assert "OrderRequest" in live
    assert "不假设立即成交" in live
    assert "from tradelearn.core" not in live
    assert "from tradelearn.backtest" not in live
    assert "OrderRequest" in contracts
    assert "OrderStatusUpdate" in contracts


def test_legacy_chinese_readme_alias_is_removed() -> None:
    assert not Path("README_zh.md").exists()
