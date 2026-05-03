from __future__ import annotations

from scripts.generate_api_reference import (
    API_REFERENCE_MODULES,
    render_api_reference,
    render_engine_api_guide,
    render_lite_api_guide,
    render_module_reference,
    render_strategy_writing_guide,
    write_api_guides,
    write_api_reference,
    write_api_reference_pages,
)


def test_render_api_reference_is_readable_overview_not_module_dump() -> None:
    rendered = render_api_reference()

    assert "## 先看这里" in rendered
    assert "## 公开模块" in rendered
    assert "| 模块 | 用途 | 常用入口 | 完整 Reference |" in rendered
    assert "[Engine API 签名](../guides/engine-api.md)" in rendered
    assert "[Lite API 签名](../guides/lite-api.md)" in rendered
    assert "[策略编写指南](../guides/strategy.md)" in rendered
    assert "[`tradelearn.engine`](reference/engine.md)" in rendered
    assert "## 按模块列出公开符号" in rendered
    assert "`Cerebro`" in rendered
    assert "`DataProvider`" in rendered
    assert "`win_rate`" in rendered
    assert "::: tradelearn.engine" not in rendered
    assert "::: tradelearn.lite" not in rendered
    assert "::: tradelearn.backtest" not in rendered
    assert "Compat Backtrader" not in rendered
    assert all(module.import_path in rendered for module in API_REFERENCE_MODULES)


def test_render_module_reference_contains_single_mkdocstrings_directive() -> None:
    rendered = render_module_reference(API_REFERENCE_MODULES[0])

    assert rendered.startswith("# Lite Reference")
    assert "::: tradelearn.lite" in rendered
    assert "::: tradelearn.engine" not in rendered
    assert "show_source: false" in rendered


def test_write_api_reference_creates_expected_page(tmp_path) -> None:
    output = write_api_reference(tmp_path)

    assert output == tmp_path / "api" / "reference.md"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# API 参考")


def test_write_api_reference_pages_creates_module_pages(tmp_path) -> None:
    outputs = write_api_reference_pages(tmp_path)

    assert tmp_path / "api" / "reference.md" in outputs
    assert tmp_path / "api" / "reference" / "engine.md" in outputs
    assert tmp_path / "api" / "reference" / "lite.md" in outputs
    assert (tmp_path / "api" / "reference" / "engine.md").exists()


def test_write_api_guides_creates_strategy_writing_guide(tmp_path) -> None:
    outputs = write_api_guides(tmp_path)

    assert tmp_path / "guides" / "strategy.md" in outputs
    assert tmp_path / "guides" / "engine-api.md" in outputs
    assert tmp_path / "guides" / "lite-api.md" in outputs
    text = (tmp_path / "guides" / "strategy.md").read_text(encoding="utf-8")
    assert "# 策略编写指南" in text
    assert "Engine 策略" in text
    assert "Lite 策略" in text


def test_render_engine_api_guide_includes_code_signatures_and_parameter_tables() -> None:
    rendered = render_engine_api_guide()

    assert "Generated from `tradelearn.engine` code signatures" in rendered
    assert "## `Cerebro.__init__`" in rendered
    assert "| `match_mode` | `str` | `'exact'` |" in rendered
    assert "## `Strategy.buy_bracket`" in rendered
    assert "| `stopprice` |" in rendered
    assert "## Engine 完整接口" in rendered
    assert "| `Cerebro.addobserver` | method |" in rendered
    assert "| `Cerebro.getwriterinfo` | method |" in rendered
    assert "| `Strategy.close` | method |" in rendered
    assert "| `Strategy.getpositionbyname` | method |" in rendered


def test_render_lite_api_guide_includes_code_signatures_and_parameter_tables() -> None:
    rendered = render_lite_api_guide()

    assert "Generated from `tradelearn.lite` code signatures" in rendered
    assert "## `Backtest.__init__`" in rendered
    assert "| `data` | `pd.DataFrame \\| dict[str, pd.DataFrame]` | `required` |" in rendered
    assert "## `Strategy.buy`" in rendered
    assert "| `ticker` | `str` | `None` |" in rendered
    assert "| `sl` | `float` | `None` |" in rendered
    assert "## Lite 完整接口" in rendered
    assert "| `Strategy.target_weights` | method |" in rendered
    assert "| `Strategy.close_all` | method |" in rendered
    assert "| `Strategy.orders` | property |" in rendered
    assert "| `LiteDataProxy.df` | property |" in rendered
    assert "| `PositionProxy.pl_pct` | property |" in rendered


def test_render_strategy_writing_guide_explains_strategy_workflow() -> None:
    rendered = render_strategy_writing_guide()

    assert "## Engine 策略" in rendered
    assert "class SmaCross(bt.Strategy)" in rendered
    assert "## Lite 策略" in rendered
    assert "class SmaCross(tl.Strategy)" in rendered
    assert "`line[0]` 是当前 bar" in rendered
    assert "uv run python benchmarks/runners/benchmark_bt.py" in rendered
