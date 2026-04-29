from __future__ import annotations

from scripts.generate_api_reference import (
    API_REFERENCE_MODULES,
    render_api_reference,
    render_engine_api_guide,
    render_lite_api_guide,
    write_api_reference,
)


def test_render_api_reference_contains_mkdocstrings_directives() -> None:
    rendered = render_api_reference()

    assert "::: tradelearn.engine" in rendered
    assert "::: tradelearn.lite" in rendered
    assert "::: tradelearn.ml" in rendered
    assert "::: tradelearn.report" in rendered
    assert "::: tradelearn.backtest" not in rendered
    assert "Compat Backtrader" not in rendered
    assert all(module.import_path in rendered for module in API_REFERENCE_MODULES)


def test_write_api_reference_creates_expected_page(tmp_path) -> None:
    output = write_api_reference(tmp_path)

    assert output == tmp_path / "api" / "reference.md"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# API Reference")


def test_render_engine_api_guide_includes_code_signatures_and_parameter_tables() -> None:
    rendered = render_engine_api_guide()

    assert "Generated from `tradelearn.engine` code signatures" in rendered
    assert "## `Cerebro.__init__`" in rendered
    assert "| `match_mode` | `str` | `'exact'` |" in rendered
    assert "## `Strategy.buy_bracket`" in rendered
    assert "| `stopprice` |" in rendered


def test_render_lite_api_guide_includes_code_signatures_and_parameter_tables() -> None:
    rendered = render_lite_api_guide()

    assert "Generated from `tradelearn.lite` code signatures" in rendered
    assert "## `Backtest.__init__`" in rendered
    assert "| `data` | `pd.DataFrame \\| dict[str, pd.DataFrame]` | `required` |" in rendered
    assert "## `Strategy.buy`" in rendered
    assert "| `ticker` | `str` | `None` |" in rendered
    assert "| `sl` | `float` | `None` |" in rendered
