from __future__ import annotations

from scripts.generate_api_reference import (
    API_REFERENCE_MODULES,
    render_api_reference,
    write_api_reference,
)


def test_render_api_reference_contains_mkdocstrings_directives() -> None:
    rendered = render_api_reference()

    assert "::: tradelearn.backtest" in rendered
    assert "::: tradelearn.compat.backtrader" in rendered
    assert "::: tradelearn.ml" in rendered
    assert "::: tradelearn.report" in rendered
    assert all(module.import_path in rendered for module in API_REFERENCE_MODULES)


def test_write_api_reference_creates_expected_page(tmp_path) -> None:
    output = write_api_reference(tmp_path)

    assert output == tmp_path / "api" / "reference.md"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# API Reference")
