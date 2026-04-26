from __future__ import annotations

from scripts.generate_comparison_page import (
    COMPARISON_PROJECTS,
    render_comparison_page,
    write_comparison_page,
)


def test_render_comparison_page_covers_required_competitors() -> None:
    rendered = render_comparison_page()
    names = {project.name for project in COMPARISON_PROJECTS}

    assert names == {"qlib", "vnpy", "backtrader", "nautilus"}
    assert "vs qlib / vnpy / backtrader / nautilus" in rendered
    assert "| qlib |" in rendered
    assert "| vnpy |" in rendered
    assert "| backtrader |" in rendered
    assert "| nautilus |" in rendered


def test_render_comparison_page_states_tradelearn_positioning() -> None:
    rendered = render_comparison_page()

    assert "Python 量化研究框架" in rendered
    assert "传统策略与 ML 策略共用 API" in rendered
    assert "compat.backtrader" in rendered
    assert "Rust 事件型撮合核" in rendered
    assert "QMT" in rendered


def test_write_comparison_page_creates_expected_markdown(tmp_path) -> None:
    output = write_comparison_page(tmp_path)

    assert output == tmp_path / "comparison.md"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# Comparison")
