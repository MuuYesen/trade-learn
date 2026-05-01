from __future__ import annotations

from pathlib import Path


def test_examples_contains_only_strategy_files() -> None:
    root = Path(__file__).parents[3]
    examples = root / "examples"
    allowed_names = {"__init__.py"}
    offenders: list[str] = []

    for path in examples.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts or "output" in path.parts:
            continue
        if path.name == ".DS_Store":
            continue
        rel = path.relative_to(examples)
        is_allowed_python = path.suffix == ".py" and (
            path.name in allowed_names or _looks_like_strategy(path)
        )
        if not is_allowed_python:
            offenders.append(str(rel))

    assert offenders == []


def test_full_workflow_examples_cover_research_to_backtest_flow() -> None:
    root = Path(__file__).parents[3]

    for name in ("full_workflow_lite.py", "full_workflow_engine.py"):
        source = (root / "examples" / name).read_text()
        assert "TradingViewProvider" in source
        assert "FactorAnalyzer" in source
        assert "factor_report" in source
        assert ".report(" in source
        assert ".plot(" in source
        assert "log_mlflow" in source or "MLflowAnalyzer" in source


def _looks_like_strategy(path: Path) -> bool:
    text = path.read_text()
    return "class " in text and "Strategy" in text
