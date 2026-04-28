from __future__ import annotations

from pathlib import Path


def test_examples_contains_only_strategy_files() -> None:
    root = Path(__file__).parents[3]
    examples = root / "examples"
    allowed_names = {"__init__.py"}
    offenders: list[str] = []

    for path in examples.rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(examples)
        if path.suffix != ".py" or path.name not in allowed_names and not _looks_like_strategy(path):
            offenders.append(str(rel))

    assert offenders == []


def _looks_like_strategy(path: Path) -> bool:
    text = path.read_text()
    return "class " in text and "Strategy" in text
