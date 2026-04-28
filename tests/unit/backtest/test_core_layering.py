from __future__ import annotations

from pathlib import Path


def test_core_layer_does_not_import_compat() -> None:
    root = Path(__file__).parents[3]
    core_roots = [root / "tradelearn" / "backtest" / "core", root / "tradelearn" / "core"]
    offenders: list[str] = []

    for core_root in core_roots:
        for path in core_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text()
            if "tradelearn.compat" in text or "compat.backtrader" in text:
                offenders.append(str(path.relative_to(root)))

    assert offenders == []
