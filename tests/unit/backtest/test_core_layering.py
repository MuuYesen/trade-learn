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


def test_backtest_core_keeps_only_shared_runtime_modules() -> None:
    root = Path(__file__).parents[3]
    core_root = root / "tradelearn" / "backtest" / "core"

    assert not (core_root / "brokers").exists()
    assert not (core_root / "metrics.py").exists()
    assert not (core_root / "resampler.py").exists()

    assert (core_root / "broker.py").exists()
    assert (root / "tradelearn" / "metrics" / "engine.py").exists()
    assert (root / "tradelearn" / "data" / "resampler.py").exists()
