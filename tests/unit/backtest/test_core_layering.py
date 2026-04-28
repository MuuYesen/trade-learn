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


def test_platform_core_does_not_import_backtest_or_facades() -> None:
    root = Path(__file__).parents[3]
    platform_core = root / "tradelearn" / "core"
    offenders: list[str] = []

    for path in platform_core.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        if "tradelearn.backtest" in text or "tradelearn.compat" in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_backtest_core_keeps_only_shared_runtime_modules() -> None:
    root = Path(__file__).parents[3]
    core_root = root / "tradelearn" / "backtest" / "core"
    expected_modules = {
        "broker.py",
        "data.py",
        "engine.py",
        "event_runner.py",
        "indicator_cache.py",
        "lines.py",
        "models.py",
        "sizer.py",
        "strategy.py",
    }

    assert not (core_root / "brokers").exists()
    assert not (core_root / "metrics.py").exists()
    assert not (core_root / "resampler.py").exists()

    actual_modules = {
        path.name
        for path in core_root.glob("*.py")
        if path.name != "__init__.py"
    }
    assert actual_modules == expected_modules

    assert (core_root / "broker.py").exists()
    assert (root / "tradelearn" / "metrics" / "engine.py").exists()
    assert (root / "tradelearn" / "data" / "resampler.py").exists()


def test_project_structure_documents_core_boundaries() -> None:
    root = Path(__file__).parents[3]
    text = (root / "docs" / "PROJECT_STRUCTURE.md").read_text()

    assert "不允许 import `tradelearn.backtest.*`" in text
    assert "回测专属 runtime 不上移到 `tradelearn/core/`" in text
