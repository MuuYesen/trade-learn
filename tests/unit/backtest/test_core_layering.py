from __future__ import annotations

from pathlib import Path


def test_core_layer_does_not_import_compat() -> None:
    root = Path(__file__).parents[3]
    core_roots = [root / "tradelearn" / "backtest", root / "tradelearn" / "core"]
    offenders: list[str] = []

    for core_root in core_roots:
        for path in core_root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path == root / "tradelearn" / "backtest" / "__init__.py":
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


def test_backtest_runtime_modules_are_flattened() -> None:
    root = Path(__file__).parents[3]
    backtest_root = root / "tradelearn" / "backtest"
    expected_modules = {
        "__init__.py",
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

    assert not (backtest_root / "core").exists()
    assert not (backtest_root / "brokers").exists()
    assert not (backtest_root / "metrics.py").exists()
    assert not (backtest_root / "resampler.py").exists()

    actual_modules = {path.name for path in backtest_root.glob("*.py")}
    assert actual_modules == expected_modules

    assert (backtest_root / "broker.py").exists()
    assert (root / "tradelearn" / "metrics" / "engine.py").exists()
    assert (root / "tradelearn" / "data" / "resampler.py").exists()


def test_project_structure_documents_core_boundaries() -> None:
    root = Path(__file__).parents[3]
    text = (root / "docs" / "PROJECT_STRUCTURE.md").read_text()

    assert "不允许 import `tradelearn.backtest.*`" in text
    assert "回测专属 runtime 不上移到 `tradelearn/core/`" in text
    assert "tradelearn/backtest/engine.py" in text


def test_rust_core_is_split_by_runtime_responsibility() -> None:
    root = Path(__file__).parents[3]
    rust_src = root / "rust" / "tradelearn-rust" / "src"
    expected_modules = {
        "engine.rs",
        "lib.rs",
        "matching.rs",
        "runner.rs",
        "types.rs",
    }

    actual_modules = {path.name for path in rust_src.glob("*.rs")}

    assert actual_modules == expected_modules
    assert not (rust_src / "core.rs").exists()


def test_runnable_tools_are_not_kept_under_tests() -> None:
    root = Path(__file__).parents[3]

    assert not (root / "tests" / "runners").exists()
    assert (root / "benchmarks" / "runners" / "benchmark_bt.py").exists()
    assert (root / "benchmarks" / "runners" / "compare_backtesting.py").exists()
    assert (root / "scripts" / "examples" / "ml_strategy.py").exists()
