from __future__ import annotations

import ast
from pathlib import Path

import pytest


BOUNDARY_RULES = [
    (
        "platform core stays neutral",
        "tradelearn/core",
        ("tradelearn.backtest", "tradelearn.engine", "tradelearn.lite"),
    ),
    (
        "backtest runtime does not import user facades",
        "tradelearn/backtest",
        ("tradelearn.engine", "tradelearn.lite"),
    ),
    (
        "backtest runtime does not import research layer",
        "tradelearn/backtest",
        ("tradelearn.research",),
    ),
    (
        "lite facade does not import engine facade",
        "tradelearn/lite",
        ("tradelearn.engine",),
    ),
    (
        "engine facade does not import lite facade",
        "tradelearn/engine",
        ("tradelearn.lite",),
    ),
]


def _find_import_offenders(root: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for module in imported
                for prefix in forbidden_prefixes
            ):
                offenders.append(str(path.relative_to(root)))
                break

    return offenders


def test_import_boundary_scanner_ignores_comments(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text(
        "# NOTE: do not import tradelearn.engine here\n"
        "value = 'tradelearn.backtest appears in plain text'\n"
        "from tradelearn.core import StreamBar\n"
    )

    assert _find_import_offenders(tmp_path, ("tradelearn.engine", "tradelearn.backtest")) == []


@pytest.mark.parametrize(("name", "root_path", "forbidden"), BOUNDARY_RULES)
def test_import_boundaries_are_enforced_by_rule_table(
    name: str,
    root_path: str,
    forbidden: tuple[str, ...],
) -> None:
    root = Path(__file__).parents[3]

    offenders = _find_import_offenders(root / root_path, forbidden)

    assert offenders == [], name


def test_core_layer_does_not_import_facades() -> None:
    root = Path(__file__).parents[3]
    core_roots = [root / "tradelearn" / "backtest", root / "tradelearn" / "core"]
    offenders: list[str] = []

    for core_root in core_roots:
        offenders.extend(
            str(core_root / offender)
            for offender in _find_import_offenders(
                core_root,
                ("tradelearn.engine", "tradelearn.lite"),
            )
        )

    assert offenders == []


def test_backtest_runtime_does_not_import_research_layer() -> None:
    root = Path(__file__).parents[3]
    backtest_root = root / "tradelearn" / "backtest"
    offenders = _find_import_offenders(backtest_root, ("tradelearn.research",))

    assert offenders == []


def test_platform_core_does_not_import_backtest_or_facades() -> None:
    root = Path(__file__).parents[3]
    platform_core = root / "tradelearn" / "core"
    offenders = _find_import_offenders(platform_core, ("tradelearn.backtest", "tradelearn.engine", "tradelearn.lite"))

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
        "feed.py",
        "history.py",
        "indicator_cache.py",
        "lines.py",
        "models.py",
        "optimize.py",
        "reporting.py",
        "runtime_config.py",
        "sizer.py",
        "strategy.py",
        "targets.py",
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
    text = (root / "design" / "PROJECT_STRUCTURE.md").read_text()

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
        "resampler.rs",
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
    assert not (root / "benchmarks" / "runners" / "compare_backtesting.py").exists()
    assert not (root / "benchmarks" / "runners" / "speed_test_backtesting.py").exists()
    assert (root / "scripts" / "examples" / "ml_strategy.py").exists()


def test_backtest_runtime_uses_timezone_aware_now() -> None:
    root = Path(__file__).parents[3]
    text = (root / "tradelearn" / "backtest" / "models.py").read_text()

    assert "Timestamp.utcnow" not in text


def test_backtest_models_has_no_backtrader_facade_only_classes() -> None:
    root = Path(__file__).parents[3]
    text = (root / "tradelearn" / "backtest" / "models.py").read_text()
    forbidden = [
        "class Params",
        "class TimeFrame",
        "class BaseAnalyzer",
        "class BaseSizer",
        "class BaseBroker",
    ]

    assert [token for token in forbidden if token in text] == []


def test_backtest_modules_do_not_import_backtrader_facade_only_classes() -> None:
    root = Path(__file__).parents[3]
    backtest_root = root / "tradelearn" / "backtest"
    forbidden = {"Params", "TimeFrame", "BaseAnalyzer", "BaseSizer", "BaseBroker"}
    offenders: list[str] = []

    for path in backtest_root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "tradelearn.backtest.models":
                continue
            imported = {alias.name for alias in node.names}
            if forbidden & imported:
                offenders.append(str(path.relative_to(root)))
                break

    assert offenders == []


def test_root_namespace_does_not_import_facade_only_classes_from_backtest_models() -> None:
    root = Path(__file__).parents[3]
    text = (root / "tradelearn" / "__init__.py").read_text()

    assert "from tradelearn.backtest.models import TimeFrame" not in text


def test_backtest_public_namespace_excludes_facade_apis() -> None:
    import tradelearn.backtest as backtest

    forbidden = {
        "Analyzer",
        "Cerebro",
        "CoreStrategy",
        "SimBroker",
        "Strategy",
        "Params",
        "TimeFrame",
        "BaseAnalyzer",
        "BaseSizer",
        "BaseBroker",
    }

    assert forbidden.isdisjoint(set(backtest.__all__))


def test_facades_do_not_import_each_other() -> None:
    root = Path(__file__).parents[3]

    lite_offenders = _find_import_offenders(
        root / "tradelearn" / "lite",
        ("tradelearn.engine",),
    )
    engine_offenders = _find_import_offenders(
        root / "tradelearn" / "engine",
        ("tradelearn.lite",),
    )

    assert lite_offenders == []
    assert engine_offenders == []
