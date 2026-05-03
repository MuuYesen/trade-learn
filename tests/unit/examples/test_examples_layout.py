from __future__ import annotations

import ast
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


def test_research_examples_cover_lite_and_engine_workflows() -> None:
    root = Path(__file__).parents[3]
    examples = root / "examples" / "research"

    direct_expected = {
        "index_enhance_lite.py": ("import tradelearn.lite as tl", "tl.Backtest"),
        "index_enhance_engine.py": ("import tradelearn.engine as bt", "bt.Cerebro"),
    }
    pipeline_expected = {
        "index_enhance_lite_pipeline.py": ("import tradelearn.lite as tl", "tl.Backtest"),
        "index_enhance_engine_pipeline.py": ("import tradelearn.engine as bt", "bt.Cerebro"),
    }

    for name, api_markers in {**direct_expected, **pipeline_expected}.items():
        source = (examples / name).read_text()
        assert "TradingViewProvider" in source
        assert "research.FeatureSet" in source
        assert "GradientBoostingRegressor" in source
        assert "target={" in source
        assert ".fit(train_features" in source
        assert "research.ModelScorer" in source
        assert "ResearchRun" in source
        assert "import tradelearn.research as research" in source
        assert "import tradelearn.research.explore as ex" in source
        assert "import tradelearn.research.preprocess as pp" in source
        assert "import tradelearn.research.portfolio as pf" in source
        assert "ex.profile" in source
        assert "pp.Winsorizer" in source
        assert "pp.Neutralizer" in source
        assert "pp.StandardScaler" in source
        assert "research.time_split" in source
        assert ".fit_transform(train_features" in source
        assert ".transform(test_features" in source
        if name in pipeline_expected:
            assert "research.Pipeline" in source
            assert "pf.Allocator" in source
            assert "pf.TopK" in source
            assert "pf.EqualWeight" in source
            assert "pf.Constraints" in source
        else:
            assert "research.Pipeline" not in source
            assert "pf.select_top" in source
            assert "pf.equal_weight" in source
            assert "pf.apply_constraints" in source
        assert "research_result.weights[0]" in source
        if name.startswith("index_enhance_lite"):
            assert "target_weights" in source
        else:
            assert "class EngineResearchIndexEnhance(bt.Strategy)" in source
            assert "order_target_percent" in source
            assert "target=weights.get(data._name)" in source
        assert ".report(" in source
        assert "os.getenv" not in source
        assert "TRADELEARN_DEMO_MLFLOW" not in source
        assert "mlflow_uri" not in source
        assert "mlflow_username" not in source
        assert "mlflow_password" not in source
        assert "127.0.0.1:5050" not in source
        assert "MLFLOW_TRACKING_USERNAME" not in source
        assert "MLFLOW_TRACKING_PASSWORD" not in source
        assert "os.environ" not in source
        for marker in api_markers:
            assert marker in source


def test_lite_live_research_example_keeps_inference_inside_strategy() -> None:
    root = Path(__file__).parents[3]
    source = (root / "examples" / "research" / "index_enhance_lite_live.py").read_text()
    tree = ast.parse(source)

    strategy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LiteLiveIndexEnhance"
    )
    main_guard = next(node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node))

    assert "GradientBoostingRegressor" in source
    assert "research.FeatureSet" in source
    assert "research.ModelScorer" in source
    assert "pf.Allocator" in source
    assert "research_result.weights[0]" not in source
    assert _contains_call(strategy, "transform")
    assert _contains_call(strategy, "predict")
    assert _contains_call(strategy, "build")
    assert _contains_call(strategy, "target_weights")
    assert not _contains_call(main_guard, "select_top")
    assert not _contains_call(main_guard, "equal_weight")
    assert not _contains_call(main_guard, "apply_constraints")


def test_engine_live_research_example_keeps_inference_inside_strategy() -> None:
    root = Path(__file__).parents[3]
    source = (root / "examples" / "research" / "index_enhance_engine_live.py").read_text()
    tree = ast.parse(source)

    strategy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineLiveIndexEnhance"
    )
    main_guard = next(node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node))

    assert "GradientBoostingRegressor" in source
    assert "research.FeatureSet" in source
    assert "research.ModelScorer" in source
    assert "pf.Allocator" in source
    assert "research_result.weights[0]" not in source
    assert "class EngineLiveIndexEnhance(bt.Strategy)" in source
    assert "target=weights.get(data._name)" in source
    assert _contains_call(strategy, "transform")
    assert _contains_call(strategy, "predict")
    assert _contains_call(strategy, "build")
    assert _contains_call(strategy, "order_target_percent")
    assert not _contains_call(main_guard, "select_top")
    assert not _contains_call(main_guard, "equal_weight")
    assert not _contains_call(main_guard, "apply_constraints")
    assert "os.getenv" not in source
    assert "TRADELEARN_DEMO_MLFLOW" not in source
    assert "mlflow_uri" not in source
    assert "MLFLOW_TRACKING_USERNAME" not in source
    assert "MLFLOW_TRACKING_PASSWORD" not in source
    assert "os.environ" not in source


def test_research_examples_keep_runtime_flow_in_main() -> None:
    root = Path(__file__).parents[3]

    for name in (
        "index_enhance_lite.py",
        "index_enhance_engine.py",
        "index_enhance_lite_pipeline.py",
        "index_enhance_engine_pipeline.py",
        "index_enhance_lite_live.py",
        "index_enhance_engine_live.py",
    ):
        path = root / "examples" / "research" / name
        tree = ast.parse(path.read_text())
        body = list(tree.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        for node in body:
            assert isinstance(
                node,
                ast.Import | ast.ImportFrom | ast.ClassDef | ast.If,
            ), (name, type(node).__name__)
            if isinstance(node, ast.If):
                assert _is_main_guard(node), name


def _looks_like_strategy(path: Path) -> bool:
    text = path.read_text()
    return "class " in text and "Strategy" in text


def _is_main_guard(node: ast.If) -> bool:
    return (
        isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _contains_call(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            return True
        if isinstance(func, ast.Attribute) and func.attr == name:
            return True
    return False
