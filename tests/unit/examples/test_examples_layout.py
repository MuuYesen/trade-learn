from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).parents[3]
RESEARCH = ROOT / "examples/research"
WRAPPERS = {
    "index_enhance_lite.py": "index_enhance_lite_pipeline",
    "index_enhance_engine.py": "index_enhance_engine_pipeline",
    "index_enhance_lite_live.py": "index_enhance_lite_pipeline",
    "index_enhance_engine_live.py": "index_enhance_engine_pipeline",
}


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call)
        and (
            isinstance(n.func, ast.Name)
            and n.func.id == name
            or isinstance(n.func, ast.Attribute)
            and n.func.attr == name
        )
        for n in ast.walk(node)
    )


def _function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _pipeline(kind):
    return ast.parse((RESEARCH / f"index_enhance_{kind}_pipeline.py").read_text())


def test_examples_contains_only_strategy_files() -> None:
    """Keep only documented strategies, explicit aliases and shared research helpers."""
    helpers = {
        "research/_workflow.py",
        "lite/_indicators.py",
        "research/components.py",
        "exploration/classifier_comparison.py",
    }
    helpers.update("research/" + name for name in WRAPPERS)
    offenders = []
    for path in (ROOT / "examples").rglob("*"):
        if path.is_dir() or "__pycache__" in path.parts or "output" in path.parts:
            continue
        if path.name == ".DS_Store":
            continue
        relative = path.relative_to(ROOT / "examples").as_posix()
        if path.suffix != ".py" or not (
            path.name == "__init__.py"
            or relative in helpers
            or "class " in path.read_text()
            and "Strategy" in path.read_text()
        ):
            offenders.append(relative)
    assert not offenders


def test_research_examples_cover_lite_and_engine_workflows() -> None:
    workflow = (RESEARCH / "_workflow.py").read_text()
    tree = ast.parse(workflow)
    train, infer = _function(tree, "train"), _function(tree, "infer")
    assert _contains_call(train, "ResearchRun") and _contains_call(train, "register")
    assert _contains_call(train, "forward_return") and _contains_call(train, "purged_splits")
    assert "label_end" in workflow and "fit_transform" in workflow
    assert _contains_call(infer, "transform") and _contains_call(infer, "predict")
    assert _contains_call(infer, "build") and not _contains_call(infer, "fit")
    inputs = _function(tree, "inputs")
    assert _contains_call(inputs, "open_dataset")
    assert _contains_call(inputs, "save_bundle") and _contains_call(inputs, "load_bundle")
    assert "TradingViewProvider" not in workflow
    for kind in ("engine", "lite"):
        main = _function(_pipeline(kind), "main")
        assert _contains_call(main, "inputs") and _contains_call(main, "run")
        assert _contains_call(main, "finish")
    for name, module in WRAPPERS.items():
        source = (RESEARCH / name).read_text()
        assert f"from examples.research.{module} import" in source
        assert "__name__" in source and "main()" in source


def _check_live_inference(kind, order_method):
    wrapper = (RESEARCH / f"index_enhance_{kind}_live.py").read_text()
    assert f"from examples.research.index_enhance_{kind}_pipeline import" in wrapper
    strategy = next(n for n in _pipeline(kind).body if isinstance(n, ast.ClassDef))
    next_method = _function(strategy, "next")
    assert _contains_call(next_method, "infer")
    assert _contains_call(next_method, order_method)
    assert not _contains_call(next_method, "fit")
    assert not _contains_call(next_method, "train")
    for token in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD", "127.0.0.1:5050"):
        assert token not in wrapper


def test_lite_live_research_example_keeps_inference_inside_strategy() -> None:
    _check_live_inference("lite", "target_weights")


def test_engine_live_research_example_keeps_inference_inside_strategy() -> None:
    _check_live_inference("engine", "order_target_percent")


def test_research_examples_keep_runtime_flow_in_main() -> None:
    for path in RESEARCH.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Assign):
                assert not any(isinstance(n, ast.Call) for n in ast.walk(node)), path.name
                continue
            assert isinstance(node, ast.If), (path.name, type(node).__name__)
            assert isinstance(node.test, ast.Compare)
            assert isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
            assert isinstance(node.test.comparators[0], ast.Constant)
            assert node.test.comparators[0].value == "__main__"
