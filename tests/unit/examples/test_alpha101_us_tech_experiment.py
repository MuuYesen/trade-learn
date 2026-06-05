from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "zoo" / "outputs" / "examples" / "alpha101_us_tech_experiment.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("alpha101_us_tech_experiment", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alpha101_us_tech_experiment_keeps_main_ranking_and_scoring_functions() -> None:
    module = _load_example_module()

    functions = [
        name
        for name, value in vars(module).items()
        if callable(value) and getattr(value, "__module__", None) == module.__name__
    ]

    assert functions == ["main", "rank_factors", "factor_score"]


def test_alpha101_us_tech_experiment_avoids_type_cast_noise() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "typing import cast" not in source
    assert "cast(" not in source


def test_alpha101_us_tech_experiment_stops_after_factor_selection() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "parts = []" not in source
    assert 'selected.set_index("column")' not in source
    assert "StandardScaler" not in source
    assert "weights =" not in source
    assert "composite =" not in source
    assert "scores =" not in source


def test_alpha101_us_tech_experiment_does_not_write_files() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert ".to_csv(" not in source
    assert ".report(" not in source
    assert ".write_text(" not in source


def test_alpha101_us_tech_experiment_uses_summary_metrics_for_ranking() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert ".summary()" in source
    assert ".factor_information_coefficient()" not in source
    assert ".monotonicity()" not in source
    assert "from tradelearn.metrics import ic_ir" not in source
    assert "ic_ir(" not in source


def test_alpha101_us_tech_experiment_prints_ranking_without_extra_top_variable() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "top = ranking.head" not in source
