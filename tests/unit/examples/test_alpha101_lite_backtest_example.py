from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = (
    ROOT
    / "zoo"
    / "wechat_docs"
    / "articles"
    / "alpha101-us-tech"
    / "outputs"
    / "examples"
)
EXAMPLE = EXAMPLES / "alpha101_us_tech_lite_backtest.py"


def _load_example_module():
    sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("alpha101_us_tech_lite_backtest", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lite_backtest_uses_current_pipeline_instead_of_saved_csvs() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "bars.csv" not in source
    assert "alpha101_factors.csv" not in source
    assert "factor_ranking.csv" not in source
    assert "load_lite_bars" not in source
    assert "import alpha101_us_tech_experiment" not in source
    assert "experiment." not in source
    assert "\ndef rank_factors(" in source
    assert "\ndef factor_score(" in source
    assert "from tradelearn.research.portfolio import topk_equal_weights" in source
    assert "factor_topk_equal_weights" not in source
    assert "\ndef topk_equal_weights(" not in source
    assert "\ndef to_lite_bars(" not in source
    assert "lite_bars" not in source
    assert "\ndef reconstruct_multi_asset_equity(" not in source
    assert "corrected_fill_summary" not in source
    assert "\ndef _return_summary(" not in source
    assert "\ndef _json_ready(" not in source
    assert "lite_backtest_summary.json" not in source
    assert "stats.equity" not in source
    assert "stats.returns" not in source
    assert "selected = ranking.head(TOP_K).copy()" in source
    assert "\ndef build_composite_scores(" in source
    assert "scores = build_composite_scores(factors, selected)" in source
    assert "weights = build_target_weights(factors, selected)" in source
    assert "weights = build_target_weights(factors, ranking)" not in source
    assert "rebalance_dates = frozenset" not in source
    assert "rebalance_dates=rebalance_dates" not in source
    assert "weights.raw.index" not in source
    assert "weights.has_current()" in source
    assert "Selected factors:" not in source
    assert '", ".join(selected' not in source
    assert 'ranking.head(10).to_markdown(index=False, floatfmt=".4f")' in source


def test_build_composite_scores_uses_selected_factor_columns() -> None:
    module = _load_example_module()
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "alpha001_101": [1.0, 2.0, 3.0, 1.0],
        }
    )
    selected = pd.DataFrame({"column": ["alpha001_101"], "direction": [1]})

    scores = module.build_composite_scores(factors, selected)

    assert scores.index.names == ["date", "symbol"]
    assert scores.name == "score"
    assert scores.groupby(level="date").mean().round(10).eq(0.0).all()


def test_build_target_weights_uses_selected_factor_columns() -> None:
    module = _load_example_module()
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "alpha001_101": [1.0, 2.0, 3.0, 1.0],
        }
    )
    selected = pd.DataFrame(
        {
            "column": ["alpha001_101"],
            "direction": [1],
        }
    )

    weights = module.build_target_weights(factors, selected, rebalance_every=1)

    assert weights.index.names == ["timestamp", "symbol"]
    assert weights.name == "weight"
    assert weights.groupby(level="timestamp").sum().eq(1.0).all()
