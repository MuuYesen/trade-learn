from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_benchmark_page import render_benchmark_page, write_benchmark_page


def test_render_benchmark_page_contains_speed_targets_and_measured_results() -> None:
    baseline = json.loads(Path("benchmarks/baseline.json").read_text(encoding="utf-8"))

    rendered = render_benchmark_page(baseline)

    assert "# Benchmark" in rendered
    assert "单品种 10 年" in rendered
    assert "500 股组合" in rendered
    assert "3.173ms" in rendered
    assert "312.569ms" in rendered
    assert "trades 0 差异" in rendered
    assert "PnL rtol=1e-4" in rendered


def test_write_benchmark_page_creates_expected_markdown(tmp_path) -> None:
    output = write_benchmark_page(tmp_path, baseline_path=Path("benchmarks/baseline.json"))

    assert output == tmp_path / "benchmark.md"
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("# Benchmark")
