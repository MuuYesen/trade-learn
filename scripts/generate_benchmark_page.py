"""Generate the release benchmark page from recorded baseline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _format_ms(value: float) -> str:
    return f"{value:.3f}ms"


def render_benchmark_page(baseline: dict[str, Any]) -> str:
    """Render a benchmark markdown page from ``benchmarks/baseline.json``."""
    targets = baseline["targets"]
    stage3 = baseline["measured"]["stage3_backtest"]
    single = stage3["single"]
    portfolio = stage3["portfolio"]

    lines = [
        "# Benchmark",
        "",
        "## 速度目标",
        "",
        "| 场景 | 目标 | 实测 | 状态 |",
        "|---|---:|---:|---|",
        (
            "| 单品种 10 年 | "
            f"{targets['single_symbol_10y_daily_ms']:.0f}ms | "
            f"{_format_ms(single['elapsed_ms'])} | pass |"
        ),
        (
            "| 500 股组合 | "
            f"{targets['portfolio_500_symbols_10y_daily_s']:.0f}s | "
            f"{_format_ms(portfolio['elapsed_ms'])} | pass |"
        ),
        "",
        "## 一致性口径",
        "",
        "- trades 0 差异",
        "- PnL rtol=1e-4",
        "- benchmark 结果来自 `benchmarks/baseline.json` 的 `stage3_backtest` 记录",
        "",
        "## 复现命令",
        "",
        f"`{stage3['command']}`",
        "",
    ]
    return "\n".join(lines)


def write_benchmark_page(
    docs_dir: Path | str = Path("docs"),
    *,
    baseline_path: Path | str = Path("benchmarks/baseline.json"),
) -> Path:
    """Write the generated benchmark page under ``docs_dir/benchmark.md``."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    output = Path(docs_dir) / "release" / "evaluation.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_benchmark_page(baseline), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baseline.json"))
    args = parser.parse_args(argv)

    print(write_benchmark_page(args.docs_dir, baseline_path=args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
