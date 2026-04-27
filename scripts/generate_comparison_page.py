"""Generate the release comparison page for the documentation site."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComparisonProject:
    """One external project compared on the release comparison page."""

    name: str
    focus: str
    tradelearn_difference: str


COMPARISON_PROJECTS: tuple[ComparisonProject, ...] = (
    ComparisonProject(
        "qlib",
        "研究平台偏重数据、模型与实验体系",
        "trade-learn 是本地优先的 Python 量化研究框架,传统策略与 ML 策略共用 API",
    ),
    ComparisonProject(
        "vnpy",
        "交易系统与实盘网关生态",
        "trade-learn 1.0 聚焦研究与回测 SDK,1.1 通过 QMT 补齐研究到实盘路径",
    ),
    ComparisonProject(
        "backtrader",
        "成熟的事件驱动回测 API",
        "trade-learn 提供 compat.backtrader 迁移层,同时接入 MLStrategy 与现代报告体系",
    ),
    ComparisonProject(
        "nautilus",
        "高性能事件驱动交易架构",
        "trade-learn 采用 Rust 事件型撮合核,但保留轻量 Python SDK 使用体验",
    ),
)


def render_comparison_page(projects: tuple[ComparisonProject, ...] = COMPARISON_PROJECTS) -> str:
    """Render a markdown comparison page from the project positioning."""
    lines = [
        "# Comparison",
        "",
        "vs qlib / vnpy / backtrader / nautilus",
        "",
        "trade-learn 是 Python 量化研究框架,让传统策略与 ML 策略共用 API。",
        "",
        "| 项目 | 侧重点 | trade-learn 差异 |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {project.name} | {project.focus} | {project.tradelearn_difference} |"
        for project in projects
    )
    lines.extend(
        [
            "",
            "## 1.0 定位",
            "",
            "- compat.backtrader 承接存量策略",
            "- Rust 事件型撮合核负责回测一致性与性能",
            "- QMT 实盘对接进入 1.1,不拖慢 1.0 发版",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_page(docs_dir: Path | str = Path("docs")) -> Path:
    """Write the generated comparison page under ``docs_dir/comparison.md``."""
    output = Path(docs_dir) / "release" / "evaluation.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_comparison_page(), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)

    print(write_comparison_page(args.docs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
