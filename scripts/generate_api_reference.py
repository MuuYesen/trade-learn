"""Generate mkdocstrings API Reference pages for the documentation site."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiReferenceModule:
    """One public API module rendered in the reference page."""

    title: str
    import_path: str
    summary: str


API_REFERENCE_MODULES: tuple[ApiReferenceModule, ...] = (
    ApiReferenceModule("Backtest", "tradelearn.backtest", "Strategy, Cerebro, broker, orders."),
    ApiReferenceModule("Compat Backtrader", "tradelearn.engine", "Migration facade."),
    ApiReferenceModule("Data", "tradelearn.data", "Bars, providers, and cache helpers."),
    ApiReferenceModule("Indicators", "tradelearn.indicators", "Technical indicator facade."),
    ApiReferenceModule("Metrics", "tradelearn.metrics", "Returns, risk, and factor metrics."),
    ApiReferenceModule("Factor", "tradelearn.factor", "Alpha formulas and FactorAnalyzer."),
    ApiReferenceModule("Report", "tradelearn.report", "HTML, Excel, and explorer reports."),
    ApiReferenceModule("ML", "tradelearn.ml", "MLStrategy, FeatureStore, registry, selector."),
)


def render_api_reference(modules: tuple[ApiReferenceModule, ...] = API_REFERENCE_MODULES) -> str:
    """Render a mkdocstrings-backed API reference markdown page."""
    parts = [
        "# API Reference",
        "",
        "This page is generated from Python docstrings via mkdocstrings.",
        "",
    ]
    for module in modules:
        parts.extend(
            [
                f"## {module.title}",
                "",
                module.summary,
                "",
                f"::: {module.import_path}",
                "    options:",
                "      show_source: true",
                "      show_root_heading: true",
                "      members_order: source",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def write_api_reference(docs_dir: Path | str = Path("docs")) -> Path:
    """Write the generated API Reference page under ``docs_dir/api/reference.md``."""
    output = Path(docs_dir) / "api" / "reference.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_api_reference(), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for local documentation generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Documentation root directory.",
    )
    args = parser.parse_args(argv)
    output = write_api_reference(args.docs_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
