"""Check that the public documentation site exposes the expected pages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_GUIDE_PAGES = (
    "README.md",
    "quickstart.md",
    "concepts/architecture.md",
    "concepts/runtime.md",
    "concepts/stats.md",
    "guides/strategy.md",
    "guides/lite.md",
    "guides/lite-api.md",
    "guides/engine.md",
    "guides/engine-api.md",
    "guides/data.md",
    "guides/indicators.md",
    "guides/extensions.md",
    "guides/live.md",
    "guides/research.md",
    "guides/factor.md",
    "guides/optimization.md",
    "guides/report.md",
    "guides/mlflow-lab-mcp.md",
)

REQUIRED_DESIGN_NOTE_PAGES = (
    "internals/contracts.md",
    "internals/event-loop.md",
    "internals/matching.md",
    "internals/portfolio.md",
    "internals/consistency.md",
    "internals/migration.md",
)

REQUIRED_BENCHMARK_PAGES = (
    "benchmarks.md",
    "release/evaluation.md",
)

REQUIRED_API_REFERENCE_PAGES = (
    "api/reference.md",
    "api/reference/lite.md",
    "api/reference/engine.md",
    "api/reference/data.md",
    "api/reference/indicators.md",
    "api/reference/metrics.md",
    "api/reference/factor.md",
    "api/reference/report.md",
    "api/reference/ml.md",
    "api/reference/research.md",
)


class _MkDocsConfigLoader(yaml.SafeLoader):
    """YAML loader for mkdocs.yml fields that are only meaningful to MkDocs."""


def _construct_python_name(
    loader: yaml.SafeLoader, suffix: str, node: yaml.nodes.Node
) -> str:
    """Treat MkDocs Python object tags as inert strings during static checks."""
    value = loader.construct_scalar(node)
    return suffix if not value else f"{suffix} {value}"


_MkDocsConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    _construct_python_name,
)


def load_mkdocs_config(config_path: Path | str = Path("mkdocs.yml")) -> dict[str, Any]:
    """Load mkdocs.yml for static tests without importing MkDocs extension objects."""
    return yaml.load(Path(config_path).read_text(encoding="utf-8"), Loader=_MkDocsConfigLoader)


@dataclass(frozen=True)
class DocsCompletenessReport:
    """Structured result for documentation navigation checks."""

    nav_paths: tuple[str, ...]
    required_guide_pages: tuple[str, ...]
    required_design_note_pages: tuple[str, ...]
    required_benchmark_pages: tuple[str, ...]
    required_api_reference_pages: tuple[str, ...]
    missing_guide_pages: tuple[str, ...]
    missing_design_note_pages: tuple[str, ...]
    missing_benchmark_pages: tuple[str, ...]
    missing_api_reference_pages: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether all required public docs are reachable from mkdocs."""
        return not (
            self.missing_guide_pages
            or self.missing_design_note_pages
            or self.missing_benchmark_pages
            or self.missing_api_reference_pages
        )


def _flatten_nav_paths(nav: list[Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for item in nav:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(_flatten_nav_paths(value))
    return tuple(paths)


def _missing(required: tuple[str, ...], nav_paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in required if path not in nav_paths)


def check_docs_completeness(project_root: Path | str = Path(".")) -> DocsCompletenessReport:
    """Check that user-facing guides, design notes, benchmarks, and API docs are linked."""
    root = Path(project_root)
    config = load_mkdocs_config(root / "mkdocs.yml")
    nav_paths = _flatten_nav_paths(config["nav"])

    return DocsCompletenessReport(
        nav_paths=nav_paths,
        required_guide_pages=REQUIRED_GUIDE_PAGES,
        required_design_note_pages=REQUIRED_DESIGN_NOTE_PAGES,
        required_benchmark_pages=REQUIRED_BENCHMARK_PAGES,
        required_api_reference_pages=REQUIRED_API_REFERENCE_PAGES,
        missing_guide_pages=_missing(REQUIRED_GUIDE_PAGES, nav_paths),
        missing_design_note_pages=_missing(REQUIRED_DESIGN_NOTE_PAGES, nav_paths),
        missing_benchmark_pages=_missing(REQUIRED_BENCHMARK_PAGES, nav_paths),
        missing_api_reference_pages=_missing(REQUIRED_API_REFERENCE_PAGES, nav_paths),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for docs completeness checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    report = check_docs_completeness(args.project_root)
    if report.ok:
        print(
            "docs-completeness:ok "
            f"guides={len(report.required_guide_pages)} "
            f"design_notes={len(report.required_design_note_pages)} "
            f"benchmarks={len(report.required_benchmark_pages)} "
            f"api_reference={len(report.required_api_reference_pages)}"
        )
        return 0

    print("docs-completeness:failed")
    if report.missing_guide_pages:
        print("missing_guide_pages=" + ",".join(report.missing_guide_pages))
    if report.missing_design_note_pages:
        print("missing_design_note_pages=" + ",".join(report.missing_design_note_pages))
    if report.missing_benchmark_pages:
        print("missing_benchmark_pages=" + ",".join(report.missing_benchmark_pages))
    if report.missing_api_reference_pages:
        print("missing_api_reference_pages=" + ",".join(report.missing_api_reference_pages))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
