"""Check release documentation coverage against the mkdocs navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REQUIRED_RELEASE_PAGES = ("README.md", "VISION.md", "RUNBOOK.md")
REQUIRED_INTERNAL_NAV = (
    "internal/event-loop.md",
    "internal/matching-design.md",
    "internal/portfolio.md",
)


@dataclass(frozen=True)
class DocsCompletenessReport:
    """Structured result for release documentation completeness checks."""

    nav_paths: tuple[str, ...]
    spec_nav_paths: tuple[str, ...]
    expected_spec_paths: tuple[str, ...]
    missing_spec_nav: tuple[str, ...]
    missing_release_pages: tuple[str, ...]
    missing_internal_nav: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether all required release documentation is reachable."""
        return not (
            self.missing_spec_nav or self.missing_release_pages or self.missing_internal_nav
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


def _expected_spec_paths(docs_dir: Path) -> tuple[str, ...]:
    specs_dir = docs_dir / "specs"
    paths = []
    for path in specs_dir.glob("*.md"):
        if path.name == "README.md":
            continue
        paths.append(path.relative_to(docs_dir).as_posix())
    return tuple(sorted(paths))


def check_docs_completeness(project_root: Path | str = Path(".")) -> DocsCompletenessReport:
    """Check that release docs expose all spec pages and required guides."""
    root = Path(project_root)
    config = yaml.safe_load((root / "mkdocs.yml").read_text(encoding="utf-8"))
    docs_dir = root / config.get("docs_dir", "docs")
    nav_paths = _flatten_nav_paths(config["nav"])
    spec_nav_paths = tuple(path for path in nav_paths if path.startswith("specs/"))
    expected_spec_paths = _expected_spec_paths(docs_dir)

    return DocsCompletenessReport(
        nav_paths=nav_paths,
        spec_nav_paths=spec_nav_paths,
        expected_spec_paths=expected_spec_paths,
        missing_spec_nav=tuple(path for path in expected_spec_paths if path not in nav_paths),
        missing_release_pages=tuple(
            path for path in REQUIRED_RELEASE_PAGES if path not in nav_paths
        ),
        missing_internal_nav=tuple(path for path in REQUIRED_INTERNAL_NAV if path not in nav_paths),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for release docs completeness checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    report = check_docs_completeness(args.project_root)
    if report.ok:
        print(
            "docs-completeness:ok "
            f"specs={len(report.expected_spec_paths)} "
            f"release_pages={len(REQUIRED_RELEASE_PAGES)} "
            f"internal={len(REQUIRED_INTERNAL_NAV)}"
        )
        return 0

    print("docs-completeness:failed")
    if report.missing_spec_nav:
        print("missing_spec_nav=" + ",".join(report.missing_spec_nav))
    if report.missing_release_pages:
        print("missing_release_pages=" + ",".join(report.missing_release_pages))
    if report.missing_internal_nav:
        print("missing_internal_nav=" + ",".join(report.missing_internal_nav))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
