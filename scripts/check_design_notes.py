#!/usr/bin/env python
"""Check Stage 2 clean-room design note readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_DESIGN_NOTES = (
    "matching-design.md",
    "event-loop.md",
    "portfolio.md",
)

REQUIRED_SECTIONS = (
    "## Scope",
    "## Clean-room Boundary",
    "## Source Notes",
    "## Implementation Decisions",
    "## Open Questions",
)

NOTE_TITLES = {
    "matching-design.md": "Matching Design",
    "event-loop.md": "Event Loop",
    "portfolio.md": "Portfolio",
}

SECTION_PROMPTS = {
    "## Scope": "Define the design boundary and the behavior this note covers.",
    "## Clean-room Boundary": "Record source separation rules before implementation starts.",
    "## Source Notes": "List reviewed references and observations without copying source code.",
    "## Implementation Decisions": "Record independent implementation decisions for v2.",
    "## Open Questions": "Track unresolved items that must be closed before the freeze.",
}


def design_note_template(filename: str) -> str:
    """Return the starter content for one clean-room design note."""
    lines = [f"# {NOTE_TITLES[filename]}", ""]
    for section in REQUIRED_SECTIONS:
        lines.extend([section, SECTION_PROMPTS[section], ""])
    return "\n".join(lines)


def init_design_notes(directory: Path) -> list[str]:
    """Create missing design note templates without overwriting drafts."""
    directory.mkdir(parents=True, exist_ok=True)
    statuses: list[str] = []
    for filename in REQUIRED_DESIGN_NOTES:
        path = directory / filename
        if path.exists():
            statuses.append(f"exists design note: {filename}")
            continue
        path.write_text(design_note_template(filename), encoding="utf-8")
        statuses.append(f"created design note: {filename}")
    return statuses


def note_errors(directory: Path, filename: str) -> list[str]:
    """Return readiness errors for one required design note."""
    path = directory / filename
    if not path.is_file():
        return [f"missing design note: {filename}"]

    content = path.read_text(encoding="utf-8")
    missing_sections = [
        f"missing section in {filename}: {section}"
        for section in REQUIRED_SECTIONS
        if section not in content
    ]
    return missing_sections


def check_design_notes(directory: Path) -> list[str]:
    """Return all design note readiness errors."""
    errors: list[str] = []
    for filename in REQUIRED_DESIGN_NOTES:
        errors.extend(note_errors(directory, filename))
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the design note readiness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("docs/internal"),
        help="Directory containing Stage 2 clean-room design notes.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create missing design note templates before checking readiness.",
    )
    args = parser.parse_args(argv)

    if args.init:
        print("\n".join(init_design_notes(args.directory)))
        return 0

    errors = check_design_notes(args.directory)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    for filename in REQUIRED_DESIGN_NOTES:
        print(f"design-note:{filename}=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
