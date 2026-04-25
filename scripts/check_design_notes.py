#!/usr/bin/env python
"""Check Stage 2 clean-room design note readiness."""

from __future__ import annotations

import argparse
import json
import re
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

STRICT_PLACEHOLDER_TOKENS = ("TODO", "TBD", "FIXME")


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


def section_body(content: str, section: str) -> str:
    """Return the body text under a required markdown section."""
    start = content.find(section)
    if start == -1:
        return ""
    body_start = start + len(section)
    next_section_starts = [
        position
        for other_section in REQUIRED_SECTIONS
        if other_section != section
        for position in [content.find(other_section, body_start)]
        if position != -1
    ]
    body_end = min(next_section_starts) if next_section_starts else len(content)
    return content[body_start:body_end].strip()


def has_placeholder_token(content: str, token: str) -> bool:
    """Return whether content contains a standalone freeze-blocking token."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", content) is not None


def note_errors(directory: Path, filename: str, *, strict: bool = False) -> list[str]:
    """Return readiness errors for one required design note."""
    path = directory / filename
    if not path.is_file():
        return [f"missing design note: {filename}"]

    content = path.read_text(encoding="utf-8")
    errors = [
        f"missing section in {filename}: {section}"
        for section in REQUIRED_SECTIONS
        if section not in content
    ]
    if strict:
        errors.extend(
            f"untouched template prompt in {filename}: {section}"
            for section, prompt in SECTION_PROMPTS.items()
            if prompt in content
        )
        errors.extend(
            f"empty section body in {filename}: {section}"
            for section in REQUIRED_SECTIONS
            if section in content and not section_body(content, section)
        )
        errors.extend(
            f"placeholder token in {filename}: {token}"
            for token in STRICT_PLACEHOLDER_TOKENS
            if has_placeholder_token(content, token)
        )
    return errors


def check_design_notes(directory: Path, *, strict: bool = False) -> list[str]:
    """Return all design note readiness errors."""
    errors: list[str] = []
    for filename in REQUIRED_DESIGN_NOTES:
        errors.extend(note_errors(directory, filename, strict=strict))
    return errors


def design_note_report(directory: Path, *, strict: bool = False) -> dict[str, object]:
    """Return a machine-readable readiness report for all design notes."""
    notes = [
        {
            "file": filename,
            "errors": note_errors(directory, filename, strict=strict),
        }
        for filename in REQUIRED_DESIGN_NOTES
    ]
    return {
        "directory": str(directory),
        "strict": strict,
        "ok": all(not note["errors"] for note in notes),
        "notes": notes,
    }


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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail if generated template prompts have not been replaced.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable readiness report.",
    )
    args = parser.parse_args(argv)

    if args.init:
        print("\n".join(init_design_notes(args.directory)))
        return 0

    if args.json:
        report = design_note_report(args.directory, strict=args.strict)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    errors = check_design_notes(args.directory, strict=args.strict)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    for filename in REQUIRED_DESIGN_NOTES:
        print(f"design-note:{filename}=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
