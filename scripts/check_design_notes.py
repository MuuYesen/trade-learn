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

NOTE_TITLE_ALIASES = {
    "matching-design.md": ("Matching Design", "撮合设计"),
    "event-loop.md": ("Event Loop", "事件循环"),
    "portfolio.md": ("Portfolio", "组合记账"),
}

SECTION_ALIASES = {
    "## Scope": ("## Scope", "## 范围"),
    "## Clean-room Boundary": ("## Clean-room Boundary", "## Clean-room 边界"),
    "## Source Notes": ("## Source Notes", "## 来源笔记"),
    "## Implementation Decisions": ("## Implementation Decisions", "## 实现决策"),
    "## Open Questions": ("## Open Questions", "## 开放问题"),
}

SECTION_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

SECTION_PROMPTS = {
    "## Scope": "Define the design boundary and the behavior this note covers.",
    "## Clean-room Boundary": "Record source separation rules before implementation starts.",
    "## Source Notes": "List reviewed references and observations without copying source code.",
    "## Implementation Decisions": "Record independent implementation decisions for v2.",
    "## Open Questions": "Track unresolved items that must be closed before the freeze.",
}

SOURCE_CHECKLISTS = {
    "matching-design.md": (
        "design/specs/BACKTEST_SPEC.md: order matching rules",
        "design/specs/STRATEGY_SPEC.md: user-facing order API",
        "design/specs/CONSISTENCY.md: decision-layer tolerance",
    ),
    "event-loop.md": (
        "design/specs/BACKTEST_SPEC.md: event loop ordering",
        "design/specs/ARCHITECTURE.md: Python/Rust boundary",
        "design/specs/STRATEGY_SPEC.md: callback lifecycle",
    ),
    "portfolio.md": (
        "design/specs/BACKTEST_SPEC.md: portfolio accounting",
        "design/specs/CONTRACTS.md: Broker contract",
        "design/specs/REPORT_SPEC.md: downstream report artifacts",
    ),
}

STRICT_PLACEHOLDER_TOKENS = ("TODO", "TBD", "FIXME")

SectionSpans = dict[str, tuple[int, int]]
SectionCounts = dict[str, int]


def normalized_heading(line: str) -> str:
    """Return a stripped markdown heading line without a leading UTF-8 BOM."""
    return line.strip().removeprefix("\ufeff")


def canonical_section_heading(line: str) -> str | None:
    """Return the canonical required section for an English or Chinese heading."""
    return SECTION_ALIAS_TO_CANONICAL.get(normalized_heading(line))


def fence_marker(line: str) -> tuple[str, int, str] | None:
    """Return fenced-code marker metadata for a markdown line."""
    stripped = line.strip()
    if not stripped or stripped[0] not in ("`", "~"):
        return None
    marker = stripped[0]
    marker_length = len(stripped) - len(stripped.lstrip(marker))
    if marker_length < 3:
        return None
    return marker, marker_length, stripped[marker_length:]


def design_note_template(filename: str) -> str:
    """Return the starter content for one clean-room design note."""
    lines = [f"# {NOTE_TITLES[filename]}", ""]
    for section in REQUIRED_SECTIONS:
        lines.extend([section, SECTION_PROMPTS[section], ""])
        if section == "## Source Notes":
            lines.extend(["Source checklist:"])
            lines.extend(f"- [ ] {item}" for item in SOURCE_CHECKLISTS[filename])
            lines.append("")
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


def design_note_paths(directory: Path) -> list[str]:
    """Return the required Stage 2 design note paths."""
    return [str(directory / filename) for filename in REQUIRED_DESIGN_NOTES]


def section_heading_spans(content: str) -> SectionSpans:
    """Return spans for required markdown section heading lines."""
    spans: SectionSpans = {}
    offset = 0
    fence: tuple[str, int] | None = None
    for line in content.splitlines(keepends=True):
        marker = fence_marker(line)
        if marker is not None:
            marker_char, marker_length, marker_rest = marker
            if fence is None:
                fence = (marker_char, marker_length)
            elif (
                marker_char == fence[0]
                and marker_length >= fence[1]
                and not marker_rest.strip()
            ):
                fence = None
            offset += len(line)
            continue
        if fence is not None:
            offset += len(line)
            continue
        heading = normalized_heading(line)
        canonical = canonical_section_heading(heading)
        if canonical is not None and canonical not in spans:
            spans[canonical] = (offset + line.index(heading), offset + len(line))
        offset += len(line)
    return spans


def section_heading_counts(content: str) -> SectionCounts:
    """Return occurrence counts for required markdown section heading lines."""
    counts: SectionCounts = {}
    fence: tuple[str, int] | None = None
    for line in content.splitlines():
        marker = fence_marker(line)
        if marker is not None:
            marker_char, marker_length, marker_rest = marker
            if fence is None:
                fence = (marker_char, marker_length)
            elif (
                marker_char == fence[0]
                and marker_length >= fence[1]
                and not marker_rest.strip()
            ):
                fence = None
            continue
        if fence is not None:
            continue
        canonical = canonical_section_heading(line)
        if canonical is not None:
            counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def top_level_headings(content: str) -> list[str]:
    """Return H1 markdown heading lines outside fenced code blocks."""
    headings: list[str] = []
    fence: tuple[str, int] | None = None
    for line in content.splitlines():
        marker = fence_marker(line)
        if marker is not None:
            marker_char, marker_length, marker_rest = marker
            if fence is None:
                fence = (marker_char, marker_length)
            elif (
                marker_char == fence[0]
                and marker_length >= fence[1]
                and not marker_rest.strip()
            ):
                fence = None
            continue
        if fence is not None:
            continue
        heading = normalized_heading(line)
        if heading.startswith("# ") and not heading.startswith("## "):
            headings.append(heading)
    return headings


def section_body(content: str, section: str, heading_spans: SectionSpans) -> str:
    """Return the body text under a required markdown section."""
    current_span = heading_spans.get(section)
    if current_span is None:
        return ""
    body_start = current_span[1]
    next_section_starts = [
        start
        for other_section, (start, _) in heading_spans.items()
        if other_section != section and start > body_start
    ]
    body_end = min(next_section_starts) if next_section_starts else len(content)
    return content[body_start:body_end].strip()


def has_placeholder_token(content: str, token: str) -> bool:
    """Return whether content contains a standalone freeze-blocking token."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", content) is not None


def has_expected_title(content: str, filename: str) -> bool:
    """Return whether a note starts with the expected top-level heading."""
    expected = {f"# {title}" for title in NOTE_TITLE_ALIASES[filename]}
    for line in content.splitlines():
        heading = normalized_heading(line)
        if heading:
            return heading in expected
    return False


def note_errors(directory: Path, filename: str, *, strict: bool = False) -> list[str]:
    """Return readiness errors for one required design note."""
    path = directory / filename
    if not path.is_file():
        return [f"missing design note: {filename}"]

    content = path.read_text(encoding="utf-8")
    heading_spans = section_heading_spans(content)
    heading_counts = section_heading_counts(content)
    errors = [
        f"missing section in {filename}: {section}"
        for section in REQUIRED_SECTIONS
        if section not in heading_spans
    ]
    if not has_expected_title(content, filename):
        errors.append(f"wrong title in {filename}: expected # {NOTE_TITLES[filename]}")
    errors.extend(
        f"duplicate top-level title in {filename}: {heading}"
        for heading in top_level_headings(content)[1:]
    )
    errors.extend(
        f"duplicate section in {filename}: {section}"
        for section in REQUIRED_SECTIONS
        if heading_counts.get(section, 0) > 1
    )
    if strict:
        errors.extend(
            f"untouched template prompt in {filename}: {section}"
            for section, prompt in SECTION_PROMPTS.items()
            if prompt in content
        )
        errors.extend(
            f"empty section body in {filename}: {section}"
            for section in REQUIRED_SECTIONS
            if section in heading_spans
            and not section_body(content, section, heading_spans)
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
            "path": str(directory / filename),
            "exists": (directory / filename).is_file(),
            "errors": note_errors(directory, filename, strict=strict),
        }
        for filename in REQUIRED_DESIGN_NOTES
    ]
    return {
        "directory": str(directory),
        "strict": strict,
        "ok": all(not note["errors"] for note in notes),
        "error_count": sum(len(note["errors"]) for note in notes),
        "missing_count": sum(not note["exists"] for note in notes),
        "notes": notes,
    }


def design_note_summary(report: dict[str, object]) -> str:
    """Return a compact human-readable summary for a readiness report."""
    notes = report["notes"]
    if not isinstance(notes, list):
        msg = "design note report has invalid notes"
        raise TypeError(msg)
    existing_count = sum(
        bool(note["exists"])
        for note in notes
        if isinstance(note, dict)
    )
    return (
        f"design-note-summary:required={len(notes)} "
        f"existing={existing_count} "
        f"missing={report['missing_count']} "
        f"errors={report['error_count']} "
        f"strict={report['strict']}"
    )


def design_note_freeze_ready_summary(report: dict[str, object]) -> str:
    """Return a compact strict readiness summary for Stage 2 note freeze."""
    return design_note_summary(report).replace(
        "design-note-summary:",
        "design-note-freeze-ready:",
        1,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the design note readiness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("design/internal"),
        help="Directory containing Stage 2 clean-room design notes.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create missing design note templates before checking readiness.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print required Stage 2 design note paths without checking files.",
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
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact readiness summary.",
    )
    parser.add_argument(
        "--freeze-ready",
        action="store_true",
        help="Print the strict Stage 2 design-note freeze readiness summary.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(design_note_paths(args.directory)))
        return 0

    if args.init:
        print("\n".join(init_design_notes(args.directory)))
        return 0

    if args.json:
        report = design_note_report(args.directory, strict=args.strict)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if args.summary:
        report = design_note_report(args.directory, strict=args.strict)
        print(design_note_summary(report))
        return 0 if report["ok"] else 1

    if args.freeze_ready:
        report = design_note_report(args.directory, strict=True)
        print(design_note_freeze_ready_summary(report))
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
