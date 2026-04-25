"""Tests for Stage 2 clean-room design note readiness checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_design_note(path: Path, title: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "## Scope",
                "Captures the design boundary.",
                "",
                "## Clean-room Boundary",
                "Records source separation rules.",
                "",
                "## Source Notes",
                "Lists references reviewed before the freeze.",
                "",
                "## Implementation Decisions",
                "Records independent implementation decisions.",
                "",
                "## Open Questions",
                "Tracks unresolved follow-up items.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_check_design_notes_accepts_required_clean_room_notes(tmp_path: Path) -> None:
    """The design-note checker accepts the three required Stage 2 notes."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    write_design_note(docs_internal / "portfolio.md", "Portfolio")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "design-note:matching-design.md=ok",
        "design-note:event-loop.md=ok",
        "design-note:portfolio.md=ok",
    ]
    assert result.stderr == ""


def test_check_design_notes_reports_missing_note(tmp_path: Path) -> None:
    """The design-note checker fails when one required note is missing."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == ["missing design note: portfolio.md"]


def test_check_design_notes_reports_missing_required_section(tmp_path: Path) -> None:
    """The design-note checker fails when a required clean-room section is absent."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    (docs_internal / "portfolio.md").write_text(
        "\n".join(
            [
                "# Portfolio",
                "",
                "## Scope",
                "Captures the design boundary.",
                "",
                "## Clean-room Boundary",
                "Records source separation rules.",
                "",
                "## Source Notes",
                "Lists references reviewed before the freeze.",
                "",
                "## Open Questions",
                "Tracks unresolved follow-up items.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "missing section in portfolio.md: ## Implementation Decisions"
    ]
