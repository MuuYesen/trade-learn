"""Tests for Stage 2 clean-room design note readiness checks."""

from __future__ import annotations

import json
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


def test_check_design_notes_reports_wrong_top_level_title(tmp_path: Path) -> None:
    """The checker fails when a design note has the wrong H1 title."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    write_design_note(docs_internal / "portfolio.md", "Event Loop")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "wrong title in portfolio.md: expected # Portfolio"
    ]


def test_check_design_notes_reports_duplicate_top_level_title(tmp_path: Path) -> None:
    """The checker fails when a design note has more than one H1 title."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    write_design_note(docs_internal / "portfolio.md", "Portfolio")
    with (docs_internal / "portfolio.md").open("a", encoding="utf-8") as handle:
        handle.write("\n# Event Loop\n")
        handle.write("A second H1 makes the clean-room note identity ambiguous.\n")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "duplicate top-level title in portfolio.md: # Event Loop"
    ]


def test_check_design_notes_accepts_crlf_top_level_title(tmp_path: Path) -> None:
    """The checker accepts design notes written with CRLF line endings."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    (docs_internal / "portfolio.md").write_text(
        "\r\n".join(
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


def test_check_design_notes_accepts_bom_before_top_level_title(tmp_path: Path) -> None:
    """The checker accepts UTF-8 BOM before the design-note H1."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    (docs_internal / "portfolio.md").write_text(
        "\ufeff"
        + "\n".join(
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


def test_check_design_notes_init_creates_required_templates(tmp_path: Path) -> None:
    """The design-note checker can create the three required note templates."""
    docs_internal = tmp_path / "docs" / "internal"

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--init", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "created design note: matching-design.md",
        "created design note: event-loop.md",
        "created design note: portfolio.md",
    ]
    assert result.stderr == ""

    check = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert check.stdout.splitlines() == [
        "design-note:matching-design.md=ok",
        "design-note:event-loop.md=ok",
        "design-note:portfolio.md=ok",
    ]


def test_check_design_notes_init_does_not_overwrite_existing_note(tmp_path: Path) -> None:
    """The init command must not overwrite existing local design drafts."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    existing = docs_internal / "portfolio.md"
    existing.write_text("# Custom Portfolio\n\nlocal draft\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--init", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "exists design note: portfolio.md" in result.stdout.splitlines()
    assert existing.read_text(encoding="utf-8") == "# Custom Portfolio\n\nlocal draft\n"


def test_check_design_notes_list_prints_required_note_paths(tmp_path: Path) -> None:
    """The list command prints the required Stage 2 design note paths."""
    docs_internal = tmp_path / "docs" / "internal"

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--list", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        str(docs_internal / "matching-design.md"),
        str(docs_internal / "event-loop.md"),
        str(docs_internal / "portfolio.md"),
    ]
    assert result.stderr == ""


def test_check_design_notes_strict_rejects_untouched_template_prompts(
    tmp_path: Path,
) -> None:
    """Strict mode fails while template prompts are still untouched."""
    docs_internal = tmp_path / "docs" / "internal"
    subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--init", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "untouched template prompt in matching-design.md: ## Scope" in (
        result.stderr.splitlines()
    )


def test_check_design_notes_strict_accepts_filled_design_notes(tmp_path: Path) -> None:
    """Strict mode passes once all template prompts have been replaced."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    for filename, title in [
        ("matching-design.md", "Matching Design"),
        ("event-loop.md", "Event Loop"),
        ("portfolio.md", "Portfolio"),
    ]:
        write_design_note(docs_internal / filename, title)

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
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


def test_check_design_notes_strict_rejects_placeholder_tokens(
    tmp_path: Path,
) -> None:
    """Strict mode fails when freeze-blocking placeholder tokens remain."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    for filename, title in [
        ("matching-design.md", "Matching Design"),
        ("event-loop.md", "Event Loop"),
        ("portfolio.md", "Portfolio"),
    ]:
        write_design_note(docs_internal / filename, title)
    with (docs_internal / "event-loop.md").open("a", encoding="utf-8") as handle:
        handle.write("\nTODO: resolve event ordering before freeze.\n")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "placeholder token in event-loop.md: TODO"
    ]


def test_check_design_notes_strict_allows_placeholder_substrings(
    tmp_path: Path,
) -> None:
    """Strict mode only treats standalone placeholder tokens as blockers."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    for filename, title in [
        ("matching-design.md", "Matching Design"),
        ("event-loop.md", "Event Loop"),
        ("portfolio.md", "Portfolio"),
    ]:
        write_design_note(docs_internal / filename, title)
    with (docs_internal / "event-loop.md").open("a", encoding="utf-8") as handle:
        handle.write("\nTODOS are tracked outside this note.\n")
        handle.write("TODO_LIST is an example variable name, not a placeholder.\n")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
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


def test_check_design_notes_strict_rejects_empty_section_body(
    tmp_path: Path,
) -> None:
    """Strict mode fails when a required section has no body content."""
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

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "empty section body in portfolio.md: ## Source Notes"
    ]


def test_check_design_notes_strict_allows_section_names_in_body_text(
    tmp_path: Path,
) -> None:
    """Strict mode treats only exact required heading lines as section boundaries."""
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
                "## Implementation Decisions are referenced as source-note prose.",
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

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--strict", str(docs_internal)],
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


def test_check_design_notes_reports_duplicate_required_section(
    tmp_path: Path,
) -> None:
    """The checker fails when a required design-note section appears twice."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    write_design_note(docs_internal / "portfolio.md", "Portfolio")
    with (docs_internal / "portfolio.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## Source Notes\n")
        handle.write("Duplicate notes would make the freeze boundary ambiguous.\n")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "duplicate section in portfolio.md: ## Source Notes"
    ]


def test_check_design_notes_ignores_required_sections_inside_fenced_blocks(
    tmp_path: Path,
) -> None:
    """The checker ignores markdown examples fenced inside design notes."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")
    write_design_note(docs_internal / "portfolio.md", "Portfolio")
    with (docs_internal / "portfolio.md").open("a", encoding="utf-8") as handle:
        handle.write("\n```markdown\n")
        handle.write("## Source Notes\n")
        handle.write("Example markdown that should not count as a real section.\n")
        handle.write("```\n")

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


def test_check_design_notes_json_reports_all_note_statuses(tmp_path: Path) -> None:
    """JSON output reports machine-readable status for all design notes."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    for filename, title in [
        ("matching-design.md", "Matching Design"),
        ("event-loop.md", "Event Loop"),
        ("portfolio.md", "Portfolio"),
    ]:
        write_design_note(docs_internal / filename, title)

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--json", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "directory": str(docs_internal),
        "strict": False,
        "ok": True,
        "notes": [
            {"file": "matching-design.md", "errors": []},
            {"file": "event-loop.md", "errors": []},
            {"file": "portfolio.md", "errors": []},
        ],
    }
    assert result.stderr == ""


def test_check_design_notes_json_reports_failures_on_stdout(
    tmp_path: Path,
) -> None:
    """JSON output keeps failures on stdout for automation consumers."""
    docs_internal = tmp_path / "docs" / "internal"
    docs_internal.mkdir(parents=True)
    write_design_note(docs_internal / "matching-design.md", "Matching Design")
    write_design_note(docs_internal / "event-loop.md", "Event Loop")

    result = subprocess.run(
        [sys.executable, "scripts/check_design_notes.py", "--json", str(docs_internal)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["notes"][-1] == {
        "file": "portfolio.md",
        "errors": ["missing design note: portfolio.md"],
    }
    assert result.stderr == ""
