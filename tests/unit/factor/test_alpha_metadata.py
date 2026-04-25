"""Tests for Alpha formula metadata helpers."""

import json
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from tradelearn.factor.alpha import (
    ALPHA101_SKIPPED,
    ALPHA101_SUPPORTED,
    ALPHA191_SKIPPED,
    ALPHA191_SUPPORTED,
)

ROOT = Path(__file__).resolve().parents[3]


def test_alpha_formula_metadata_lists_supported_and_skipped_formulas() -> None:
    """Callers can discover supported and intentionally skipped Alpha formulas."""
    from tradelearn.factor import alpha as alpha_package

    alpha_formula_metadata = alpha_package.alpha_formula_metadata
    metadata = alpha_formula_metadata()

    assert metadata == {
        "alpha101": {
            "supported": tuple(sorted(ALPHA101_SUPPORTED)),
            "supported_count": len(ALPHA101_SUPPORTED),
            "skipped": ALPHA101_SKIPPED,
            "skipped_count": len(ALPHA101_SKIPPED),
        },
        "alpha191": {
            "supported": tuple(sorted(ALPHA191_SUPPORTED)),
            "supported_count": len(ALPHA191_SUPPORTED),
            "skipped": ALPHA191_SKIPPED,
            "skipped_count": len(ALPHA191_SKIPPED),
        },
    }


def test_alpha_formula_metadata_returns_skipped_copies() -> None:
    """Mutating metadata from one call must not change the package constants."""
    from tradelearn.factor import alpha as alpha_package

    alpha_formula_metadata = alpha_package.alpha_formula_metadata
    metadata = alpha_formula_metadata()

    metadata["alpha101"]["skipped"]["alpha999"] = "local mutation"
    metadata["alpha191"]["skipped"]["alpha999"] = "local mutation"

    fresh = alpha_formula_metadata()
    assert "alpha999" not in ALPHA101_SKIPPED
    assert "alpha999" not in ALPHA191_SKIPPED
    assert "alpha999" not in fresh["alpha101"]["skipped"]
    assert "alpha999" not in fresh["alpha191"]["skipped"]


def test_alpha_formula_metadata_includes_formula_counts() -> None:
    """Metadata exposes deterministic counts for progress and docs checks."""
    from tradelearn.factor import alpha as alpha_package

    metadata = alpha_package.alpha_formula_metadata()

    assert metadata["alpha101"]["supported_count"] == len(ALPHA101_SUPPORTED)
    assert metadata["alpha101"]["skipped_count"] == len(ALPHA101_SKIPPED)
    assert metadata["alpha191"]["supported_count"] == len(ALPHA191_SUPPORTED)
    assert metadata["alpha191"]["skipped_count"] == len(ALPHA191_SKIPPED)


def test_alpha_formula_metadata_uses_public_typed_dict() -> None:
    """The metadata helper exposes a stable type contract for consumers."""
    import tradelearn.factor.alpha as alpha_package

    hints = get_type_hints(alpha_package.alpha_formula_metadata)

    assert hints["return"] == dict[str, alpha_package.AlphaFormulaFamilyMetadata]
    assert "AlphaFormulaFamilyMetadata" in alpha_package.__all__


def test_alpha_formula_metadata_type_is_exported_from_factor_package() -> None:
    """The top-level factor facade exposes the metadata type contract."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.AlphaFormulaFamilyMetadata is alpha_package.AlphaFormulaFamilyMetadata
    assert "AlphaFormulaFamilyMetadata" in factor_package.__all__


def test_validate_alpha_formula_metadata_accepts_current_metadata() -> None:
    """The public validator accepts the packaged Alpha metadata."""
    import tradelearn.factor.alpha as alpha_package

    assert alpha_package.validate_alpha_formula_metadata() is None


def test_validated_alpha_formula_metadata_returns_current_metadata() -> None:
    """Consumers can fetch metadata that has already passed validation."""
    import tradelearn.factor.alpha as alpha_package

    metadata = alpha_package.validated_alpha_formula_metadata()

    assert metadata == alpha_package.alpha_formula_metadata()


def test_check_alpha_metadata_script_reports_validated_counts() -> None:
    """The metadata check script reports counts from validated metadata."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [sys.executable, "scripts/check_alpha_metadata.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert result.stdout.splitlines() == [
        f"{family}: supported={family_metadata['supported_count']} "
        f"skipped={family_metadata['skipped_count']}"
        for family, family_metadata in sorted(metadata.items())
    ]
    assert result.stderr == ""


def test_check_alpha_metadata_script_reports_json_counts() -> None:
    """The metadata check script can emit machine-readable counts."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [sys.executable, "scripts/check_alpha_metadata.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert json.loads(result.stdout) == {
        family: {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
        }
        for family, family_metadata in sorted(metadata.items())
    }
    assert result.stderr == ""


def test_check_alpha_metadata_script_filters_text_by_family() -> None:
    """The metadata check script can report one Alpha family in text mode."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [sys.executable, "scripts/check_alpha_metadata.py", "--family", "alpha191"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    family_metadata = validated_alpha_formula_metadata()["alpha191"]

    assert result.stdout.splitlines() == [
        "alpha191: supported="
        f"{family_metadata['supported_count']} skipped={family_metadata['skipped_count']}"
    ]
    assert result.stderr == ""


def test_check_alpha_metadata_script_filters_json_by_family() -> None:
    """The metadata check script can report one Alpha family in JSON mode."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_alpha_metadata.py",
            "--json",
            "--include-all",
            "--family",
            "alpha101",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    family_metadata = validated_alpha_formula_metadata()["alpha101"]

    assert json.loads(result.stdout) == {
        "alpha101": {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
            "supported": list(family_metadata["supported"]),
            "skipped": family_metadata["skipped"],
        }
    }
    assert result.stderr == ""


def test_check_alpha_metadata_script_lists_families() -> None:
    """The metadata check script can list available Alpha families."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [sys.executable, "scripts/check_alpha_metadata.py", "--list-families"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == sorted(validated_alpha_formula_metadata())
    assert result.stderr == ""


@pytest.mark.parametrize(
    "detail_flag",
    ["--include-skipped", "--include-supported", "--include-all"],
)
def test_check_alpha_metadata_detail_flags_require_json(detail_flag: str) -> None:
    """Detail flags are explicit JSON-only options."""
    result = subprocess.run(
        [sys.executable, "scripts/check_alpha_metadata.py", detail_flag],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert f"{detail_flag} requires --json" in result.stderr


def test_check_alpha_metadata_script_reports_json_skipped_reasons() -> None:
    """The metadata check script can include skipped reasons in JSON output."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_alpha_metadata.py",
            "--json",
            "--include-skipped",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert json.loads(result.stdout) == {
        family: {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
            "skipped": family_metadata["skipped"],
        }
        for family, family_metadata in sorted(metadata.items())
    }
    assert result.stderr == ""


def test_check_alpha_metadata_script_reports_json_supported_formulas() -> None:
    """The metadata check script can include supported formulas in JSON output."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_alpha_metadata.py",
            "--json",
            "--include-supported",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert json.loads(result.stdout) == {
        family: {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
            "supported": list(family_metadata["supported"]),
        }
        for family, family_metadata in sorted(metadata.items())
    }
    assert result.stderr == ""


def test_check_alpha_metadata_script_reports_all_json_formula_metadata() -> None:
    """The metadata check script can include all formula details in JSON output."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_alpha_metadata.py",
            "--json",
            "--include-all",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert json.loads(result.stdout) == {
        family: {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
            "supported": list(family_metadata["supported"]),
            "skipped": family_metadata["skipped"],
        }
        for family, family_metadata in sorted(metadata.items())
    }
    assert result.stderr == ""


def test_render_alpha_known_differences_script_reports_skipped_formulas() -> None:
    """The known differences script renders skipped Alpha formulas as markdown."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [sys.executable, "scripts/render_alpha_known_differences.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = validated_alpha_formula_metadata()

    assert result.stdout.splitlines() == [
        line
        for family, family_metadata in sorted(metadata.items())
        for line in (
            f"### Alpha {family} skipped formulas",
            "",
            "| Formula | Reason |",
            "|---|---|",
            *(
                f"| `{formula}` | {reason} |"
                for formula, reason in sorted(family_metadata["skipped"].items())
            ),
            "",
        )
    ]
    assert result.stderr == ""


def test_render_alpha_known_differences_script_filters_family() -> None:
    """The known differences script can render one Alpha family."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha101",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    family_metadata = validated_alpha_formula_metadata()["alpha101"]

    assert result.stdout.splitlines() == [
        "### Alpha alpha101 skipped formulas",
        "",
        "| Formula | Reason |",
        "|---|---|",
        *(
            f"| `{formula}` | {reason} |"
            for formula, reason in sorted(family_metadata["skipped"].items())
        ),
        "",
    ]
    assert result.stderr == ""


def test_render_alpha_known_differences_script_lists_families() -> None:
    """The known differences script can list available Alpha families."""
    from tradelearn.factor.alpha import validated_alpha_formula_metadata

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--list-families",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == sorted(validated_alpha_formula_metadata())
    assert result.stderr == ""


def test_render_alpha_known_differences_script_check_passes_when_content_exists(
    tmp_path: Path,
) -> None:
    """The known differences script can verify rendered content in a file."""
    rendered = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha101",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    target = tmp_path / "MIGRATION.md"
    target.write_text(f"# MIGRATION\n\n{rendered}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha101",
            "--check",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == (
        f"Alpha known differences sections present in {target}: alpha101\n"
    )
    assert result.stderr == ""


def test_render_alpha_known_differences_script_check_fails_for_missing_content(
    tmp_path: Path,
) -> None:
    """The known differences script reports missing rendered content."""
    target = tmp_path / "MIGRATION.md"
    target.write_text("# MIGRATION\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha191",
            "--check",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "Missing Alpha known differences sections: alpha191\n"


def test_render_alpha_known_differences_script_check_fails_for_missing_file(
    tmp_path: Path,
) -> None:
    """The known differences script reports an unreadable target file."""
    target = tmp_path / "missing.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--check",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Cannot read Alpha known differences target: {target}\n"


def test_render_alpha_known_differences_script_check_reports_all_families(
    tmp_path: Path,
) -> None:
    """The known differences script reports every checked Alpha family."""
    rendered = subprocess.run(
        [sys.executable, "scripts/render_alpha_known_differences.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    target = tmp_path / "MIGRATION.md"
    target.write_text(f"# MIGRATION\n\n{rendered}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--check",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == (
        f"Alpha known differences sections present in {target}: alpha101, alpha191\n"
    )
    assert result.stderr == ""


def test_render_alpha_known_differences_script_writes_output_file(
    tmp_path: Path,
) -> None:
    """The known differences script can write rendered Markdown to a file."""
    expected = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha101",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    target = tmp_path / "alpha-known-differences.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--family",
            "alpha101",
            "--output",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert target.read_text(encoding="utf-8") == expected
    assert result.stdout == (
        f"Alpha known differences sections written to {target}: alpha101\n"
    )
    assert result.stderr == ""


def test_render_alpha_known_differences_script_output_rejects_check(
    tmp_path: Path,
) -> None:
    """The known differences script keeps output and check modes separate."""
    target = tmp_path / "MIGRATION.md"
    output = tmp_path / "alpha-known-differences.md"
    target.write_text("# MIGRATION\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--check",
            str(target),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--output cannot be used with --check" in result.stderr
    assert not output.exists()


def test_render_alpha_known_differences_script_output_reports_write_error(
    tmp_path: Path,
) -> None:
    """The known differences script reports output write failures cleanly."""
    target = tmp_path / "missing-parent" / "alpha-known-differences.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_alpha_known_differences.py",
            "--output",
            str(target),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == f"Cannot write Alpha known differences target: {target}\n"


def test_validate_alpha_formula_metadata_rejects_inconsistent_counts() -> None:
    """The validator catches stale counts before docs consume metadata."""
    import tradelearn.factor.alpha as alpha_package

    metadata = alpha_package.alpha_formula_metadata()
    metadata["alpha101"]["supported_count"] += 1

    with pytest.raises(ValueError, match="alpha101 supported_count"):
        alpha_package.validate_alpha_formula_metadata(metadata)


def test_validate_alpha_formula_metadata_rejects_supported_skipped_overlap() -> None:
    """The validator catches formulas marked both supported and skipped."""
    import tradelearn.factor.alpha as alpha_package

    metadata = alpha_package.alpha_formula_metadata()
    metadata["alpha101"]["skipped"]["alpha001"] = "local overlap"
    metadata["alpha101"]["skipped_count"] += 1

    with pytest.raises(
        ValueError,
        match="alpha101 formulas cannot be both supported and skipped: alpha001",
    ):
        alpha_package.validate_alpha_formula_metadata(metadata)


def test_validate_alpha_formula_metadata_is_exported_from_factor_package() -> None:
    """The top-level factor facade exposes the metadata validator."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert (
        factor_package.validate_alpha_formula_metadata
        is alpha_package.validate_alpha_formula_metadata
    )
    assert "validate_alpha_formula_metadata" in factor_package.__all__


def test_validated_alpha_formula_metadata_is_exported_from_factor_package() -> None:
    """The top-level factor facade exposes the validated metadata helper."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert (
        factor_package.validated_alpha_formula_metadata
        is alpha_package.validated_alpha_formula_metadata
    )
    assert "validated_alpha_formula_metadata" in factor_package.__all__


def test_alpha_formula_metadata_is_exported_from_package_all() -> None:
    """The helper is part of the public alpha facade."""
    import tradelearn.factor.alpha as alpha_package

    assert "alpha_formula_metadata" in alpha_package.__all__
    assert "validated_alpha_formula_metadata" in alpha_package.__all__
    assert "validate_alpha_formula_metadata" in alpha_package.__all__


def test_alpha_formula_metadata_is_exported_from_factor_package() -> None:
    """The top-level factor facade exposes Alpha formula metadata."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.alpha_formula_metadata() == alpha_package.alpha_formula_metadata()
    assert "alpha_formula_metadata" in factor_package.__all__


def test_alpha_formula_constants_are_exported_from_factor_package() -> None:
    """The top-level factor facade exposes Alpha formula metadata constants."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.ALPHA101_SUPPORTED == alpha_package.ALPHA101_SUPPORTED
    assert factor_package.ALPHA191_SUPPORTED == alpha_package.ALPHA191_SUPPORTED
    assert factor_package.ALPHA101_SKIPPED == alpha_package.ALPHA101_SKIPPED
    assert factor_package.ALPHA191_SKIPPED == alpha_package.ALPHA191_SKIPPED
    assert {
        "ALPHA101_SKIPPED",
        "ALPHA101_SUPPORTED",
        "ALPHA191_SKIPPED",
        "ALPHA191_SUPPORTED",
    }.issubset(factor_package.__all__)


def test_alpha_formula_facades_are_exported_from_factor_package() -> None:
    """The top-level factor facade exposes both Alpha formula callable facades."""
    import tradelearn.factor as factor_package
    import tradelearn.factor.alpha as alpha_package

    assert factor_package.alpha101 is alpha_package.alpha101
    assert factor_package.alpha191 is alpha_package.alpha191
    assert {"alpha101", "alpha191"}.issubset(factor_package.__all__)
