from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_reference_oracle_layout_exists() -> None:
    reference = ROOT / "reference" / "tradelearn_1x"

    assert reference.is_dir()
    assert (reference / "query" / "query.py").is_file()


def test_check_oracle_cli_reports_readiness() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_oracle.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "oracle=ok" in result.stdout
    assert "provider:" in result.stdout


def test_check_oracle_cli_prints_oracle_group_hint_when_provider_missing() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_oracle.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    if "provider:yahoo=missing" in result.stdout:
        assert "hint=uv sync --group oracle --extra dev" in result.stdout


def test_pyproject_has_oracle_dependency_group() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    oracle = pyproject["dependency-groups"]["oracle"]

    assert any(dep.startswith("yfinance") for dep in oracle)
    assert any(dep.startswith("mootdx") for dep in oracle)
    tvdatafeed = "tvdatafeed @ git+https://github.com/rongardF/tvdatafeed.git"
    assert any(dep.startswith(tvdatafeed) for dep in oracle)


def test_build_golden_default_failure_writes_no_expected_json(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert not list(tmp_path.glob("*.json"))


def test_build_golden_dry_run_is_read_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--out",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "jobs=100" in result.stdout
    assert not any(tmp_path.iterdir())
