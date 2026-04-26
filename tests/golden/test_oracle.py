from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import tomllib

from scripts import build_golden

ROOT = Path(__file__).resolve().parents[2]
RETIRED_PROVIDER_NAMES = ("pytd" + "x2", "moot" + "dx")


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
    if "provider:" in result.stdout and "=missing" in result.stdout:
        assert "hint=uv sync --group oracle --extra dev" in result.stdout


def test_pyproject_has_oracle_dependency_group() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    oracle = pyproject["dependency-groups"]["oracle"]

    assert any(dep.startswith("opentdx") for dep in dependencies)
    assert not any(dep.startswith(RETIRED_PROVIDER_NAMES[0]) for dep in dependencies)
    assert not any(dep.startswith("yfinance") for dep in oracle)
    assert any(dep.startswith("opentdx") for dep in oracle)
    assert not any(dep.startswith(RETIRED_PROVIDER_NAMES[1]) for dep in oracle)
    tvdatafeed = "tvdatafeed @ git+https://github.com/rongardF/tvdatafeed.git"
    assert any(dep.startswith(tvdatafeed) for dep in oracle)


def test_current_provider_source_uses_opentdx_not_retired_names() -> None:
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "scripts" / "build_golden.py",
        ROOT / "scripts" / "check_oracle.py",
        ROOT / "tradelearn" / "data" / "providers.py",
        ROOT / "tradelearn" / "query" / "query.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "opentdx" in combined
    assert RETIRED_PROVIDER_NAMES[0] not in combined
    assert RETIRED_PROVIDER_NAMES[1] not in combined


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


def test_build_golden_datasets_only_uses_provider_stubs(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_load_reference_query(*, allow_provider_stubs: bool = False):
        calls.append(allow_provider_stubs)
        query = Mock()
        query.history_ohlc.side_effect = build_golden.GoldenDataError("network unavailable")
        return query

    monkeypatch.setattr(build_golden, "load_reference_query", fake_load_reference_query)
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": True, "tv": True})

    result = build_golden.main(
        ["--version", "1.x", "--out", str(tmp_path), "--datasets-only"]
    )

    assert result == 2
    assert calls == [True]


def test_build_golden_datasets_only_reports_unavailable_opentdx(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(build_golden, "provider_statuses", lambda: {"tdx": False, "tv": True})

    result = build_golden.main(
        ["--version", "1.x", "--out", str(tmp_path), "--datasets-only"]
    )

    assert result == 2
    assert "dataset provider unavailable: tdx:opentdx.tdxClient" in capsys.readouterr().err
