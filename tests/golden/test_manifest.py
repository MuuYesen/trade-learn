from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "golden" / "manifest.json"


def test_manifest_matches_documented_shape() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["source"] == "docs/specs/CONSISTENCY.md"
    assert len(manifest["datasets"]) == 10
    assert len(manifest["strategies"]) == 10
    assert {item["engine"] for item in manifest["datasets"]} == {"tv", "tdx"}


def test_golden_directories_exist() -> None:
    for relative in [
        "tests/golden/datasets/tv",
        "tests/golden/datasets/tdx",
        "tests/golden/strategies",
        "tests/golden/expected/v1.0",
        "tests/golden/indicators/core",
        "tests/golden/indicators/tdx",
        "tests/golden/indicators/tv",
        "tests/golden/returns",
    ]:
        assert (ROOT / relative).is_dir()


def test_synthetic_returns_has_10_fixtures() -> None:
    returns = pd.read_csv(ROOT / "tests" / "golden" / "returns" / "synthetic_returns_10.csv")

    assert "timestamp" in returns.columns
    assert len([column for column in returns.columns if column != "timestamp"]) == 10


def test_documented_strategy_scripts_exist_and_import() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for strategy in manifest["strategies"]:
        name = strategy["name"]
        path = ROOT / "tests" / "golden" / "strategies" / f"{name}.py"
        assert path.exists(), f"missing strategy script: {path}"
        namespace = runpy.run_path(str(path))
        assert namespace["STRATEGY_NAME"] == name
        classes = [
            value
            for key, value in namespace.items()
            if key.endswith("Strategy") and isinstance(value, type)
        ]
        assert len(classes) == 1


def test_build_golden_datasets_only_reports_unavailable_provider() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--out",
            "tests/golden/expected/v1.0/",
            "--datasets-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "dataset generation failed" in result.stderr


def test_build_golden_default_mode_does_not_write_fake_expected(tmp_path: Path) -> None:
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


def test_build_golden_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--out",
            "tests/golden/expected/v1.0/",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "jobs=100" in result.stdout
