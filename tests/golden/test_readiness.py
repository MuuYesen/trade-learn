from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_golden_readiness_reports_missing_real_artifacts_as_blocked() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_golden_readiness.py", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["summary"]["datasets_total"] == 10
    assert 0 <= payload["summary"]["datasets_ready"] < 10
    assert payload["summary"]["expected_total"] == 100
    assert payload["summary"]["expected_ready"] == 0
    assert payload["summary"]["strategies_total"] == 10
    assert payload["summary"]["strategies_ready"] == 0
    if payload["blockers"]["datasets"]:
        assert payload["blockers"]["datasets"][0]["reason"] == "missing real parquet"
    assert payload["blockers"]["expected"][0]["reason"] == "missing expected v1.0 result"
    assert payload["blockers"]["strategies"][0]["reason"] == "adapter still blocks execution"


def test_golden_readiness_text_output_is_actionable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_golden_readiness.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "golden-readiness:ok=False" in result.stdout
    assert "datasets=" in result.stdout
    assert "/10" in result.stdout
    assert "expected=0/100" in result.stdout
    assert "strategies=0/10" in result.stdout
    assert "blocked: missing expected v1.0 result" in result.stdout


def test_golden_readiness_can_gate_tv_dataset_subset(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True)
    manifest = json.loads((ROOT / "tests" / "golden" / "manifest.json").read_text())
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
        },
        index=pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC"),
    )
    for dataset in manifest["datasets"]:
        if dataset["engine"] != "tv":
            continue
        filename = (
            f"{dataset['symbol']}_{dataset['start']}_{dataset['end']}_{dataset['freq']}.parquet"
        )
        frame.to_parquet(tv_dir / filename)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_golden_readiness.py",
            "--json",
            "--require-tv-subset",
            "--datasets-root",
            str(datasets_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is False
    assert payload["subsets"]["tv"]["datasets_ready"] == 5
    assert payload["subsets"]["tv"]["datasets_total"] == 5
    assert payload["subsets"]["tv"]["ok"] is True
    assert payload["subsets"]["tdx"]["ok"] is False
