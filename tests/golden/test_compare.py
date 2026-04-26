from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _write_tv_subset(datasets_root: Path) -> None:
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True)
    manifest = json.loads((ROOT / "tests" / "golden" / "manifest.json").read_text())
    for index, dataset in enumerate(manifest["datasets"]):
        if dataset["engine"] != "tv":
            continue
        base = 100 + index * 10
        frame = pd.DataFrame(
            {
                "open": [base, base + 1, base + 2, base + 3, base + 4, base + 5],
                "high": [base + 2, base + 3, base + 4, base + 5, base + 6, base + 7],
                "low": [base - 1, base, base + 1, base + 2, base + 3, base + 4],
                "close": [base + 1, base + 2, base + 3, base + 4, base + 5, base + 6],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            },
            index=pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
        )
        filename = (
            f"{dataset['symbol']}_{dataset['start']}_{dataset['end']}_{dataset['freq']}.parquet"
        )
        frame.to_parquet(tv_dir / filename)


def _build_expected(datasets_root: Path, expected_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_golden.py",
            "--version",
            "1.x",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--out",
            str(expected_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_compare_golden_tv_subset_passes_against_expected(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    expected_root = tmp_path / "expected"
    _write_tv_subset(datasets_root)
    _build_expected(datasets_root, expected_root)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_golden.py",
            "--json",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--expected-root",
            str(expected_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["summary"] == {"compared": 50, "failed": 0}


def test_compare_golden_detects_trade_differences(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    expected_root = tmp_path / "expected"
    _write_tv_subset(datasets_root)
    _build_expected(datasets_root, expected_root)
    expected_file = expected_root / "sma_cross__GOOG.json"
    payload = json.loads(expected_file.read_text())
    payload["trades"].append({"size": 99})
    expected_file.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_golden.py",
            "--json",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--expected-root",
            str(expected_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["summary"]["failed"] == 1
    assert payload["failures"][0]["reason"] == "trades differ"
