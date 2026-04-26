from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _write_tv_fixture(datasets_root: Path, symbol: str, offset: float = 0.0) -> None:
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "open": [10.0 + offset, 11.0 + offset, 12.0 + offset, 13.0 + offset],
            "high": [11.0 + offset, 12.0 + offset, 13.0 + offset, 14.0 + offset],
            "low": [9.0 + offset, 10.0 + offset, 11.0 + offset, 12.0 + offset],
            "close": [10.5 + offset, 11.5 + offset, 12.5 + offset, 13.5 + offset],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )
    frame.to_parquet(tv_dir / f"{symbol}_2020-01-01_2024-12-31_1d.parquet")


def test_tv_subset_smoke_runs_every_manifest_tv_dataset(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    for offset, symbol in enumerate(["GOOG", "AAPL", "MSFT", "SPY", "BTCUSDT"]):
        _write_tv_fixture(datasets_root, symbol, float(offset))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_golden_smoke.py",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["engine"] == "tv"
    assert payload["ok"] is True
    assert payload["summary"] == {"requested": 5, "ran": 5, "missing": 0, "failed": 0}
    assert [item["dataset"] for item in payload["results"]] == [
        "GOOG",
        "AAPL",
        "MSFT",
        "SPY",
        "BTCUSDT",
    ]
    assert payload["results"][0]["trades"] == [
        {"size": 2.0, "price": 11.0, "pnl": 0.0, "isclosed": False},
        {"size": 0.0, "price": 13.0, "pnl": 4.0, "isclosed": True},
    ]
    assert payload["results"][0]["equity"] == [100.0, 101.0, 103.0, 104.0]
    assert payload["results"][0]["final_cash"] == 104.0
    assert payload["results"][0]["pnl"] == 4.0
    assert all(len(item["trades"]) == 2 for item in payload["results"])


def test_tv_subset_smoke_requires_all_tv_datasets_by_default(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    _write_tv_fixture(datasets_root, "GOOG")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_golden_smoke.py",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ok"] is False
    assert payload["summary"] == {"requested": 5, "ran": 1, "missing": 4, "failed": 0}
    assert [item["dataset"] for item in payload["missing"]] == [
        "AAPL",
        "MSFT",
        "SPY",
        "BTCUSDT",
    ]


def test_tv_subset_smoke_can_run_available_subset_in_allow_missing_mode(
    tmp_path: Path,
) -> None:
    datasets_root = tmp_path / "datasets"
    _write_tv_fixture(datasets_root, "GOOG")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_small_golden_smoke.py",
            "--engine",
            "tv",
            "--datasets-root",
            str(datasets_root),
            "--allow-missing",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["summary"] == {"requested": 5, "ran": 1, "missing": 4, "failed": 0}
