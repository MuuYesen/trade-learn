from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_tv_subset_smoke_runs_cerebro_chain_from_parquet(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    tv_dir = datasets_root / "tv"
    tv_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5, 13.5],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
    )
    frame.to_parquet(tv_dir / "GOOG_2020-01-01_2024-12-31_1d.parquet")

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
    assert payload["dataset"] == "GOOG"
    assert payload["trades"] == [
        {"size": 2.0, "price": 11.0, "pnl": 0.0, "isclosed": False},
        {"size": 0.0, "price": 13.0, "pnl": 4.0, "isclosed": True},
    ]
    assert payload["equity"] == [100.0, 101.0, 103.0, 104.0]
    assert payload["final_cash"] == 104.0
