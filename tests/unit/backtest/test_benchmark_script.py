from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_stage3_benchmark.py"


def test_stage3_benchmark_script_emits_machine_readable_results() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--single-bars",
            "5",
            "--portfolio-bars",
            "4",
            "--portfolio-symbols",
            "2",
            "--max-single-ms",
            "1000",
            "--max-portfolio-ms",
            "1000",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["single"]["bars"] == 5
    assert payload["portfolio"]["bars"] == 4
    assert payload["portfolio"]["symbols"] == 2
    assert payload["single"]["elapsed_ms"] >= 0.0
    assert payload["portfolio"]["elapsed_ms"] >= 0.0


def test_stage3_benchmark_script_fails_when_threshold_is_exceeded() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--single-bars",
            "5",
            "--portfolio-bars",
            "4",
            "--portfolio-symbols",
            "2",
            "--max-single-ms",
            "0",
            "--max-portfolio-ms",
            "0",
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["single"]["ok"] is False
    assert payload["portfolio"]["ok"] is False
