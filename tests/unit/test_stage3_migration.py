from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_stage3_migration.py"
SNAPSHOT = ROOT / "benchmarks" / "stage3_migration_blockers.json"


def test_stage3_migration_blocker_snapshot_matches_readiness_gates() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    entries = {entry["id"]: entry for entry in payload["entries"]}

    assert payload["snapshot"] == str(SNAPSHOT.relative_to(ROOT))
    assert payload["ok"] is True
    assert entries["golden-datasets"]["status"] == "blocked"
    assert entries["golden-datasets"]["ready"] == 0
    assert entries["golden-datasets"]["total"] == 10
    assert entries["golden-expected-v1"]["ready"] == 0
    assert entries["golden-expected-v1"]["total"] == 100
    assert entries["golden-strategy-adapters"]["ready"] == 0
    assert entries["golden-strategy-adapters"]["total"] == 10
    assert entries["full-golden-comparison"]["status"] == "blocked"
