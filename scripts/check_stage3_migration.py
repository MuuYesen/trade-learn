from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_golden_readiness import build_report  # noqa: E402

SNAPSHOT = ROOT / "benchmarks" / "stage3_migration_blockers.json"
REQUIRED_IDS = {
    "golden-datasets",
    "golden-expected-v1",
    "golden-strategy-adapters",
    "full-golden-comparison",
}
ALLOWED_STATUSES = {"blocked", "waiting", "accepted"}


def load_snapshot(path: Path = SNAPSHOT) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = snapshot.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    mapped = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("entries must contain objects")
        entry_id = entry.get("id")
        if not isinstance(entry_id, str):
            raise ValueError("entry id must be a string")
        if entry_id in mapped:
            raise ValueError(f"duplicate entry id: {entry_id}")
        mapped[entry_id] = entry
    return mapped


def validate_snapshot(snapshot: dict[str, Any], readiness: dict[str, Any]) -> list[str]:
    errors = []
    entries = _entry_map(snapshot)
    missing = sorted(REQUIRED_IDS - set(entries))
    if missing:
        errors.append("missing entries: " + ", ".join(missing))
    for entry_id, entry in entries.items():
        status = entry.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{entry_id}: invalid status {status!r}")
        if not entry.get("reason"):
            errors.append(f"{entry_id}: missing reason")
        if not entry.get("next"):
            errors.append(f"{entry_id}: missing next")

    summary = readiness["summary"]
    expected_counts = {
        "golden-datasets": (
            summary["datasets_ready"],
            summary["datasets_total"],
        ),
        "golden-expected-v1": (
            summary["expected_ready"],
            summary["expected_total"],
        ),
        "golden-strategy-adapters": (
            summary["strategies_ready"],
            summary["strategies_total"],
        ),
    }
    for entry_id, (ready, total) in expected_counts.items():
        entry = entries.get(entry_id)
        if entry is None:
            continue
        if entry.get("ready") != ready or entry.get("total") != total:
            errors.append(
                f"{entry_id}: expected ready/total {ready}/{total}, "
                f"got {entry.get('ready')}/{entry.get('total')}"
            )

    full_golden = entries.get("full-golden-comparison")
    if full_golden is not None and not readiness["ok"]:
        if full_golden.get("status") != "blocked":
            errors.append("full-golden-comparison must be blocked until readiness ok=true")
    return errors


def build_payload(path: Path = SNAPSHOT) -> dict[str, Any]:
    snapshot = load_snapshot(path)
    readiness = build_report(engine="tv")
    errors = validate_snapshot(snapshot, readiness)
    return {
        "ok": not errors,
        "snapshot": str(path.relative_to(ROOT)),
        "entries": snapshot["entries"],
        "readiness": readiness["summary"],
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage 3 migration blocker snapshot.")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.snapshot)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(
            "stage3-migration:"
            f"ok={payload['ok']}"
            f" entries={len(payload['entries'])}"
            f" errors={len(payload['errors'])}"
        )
        for error in payload["errors"]:
            print(f"error: {error}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
