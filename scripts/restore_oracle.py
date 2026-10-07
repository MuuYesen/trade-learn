#!/usr/bin/env python
"""Restore the minimal frozen 1.x test oracle from verified local Git blobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("oracle_manifest.json")


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def restore(destination: Path = ROOT) -> list[str]:
    """Verify every source and destination before writing missing oracle files."""
    metadata = manifest()
    pending = []
    for item in metadata["files"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid oracle path: {path}")
        target = destination / path
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"refusing to overwrite modified oracle: {target}")
            continue
        data = subprocess.check_output(
            ["git", "show", f"{metadata['commit']}:{path.as_posix()}"], cwd=ROOT
        )
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"oracle source checksum mismatch: {path}")
        pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return [str(target) for target, _ in pending]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        restored = restore(args.destination)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"oracle restore failed: {exc}\n")
    print(f"oracle=verified restored={len(restored)} commit={manifest()['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
