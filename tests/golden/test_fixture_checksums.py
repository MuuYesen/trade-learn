"""Offline oracle inputs are immutable and present in a clean checkout."""
import hashlib
import json
from pathlib import Path


def test_frozen_fixture_checksums():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "tests/golden/fixture_checksums.json").read_text())
    for relative, expected in manifest.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
