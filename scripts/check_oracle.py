#!/usr/bin/env python
"""Check readiness of the frozen 1.x oracle without fetching remote data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_golden import (  # noqa: E402
    REFERENCE,
    load_reference_query,
    provider_statuses,
    validate_reference,
)
from tradelearn.core.errors import GoldenDataError  # noqa: E402


def main() -> int:
    """Run oracle readiness checks."""

    try:
        validate_reference()
        query = load_reference_query(allow_provider_stubs=True)
    except GoldenDataError as exc:
        print(f"oracle=error {exc}", file=sys.stderr)
        return 2

    print("oracle=ok")
    print(f"reference={REFERENCE}")
    print(f"query={query.__module__}.{query.__name__}")
    statuses = provider_statuses()
    for engine, available in statuses.items():
        status = "ok" if available else "missing"
        print(f"provider:{engine}={status}")
    if not all(statuses.values()):
        print("hint=uv sync --group oracle --extra dev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
