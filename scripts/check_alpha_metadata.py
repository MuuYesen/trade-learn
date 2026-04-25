#!/usr/bin/env python
"""Check Alpha formula metadata consistency."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402


def main() -> int:
    """Print validated Alpha metadata counts."""
    metadata = validated_alpha_formula_metadata()
    for family, family_metadata in sorted(metadata.items()):
        print(
            f"{family}: supported={family_metadata['supported_count']} "
            f"skipped={family_metadata['skipped_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
