#!/usr/bin/env python
"""Render Alpha skipped formula known differences as Markdown."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402


def main() -> int:
    """Print Markdown tables for skipped Alpha formulas."""
    metadata = validated_alpha_formula_metadata()
    for family, family_metadata in sorted(metadata.items()):
        print(f"### Alpha {family} skipped formulas")
        print()
        print("| Formula | Reason |")
        print("|---|---|")
        for formula, reason in sorted(family_metadata["skipped"].items()):
            print(f"| `{formula}` | {reason} |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
