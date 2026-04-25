#!/usr/bin/env python
"""Render Alpha skipped formula known differences as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print Markdown tables for skipped Alpha formulas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("alpha101", "alpha191"),
        help="render only one Alpha formula family",
    )
    args = parser.parse_args(argv)

    metadata = validated_alpha_formula_metadata()
    families = [args.family] if args.family else sorted(metadata)
    for family in families:
        family_metadata = metadata[family]
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
