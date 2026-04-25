#!/usr/bin/env python
"""Check Alpha formula metadata consistency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Print validated Alpha metadata counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit metadata counts as JSON",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help="include skipped formula reasons in JSON output",
    )
    parser.add_argument(
        "--include-supported",
        action="store_true",
        help="include supported formula names in JSON output",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="include supported and skipped formula details in JSON output",
    )
    args = parser.parse_args(argv)

    metadata = validated_alpha_formula_metadata()
    counts = {}
    for family, family_metadata in sorted(metadata.items()):
        family_counts = {
            "supported_count": family_metadata["supported_count"],
            "skipped_count": family_metadata["skipped_count"],
        }
        if args.include_supported or args.include_all:
            family_counts["supported"] = list(family_metadata["supported"])
        if args.include_skipped or args.include_all:
            family_counts["skipped"] = family_metadata["skipped"]
        counts[family] = family_counts
    if args.json:
        print(json.dumps(counts, sort_keys=True))
        return 0

    for family, family_metadata in sorted(metadata.items()):
        print(
            f"{family}: supported={family_metadata['supported_count']} "
            f"skipped={family_metadata['skipped_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
