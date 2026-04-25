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
    parser.add_argument(
        "--family",
        help="check only one Alpha formula family",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="list available Alpha formula families",
    )
    args = parser.parse_args(argv)

    detail_flags = [
        flag
        for enabled, flag in (
            (args.include_skipped, "--include-skipped"),
            (args.include_supported, "--include-supported"),
            (args.include_all, "--include-all"),
        )
        if enabled
    ]
    if detail_flags and not args.json:
        parser.error(", ".join(detail_flags) + " requires --json")

    metadata = validated_alpha_formula_metadata()
    if args.list_families:
        families = sorted(metadata)
        if args.json:
            print(json.dumps(families))
            return 0
        for family in families:
            print(family)
        return 0

    if args.family is not None and args.family not in metadata:
        parser.error(
            f"Unknown Alpha family: {args.family}. Available: "
            + ", ".join(sorted(metadata))
        )

    families = [args.family] if args.family else sorted(metadata)
    counts = {}
    for family in families:
        family_metadata = metadata[family]
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

    for family in families:
        family_metadata = metadata[family]
        print(
            f"{family}: supported={family_metadata['supported_count']} "
            f"skipped={family_metadata['skipped_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
