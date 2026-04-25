#!/usr/bin/env python
"""Render Alpha skipped formula known differences as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402


def render_family(family: str, skipped: dict[str, str]) -> str:
    """Render one Alpha family skipped formula table."""
    lines = [
        f"### Alpha {family} skipped formulas",
        "",
        "| Formula | Reason |",
        "|---|---|",
        *(f"| `{formula}` | {reason} |" for formula, reason in sorted(skipped.items())),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print Markdown tables for skipped Alpha formulas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("alpha101", "alpha191"),
        help="render only one Alpha formula family",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="verify the rendered Markdown is present in a file",
    )
    args = parser.parse_args(argv)

    metadata = validated_alpha_formula_metadata()
    families = [args.family] if args.family else sorted(metadata)

    rendered_by_family = {
        family: render_family(family, metadata[family]["skipped"]) for family in families
    }

    if args.check is not None:
        content = args.check.read_text(encoding="utf-8")
        missing = [
            family
            for family, rendered in rendered_by_family.items()
            if rendered not in content
        ]
        if missing:
            print(
                "Missing Alpha known differences sections: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 1
        print(f"Alpha known differences are present in {args.check}")
        return 0

    for family in families:
        print(rendered_by_family[family], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
