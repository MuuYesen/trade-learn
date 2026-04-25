#!/usr/bin/env python
"""Render Alpha skipped formula known differences as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_metadata_cli import (  # noqa: E402
    alpha_metadata_families,
    selected_alpha_metadata_families,
)

from tradelearn.factor.alpha import validated_alpha_formula_metadata  # noqa: E402

MIGRATION_KNOWN_DIFFERENCES_HEADING = "### 3.2 登记示例(待填充)"
MIGRATION_KNOWN_DIFFERENCES_END = "\n---"


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


def update_migration_known_differences(content: str, rendered: str) -> str:
    """Replace the MIGRATION known-differences block with rendered Alpha content."""
    heading_start = content.find(MIGRATION_KNOWN_DIFFERENCES_HEADING)
    if heading_start == -1:
        raise ValueError("missing known differences heading")
    heading_end = heading_start + len(MIGRATION_KNOWN_DIFFERENCES_HEADING)
    block_end = content.find(MIGRATION_KNOWN_DIFFERENCES_END, heading_end)
    if block_end == -1:
        raise ValueError("missing known differences block delimiter")
    replacement = "\n\n" + rendered.rstrip() + "\n\n"
    return content[:heading_end] + replacement + content[block_end:]


def main(argv: list[str] | None = None) -> int:
    """Print Markdown tables for skipped Alpha formulas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        help="render only one Alpha formula family",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="verify the rendered Markdown is present in a file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the rendered Markdown to a file",
    )
    parser.add_argument(
        "--update",
        type=Path,
        help="replace the MIGRATION known differences block in a file",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        help="list available Alpha formula families",
    )
    args = parser.parse_args(argv)
    write_modes = [
        mode
        for enabled, mode in (
            (args.check is not None, "--check"),
            (args.output is not None, "--output"),
            (args.update is not None, "--update"),
        )
        if enabled
    ]
    if args.check is not None and args.output is not None and args.update is None:
        parser.error("--output cannot be used with --check")
    if len(write_modes) > 1:
        parser.error(", ".join(write_modes) + " cannot be used together")

    metadata = validated_alpha_formula_metadata()
    if args.list_families:
        for family in alpha_metadata_families(metadata):
            print(family)
        return 0

    families = selected_alpha_metadata_families(parser, metadata, args.family)

    rendered_by_family = {
        family: render_family(family, metadata[family]["skipped"]) for family in families
    }

    if args.check is not None:
        try:
            content = args.check.read_text(encoding="utf-8")
        except OSError:
            print(
                f"Cannot read Alpha known differences target: {args.check}",
                file=sys.stderr,
            )
            return 1
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
        print(
            f"Alpha known differences sections present in {args.check}: "
            + ", ".join(families)
        )
        return 0

    rendered_output = "".join(rendered_by_family[family] for family in families)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered_output, encoding="utf-8")
        except OSError:
            print(
                f"Cannot write Alpha known differences target: {args.output}",
                file=sys.stderr,
            )
            return 1
        print(
            f"Alpha known differences sections written to {args.output}: "
            + ", ".join(families)
        )
        return 0

    if args.update is not None:
        try:
            content = args.update.read_text(encoding="utf-8")
            updated = update_migration_known_differences(content, rendered_output)
            args.update.write_text(updated, encoding="utf-8")
        except (OSError, ValueError):
            print(
                f"Cannot update Alpha known differences target: {args.update}",
                file=sys.stderr,
            )
            return 1
        print(
            f"Alpha known differences sections updated in {args.update}: "
            + ", ".join(families)
        )
        return 0

    for family in families:
        print(rendered_by_family[family], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
