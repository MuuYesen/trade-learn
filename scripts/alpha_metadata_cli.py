"""Shared helpers for Alpha metadata command-line tools."""

from __future__ import annotations

import argparse
from typing import Any


def alpha_metadata_families(metadata: dict[str, Any]) -> list[str]:
    """Return available Alpha metadata family names in deterministic order."""
    return sorted(metadata)


def selected_alpha_metadata_families(
    parser: argparse.ArgumentParser,
    metadata: dict[str, Any],
    family: str | None,
) -> list[str]:
    """Return selected Alpha families or report a stable argparse error."""
    families = alpha_metadata_families(metadata)
    if family is not None and family not in metadata:
        parser.error(
            f"Unknown Alpha family: {family}. Available: " + ", ".join(families)
        )
    return [family] if family else families
