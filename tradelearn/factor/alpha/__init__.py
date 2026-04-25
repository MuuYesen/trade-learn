"""Alpha factor formula facades."""

from typing import TypedDict

from tradelearn.factor.alpha.alpha101 import ALPHA101_SKIPPED, ALPHA101_SUPPORTED, alpha101
from tradelearn.factor.alpha.alpha191 import ALPHA191_SKIPPED, ALPHA191_SUPPORTED, alpha191


class AlphaFormulaFamilyMetadata(TypedDict):
    """Metadata for a single Alpha formula family."""

    supported: tuple[str, ...]
    supported_count: int
    skipped: dict[str, str]
    skipped_count: int


def alpha_formula_metadata() -> dict[str, AlphaFormulaFamilyMetadata]:
    """Return supported and intentionally skipped Alpha formula metadata."""
    return {
        "alpha101": {
            "supported": tuple(sorted(ALPHA101_SUPPORTED)),
            "supported_count": len(ALPHA101_SUPPORTED),
            "skipped": dict(ALPHA101_SKIPPED),
            "skipped_count": len(ALPHA101_SKIPPED),
        },
        "alpha191": {
            "supported": tuple(sorted(ALPHA191_SUPPORTED)),
            "supported_count": len(ALPHA191_SUPPORTED),
            "skipped": dict(ALPHA191_SKIPPED),
            "skipped_count": len(ALPHA191_SKIPPED),
        },
    }


__all__ = [
    "AlphaFormulaFamilyMetadata",
    "ALPHA101_SKIPPED",
    "ALPHA101_SUPPORTED",
    "ALPHA191_SKIPPED",
    "ALPHA191_SUPPORTED",
    "alpha101",
    "alpha191",
    "alpha_formula_metadata",
]
