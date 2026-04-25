"""Structural tests for legacy report vendor cleanup."""

from pathlib import Path


def test_legacy_report_vendor_trees_are_removed() -> None:
    """Stage 2 report/factor facades must not ship vendored report engines."""
    root = Path("tradelearn/strategy")

    assert not (root / "evaluate" / "empyrical").exists()
    assert not (root / "evaluate" / "pyfolio").exists()
    assert not (root / "examine" / "alphalens").exists()


def test_legacy_report_facades_remain_importable() -> None:
    """Compatibility facades should import without vendored pyfolio/alphalens."""
    from tradelearn.strategy.evaluate import Evaluate
    from tradelearn.strategy.examine import Examine

    assert Evaluate.__name__ == "Evaluate"
    assert Examine.__name__ == "Examine"
