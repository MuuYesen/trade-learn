"""Structural tests for removed legacy strategy facades."""

import importlib.util
from pathlib import Path


def test_legacy_strategy_package_is_removed() -> None:
    """The old strategy namespace should not ship as a user API."""
    root = Path("tradelearn/strategy")

    assert not root.exists()
    assert importlib.util.find_spec("tradelearn.strategy") is None
