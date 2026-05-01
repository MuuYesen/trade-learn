"""Tests for removed legacy query import paths."""

import importlib.util
from pathlib import Path


def test_legacy_query_package_is_removed() -> None:
    """The old query facade should not ship as a user API."""
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "tradelearn" / "query").exists()
    assert importlib.util.find_spec("tradelearn.query") is None
