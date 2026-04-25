"""Tests for removing yfinance from default import paths."""

import sys


def test_legacy_query_import_does_not_require_yfinance() -> None:
    """The legacy Query module must import without yfinance installed."""
    sys.modules.pop("yfinance", None)

    from tradelearn.query import Query

    assert Query.__name__ == "Query"
