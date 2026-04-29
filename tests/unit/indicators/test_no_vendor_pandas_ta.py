"""Tests ensuring pandas-ta vendor code is removed."""

from pathlib import Path

import pandas as pd
import pandas_ta_classic as pta


def test_legacy_pandas_ta_vendor_tree_is_removed() -> None:
    """The old vendored pandas_ta tree should not ship in tradelearn.query."""
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "tradelearn" / "query" / "tec" / "pandas_ta").exists()


def test_backtesting_util_imports_without_vendor_pandas_ta() -> None:
    """Backtesting compatibility code should not import the removed vendor tree."""
    from tradelearn.lite import util

    assert util._TA is not None


def test_backtesting_ta_accessor_delegates_to_pandas_ta_classic() -> None:
    """Backtesting data.ta methods should use pandas-ta-classic after vendor removal."""
    from tradelearn.lite.util import _TA

    data = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 4.0, 8.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )

    result = _TA(data).roc(2)

    pd.testing.assert_series_equal(result, pta.roc(data["close"], length=2))
