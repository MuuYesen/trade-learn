"""Tests ensuring pandas-ta vendor code is removed."""

from pathlib import Path

import pandas as pd
import pandas_ta_classic as pta


def test_legacy_pandas_ta_vendor_tree_is_removed() -> None:
    """The vendored pandas_ta tree should not ship in runtime packages."""
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "tradelearn" / "query" / "tec" / "pandas_ta").exists()


def test_lite_no_longer_exports_legacy_ta_util() -> None:
    """Lite no longer exposes the chain-style indicator helper."""
    import tradelearn.lite as lite

    assert not hasattr(lite, "_TA")


def test_talib_namespace_delegates_to_pandas_ta_classic() -> None:
    """TA-Lib-style namespace should use pandas-ta-classic after vendor removal."""
    import tradelearn as tl

    data = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [0.5, 1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 4.0, 8.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )

    result = tl.talib.SMA(data["close"], timeperiod=2)

    pd.testing.assert_series_equal(result, pta.sma(data["close"], length=2))
