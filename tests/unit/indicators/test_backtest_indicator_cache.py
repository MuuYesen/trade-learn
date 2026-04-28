from __future__ import annotations

import pandas as pd

from tradelearn import ta
from tradelearn.backtest.core.indicator_cache import IndicatorCache


def test_indicator_cache_precomputes_core_tdx_and_tv_namespaces() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [2.0, 3.0, 4.0, 5.0, 6.0],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        },
        index=pd.date_range("2026-01-01", periods=5, tz="UTC"),
    )

    cache = IndicatorCache(frame)
    core_sma = cache.precompute("core.sma", ta.sma, "close", period=3)
    tdx_ma = cache.precompute("tdx.ma", ta.tdx.ma, "close", n=3)
    tv_sma = cache.precompute("tv.sma", ta.tv.sma, "close", length=3)

    cache.advance(4)

    assert core_sma[0] == 4.0
    assert core_sma[-1] == 3.0
    assert tdx_ma[0] == 4.0
    assert tv_sma[0] == 4.0
    assert cache.get("core.sma", period=3) is core_sma
