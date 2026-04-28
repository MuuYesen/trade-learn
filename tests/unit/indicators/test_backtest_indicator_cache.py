from __future__ import annotations

import pandas as pd

from tradelearn import ta
from tradelearn.backtest.core.indicator_cache import (
    BatchIndicatorCache,
    IndicatorCache,
    RollingIndicatorCache,
)


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


def test_batch_indicator_cache_splits_multi_output_dataframe_lines() -> None:
    frame = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=pd.date_range("2026-01-01", periods=5, tz="UTC"),
    )

    cache = BatchIndicatorCache(frame)
    lines = cache.precompute_many(
        "bands",
        lambda close: pd.DataFrame({"mid": close.rolling(2).mean()}),
        "close",
    )
    cache.advance(4)

    assert lines["mid"][0] == 4.5


def test_rolling_indicator_cache_recomputes_recent_window_without_future_data() -> None:
    cache = RollingIndicatorCache(window=3)
    line = cache.register("core.sma", ta.sma, "close", period=3)

    for i, close in enumerate([1.0, 2.0, 3.0, 4.0]):
        cache.append_bar(
            {
                "datetime": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1.0,
            }
        )

    assert line[0] == 3.0
    assert line[-1] == 2.0
