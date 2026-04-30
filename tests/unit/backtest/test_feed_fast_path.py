from __future__ import annotations

import numpy as np
import pandas as pd

from tradelearn.backtest.feed import RuntimeDataFeed, build_data_feeds, is_normalized_ohlcv_frame


def _normalized_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": np.array([10.0, 11.0], dtype=np.float64),
            "high": np.array([11.0, 12.0], dtype=np.float64),
            "low": np.array([9.0, 10.0], dtype=np.float64),
            "close": np.array([10.5, 11.5], dtype=np.float64),
            "volume": np.array([100.0, 120.0], dtype=np.float64),
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
    )


def test_normalized_ohlcv_detector_accepts_fast_path_frames() -> None:
    assert is_normalized_ohlcv_frame(_normalized_frame())


def test_normalized_ohlcv_detector_rejects_frames_requiring_normalize() -> None:
    frame = _normalized_frame().rename(columns={"open": "Open"})

    assert not is_normalized_ohlcv_frame(frame)


def test_runtime_data_feed_can_use_internal_no_copy_fast_path() -> None:
    frame = _normalized_frame()

    feed = RuntimeDataFeed(frame, assume_normalized=True, copy=False)

    assert feed._frame is frame
    assert np.shares_memory(feed.get_array("close"), frame["close"].to_numpy())


def test_build_data_feeds_preserves_default_copy_semantics_on_fast_path() -> None:
    frame = _normalized_frame()

    feed = build_data_feeds({"AAA": frame}, assume_normalized=True)[0]
    frame.loc[frame.index[0], "close"] = 99.0

    assert feed._frame is not frame
    assert feed.get_array("close")[0] == 10.5
