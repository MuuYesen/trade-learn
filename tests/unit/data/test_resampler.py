from __future__ import annotations

import pandas as pd
import pytest

import tradelearn.data.resampler as resampler
from tradelearn.data.resampler import _DAYS, _MINUTES, resample_frame


def _minute_bars() -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:30", periods=7, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
            "volume": [100, 200, 300, 400, 500, 600, 700],
        },
        index=index,
    )


def test_resample_frame_matches_existing_pandas_minute_semantics() -> None:
    bars = _minute_bars()

    result = resample_frame(bars, timeframe=_MINUTES, compression=3)
    expected = (
        bars.resample("3min", label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_resample_frame_falls_back_for_extra_columns() -> None:
    bars = _minute_bars().assign(signal=range(7))

    result = resample_frame(bars, timeframe=_MINUTES, compression=3)

    assert "signal" in result.columns
    assert result["signal"].tolist() == [0, 3, 6]


def test_resample_frame_uses_rust_fast_path_for_plain_ohlcv(monkeypatch) -> None:
    calls: list[int] = []

    def fake_rust_resample_ohlcv(
        timestamps,
        opens,
        highs,
        lows,
        closes,
        volumes,
        period_seconds,
    ):
        calls.append(period_seconds)
        return (
            [pd.Timestamp("2026-01-01 09:30", tz="UTC").value // 1_000_000_000],
            [10.0],
            [11.0],
            [9.0],
            [10.5],
            [100.0],
        )

    monkeypatch.setattr(resampler, "_rust_resample_ohlcv", fake_rust_resample_ohlcv)

    result = resample_frame(_minute_bars(), timeframe=_MINUTES, compression=3)

    assert calls == [180]
    assert result.index[0] == pd.Timestamp("2026-01-01 09:30", tz="UTC")
    assert result.iloc[0].to_dict() == {
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100.0,
    }


def test_resample_frame_rejects_non_datetime_index() -> None:
    bars = _minute_bars().reset_index(drop=True)

    with pytest.raises(TypeError, match="DatetimeIndex"):
        resample_frame(bars, timeframe=_DAYS, compression=1)
