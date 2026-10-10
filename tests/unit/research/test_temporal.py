import pandas as pd
import pytest

from tradelearn.research.temporal import assessment, forward_return, purged_splits
from tradelearn.data.bars import normalize_bars


def _bars() -> pd.DataFrame:
    times = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:03Z"])
    index = pd.MultiIndex.from_product([times, ["A", "B"]], names=["timestamp", "symbol"])
    return pd.DataFrame({"close": [100, 200, 101, 201, 103, 203]}, index=index)


def test_seconds_require_exact_future_timestamp():
    labels = forward_return(_bars(), horizon=1, unit="seconds")
    assert labels.loc[(pd.Timestamp("2026-01-01T00:00:00Z"), "A"), "forward_return"] == pytest.approx(.01)
    assert pd.isna(labels.loc[(pd.Timestamp("2026-01-01T00:00:01Z"), "A"), "forward_return"])


def test_normalized_bars_accept_one_second_frequency():
    raw = _bars().reset_index()
    raw['open'] = raw.close
    raw['high'] = raw.close + 1
    raw['low'] = raw.close - 1
    raw['volume'] = 1
    bars = normalize_bars(raw, market='CRYPTO', freq='1s', engine='fixture', source='test', adjust='none')
    assert bars.attrs['freq'] == '1s'


def test_observation_horizon_does_not_mean_elapsed_seconds():
    labels = forward_return(_bars(), horizon=1)
    assert labels.loc[(pd.Timestamp("2026-01-01T00:00:01Z"), "A"), "label_end"] == pd.Timestamp("2026-01-01T00:00:03Z")


def test_single_asset_report_is_explicitly_temporal():
    bars = _bars().xs("A", level="symbol", drop_level=False)
    labels = forward_return(bars, horizon=1)
    factors = bars.rename(columns={"close": "momentum"})
    result = assessment(factors, labels, factors=["momentum"], min_pairs=3)
    assert result["mode"] == "time_series"
    assert result["factors"]["momentum"]["A"]["reason"] == "insufficient_pairs"


def test_purge_removes_labels_crossing_split_boundaries():
    bars = _bars().xs("A", level="symbol", drop_level=False)
    labels = forward_return(bars, horizon=1)
    parts = purged_splits(labels, train_end="2026-01-01T00:00:01Z", validation_end="2026-01-01T00:00:03Z")
    assert len(parts["train"]) == 0
    assert len(parts["validation"]) == 0
    assert len(parts["test"]) == 0


def test_purge_keeps_only_labels_ending_within_section():
    times = pd.date_range('2026-01-01', periods=6, freq='s', tz='UTC')
    index = pd.MultiIndex.from_product([times, ['A']], names=['timestamp', 'symbol'])
    labels = forward_return(pd.DataFrame({'close': [1, 2, 3, 4, 5, 6]}, index=index), horizon=1, unit='seconds')
    parts = purged_splits(labels, train_end=times[2], validation_end=times[4])
    assert [part.get_level_values('timestamp').tolist() for part in parts.values()] == [[times[0]], [times[2]], [times[4]]]
