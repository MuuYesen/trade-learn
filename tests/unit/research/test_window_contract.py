from types import SimpleNamespace

import pandas as pd
import pytest

from tradelearn.backtest import history
from tradelearn.research import FeatureSet


def feed(name="A", cursor=3):
    index = pd.date_range("2026-01-01", periods=6, tz="UTC", name="timestamp")
    return SimpleNamespace(
        _name=name,
        _cursor=cursor,
        _frame=pd.DataFrame(
            {field: range(1, 7) for field in history.DEFAULT_HISTORY_FIELDS}, index=index
        ),
    )


def test_single_snapshot_identity_ago_and_feature_set():
    source = feed()
    result = history.window(source, size=2, ago=-1)
    assert result.index.equals(source._frame.index[1:3])
    assert result.attrs["symbol"] == "A"
    features = FeatureSet({"price": lambda p: p.close}).transform(result)
    assert features.index.get_level_values("symbol").tolist() == ["A", "A"]
    result.iloc[0, 0] = -100
    assert source._frame.iloc[1, 0] == 2


def test_multifeed_cutoff_precedes_size_and_ago():
    a, b = feed(cursor=2), feed("B", cursor=4)
    owner = SimpleNamespace(data=a, datas=[a, b], current_datetime=lambda: a._frame.index[2])
    result = history.window(owner, data=owner.datas, size=2, ago=-1)
    assert result.index.get_level_values("timestamp").max() == a._frame.index[1]
    assert result.groupby(level="symbol").size().to_dict() == {"A": 2, "B": 2}


@pytest.mark.parametrize(
    "kwargs", [{"size": 0}, {"size": True}, {"size": 1.5}, {"size": 2, "ago": 1}]
)
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        history.window(feed(), **kwargs)


def test_missing_field_and_missing_identity_are_rejected():
    with pytest.raises(ValueError, match="field"):
        history.window(feed(), size=2, fields=["absent"])
    frame = feed()._frame
    with pytest.raises(ValueError, match="symbol"):
        FeatureSet({"price": lambda p: p.close}).transform(frame)


def test_unstarted_feed_has_no_future_rows():
    assert history.window(feed(cursor=-1), size=2).empty


def test_single_window_preserves_source_timezone_features_normalize_utc():
    source = feed()
    source._frame.index = source._frame.index.tz_convert("Asia/Tokyo")
    snapshot = history.window(source, size=2)
    assert str(snapshot.index.tz) == "Asia/Tokyo"
    features = FeatureSet({"price": lambda p: p.close}).transform(snapshot)
    assert str(features.index.get_level_values("timestamp").tz) == "UTC"


def test_feature_set_never_drops_an_extra_identity_level():
    from tradelearn.core.errors import ContractError

    frame = feed()._frame.copy()
    frame.index = pd.MultiIndex.from_product(
        [frame.index, ["A"], ["revision1"]], names=["timestamp", "symbol", "revision"]
    )
    with pytest.raises(ContractError, match="two"):
        FeatureSet({"price": lambda p: p.close}).transform(frame)


def test_window_metadata_snapshot_is_independent():
    source = feed()
    source._frame.attrs["source"] = {"timezone": "UTC"}
    snapshot = history.window(source, size=2)
    snapshot.attrs["source"]["timezone"] = "changed"
    assert source._frame.attrs["source"]["timezone"] == "UTC"


def test_internal_decision_clock_precedes_public_fallback():
    a, b = feed(cursor=4), feed("B", cursor=4)
    owner = SimpleNamespace(
        data=a,
        datas=[a, b],
        _decision_datetime=lambda: a._frame.index[2],
        current_datetime=lambda: a._frame.index[4],
    )
    result = history.window(owner, data=owner.datas, size=2)
    assert result.index.get_level_values("timestamp").max() == a._frame.index[2]
