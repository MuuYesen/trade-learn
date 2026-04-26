from __future__ import annotations

import pandas as pd

from tradelearn.ml import FeatureStore, feature


def _bars() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3, tz="UTC"), ["AAA"]],
        names=["timestamp", "symbol"],
    )
    bars = pd.DataFrame(
        {
            "open": [10.0, 11.0, 13.0],
            "high": [11.0, 12.0, 16.0],
            "low": [9.0, 10.0, 12.0],
            "close": [10.0, 12.0, 15.0],
            "volume": [100.0, 120.0, 130.0],
        },
        index=index,
    )
    bars.attrs.update({"engine": "fixture", "freq": "1d", "market": "US"})
    return bars


def test_feature_decorator_sets_name_version_and_kind() -> None:
    @feature(name="momentum", version=2, factor_type="momentum", horizon=5)
    def momentum(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].pct_change()

    assert momentum.feature_name == "momentum"
    assert momentum.feature_version == "2"
    assert momentum.feature_type == "momentum"
    assert momentum.feature_horizon == 5


def test_feature_store_computes_factor_contract_and_metadata(tmp_path) -> None:
    @feature(name="momentum", version=1, factor_type="momentum", horizon=2)
    def momentum(bars: pd.DataFrame) -> pd.Series:
        return bars["close"].pct_change()

    store = FeatureStore(tmp_path)
    store.register(momentum)

    factors = store.compute(_bars(), ["momentum"])

    assert list(factors.columns) == ["momentum"]
    assert factors.index.equals(_bars().index)
    assert factors.attrs["factor_type"] == "momentum"
    assert factors.attrs["horizon"] == 2
    assert factors.attrs["version"] == {"momentum": "1"}
    assert float(factors["momentum"].iloc[-1]) == 0.25
    assert store.exists(_bars(), "momentum")


def test_feature_store_reuses_cached_feature_by_bars_fingerprint(tmp_path) -> None:
    calls = 0

    @feature(name="close_x2", version=1)
    def close_x2(bars: pd.DataFrame) -> pd.Series:
        nonlocal calls
        calls += 1
        return bars["close"] * 2

    store = FeatureStore(tmp_path)
    store.register(close_x2)

    first = store.compute(_bars(), ["close_x2"])
    second = store.compute(_bars(), ["close_x2"])

    assert calls == 1
    pd.testing.assert_frame_equal(first, second)
