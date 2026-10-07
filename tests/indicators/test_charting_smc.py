import json

import numpy as np
import pandas as pd
import pytest

from tradelearn.indicators.charting import calculate, catalog
from tradelearn.indicators.smc import main


def bars(count=160):
    rng = np.random.default_rng(81)
    close = 100 + rng.normal(size=count).cumsum()
    opening = close + rng.normal(scale=0.5, size=count)
    return pd.DataFrame(
        dict(
            open=opening,
            high=np.maximum(opening, close) + 0.5,
            low=np.minimum(opening, close) - 0.5,
            close=close,
            volume=np.ones(count),
        ),
        index=pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC"),
    )


def test_smc_catalog_and_legacy_geometry():
    frame = bars()
    entry = next(x for x in catalog() if x["name"] == "Smart Money Concepts")
    assert entry["overlay"] and entry["plots"] == []
    result = calculate(frame, entry["id"], {"depth": 3})
    json.dumps(result, allow_nan=False)
    assert result["series"] == []
    drawings = result["drawings"]
    legacy = main(frame.rename(columns=str.title).reset_index(drop=True), 3)
    def ts(i):
        return int(frame.index[int(i)].timestamp())
    assert drawings["levels"] == [
        dict(
            time=ts(i),
            price=float(price),
            confirmedAt=ts(legacy.confirmations[int(i)])
            if int(i) in legacy.confirmations
            else None,
            provisional=int(i) not in legacy.confirmations,
        )
        for i, price in legacy.lev
    ]
    for key, attr in [("bosLL", "bos_ll"), ("bosHH", "bos_hh")]:
        assert drawings[key] == [
            dict(timeA=ts(a), price=float(p), timeB=ts(b)) for a, p, b, _ in getattr(legacy, attr)
        ]
    for key, attr, fields in [
        ("zoneD", "demand_zone_values", ("priceA", "priceB")),
        ("zoneS", "supply_zone_values", ("priceA", "priceB")),
        ("impHH", "imp_hh", ("valueA", "valueB")),
        ("impLL", "imp_ll", ("valueA", "valueB")),
    ]:
        assert drawings[key] == [
            dict(timeA=ts(a), timeB=ts(b), **{fields[0]: float(p), fields[1]: float(q)})
            for a, p, b, q in getattr(legacy, attr)
        ]


def test_smc_prefix_isolated_from_future_and_confirmed_pivots_stable():
    frame = bars()
    before = calculate(frame.iloc[:100], "Smart Money Concepts", {"depth": 3})
    changed = frame.copy()
    changed.iloc[100:, :4] *= 10
    assert calculate(changed.iloc[:100], "Smart Money Concepts", {"depth": 3}) == before
    end = int(frame.index[99].timestamp())
    full = calculate(frame, "Smart Money Concepts", {"depth": 3})["drawings"]
    def confirmed(ds):
        return [
            x for x in ds["levels"] if x["confirmedAt"] is not None and x["confirmedAt"] <= end
        ]
    assert confirmed(before["drawings"]) == confirmed(full)
    for drawings in before["drawings"].values():
        for drawing in drawings:
            for field in ("time", "timeA", "timeB", "confirmedAt"):
                if drawing.get(field) is not None:
                    assert drawing[field] <= end


def test_smc_bounds_empty_and_depth():
    assert all(
        not values for values in calculate(bars(0), "Smart Money Concepts")["drawings"].values()
    )
    assert calculate(bars(3), "Smart Money Concepts")["drawings"]["levels"] == []
    with pytest.raises(ValueError, match="5000"):
        calculate(bars(5001), "Smart Money Concepts")
    for depth in (0, 1001, 1.5, True):
        with pytest.raises(ValueError):
            calculate(bars(), "Smart Money Concepts", {"depth": depth})
    assert calculate(bars(), "Smart Money Concepts", {"depth": 2}) != calculate(
        bars(), "Smart Money Concepts", {"depth": 8}
    )
