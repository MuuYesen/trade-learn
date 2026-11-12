from types import SimpleNamespace

import pandas as pd

from tradelearn.research import ModelScorer, ResearchResult
from tradelearn.research.run import bind_research_result


def test_empty_prediction_never_calls_model_or_reuses_stale_rows():
    class Model:
        def predict(self, frame):
            raise AssertionError("must not predict without current valid inputs")

    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], tz="UTC"), []], names=["timestamp", "symbol"]
    )
    assert ModelScorer(Model(), features=["x"]).predict(pd.DataFrame({"x": []}, index=index)).empty
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=2, tz="UTC"), ["A"]], names=["timestamp", "symbol"]
    )
    assert (
        ModelScorer(Model(), features=["x"])
        .predict(pd.DataFrame({"x": [1.0, float("nan")]}, index=index))
        .empty
    )


def test_weight_view_does_not_read_later_same_day_sample():
    times = pd.to_datetime(["2026-01-01T09:00Z", "2026-01-01T15:00Z"])
    weights = pd.Series(
        [0.3, 0.8], index=pd.MultiIndex.from_product([times, ["A"]], names=["timestamp", "symbol"])
    )
    strategy = SimpleNamespace(data=SimpleNamespace(now=pd.Timestamp("2026-01-01T10:00Z")))
    bound = bind_research_result(ResearchResult("pit", weights=weights), strategy)
    assert bound.weights[0].to_dict() == {"A": 0.3}
    strategy.data.now = pd.Timestamp("2026-01-01T08:00Z")
    assert not bound.weights.has_current()
    assert bound.weights[0].empty


def test_research_weights_prefer_internal_clock_over_bt_ordinals():
    times = pd.to_datetime(["2026-01-01T09:00Z", "2026-01-01T15:00Z"])
    weights = pd.Series(
        [0.3, 0.8], index=pd.MultiIndex.from_product([times, ["A"]], names=["timestamp", "symbol"])
    )
    strategy = SimpleNamespace(
        _decision_datetime=lambda: times[0],
        current_datetime=lambda: times[1],
        data=SimpleNamespace(datetime=[739617.0]),
    )
    result = bind_research_result(ResearchResult("clock", weights=weights), strategy)
    assert result.weights[0].to_dict() == {"A": 0.3}
