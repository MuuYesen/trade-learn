import json

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from tradelearn.research import ResearchRun


def test_external_model_registration_records_only_real_calls():
    model = LinearRegression()
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with ResearchRun("external") as run:
        assert run.register("model", model) is model
        assert run.steps == []
        run.call("model", "fit", frame, pd.Series([2.0, 4.0, 6.0]))
        scores = run.call("model", "predict", frame)
        result = run.finish(scores=scores, model=model)
    payload = result.to_dict()
    assert payload["artifacts"]["components"]["model"]["type"] == "LinearRegression"
    assert [step.params["method"] for step in result.steps] == ["fit", "predict"]
    assert result.steps[0].params["inputs"]["args"][0]["rows"] == 3
    assert result.steps[0].params["status"] == "completed"
    json.dumps(payload)


def test_failed_call_recorded_and_registration_conflict_rejected():
    with ResearchRun("failure") as run:
        run.register("model", LinearRegression())
        with pytest.raises(ValueError, match="registered"):
            run.register("model", LinearRegression())
        with pytest.raises(NotFittedError):
            run.call("model", "predict", pd.DataFrame({"x": [1.0]}))
    assert run.steps[-1].params["status"] == "failed"
    assert run.steps[-1].params["error"] == "NotFittedError"


def test_registered_pipeline_retains_each_step_columns():
    from tradelearn.research import Pipeline, preprocess

    pipeline = Pipeline(
        [preprocess.StandardScaler(columns=["x"]), preprocess.StandardScaler(columns=["y"])]
    )
    with ResearchRun("pipeline") as run:
        run.register("pipeline", pipeline)
        out = run.call(
            "pipeline", "fit_transform", pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]})
        )
        result = run.finish(features=out)
    steps = result.artifacts["components"]["pipeline"]["params"]["steps"]
    assert [step["columns"] for step in steps] == [["x"], ["y"]]
