import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from tradelearn.ml._lazy_predict import LazyClassifier, LazyRegressor


class Probability(BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array(["yes", "no"])
        return self

    def predict(self, X):
        return np.array(["yes"] * len(X))

    def predict_proba(self, X):
        return np.array([[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]])


class Decision(Probability):
    predict_proba = None

    def decision_function(self, X):
        return np.array([-3.0, -1.0, 1.0, 3.0])


class LabelsOnly(Probability):
    predict_proba = None


class BrokenFit(Probability):
    def fit(self, X, y):
        raise RuntimeError("fit failed")


class BrokenPredict(Probability):
    def predict(self, X):
        raise LookupError("predict failed")


@pytest.fixture
def data():
    X = pd.DataFrame({"x": range(4)})
    return X, np.array(["yes", "yes", "no", "no"])


@pytest.mark.parametrize("model", [Probability, Decision])
def test_auc_continuous_scores_and_class_order(data, model):
    X, y = data
    learner = LazyClassifier(classifiers=[model])
    scores, _ = learner.fit(X, X, y, y)
    assert scores.loc[model.__name__, "ROC AUC"] == 1
    assert learner.score_errors_ == {}


@pytest.mark.parametrize("single", [False, True])
def test_auc_unavailable_is_nan_with_reason(data, single):
    X, y = data
    learner = LazyClassifier(classifiers=[LabelsOnly])
    scores, _ = learner.fit(X, X, y, np.array(["yes"] * 4) if single else y)
    assert np.isnan(scores.loc["LabelsOnly", "ROC AUC"])
    assert learner.score_errors_["LabelsOnly"]["ROC AUC"]["message"]


@pytest.mark.parametrize("kind", [LazyClassifier, LazyRegressor])
@pytest.mark.parametrize("model,stage", [(BrokenFit, "fit"), (BrokenPredict, "predict")])
def test_failures_are_visible_and_refit_resets(data, kind, model, stage):
    X, y = data
    key = "classifiers" if kind is LazyClassifier else "regressors"
    learner = kind(**{key: [model]})
    learner.fit(X, X, y, y)
    assert learner.failures_[model.__name__]["stage"] == stage
    assert learner.failures_[model.__name__]["error_type"]
    setattr(learner, key, [])
    learner.fit(X, X, y, y)
    assert learner.failures_ == learner.score_errors_ == learner.models == {}


def test_custom_scoring_failure_does_not_corrupt_columns(data):
    X, y = data

    def broken(y, prediction):
        raise ArithmeticError("custom metric failed")

    learner = LazyClassifier(classifiers=[Probability], custom_metric=broken)
    scores, _ = learner.fit(X, X, y, y)
    assert np.isnan(scores.loc["Probability", "broken"])
    assert learner.score_errors_["Probability"]["broken"]["error_type"] == "ArithmeticError"


class BrokenProbability(Probability):
    def predict_proba(self, X):
        raise RuntimeError("scores unavailable")


class Nonfinite(Probability):
    def predict_proba(self, X):
        return np.full((len(X), 2), np.nan)


@pytest.mark.parametrize("model", [BrokenProbability, Nonfinite])
def test_invalid_scores_preserve_model_row(data, model):
    X, y = data
    learner = LazyClassifier(classifiers=[model])
    scores, _ = learner.fit(X, X, y, y)
    assert np.isnan(scores.loc[model.__name__, "ROC AUC"])
    assert learner.score_errors_[model.__name__]["ROC AUC"]["message"]


def test_multiclass_auc_is_explicitly_unavailable(data):
    X, y = data

    class Multiclass(Probability):
        def fit(self, X, y):
            self.classes_ = np.array(["yes", "no", "maybe"])
            return self

    learner = LazyClassifier(classifiers=[Multiclass])
    scores, _ = learner.fit(X, X, y, y)
    assert np.isnan(scores.loc["Multiclass", "ROC AUC"])
    assert "binary" in learner.score_errors_["Multiclass"]["ROC AUC"]["message"]


def test_repeated_fit_keeps_configuration_and_clears_score_errors(data):
    X, y = data
    learner = LazyClassifier(classifiers=[Probability])
    learner.fit(X, X, y, np.array(["yes"] * 4))
    learner.fit(X, X, y, y)
    assert learner.score_errors_ == {}
    assert "Probability" in learner.models


class BrokenConstruct(BaseEstimator):
    def __init__(self):
        raise TypeError("constructor failed")


class Regression(BaseEstimator):
    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.arange(len(X), dtype=float)


@pytest.mark.parametrize(
    "kind,key", [(LazyClassifier, "classifiers"), (LazyRegressor, "regressors")]
)
def test_construct_failure_has_stage(data, kind, key):
    X, y = data
    learner = kind(**{key: [BrokenConstruct]})
    scores, _ = learner.fit(X, X, y, y)
    assert scores.empty
    assert learner.failures_["BrokenConstruct"] == {
        "stage": "construct",
        "error_type": "TypeError",
        "message": "constructor failed",
    }


def test_regression_scoring_failure_preserves_predictions(data):
    X, _ = data

    def broken(y, prediction):
        raise ArithmeticError("custom metric failed")

    learner = LazyRegressor(regressors=[Regression], predictions=True, custom_metric=broken)
    scores, predictions = learner.fit(X, X, np.arange(4), np.arange(4))
    assert scores.loc["Regression", "R-Squared"] == 1
    assert np.isnan(scores.loc["Regression", "broken"])
    assert learner.score_errors_["Regression"]["broken"]["error_type"] == "ArithmeticError"
    assert predictions["Regression"].tolist() == list(range(4))


def test_probability_shape_rejected_without_decision_fallback(data):
    X, y = data

    class Invalid(Decision):
        def predict_proba(self, X):
            return np.ones(len(X))

    learner = LazyClassifier(classifiers=[Invalid])
    scores, _ = learner.fit(X, X, y, y)
    assert np.isnan(scores.loc["Invalid", "ROC AUC"])
    assert "shape" in learner.score_errors_["Invalid"]["ROC AUC"]["message"]


@pytest.mark.parametrize(
    "kind,key,model",
    [
        (LazyClassifier, "classifiers", Probability),
        (LazyRegressor, "regressors", Regression),
    ],
)
def test_callable_metric_without_name_records_error(data, kind, key, model):
    X, y = data

    class Metric:
        def __call__(self, y, prediction):
            raise ValueError("metric unavailable")

    learner = kind(**{key: [model]}, custom_metric=Metric())
    target = y if kind is LazyClassifier else np.arange(4)
    scores, _ = learner.fit(X, X, target, target)
    assert np.isnan(scores.loc[model.__name__, "Metric"])
    assert learner.score_errors_[model.__name__]["Metric"]["error_type"] == "ValueError"
