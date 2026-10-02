"""Optimized factor windows preserve finite observations and rank inputs."""

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("name", ["alpha101", "alpha191"])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_window_is_missing_then_recovers(name, jit, invalid, monkeypatch):
    module = importlib.import_module(f"tradelearn.factor.alpha.{name}")
    if not jit:
        monkeypatch.setattr(module, "njit", None)
    frame = pd.DataFrame({"asset": [1.0, invalid, 3.0, 4.0, 5.0]})
    for fn in (module._decay_linear, module._ts_rank):
        result = fn(frame, 3)
        assert result.iloc[:4].isna().all().all()
        assert np.isfinite(result.iloc[4]).all()


@pytest.mark.parametrize("name", ["alpha101", "alpha191"])
@pytest.mark.parametrize("jit", [False, True])
def test_decay_preserves_numpy_reduction_before_discontinuous_rank(name, jit, monkeypatch):
    module = importlib.import_module(f"tradelearn.factor.alpha.{name}")
    if not jit:
        monkeypatch.setattr(module, "njit", None)
    # Cancellation makes reduction order observable. Approximate equality here
    # is insufficient: the result feeds rank, which can amplify one ULP.
    weights = np.arange(1, 9, dtype=float)
    products = np.array([1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0])
    frame = pd.DataFrame({"asset": products / weights})
    expected = np.sum(frame["asset"].to_numpy() * weights) / np.sum(weights)
    assert module._decay_linear(frame, 8).iloc[-1, 0] == expected


def test_weighted_window_longer_than_input_stays_missing():
    for name in ("alpha101", "alpha191"):
        module = importlib.import_module(f"tradelearn.factor.alpha.{name}")
        assert module._decay_linear(pd.DataFrame({"asset": [1.0, 2.0]}), 3).isna().all().all()
