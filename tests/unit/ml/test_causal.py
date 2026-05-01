from __future__ import annotations

import pandas as pd

from tradelearn.ml import CausalSelector


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strong": [0.0, 1.0, 2.0, 3.0, 4.0],
            "weak": [1.0, 1.0, 2.0, 1.0, 2.0],
            "noise": [5.0, 3.0, 4.0, 2.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
    )


def _dataset() -> pd.DataFrame:
    data = _features().copy()
    data["target"] = [0.0, 1.1, 1.9, 3.1, 4.0]
    return data


def test_causal_selector_selects_top_features_by_target_association() -> None:
    selector = CausalSelector(target="target", max_features=2)

    selector.fit(_dataset())

    assert selector.selected_features_ == ["strong", "noise"]
    assert selector.scores_["strong"] > selector.scores_["weak"]


def test_causal_selector_transforms_dataframe_to_selected_columns() -> None:
    selector = CausalSelector(target="target", max_features=1)

    selected = selector.fit_transform(_dataset())

    assert list(selected.columns) == ["strong"]
    pd.testing.assert_index_equal(selected.index, _features().index)


def test_causal_selector_accepts_custom_backend_scores() -> None:
    def backend(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        return {"weak": 10.0, "strong": 1.0, "noise": 0.0}

    selector = CausalSelector(target="target", max_features=1, backend=backend)

    selector.fit(_dataset())

    assert selector.selected_features_ == ["weak"]


def test_causal_selector_pc_uses_causal_learn_runner() -> None:
    """PC mode scores features adjacent to the target in a causal-learn graph."""

    selector = CausalSelector(target="target", method="pc", pc_runner=_fake_pc_runner)

    selector.fit(_dataset())

    assert selector.selected_features_ == ["strong"]
    assert selector.scores_ == {"strong": 1.0, "weak": 0.0, "noise": 0.0}


def test_causal_selector_fci_uses_causal_learn_runner() -> None:
    """FCI mode scores features adjacent to the target in a causal-learn graph."""

    selector = CausalSelector(target="target", method="fci", fci_runner=_fake_fci_runner)

    selector.fit(_dataset())

    assert selector.selected_features_ == ["weak"]
    assert selector.scores_ == {"strong": 0.0, "weak": 1.0, "noise": 0.0}


def test_causal_selector_can_limit_feature_columns() -> None:
    selector = CausalSelector(target="target", features=("weak", "strong"), max_features=1)

    selector.fit(_dataset())

    assert selector.feature_names_ == ["weak", "strong"]
    assert selector.selected_features_ == ["strong"]


class _FakeCausalGraph:
    def __init__(self, adjacent_to_target: list[int], size: int = 4) -> None:
        self.graph = [[0 for _ in range(size)] for _ in range(size)]
        target_idx = size - 1
        for idx in adjacent_to_target:
            self.graph[idx][target_idx] = 1
            self.graph[target_idx][idx] = 1


class _FakePCResult:
    def __init__(self, adjacent_to_target: list[int]) -> None:
        self.G = _FakeCausalGraph(adjacent_to_target)


def _fake_pc_runner(data, **kwargs):
    assert kwargs["node_names"] == ["strong", "weak", "noise", "__target__"]
    assert kwargs["indep_test"] == "fisherz"
    return _FakePCResult([0])


def _fake_fci_runner(data, **kwargs):
    assert kwargs["node_names"] == ["strong", "weak", "noise", "__target__"]
    assert kwargs["independence_test_method"] == "fisherz"
    return _FakeCausalGraph([1]), []
