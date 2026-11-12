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


class _Node:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _Endpoint:
    def __init__(self, name):
        self.name = name


class _Edge:
    def __init__(self, left, right, first, second):
        self.left, self.right = left, right
        self.first, self.second = _Endpoint(first), _Endpoint(second)

    def get_node1(self):
        return self.left

    def get_node2(self):
        return self.right

    def get_endpoint1(self):
        return self.first

    def get_endpoint2(self):
        return self.second


class _EdgeGraph:
    def __init__(self, first, second):
        self.nodes = [_Node('feature "A" <tag>'), _Node("target")]
        self.edge = _Edge(*self.nodes, first, second)

    def get_nodes(self):
        return self.nodes

    def get_graph_edges(self):
        return [self.edge]


def test_mermaid_preserves_actual_edge_endpoints_and_safe_node_identifiers():
    cases = [
        ("TAIL", "ARROW", "n0 --> n1"),
        ("ARROW", "TAIL", "n1 --> n0"),
        ("TAIL", "TAIL", "n0 --- n1"),
        ("ARROW", "ARROW", "n0 <--> n1"),
        ("CIRCLE", "CIRCLE", "n0 o--o n1"),
        ("CIRCLE", "ARROW", "n0 ---|CIRCLE / ARROW| n1"),
        ("ARROW", "CIRCLE", "n0 ---|ARROW / CIRCLE| n1"),
        ("TAIL", "CIRCLE", "n0 ---|TAIL / CIRCLE| n1"),
        ("CIRCLE", "TAIL", "n0 ---|CIRCLE / TAIL| n1"),
        ("STAR", "NULL", "n0 ---|STAR / NULL| n1"),
        ("TAIL_AND_ARROW", "ARROW_AND_ARROW", "n0 ---|TAIL_AND_ARROW / ARROW_AND_ARROW| n1"),
    ]
    for first, second, expected in cases:
        selector = CausalSelector(target="target", method="fci")
        selector.graph_ = _EdgeGraph(first, second)
        rendered = selector.to_mermaid()
        assert expected in rendered
        assert 'n0["feature &quot;A&quot; &lt;tag&gt;"]' in rendered


def test_refit_correlation_does_not_keep_an_old_discovery_graph():
    selector = CausalSelector(target="target")
    selector.graph_ = _EdgeGraph("TAIL", "ARROW")
    selector.fit(_dataset())
    assert not hasattr(selector, "graph_")
    assert "No_Data" in selector.to_mermaid()


def test_unknown_graph_endpoint_is_not_silently_rendered_as_causal_arrow():
    import pytest

    selector = CausalSelector(target="target")
    selector.graph_ = _EdgeGraph("UNKNOWN", "ARROW")
    with pytest.raises(ValueError, match="endpoint"):
        selector.to_mermaid()


def test_mermaid_does_not_collapse_multiple_edges_between_the_same_nodes():
    graph = _EdgeGraph("TAIL", "ARROW")
    second = _Edge(*graph.nodes, "ARROW", "ARROW")
    graph.get_graph_edges = lambda: [graph.edge, second]
    selector = CausalSelector(target="target")
    selector.graph_ = graph
    rendered = selector.to_mermaid()
    assert "n0 --> n1" in rendered and "n0 <--> n1" in rendered


def test_existing_pipeline_and_selector_use_explicit_train_label_only():
    import pytest

    from tradelearn.research import FeatureSet, Pipeline
    from tradelearn.research.preprocess import StandardScaler
    from tradelearn.research.temporal import forward_return, purged_splits

    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10, tz="UTC"), ["SYNTH"]],
        names=["timestamp", "symbol"],
    )
    bars = pd.DataFrame(
        {name: range(10, 20) for name in ("open", "high", "low", "close", "volume")}, index=index
    )
    labels = forward_return(bars, horizon=1)
    builder = FeatureSet(
        {"price": lambda panel: panel.close}, target={"label": labels.forward_return}
    )
    seen = {}

    def backend(X, y):
        seen["X"], seen["y"] = X.copy(), y.copy()
        return {"price": 1.0}

    selector = CausalSelector(target="label", features=["price"], backend=backend)
    with pytest.raises(ValueError, match="target column"):
        selector.fit(builder.fit_transform(bars))
    dataset = builder.fit_transform(bars, include_target=True)
    split = purged_splits(labels, train_end="2024-01-07", validation_end="2024-01-09")
    train = dataset.loc[split["train"]]
    pipeline = Pipeline([StandardScaler(columns=["price"])])
    X_train = pipeline.fit_transform(train[["price"]])
    selector.fit(X_train.join(train[["label"]]))
    validation = dataset.loc[split["validation"], ["price"]]
    output = selector.transform(pipeline.transform(validation))
    assert seen["X"].index.equals(split["train"])
    assert (labels.loc[seen["y"].index, "label_end"] < pd.Timestamp("2024-01-07", tz="UTC")).all()
    assert output.index.equals(validation.index) and list(output) == ["price"]
    assert "label" not in output and not seen["X"].index.intersection(validation.index).size


def test_correlation_and_report_section_do_not_import_causal_learn():
    import subprocess
    import sys

    code = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] == 'causallearn': raise AssertionError('unexpected discovery import')
sys.meta_path.insert(0, Block())
import pandas as pd
from tradelearn.ml import CausalSelector
selector = CausalSelector(target='label').fit(pd.DataFrame({'x':[1,2,3], 'label':[2,4,6]}))
assert 'No_Data' in selector.to_section()['html']
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_causal_metadata_records_parameters_and_selected_state():
    selector = CausalSelector(target="label", features=["x"], method="correlation")
    assert selector.get_params()["target"] == "label"
    selector.fit(pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": [2.0, 4.0, 6.0]}))
    artifact = selector.to_dict()
    assert artifact["selected_features"] == ["x"]
    assert artifact["input_columns"] == ["x"]
    assert artifact["method"] == "correlation"
