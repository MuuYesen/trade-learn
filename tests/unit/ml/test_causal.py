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


def test_causal_selector_selects_top_features_by_target_association() -> None:
    selector = CausalSelector(max_features=2)
    y = pd.Series([0.0, 1.1, 1.9, 3.1, 4.0], index=_features().index)

    selector.fit(_features(), y)

    assert selector.selected_features_ == ["strong", "noise"]
    assert selector.scores_["strong"] > selector.scores_["weak"]


def test_causal_selector_transforms_dataframe_to_selected_columns() -> None:
    selector = CausalSelector(max_features=1)
    y = pd.Series([0.0, 1.1, 1.9, 3.1, 4.0], index=_features().index)

    selected = selector.fit_transform(_features(), y)

    assert list(selected.columns) == ["strong"]
    pd.testing.assert_index_equal(selected.index, _features().index)


def test_causal_selector_accepts_custom_backend_scores() -> None:
    def backend(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        return {"weak": 10.0, "strong": 1.0, "noise": 0.0}

    selector = CausalSelector(max_features=1, backend=backend)

    selector.fit(_features(), pd.Series([0.0, 1.0, 2.0, 3.0, 4.0]))

    assert selector.selected_features_ == ["weak"]
