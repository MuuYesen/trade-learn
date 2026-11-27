"""Checks for choosing stable regions on a preregistered parameter grid."""

import pandas as pd
import pytest

from tradelearn.research import parameter_surface


def test_disconnected_regions_are_reported_without_merging():
    scores = pd.Series([2., 2., 0., 0., 0., 2., 2.], index=pd.Index(range(7), name="period"))
    result = parameter_surface.stable_region(scores, {"period": range(7)}, standard_error=0.01)
    assert result.smoothed.loc[1] == pytest.approx(4 / 3)
    assert result.components == (((0,),), ((6,),))
    assert result.boundary_cells == ((0,), (6,))
    assert result.coverage == pytest.approx(2 / 7)


def test_surface_reports_unusable_and_absent_grid_cells():
    axes = {"fast": [5, 10, 15], "slow": [20, 25, 30]}
    index = pd.MultiIndex.from_tuples([(5, 20), (5, 25), (10, 20)], names=axes)
    scores = pd.Series([1.0, 2.0, float("nan")], index=index)
    result = parameter_surface.stable_region(scores, axes, standard_error=0.0)
    assert result.smoothed.to_list() == [1.5, 1.5]
    assert result.components == (((5, 20), (5, 25)),)
    assert result.unavailable_cells == 7


def test_surface_validates_axis_labels_and_threshold():
    scores = pd.Series([1., 2.], index=pd.Index([1, 2], name="period"))
    with pytest.raises(ValueError, match="outside"):
        parameter_surface.stable_region(scores, {"period": [3, 4]}, standard_error=0.1)
    with pytest.raises(ValueError, match="standard_error"):
        parameter_surface.stable_region(scores, {"period": [1, 2]}, standard_error=-0.1)
