"""Only qualified persistent functions are discovered in vendor facades."""

import pytest

from tradelearn.indicators.base import FunctionIndicator


def test_talib_public_exports_are_persistent_and_no_shifted_names():
    from tradelearn.indicators import talib

    assert len(talib.QUALIFIED) > 5
    for name in talib.__all__:
        value = getattr(talib, name)
        if isinstance(value, FunctionIndicator):
            assert value.supports_streaming
    for name in ["DPO", "FRACTAL", "ICHIMOKU", "ZIGZAG"]:
        assert not hasattr(talib, name)


def test_no_pta_namespace_or_transitive_import():
    from tradelearn import indicators

    assert "pta" not in indicators.__all__
    with pytest.raises(AttributeError):
        _ = indicators.pta
    # Other test modules may deliberately import the retired upstream package.
    # Real import isolation is verified in the separate subprocess test.
