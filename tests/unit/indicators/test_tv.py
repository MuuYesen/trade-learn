"""Real PyneCore 6.13.3 source formulas and public discovery."""

import numpy as np
import pandas as pd

from tradelearn.indicators import tv
from tradelearn.indicators.base import FunctionIndicator


def test_source_sma_and_macd_keep_seed_and_output_order():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    np.testing.assert_allclose(
        tv.sma(close, length=3), [np.nan, np.nan, 11, 12, 13, 14], equal_nan=True
    )
    result = tv.macd(close, fastlen=2, slowlen=3, siglen=2)
    assert list(result) == ["macd", "signal", "hist"]
    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[3:], [[0.5, 0.5, 0.0]] * 3)


def test_full_fixed_upstream_is_inventoried_beyond_old_41():
    assert len(tv.QUALIFIED) == 66
    assert tv.BACKEND == "pynesys-pynecore==6.13.3"
    for name in tv.QUALIFIED:
        function = getattr(tv, name)
        assert isinstance(function, FunctionIndicator) and function.supports_streaming
        assert function is getattr(tv, name.upper())
    for name in ["rci", "ichimoku", "adx", "bbands"]:
        assert not hasattr(tv, name)


def test_ohlcv_uses_declared_minimal_schema():
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    high, low = close + 1, close - 1
    assert len(tv.atr(high, low, close, length=2)) == 5
    assert len(tv.stoch(high, low, close, length=2)) == 5
    assert tv.vwma(close, pd.Series([1.0] * 5), length=2).iloc[-1] == 13.5
