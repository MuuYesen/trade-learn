from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.lite import Backtest, Strategy
import tradelearn.lite as tl


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 140.0],
        },
        index=pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"),
    )


def test_lite_facade_accepts_lite_strategy_surface() -> None:
    storage: dict[str, object] = {}
    seen: dict[str, object] = {}

    class SurfaceStrategy(Strategy):
        threshold = 12

        def init(self) -> None:
            seen["init_len"] = len(self.data.close)
            seen["df_columns"] = list(self.data.df.columns)
            seen["index"] = self.data.index
            seen["ticker"] = self.data.the_ticker
            self.start_on_day(2)
            self.direct = self.data.close - 9
            self.callable = tl.talib.SMA(self.data.close, timeperiod=2)

        def next(self) -> None:
            seen["first_next_len"] = seen.setdefault("first_next_len", len(self.data.close))
            self.storage["last_close"] = float(self.data.close[0])
            self.record(signal={"direct": self.direct[0], "threshold": self.threshold})
            if len(self.data.close) == 3:
                assert self.data.close[0] == 12.0
                assert self.data.close[-1] == 11.0
                assert len(self.data.close.to_series())==len(self.data.index)
                assert self.direct[0] == 3
                assert self.position().size == 0
                assert self.position().is_long is False
                self.buy(size=1, tag="entry")
            elif len(self.data.close) == 4:
                assert self.equity > 0
                assert isinstance(self.orders, tuple)
                self.position().close()

    stats = Backtest(_data(), SurfaceStrategy, cash=1000.0, storage=storage).run(threshold=13)

    assert stats["final_value"] > 0
    assert seen["init_len"] == 0
    assert seen["first_next_len"] == 3
    assert seen["df_columns"] == ["open", "high", "low", "close", "volume"]
    assert seen["ticker"] == "Asset"
    assert storage["last_close"] == 14.0


def test_lite_facade_rejects_removed_capitalized_data_aliases() -> None:
    class CapitalizedStrategy(Strategy):
        def init(self) -> None:
            _ = self.data.Close

        def next(self) -> None:
            pass

    with pytest.raises(AttributeError, match="use 'close' instead"):
        Backtest(_data(), CapitalizedStrategy).run()


def test_lite_facade_rejects_unknown_strategy_params() -> None:
    class ParamStrategy(Strategy):
        known = 1

        def init(self) -> None:
            pass

        def next(self) -> None:
            pass

    with pytest.raises(AttributeError, match="missing parameter 'unknown'"):
        Backtest(_data(), ParamStrategy).run(unknown=2)


def test_lite_facade_retired_I_not_exposed() -> None:
    assert not hasattr(Strategy, "I")
