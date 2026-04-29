from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.compat.backtesting import Backtest, Strategy


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


def test_backtesting_facade_accepts_tradelearn_1x_strategy_surface() -> None:
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
            self.direct = self.I(
                pd.Series([1, 2, 3, 4, 5], index=self.data.index, name="direct"),
                name="Direct",
                plot=False,
            )
            self.callable = self.I(
                lambda close, n=2: pd.Series(close).rolling(n).mean(),
                self.data.close,
                n=2,
            )

        def next(self) -> None:
            seen["first_next_len"] = seen.setdefault("first_next_len", len(self.data.close))
            self.storage["last_close"] = float(self.data.close[-1])
            self.record(signal={"direct": self.direct[-1], "threshold": self.threshold})
            if len(self.data.close) == 3:
                assert self.data.close[-1] == 12.0
                assert self.data.close.df.index.equals(self.data.index)
                assert self.direct[-1] == 3
                assert self.position().size == 0
                assert self.position().is_long is False
                self.buy(size=1, tag="entry")
            elif len(self.data.close) == 4:
                assert self.equity > 0
                assert isinstance(self.orders, tuple)
                self.position().close()

    stats = Backtest(_data(), SurfaceStrategy, cash=1000.0, storage=storage).run(threshold=13)

    assert stats["Equity Final [$]"] > 0
    assert seen["init_len"] == 5
    assert seen["first_next_len"] == 3
    assert seen["df_columns"] == ["open", "high", "low", "close", "volume"]
    assert seen["ticker"] == "Asset"
    assert storage["last_close"] == 14.0


def test_backtesting_facade_rejects_backtesting_py_capitalized_data_aliases() -> None:
    class CapitalizedStrategy(Strategy):
        def init(self) -> None:
            _ = self.data.Close

        def next(self) -> None:
            pass

    with pytest.raises(AttributeError, match="use 'close' instead"):
        Backtest(_data(), CapitalizedStrategy).run()


def test_backtesting_facade_rejects_unknown_strategy_params() -> None:
    class ParamStrategy(Strategy):
        known = 1

        def init(self) -> None:
            pass

        def next(self) -> None:
            pass

    with pytest.raises(AttributeError, match="missing parameter 'unknown'"):
        Backtest(_data(), ParamStrategy).run(unknown=2)


def test_backtesting_facade_validates_indicator_index() -> None:
    class BadIndicatorStrategy(Strategy):
        def init(self) -> None:
            bad = pd.Series([1, 2, 3], index=pd.date_range("2025-01-01", periods=3, tz="UTC"))
            self.I(bad)

        def next(self) -> None:
            pass

    with pytest.raises(ValueError, match="same index"):
        Backtest(_data(), BadIndicatorStrategy).run()
