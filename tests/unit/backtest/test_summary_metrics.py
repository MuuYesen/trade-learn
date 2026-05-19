from __future__ import annotations

import math

import pandas as pd
import pytest

from tradelearn.backtest.engine import _summary_trade_stats, _trade_summary


class BrokerWithSummary:
    def trade_summary(self) -> tuple[float, float]:
        return 4.0, 4.0


def test_summary_sqn_uses_backtrader_population_std() -> None:
    trades = pd.DataFrame(
        {
            "pnlcomm": [100.0, -50.0],
            "isclosed": [True, True],
            "dtopen": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "dtclose": pd.to_datetime(["2026-01-02", "2026-01-03"]),
        }
    )

    stats = _summary_trade_stats(trades, start_cash=100_000.0)

    assert stats["sqn"] == pytest.approx(math.sqrt(2) * 25.0 / 75.0)


def test_trade_summary_prefers_closed_trades_with_backtrader_win_semantics() -> None:
    trades = pd.DataFrame(
        {
            "pnl": [10.0, 0.0, -1.0, 999.0],
            "pnlcomm": [10.0, 0.0, -1.0, 999.0],
            "isclosed": [True, True, True, False],
        }
    )

    total, wins = _trade_summary(BrokerWithSummary(), trades)

    assert total == 3.0
    assert wins == 2.0
