from __future__ import annotations

import math

import pandas as pd
import pytest

from tradelearn.backtest.engine import _summary_exposure_pct, _summary_trade_stats, _trade_summary
from tradelearn.backtest.models import SummaryDict


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


def test_summary_exposure_uses_daily_position_snapshots() -> None:
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 09:30",
                    "2024-01-01 10:00",
                    "2024-01-02 09:30",
                    "2024-01-03 09:30",
                ],
                utc=True,
            ),
            "data": ["AAA", "AAA", "AAA", "AAA"],
            "size": [10.0, 0.0, 0.0, 5.0],
            "value": [100.0, 0.0, 0.0, 50.0],
        }
    )

    assert _summary_exposure_pct(positions) == pytest.approx(100.0 / 3.0)


def test_trade_pct_summary_labels_disclose_initial_equity_denominator() -> None:
    text = str(
        SummaryDict(
            {
                "best_trade_pct": 1.0,
                "worst_trade_pct": -1.0,
                "avg_trade_pct": 0.0,
            }
        )
    )

    assert "Best Trade PnL / Initial Equity [%]" in text
    assert "Worst Trade PnL / Initial Equity [%]" in text
    assert "Avg. Trade PnL / Initial Equity [%]" in text
