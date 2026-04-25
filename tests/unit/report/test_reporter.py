"""Tests for report facade summary statistics."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tradelearn import metrics
from tradelearn.report import Reporter


def test_reporter_summary_uses_metrics_functions() -> None:
    """Reporter.summary returns the spec's stable key metrics."""
    stats = _stats()
    benchmark = _benchmark()

    summary = Reporter(stats, periods=252).summary(benchmark=benchmark)

    assert list(summary) == [
        "annual_return",
        "cumulative_return",
        "annual_volatility",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "max_drawdown",
        "max_dd_duration",
        "alpha",
        "beta",
        "information_ratio",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "total_trades",
        "turnover",
    ]
    assert summary["annual_return"] == metrics.annual_return(stats.returns, periods=252)
    assert summary["cumulative_return"] == metrics.cum_returns(stats.returns).iloc[-1]
    assert summary["annual_volatility"] == metrics.volatility(stats.returns, periods=252)
    assert summary["sharpe_ratio"] == metrics.sharpe(stats.returns, periods=252)
    assert summary["calmar_ratio"] == metrics.calmar(stats.returns, periods=252)
    assert summary["sortino_ratio"] == metrics.sortino(stats.returns, periods=252)
    assert summary["max_drawdown"] == metrics.max_drawdown(stats.returns)
    assert summary["alpha"] == metrics.alpha(stats.returns, benchmark, periods=252)
    assert summary["beta"] == metrics.beta(stats.returns, benchmark)
    assert summary["information_ratio"] == metrics.information_ratio(stats.returns, benchmark, 252)
    assert summary["win_rate"] == metrics.win_rate(stats.trades)
    assert summary["profit_factor"] == metrics.profit_factor(stats.trades)
    assert summary["avg_win"] == metrics.avg_win(stats.trades)
    assert summary["avg_loss"] == metrics.avg_loss(stats.trades)
    assert summary["total_trades"] == 4


def test_reporter_summary_accepts_mapping_stats_and_existing_summary() -> None:
    """Reporter accepts dict-shaped stats and lets explicit summary values win."""
    stats = {
        "returns": _returns(),
        "trades": _trades(),
        "summary": {"strategy_name": "demo"},
    }

    summary = Reporter(stats, periods=252).summary()

    assert summary["strategy_name"] == "demo"
    assert "alpha" not in summary
    assert "beta" not in summary
    assert "information_ratio" not in summary


def test_reporter_equity_curve_and_drawdown_delegate_metrics() -> None:
    """Reporter exposes reusable return series for later HTML and Excel outputs."""
    stats = _stats()
    reporter = Reporter(stats)

    pd.testing.assert_series_equal(reporter.equity_curve(), metrics.cum_returns(stats.returns, 1.0))
    pd.testing.assert_series_equal(reporter.drawdown(), metrics.drawdown_series(stats.returns))


def test_reporter_monthly_heatmap_pivots_monthly_returns() -> None:
    """Reporter.monthly_heatmap returns year/month return matrix."""
    returns = pd.Series(
        [0.10, -0.05, 0.02, 0.03],
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-31",
                "2024-02-01",
                "2024-02-29",
            ],
            utc=True,
        ),
        name="returns",
    )
    reporter = Reporter({"returns": returns, "trades": pd.DataFrame()})

    heatmap = reporter.monthly_heatmap()

    assert heatmap.loc[2024, 1] == pytest.approx((1.10 * 0.95) - 1.0)
    assert heatmap.loc[2024, 2] == pytest.approx((1.02 * 1.03) - 1.0)
    assert heatmap.loc[2024, "year_total"] == pytest.approx(
        (1.10 * 0.95 * 1.02 * 1.03) - 1.0
    )
    assert "month_avg" in heatmap.index


def test_reporter_rolling_sharpe_uses_windowed_metrics() -> None:
    """Reporter.rolling_sharpe returns a rolling metrics.sharpe series."""
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
        name="returns",
    )
    reporter = Reporter({"returns": returns, "trades": pd.DataFrame()}, periods=252)

    rolling = reporter.rolling_sharpe(window=3)

    assert pd.isna(rolling.iloc[0])
    assert pd.isna(rolling.iloc[1])
    assert rolling.iloc[2] == metrics.sharpe(returns.iloc[:3], periods=252)
    assert rolling.iloc[3] == metrics.sharpe(returns.iloc[1:4], periods=252)


def test_reporter_top_drawdowns_returns_largest_episodes() -> None:
    """Reporter.top_drawdowns returns the largest drawdown episodes first."""
    returns = pd.Series(
        [0.10, -0.20, -0.10, 0.50, -0.05, -0.05, 0.12],
        index=pd.date_range("2024-01-01", periods=7, tz="UTC"),
        name="returns",
    )
    reporter = Reporter({"returns": returns, "trades": pd.DataFrame()})

    drawdowns = reporter.top_drawdowns(limit=2)

    assert list(drawdowns.columns) == [
        "peak",
        "valley",
        "recovery",
        "max_drawdown",
        "duration",
    ]
    assert len(drawdowns) == 2
    assert drawdowns.iloc[0]["max_drawdown"] < drawdowns.iloc[1]["max_drawdown"]
    assert drawdowns.iloc[0]["valley"] == pd.Timestamp("2024-01-03", tz="UTC")
    assert drawdowns.iloc[0]["recovery"] == pd.Timestamp("2024-01-04", tz="UTC")


def _stats() -> SimpleNamespace:
    returns = _returns()
    return SimpleNamespace(
        returns=returns,
        equity=metrics.cum_returns(returns, starting_value=100_000.0),
        trades=_trades(),
        positions=pd.DataFrame(),
        orders=pd.DataFrame(),
        summary={},
        analyzers={},
        config={"strategy": "demo"},
    )


def _returns() -> pd.Series:
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="returns",
    )


def _benchmark() -> pd.Series:
    return pd.Series(
        [0.01, -0.005, 0.01, -0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="benchmark",
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame({"pnl": [100.0, -50.0, 25.0, -10.0]})
