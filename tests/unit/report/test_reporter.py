"""Tests for report facade summary statistics."""

import logging
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from bokeh.plotting import figure

from tradelearn import metrics
from tradelearn.report import Reporter
from tradelearn.backtest.reporting import market_data_from_datas


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
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "total_trades",
        "turnover",
        "alpha",
        "beta",
        "information_ratio",
        "active_return",
        "tracking_error",
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
    assert summary["active_return"] == pytest.approx(
        ((1.0 + stats.returns).prod() - 1.0) - ((1.0 + benchmark).prod() - 1.0)
    )
    assert summary["tracking_error"] > 0
    assert summary["win_rate"] == metrics.win_rate(stats.trades)
    assert summary["profit_factor"] == metrics.profit_factor(stats.trades)
    assert summary["avg_win"] == metrics.avg_win(stats.trades)
    assert summary["avg_loss"] == metrics.avg_loss(stats.trades)
    assert summary["total_trades"] == 4


def test_reporter_report_logs_output_path_and_format(tmp_path, caplog) -> None:
    path = tmp_path / "report.html"
    caplog.set_level(logging.DEBUG, logger="tradelearn.report")

    Reporter(_stats(), periods=252).report(path, benchmark=_benchmark())

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Report written" in message
        and "format=html" in message
        and "benchmark=True" in message
        and str(path) in message
        for message in messages
    )


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


def test_reporter_summary_counts_closed_trades_when_available() -> None:
    """Reporter summary uses the same closed-trade count as the backtest engine."""
    stats = _stats()
    stats.trades = pd.DataFrame(
        {
            "pnl": [0.0, 100.0, 0.0, -25.0, 0.0],
            "isclosed": [False, True, False, True, False],
        }
    )

    summary = Reporter(stats, periods=252).summary()

    assert summary["total_trades"] == 2
    assert summary["win_rate"] == 0.5


def test_reporter_reads_artifact_attributes_before_mapping_summary() -> None:
    """LiteStats-like mappings expose summary keys plus artifact attributes."""

    class StatsMapping(Mapping[str, Any]):
        summary = {"strategy_name": "demo"}
        returns = _returns()
        trades = pd.DataFrame({"pnl": [0.0, 100.0], "isclosed": [False, True]})
        positions = pd.DataFrame()
        orders = pd.DataFrame()
        config = {"strategy": "demo"}
        analyzers = {}

        def __getitem__(self, key: str) -> Any:
            return self.summary[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self.summary)

        def __len__(self) -> int:
            return len(self.summary)

        def get(self, key: str, default: Any = None) -> Any:
            return self.summary.get(key, default)

    summary = Reporter(StatsMapping(), periods=252).summary()

    assert summary["total_trades"] == 1
    assert summary["strategy_name"] == "demo"


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


def test_reporter_rolling_beta_uses_windowed_metrics() -> None:
    """Reporter.rolling_beta returns a rolling metrics.beta series."""
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
        name="returns",
    )
    benchmark = pd.Series(
        [0.02, 0.01, -0.02, 0.04],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
        name="benchmark",
    )
    reporter = Reporter({"returns": returns, "trades": pd.DataFrame()})

    rolling = reporter.rolling_beta(benchmark, window=3)

    assert pd.isna(rolling.iloc[0])
    assert pd.isna(rolling.iloc[1])
    assert rolling.iloc[2] == metrics.beta(returns.iloc[:3], benchmark.iloc[:3])
    assert rolling.iloc[3] == metrics.beta(returns.iloc[1:4], benchmark.iloc[1:4])


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


def test_reporter_trade_distribution_bins_trade_pnl() -> None:
    """Reporter.trade_distribution returns histogram bins and summary stats."""
    reporter = Reporter({"returns": _returns(), "trades": _trades()})

    distribution = reporter.trade_distribution(bins=2)

    assert list(distribution.columns) == ["left", "right", "count"]
    assert distribution["count"].sum() == len(_trades())
    assert distribution.attrs["mean"] == pytest.approx(_trades()["pnl"].mean())
    assert distribution.attrs["median"] == pytest.approx(_trades()["pnl"].median())


def test_reporter_trade_distribution_uses_closed_trades_when_available() -> None:
    """Trade distribution ignores open trade rows from engine-style trade frames."""
    trades = pd.DataFrame(
        {
            "pnl": [0.0, 100.0, 0.0, -50.0, 0.0],
            "isclosed": [False, True, False, True, False],
        }
    )
    reporter = Reporter({"returns": _returns(), "trades": trades})

    distribution = reporter.trade_distribution(bins=5)

    assert distribution["count"].sum() == 2
    assert distribution.attrs["mean"] == pytest.approx(25.0)
    assert distribution.attrs["median"] == pytest.approx(25.0)


def test_reporter_exposure_pivots_multi_asset_positions() -> None:
    """Reporter.exposure returns daily symbol exposure weights."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "value": [60.0, 40.0, 25.0, 75.0],
        }
    )
    reporter = Reporter({"returns": _returns(), "trades": pd.DataFrame(), "positions": positions})

    exposure = reporter.exposure()

    assert list(exposure.columns) == ["AAA", "BBB"]
    assert exposure.loc[pd.Timestamp("2024-01-01", tz="UTC"), "AAA"] == pytest.approx(0.6)
    assert exposure.loc[pd.Timestamp("2024-01-01", tz="UTC"), "BBB"] == pytest.approx(0.4)
    assert exposure.loc[pd.Timestamp("2024-01-02", tz="UTC"), "AAA"] == pytest.approx(0.25)
    assert exposure.loc[pd.Timestamp("2024-01-02", tz="UTC"), "BBB"] == pytest.approx(0.75)


def test_reporter_exposure_accepts_engine_position_column_names() -> None:
    """Engine position frames use datetime/data/value columns."""
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                utc=True,
            ),
            "data": ["AAA", "BBB", "AAA", "BBB"],
            "value": [80.0, 20.0, 50.0, 50.0],
        }
    )
    reporter = Reporter({"returns": _returns(), "trades": pd.DataFrame(), "positions": positions})

    exposure = reporter.exposure()
    correlation = reporter.correlation_matrix()

    assert list(exposure.columns) == ["AAA", "BBB"]
    assert exposure.loc[pd.Timestamp("2024-01-01", tz="UTC"), "AAA"] == pytest.approx(0.8)
    assert exposure.loc[pd.Timestamp("2024-01-02", tz="UTC"), "BBB"] == pytest.approx(0.5)
    assert correlation.shape == (2, 2)


def test_market_data_from_datas_preserves_multi_asset_frames() -> None:
    """Report glue keeps every data feed for portfolio replay charts."""
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    first = SimpleNamespace(
        _name="AAA",
        _frame=pd.DataFrame({"close": [10.0, 11.0]}, index=dates),
    )
    second = SimpleNamespace(
        _name="BBB",
        _frame=pd.DataFrame({"close": [20.0, 21.0]}, index=dates),
    )

    market_data = market_data_from_datas([first, second])

    assert set(market_data) == {"AAA", "BBB"}
    pd.testing.assert_frame_equal(market_data["AAA"], first._frame)
    pd.testing.assert_frame_equal(market_data["BBB"], second._frame)


def test_reporter_index_enhance_tables_align_benchmark_and_positions() -> None:
    """Reporter exposes benchmark-aware index enhancement tables."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "value": [60.0, 40.0, 25.0, 75.0],
        }
    )
    benchmark_weights = pd.DataFrame(
        {"AAA": [0.5, 0.5], "BBB": [0.5, 0.5]},
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )
    reporter = Reporter(
        {
            "returns": _returns(),
            "trades": pd.DataFrame(),
            "positions": positions,
        },
        periods=252,
    )

    active_returns = reporter.active_returns(_benchmark())
    active_weights = reporter.active_weights(benchmark_weights=benchmark_weights)
    attribution = reporter.performance_attribution(_benchmark())

    assert active_returns.name == "active_return"
    assert active_returns.iloc[0] == pytest.approx(_returns().iloc[0] - _benchmark().iloc[0])
    assert active_weights.loc[pd.Timestamp("2024-01-01", tz="UTC"), "AAA"] == pytest.approx(0.1)
    assert active_weights.loc[pd.Timestamp("2024-01-02", tz="UTC"), "BBB"] == pytest.approx(0.25)
    assert list(attribution.columns) == ["component", "value"]
    assert set(attribution["component"]) == {
        "strategy_return",
        "benchmark_return",
        "active_return",
    }


def test_reporter_correlation_matrix_uses_multi_asset_exposure() -> None:
    """Reporter.correlation_matrix returns symbol correlations from exposure."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "value": [80.0, 20.0, 50.0, 50.0, 20.0, 80.0],
        }
    )
    reporter = Reporter({"returns": _returns(), "trades": pd.DataFrame(), "positions": positions})

    correlation = reporter.correlation_matrix()

    assert list(correlation.index) == ["AAA", "BBB"]
    assert list(correlation.columns) == ["AAA", "BBB"]
    assert correlation.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert correlation.loc["BBB", "BBB"] == pytest.approx(1.0)
    assert correlation.loc["AAA", "BBB"] == pytest.approx(-1.0)


def test_reporter_factor_quantile_returns_uses_factor_analyzer() -> None:
    """Reporter.factor_quantile_returns returns analyzer quantile cumulative returns."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    quantiles = reporter.factor_quantile_returns()

    pd.testing.assert_frame_equal(quantiles, analyzer.quantile_cumulative_returns())


def test_reporter_factor_long_short_returns_uses_factor_analyzer() -> None:
    """Reporter.factor_long_short_returns returns analyzer long-short cumulative returns."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    pd.testing.assert_frame_equal(
        reporter.factor_long_short_returns(),
        analyzer.long_short_cumulative_returns(),
    )


def test_reporter_factor_ic_uses_factor_analyzer() -> None:
    """Reporter.factor_ic returns analyzer IC series."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    pd.testing.assert_series_equal(reporter.factor_ic(), analyzer.ic())


def test_reporter_factor_rank_ic_uses_factor_analyzer() -> None:
    """Reporter.factor_rank_ic returns analyzer rank IC series."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    pd.testing.assert_series_equal(reporter.factor_rank_ic(), analyzer.factor_information_coefficient())


def test_reporter_factor_turnover_uses_factor_analyzer() -> None:
    """Reporter.factor_turnover returns analyzer turnover series."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    pd.testing.assert_series_equal(
        reporter.factor_turnover(),
        analyzer.quantile_turnover().mean(axis=1).rename("turnover"),
    )


def test_reporter_factor_autocorrelation_uses_factor_analyzer() -> None:
    """Reporter.factor_autocorrelation returns analyzer autocorrelation series."""
    analyzer = _FactorAnalyzerStub()
    reporter = Reporter(
        {"returns": _returns(), "trades": pd.DataFrame(), "analyzers": {"factor": analyzer}}
    )

    pd.testing.assert_series_equal(
        reporter.factor_autocorrelation(),
        analyzer.factor_rank_autocorrelation(),
    )


def test_reporter_summary_includes_factor_analyzer_metrics() -> None:
    """Reporter.summary prefixes factor analyzer summary metrics when available."""
    reporter = Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "analyzers": {"factor": _FactorAnalyzerStub()},
        }
    )

    summary = reporter.summary()

    assert summary["factor_ic_mean"] == 0.12
    assert summary["factor_factor_information_coefficient_mean"] == 0.23


def test_reporter_chart_facade_returns_bokeh_figures() -> None:
    """Reporter exposes notebook-ready Bokeh chart methods beside series APIs."""
    reporter = Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "positions": _positions(),
            "analyzers": {"factor": _FactorAnalyzerStub()},
        }
    )
    benchmark = _benchmark()

    plots = [
        reporter.equity_curve_chart(benchmark=benchmark),
        reporter.drawdown_chart(),
        reporter.monthly_heatmap_chart(),
        reporter.rolling_sharpe_chart(window=3),
        reporter.rolling_beta_chart(benchmark, window=3),
        reporter.trade_distribution_chart(bins=2),
        reporter.exposure_chart(),
        reporter.correlation_matrix_chart(),
        reporter.factor_quantile_returns_chart(),
        reporter.factor_long_short_returns_chart(),
        reporter.factor_ic_chart(),
        reporter.factor_rank_ic_chart(),
        reporter.factor_turnover_chart(),
    ]

    assert all(isinstance(plot, type(figure())) for plot in plots)


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


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "value": [80.0, 20.0, 50.0, 50.0, 20.0, 80.0],
        }
    )


class _FactorAnalyzerStub:
    def ic(self) -> pd.Series:
        """Return factor IC series for reporter tests."""
        return pd.Series(
            [0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="ic",
        )

    def factor_information_coefficient(self) -> pd.Series:
        """Return factor rank IC series for reporter tests."""
        return pd.Series(
            [0.15, 0.25],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="factor_information_coefficient",
        )

    def quantile_turnover(self) -> pd.DataFrame:
        """Return factor turnover series for reporter tests."""
        return pd.DataFrame(
            {1: [0.20, 0.30], 2: [0.40, 0.50]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def factor_rank_autocorrelation(self) -> pd.Series:
        """Return factor autocorrelation series for reporter tests."""
        return pd.Series(
            [0.60, 0.70],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="factor_rank_autocorrelation",
        )

    def quantile_cumulative_returns(self) -> pd.DataFrame:
        """Return factor quantile cumulative returns for reporter tests."""
        return pd.DataFrame(
            {1: [0.01, 0.02], 2: [0.03, 0.04]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def long_short_cumulative_returns(self) -> pd.DataFrame:
        """Return factor long-short cumulative returns for reporter tests."""
        return pd.DataFrame(
            {
                "long": [0.03, 0.04],
                "short": [0.01, 0.02],
                "spread": [0.02, 0.02],
            },
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def summary(self) -> dict[str, float]:
        """Return scalar factor diagnostics for reporter tests."""
        return {"ic_mean": 0.12, "factor_information_coefficient_mean": 0.23}
