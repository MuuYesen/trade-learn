"""Tests for HTML report export."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tradelearn import metrics
from tradelearn.report import Reporter


def test_html_template_exists() -> None:
    """HTML report uses the spec's tear sheet template file."""
    template = Path("tradelearn/report/templates/tear_sheet.html")

    assert template.exists()
    assert "Summary Stats" in template.read_text()


def test_reporter_html_writes_single_file_tear_sheet(tmp_path) -> None:
    """Reporter.html writes a shareable HTML tear sheet."""
    path = tmp_path / "report.html"
    stats = _stats()

    result = Reporter(stats, periods=252).html(path)

    assert result == path
    html = path.read_text()
    assert "<!doctype html>" in html.lower()
    assert "Summary Stats" in html
    assert "demo-strategy" in html
    assert "run-001" in html
    assert "Generated" in html
    assert "Equity Curve" in html
    assert "Drawdown" in html
    assert "Annual Returns" in html
    assert "Top 10 Drawdowns" in html
    assert "Monthly Returns Heatmap" in html
    assert "Monthly Returns Distribution" in html
    assert "Rolling Returns" in html
    assert "Rolling Volatility" in html
    assert "Return Quantiles" in html
    assert "Trade Distribution" in html
    assert "Rolling Sharpe" not in html
    assert "Tradelearn" in html
    assert "Bokeh" in html
    assert "annual_return" in html


def test_reporter_report_dispatches_html_by_suffix(tmp_path) -> None:
    """Reporter.report writes HTML when the output suffix is .html."""
    path = tmp_path / "report.html"

    result = Reporter(_stats(), periods=252).report(path)

    assert result == path
    assert "Summary Stats" in path.read_text()


def test_reporter_report_uses_format_when_suffix_missing(tmp_path) -> None:
    """Reporter.report can choose an output type from format=."""
    path = tmp_path / "report"

    result = Reporter(_stats(), periods=252).report(path, format="html")

    assert result == tmp_path / "report.html"
    assert result.exists()


def test_reporter_html_adds_price_trades_chart_when_market_data_exists(tmp_path) -> None:
    """Reporter.html keeps the tear sheet and adds optional market replay."""
    path = tmp_path / "market-report.html"
    stats = _stats()
    market_data = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.8, 11.2, 11.7],
            "low": [9.8, 10.2, 10.8],
            "close": [10.6, 11.0, 11.5],
            "volume": [100.0, 110.0, 120.0],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )
    stats.fills = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "side": ["buy", "sell"],
            "price": [10.6, 11.0],
        }
    )

    Reporter(stats, market_data=market_data).html(path)

    html = path.read_text()
    assert "Price / Trades" in html
    assert "Buy" in html
    assert "Sell" in html
    assert "Equity Curve" in html
    assert "Drawdown" in html
    assert "Monthly Returns Heatmap" in html


def test_reporter_html_accepts_mapping_stats(tmp_path) -> None:
    """Reporter.html accepts dict-shaped stats."""
    path = tmp_path / "mapping-report.html"

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "summary": {"strategy_name": "demo"},
            "config": {"strategy": "demo"},
        }
    ).html(path)

    assert path.exists()


def test_reporter_html_writes_report_artifacts(tmp_path) -> None:
    """Reporter.html writes colocated report artifacts from the spec."""
    path = tmp_path / "report.html"

    Reporter(_stats(), periods=252).html(path)

    assert (tmp_path / "equity.parquet").exists()
    assert (tmp_path / "trades.parquet").exists()
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["summary"]["strategy_name"] == "demo-strategy"
    assert stats["config"]["strategy"] == "demo"


def test_reporter_html_expands_pipeline_parameters_in_experiment_section(tmp_path) -> None:
    """Reporter.html renders pipeline config as readable experiment parameters."""
    path = tmp_path / "pipeline-report.html"

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "summary": {"strategy_name": "pipeline-demo"},
            "config": {
                "strategy": "pipeline-demo",
                "pipeline": {
                    "steps": ["features", "model", "selector"],
                    "features": {
                        "type": "FactorTransformer",
                        "features": ["value", "quality"],
                        "feature_store": True,
                    },
                    "selector": {"type": "TopKSelector", "k": 10},
                },
            },
        },
        periods=252,
    ).html(path)

    html = path.read_text()
    assert "Pipeline Parameters" in html
    assert "pipeline.features.features" in html
    assert "value, quality" in html
    assert "pipeline.selector.k" in html
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["config"]["pipeline"]["selector"]["k"] == 10


def test_reporter_html_adds_exposure_chart_for_multi_asset_positions(tmp_path) -> None:
    """Reporter.html adds the multi-asset exposure section when positions contain symbols."""
    path = tmp_path / "multi-asset-report.html"

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "positions": pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                        utc=True,
                    ),
                    "symbol": ["AAA", "BBB", "AAA", "BBB"],
                    "value": [60.0, 40.0, 25.0, 75.0],
                }
            ),
            "summary": {"strategy_name": "multi"},
            "config": {"strategy": "multi"},
        }
    ).html(path)

    html = path.read_text()
    assert "Correlation Matrix" in html
    assert "Exposure Chart" in html
    assert "Holdings" in html
    assert "Long/Short Holdings" in html
    assert "Gross Leverage" in html
    assert "Position Concentration" in html
    assert "AAA" in html
    assert "BBB" in html


def test_reporter_html_accepts_benchmark_series(tmp_path) -> None:
    """Reporter.html adds benchmark metrics, overlay, and rolling beta when provided."""
    path = tmp_path / "benchmark-report.html"
    benchmark = pd.Series(
        [0.01, -0.005, 0.01, -0.02, 0.03],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="benchmark",
    )

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "summary": {"strategy_name": "benchmark-demo"},
            "config": {"strategy": "benchmark-demo"},
        },
        periods=252,
    ).html(path, benchmark=benchmark)

    html = path.read_text()
    assert "Benchmark" in html
    assert "alpha" in html
    assert "beta" in html
    assert "information_ratio" in html
    assert "Rolling Beta" not in html
    assert (tmp_path / "rolling_beta.parquet").exists()


def test_reporter_html_adds_factor_quantile_chart_when_analyzer_exists(tmp_path) -> None:
    """Reporter.html adds factor quantile chart and artifacts from analyzers."""
    path = tmp_path / "factor-report.html"

    Reporter(
        {
            "returns": _returns(),
            "trades": _trades(),
            "analyzers": {"factor": _FactorAnalyzerStub()},
            "summary": {"strategy_name": "factor-demo"},
            "config": {"strategy": "factor-demo"},
        },
        periods=252,
    ).html(path)

    html = path.read_text()
    assert "Factor Quantile Returns" in html
    assert "Factor Mean Return by Quantile" in html
    assert "Factor Quantile Returns Violin" in html
    assert "Factor Quantile Spread" in html
    assert "Factor Events Distribution" in html
    assert "Factor Quantile Counts" in html
    assert "Factor IC" in html
    assert "Factor IC Histogram" in html
    assert "Factor IC QQ" in html
    assert "Factor Rank IC" in html
    assert "Factor Turnover" in html
    assert "Factor Long-Short Returns" in html
    assert (tmp_path / "factor_ic.parquet").exists()
    assert (tmp_path / "factor_rank_ic.parquet").exists()
    assert (tmp_path / "factor_turnover.parquet").exists()
    assert (tmp_path / "factor_autocorrelation.parquet").exists()
    assert (tmp_path / "factor_long_short_returns.parquet").exists()
    assert (tmp_path / "factor_quantile_returns.parquet").exists()


class _FactorAnalyzerStub:
    def ic(self) -> pd.Series:
        """Return factor IC series for report tests."""
        return pd.Series(
            [0.10, 0.20],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="ic",
        )

    def rank_ic(self) -> pd.Series:
        """Return factor rank IC series for report tests."""
        return pd.Series(
            [0.15, 0.25],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="rank_ic",
        )

    def turnover(self) -> pd.Series:
        """Return factor turnover series for report tests."""
        return pd.Series(
            [0.30, 0.40],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="turnover",
        )

    def autocorrelation(self) -> pd.Series:
        """Return factor autocorrelation series for report tests."""
        return pd.Series(
            [0.60, 0.70],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="autocorrelation",
        )

    def quantile_cumulative_returns(self) -> pd.DataFrame:
        """Return factor quantile cumulative returns for report tests."""
        return pd.DataFrame(
            {1: [0.01, 0.02], 2: [0.03, 0.04]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def long_short_cumulative_returns(self) -> pd.DataFrame:
        """Return factor long-short cumulative returns for report tests."""
        return pd.DataFrame(
            {
                "long": [0.03, 0.04],
                "short": [0.01, 0.02],
                "spread": [0.02, 0.02],
            },
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def quantile_stats(self) -> pd.DataFrame:
        """Return factor quantile statistics for report tests."""
        return pd.DataFrame(
            {"mean": [0.01, 0.03], "std": [0.02, 0.01], "count": [3, 3]},
            index=[1, 2],
        )

    def quantile_spread(self) -> pd.Series:
        """Return factor quantile spread for report tests."""
        return pd.Series(
            [0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
            name="quantile_spread",
        )

    def quantile_counts(self) -> pd.DataFrame:
        """Return factor quantile counts for report tests."""
        return pd.DataFrame(
            {1: [2, 3], 2: [3, 2]},
            index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
        )

    def quantile_forward_returns(self) -> pd.DataFrame:
        """Return raw forward returns grouped by quantile for report tests."""
        return pd.DataFrame(
            {
                "quantile": [1, 1, 2, 2],
                "forward_return": [-0.01, 0.02, 0.03, 0.04],
            }
        )

    def events_distribution(self) -> pd.DataFrame:
        """Return event rows for distribution charts."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, tz="UTC"),
                "symbol": ["AAA", "BBB", "CCC"],
            }
        )


def _stats() -> SimpleNamespace:
    returns = _returns()
    return SimpleNamespace(
        returns=returns,
        equity=metrics.cum_returns(returns, starting_value=100_000.0),
        trades=_trades(),
        positions=pd.DataFrame(),
        orders=pd.DataFrame(),
        summary={"strategy_name": "demo-strategy"},
        analyzers={},
        config={"strategy": "demo", "run_id": "run-001"},
    )


def _returns() -> pd.Series:
    return pd.Series(
        [0.02, -0.01, 0.015, -0.03, 0.04],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="returns",
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame({"pnl": [100.0, -50.0, 25.0, -10.0]})
