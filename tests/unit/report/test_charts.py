"""Tests for reusable Bokeh report charts."""

import pandas as pd
from bokeh.plotting import figure

from tradelearn.report.charts import (
    correlation_matrix,
    drawdown,
    equity_curve,
    exposure,
    monthly_heatmap,
    rolling_sharpe,
    trade_distribution,
)


def test_report_charts_return_bokeh_figures() -> None:
    """Chart helpers return notebook-embeddable Bokeh figures."""
    plots = [
        equity_curve(_series("equity")),
        drawdown(_series("drawdown")),
        monthly_heatmap(_monthly_returns()),
        rolling_sharpe(_series("rolling_sharpe")),
        trade_distribution(_trade_distribution()),
        exposure(_exposure()),
        correlation_matrix(_correlation()),
    ]

    assert all(isinstance(plot, type(figure())) for plot in plots)


def _series(name: str) -> pd.Series:
    return pd.Series(
        [1.0, 1.1, 1.05],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        name=name,
    )


def _monthly_returns() -> pd.DataFrame:
    return pd.DataFrame({1: [0.01], 2: [-0.02]}, index=[2024])


def _trade_distribution() -> pd.DataFrame:
    result = pd.DataFrame({"left": [-1.0, 0.0], "right": [0.0, 1.0], "count": [1, 2]})
    result.attrs["mean"] = 0.2
    result.attrs["median"] = 0.3
    return result


def _exposure() -> pd.DataFrame:
    return pd.DataFrame(
        {"AAA": [0.6, 0.4], "BBB": [0.4, 0.6]},
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )


def _correlation() -> pd.DataFrame:
    return pd.DataFrame({"AAA": [1.0, -1.0], "BBB": [-1.0, 1.0]}, index=["AAA", "BBB"])
