"""Tests for reusable Bokeh report charts."""

import pandas as pd
from bokeh.plotting import figure
from bokeh.models import GlyphRenderer
from bokeh.models.glyphs import MultiLine

from tradelearn.report.charts import (
    annual_returns,
    correlation_matrix,
    daily_volume,
    drawdown,
    equity_curve,
    exposure,
    factor_events_distribution,
    factor_ic,
    factor_ic_histogram,
    factor_ic_qq,
    factor_long_short_returns,
    factor_quantile_returns_bar,
    factor_quantile_returns_violin,
    factor_quantile_spread,
    factor_rank_ic,
    factor_turnover,
    gross_leverage,
    holdings,
    long_short_holdings,
    market_replay,
    monthly_heatmap,
    monthly_returns_distribution,
    monthly_returns_timeseries,
    position_concentration,
    quantile_counts,
    quantile_returns,
    return_quantiles,
    rolling_beta,
    rolling_returns,
    rolling_sharpe,
    rolling_volatility,
    trade_distribution,
    transaction_time_histogram,
    turnover,
)


def test_report_charts_return_bokeh_figures() -> None:
    """Chart helpers return notebook-embeddable Bokeh figures."""
    plots = [
        equity_curve(_series("equity")),
        drawdown(_series("drawdown")),
        annual_returns(_series("returns")),
        monthly_heatmap(_monthly_returns()),
        monthly_returns_distribution(_series("returns")),
        monthly_returns_timeseries(_series("returns")),
        rolling_returns(_series("returns")),
        rolling_volatility(_series("returns")),
        return_quantiles(_series("returns")),
        rolling_sharpe(_series("rolling_sharpe")),
        trade_distribution(_trade_distribution()),
        exposure(_exposure()),
        correlation_matrix(_correlation()),
        quantile_returns(_quantile_returns()),
        factor_quantile_returns_bar(_quantile_stats()),
        factor_quantile_returns_violin(_quantile_forward_returns()),
        factor_quantile_spread(_series("quantile_spread")),
        factor_events_distribution(_factor_events()),
        quantile_counts(_quantile_counts()),
        rolling_beta(_series("rolling_beta")),
        factor_ic(_series("ic")),
        factor_ic_histogram(_series("ic")),
        factor_ic_qq(_series("ic")),
        factor_rank_ic(_series("rank_ic")),
        factor_turnover(_series("turnover"), _series("autocorrelation")),
        factor_long_short_returns(_long_short_returns()),
        holdings(_positions()),
        long_short_holdings(_positions()),
        gross_leverage(_positions()),
        position_concentration(_positions()),
        turnover(_transactions(), _positions()),
        daily_volume(_transactions()),
        transaction_time_histogram(_transactions()),
    ]

    assert all(isinstance(plot, type(figure())) for plot in plots)


def test_equity_curve_marks_top_drawdown_peak_and_valley() -> None:
    """Equity curve marks top drawdown peak and valley dates."""
    drawdowns = pd.DataFrame(
        {
            "peak": [pd.Timestamp("2024-01-01", tz="UTC")],
            "valley": [pd.Timestamp("2024-01-03", tz="UTC")],
        }
    )

    plot = equity_curve(_series("equity"), drawdowns=drawdowns)

    assert any(renderer.name == "drawdown_peak" for renderer in plot.renderers)
    assert any(renderer.name == "drawdown_valley" for renderer in plot.renderers)


def test_support_charts_are_static_but_market_replay_keeps_toolbar() -> None:
    """Only the market replay chart keeps visible interactive controls in reports."""
    static_plot = equity_curve(_series("equity"))
    replay = market_replay(_market_data(), fills=_fills(), equity=_series("equity"))

    assert static_plot.toolbar_location is None
    assert static_plot.toolbar.tools == []
    assert replay.toolbar_location == "right"
    assert replay.toolbar.tools


def test_market_replay_does_not_connect_trade_markers_with_multiline() -> None:
    """OHLC / Trades should not draw diagonal entry-exit trade connector lines."""
    replay = market_replay(_market_data(), fills=_fills(), equity=_series("equity"))
    renderers = _collect_glyph_renderers(replay)

    assert not any(isinstance(renderer.glyph, MultiLine) for renderer in renderers)


def test_market_replay_uses_portfolio_layout_for_multi_asset_inputs() -> None:
    """Multi-asset reports should show portfolio allocation and normalized assets."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_multi_asset_fills(),
        equity=_series("equity"),
        positions=_portfolio_positions(),
    )

    titles = _collect_plot_titles(replay)

    assert "Allocation" in titles
    assert "Assets / Trades" in titles
    assert "OHLC / Trades" not in titles


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


def _quantile_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {1: [0.01, 0.03], 2: [0.02, 0.05]},
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )


def _quantile_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {"mean": [0.01, 0.03], "std": [0.02, 0.04], "count": [10, 10]},
        index=[1, 2],
    )


def _quantile_counts() -> pd.DataFrame:
    return pd.DataFrame(
        {1: [3, 2], 2: [2, 3]},
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )


def _long_short_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "long": [0.01, 0.04],
            "short": [-0.01, -0.02],
            "spread": [0.02, 0.06],
        },
        index=pd.date_range("2024-01-01", periods=2, tz="UTC"),
    )


def _quantile_forward_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quantile": [1, 1, 1, 2, 2, 2],
            "forward_return": [-0.03, -0.01, 0.02, 0.01, 0.04, 0.06],
        }
    )


def _factor_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, tz="UTC"),
            "symbol": ["AAA", "BBB", "AAA", "CCC"],
        }
    )


def _positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "value": [60_000.0, -20_000.0, 30_000.0, 40_000.0],
        }
    )


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 10:00", periods=3, freq="h", tz="UTC"),
            "symbol": ["AAA", "BBB", "AAA"],
            "size": [100.0, -50.0, 75.0],
            "price": [10.0, 20.0, 11.0],
        }
    )


def _market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.8, 11.2, 11.7],
            "low": [9.8, 10.2, 10.8],
            "close": [10.6, 11.0, 11.5],
            "volume": [100.0, 110.0, 120.0],
        },
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
    )


def _portfolio_positions() -> pd.DataFrame:
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
            "value": [60.0, 40.0, 25.0, 75.0, 50.0, 50.0],
        }
    )


def _multi_asset_fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=4, tz="UTC"),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "side": ["buy", "buy", "sell", "sell"],
            "size": [10.0, 5.0, 4.0, 3.0],
            "price": [10.6, 15.9, 11.0, 16.5],
        }
    )


def _collect_plot_titles(layout) -> list[str]:
    titles: list[str] = []
    if hasattr(layout, "title") and getattr(layout.title, "text", None):
        titles.append(layout.title.text)
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        titles.extend(_collect_plot_titles(item))
    return titles


def _collect_glyph_renderers(layout) -> list[GlyphRenderer]:
    renderers: list[GlyphRenderer] = []
    if hasattr(layout, "renderers"):
        renderers.extend(
            renderer for renderer in layout.renderers if isinstance(renderer, GlyphRenderer)
        )
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        renderers.extend(_collect_glyph_renderers(item))
    return renderers


def _fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "side": ["buy", "sell"],
            "price": [10.6, 11.0],
        }
    )
