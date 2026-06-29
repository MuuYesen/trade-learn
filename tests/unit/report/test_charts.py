"""Tests for reusable Bokeh report charts."""

import pandas as pd
from bokeh.plotting import figure
from bokeh.models import BoxAnnotation, FixedTicker, GlyphRenderer, HoverTool, Legend, Spacer, Span
from bokeh.models.glyphs import HBar, MultiLine, Scatter, Segment, VBar
from bokeh.models.widgets import Select

from tradelearn.report.charts import (
    annual_returns,
    daily_volume,
    drawdown,
    equity_curve,
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
from tradelearn.report.charts.core import (
    _allocation_selector_js,
    _positions_wide,
    _trade_activity_frame,
    _trade_activity_height,
    _trade_activity_selector_js,
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


def test_report_charts_use_report_title_style() -> None:
    """Bokeh titles should match the report visual system."""
    static_plots = [
        monthly_heatmap(_monthly_returns()),
    ]

    for plot in static_plots:
        for item in _collect_plots(plot):
            if getattr(item.title, "text", None):
                assert item.title.text_color == "#2f3b52"
                assert item.title.text_font_size == "16px"
                assert item.title.text_font_style == "bold"
                assert item.title.align == "left"
                assert item.title.text_align == "left"
                assert item.title.offset == 0
                assert item.title.standoff == 32

    replay = market_replay(_market_data(), fills=_fills(), equity=_series("equity"))
    for item in _collect_plots(replay):
        if getattr(item.title, "text", None):
            assert item.title.text_color == "#2f3b52"
            assert item.title.text_font_size == "16px"
            assert item.title.text_font_style == "bold"
            assert item.title.align == "left"
            assert item.title.text_align == "left"
            assert item.title.offset == 0
            assert item.title.standoff == 10


def test_compact_support_charts_have_room_to_breathe() -> None:
    """Two-column support charts should not look undersized inside report cards."""
    plots = [
        drawdown(_series("drawdown")),
        rolling_returns(_series("returns")),
        rolling_volatility(_series("returns")),
        return_quantiles(_series("returns")),
        rolling_sharpe(_series("rolling_sharpe")),
        trade_distribution(_trade_distribution()),
    ]

    assert all(plot.height == 300 for plot in plots)


def test_rolling_returns_uses_windowed_returns_not_cumulative_curve() -> None:
    """Rolling returns should describe a lookback window instead of duplicating equity."""
    returns = pd.Series(
        [0.10, 0.10, -0.05, 0.20],
        index=pd.date_range("2024-01-01", periods=4, tz="UTC"),
    )

    plot = rolling_returns(returns, window=3)
    renderer = next(renderer for renderer in plot.renderers if isinstance(renderer, GlyphRenderer))

    assert plot.title.text == "Rolling Return (3-Bar)"
    assert list(renderer.data_source.data["rolling_return"]) == [
        (1.10 * 1.10 * 0.95) - 1.0,
        (1.10 * 0.95 * 1.20) - 1.0,
    ]


def test_support_chart_titles_describe_metric_units() -> None:
    """Ambiguous support charts should name the metric they actually plot."""
    assert trade_distribution(_trade_distribution()).title.text == "Closed Trade PnL Distribution"
    assert trade_distribution(_trade_distribution()).xaxis[0].axis_label == "PnL"
    assert turnover(_transactions(), _positions()).title.text == "Daily Turnover"
    assert daily_volume(_transactions()).title.text == "Daily Fill Volume"
    assert rolling_sharpe(_series("rolling_sharpe"), window=3).title.text == "Rolling Sharpe (3-Bar)"
    assert rolling_volatility(_series("returns"), window=3).title.text == "Rolling Volatility (3-Bar)"


def test_large_summary_charts_match_heatmap_height() -> None:
    """Annual returns should balance the monthly heatmap in the two-column report row."""
    assert annual_returns(_series("returns")).height == 380
    assert monthly_heatmap(_monthly_returns()).height == 380


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


def test_equity_curve_highlights_drawdown_period_with_left_legend() -> None:
    """Equity curve highlights drawdown periods with a compact in-chart legend."""
    drawdowns = pd.DataFrame(
        {
            "peak": [pd.Timestamp("2024-01-01", tz="UTC")],
            "valley": [pd.Timestamp("2024-01-03", tz="UTC")],
        }
    )

    plot = equity_curve(_series("equity"), drawdowns=drawdowns)

    assert any(
        isinstance(annotation, BoxAnnotation) and annotation.name == "max_drawdown_period"
        for annotation in plot.center
    )
    assert plot.legend[0].location == "top_left"
    assert plot.legend[0].label_text_font_size == "9px"
    assert plot.legend[0].glyph_width == 12
    assert [item.label.value for item in plot.legend[0].items] == ["Peak", "Valley"]


def test_monthly_heatmap_uses_readable_neutral_color_and_hover() -> None:
    """Near-zero monthly returns should not look like blank white cells."""
    plot = monthly_heatmap(pd.DataFrame({1: [0.01], 2: [0.0], 3: [-0.02], 4: [pd.NA]}, index=[2024]))

    assert plot.height == 380
    rect = plot.renderers[0]
    mapper = rect.glyph.fill_color["transform"]
    assert "#f7f7f7" not in mapper.palette
    assert mapper.nan_color != "white"
    assert mapper.low == -0.10
    assert mapper.high == 0.10
    assert rect.data_source.data["label"] == ["+1.0%", "+0.0%", "-2.0%", "--"]
    assert set(rect.data_source.data["label_color"]) == {"#243247"}
    label_renderer = next(renderer for renderer in plot.renderers if renderer.name == "monthly_return_labels")
    assert label_renderer.glyph.text_color.field == "label_color"
    assert label_renderer.glyph.text_font_size == "8pt"
    assert label_renderer.glyph.text_font_style == "normal"
    assert label_renderer.level == "annotation"
    hover_tools = [tool for tool in plot.tools if isinstance(tool, HoverTool)]
    assert hover_tools
    assert ("Return", "@return{+0.00%}") in hover_tools[0].tooltips


def test_support_charts_are_static_but_market_replay_keeps_toolbar() -> None:
    """Only the market replay chart keeps visible interactive controls in reports."""
    static_plot = equity_curve(_series("equity"))
    replay = market_replay(_market_data(), fills=_fills(), equity=_series("equity"))

    assert static_plot.toolbar_location is None
    assert static_plot.toolbar.tools == []
    assert replay.toolbar_location == "right"
    assert replay.toolbar.tools


def test_position_charts_carry_forward_asset_positions() -> None:
    """Position support charts should treat missing asset rows as unchanged positions."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "symbol": ["AAA", "BBB"],
            "value": [100.0, 50.0],
        }
    )

    plot = holdings(positions)
    source = plot.renderers[0].data_source

    assert list(source.data["holdings"]) == [1, 2]


def test_positions_wide_uses_latest_intraday_position_state() -> None:
    """Repeated same-day position rows are state updates and must not be summed."""
    positions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 09:30", "2024-01-01 09:30", "2024-01-01 09:30"],
                utc=True,
            ),
            "data": ["AAA", "AAA", "BBB"],
            "value": [100.0, 40.0, 60.0],
        }
    )

    wide = _positions_wide(positions)

    assert wide.loc[pd.Timestamp("2024-01-01 09:30"), "AAA"] == 40.0
    assert wide.loc[pd.Timestamp("2024-01-01 09:30"), "BBB"] == 60.0


def test_position_concentration_median_ignores_inactive_assets() -> None:
    """Median concentration should describe active positions, not zero placeholders."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01"],
                utc=True,
            ),
            "symbol": ["AAA", "BBB", "CCC"],
            "value": [60.0, 40.0, 0.0],
        }
    )

    plot = position_concentration(positions)
    median_line = next(
        renderer for renderer in plot.renderers if getattr(renderer, "name", None) == "median"
    )

    assert list(median_line.data_source.data["median"]) == [0.5]


def test_gross_leverage_and_turnover_use_account_equity() -> None:
    """Exposure ratios should use account equity when it is available."""
    positions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"], utc=True),
            "symbol": ["AAA"],
            "value": [50.0],
        }
    )
    equity = pd.Series([100.0], index=pd.to_datetime(["2024-01-01"], utc=True))
    transactions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01"], utc=True),
            "size": [2.0],
            "price": [10.0],
        }
    )

    leverage_plot = gross_leverage(positions, equity=equity)
    turnover_plot = turnover(transactions, positions, equity=equity)

    assert list(leverage_plot.renderers[0].data_source.data["gross_leverage"]) == [0.5]
    assert list(turnover_plot.renderers[0].data_source.data["turnover"]) == [0.2]
    assert turnover_plot.yaxis[0].formatter.format == "0.[00]%"


def test_transaction_time_histogram_labels_bar_hours() -> None:
    """Transaction-hour histograms should describe bar timestamp hours."""
    plot = transaction_time_histogram(_transactions())

    assert plot.title.text == "Fill Bar Time Histogram"
    assert plot.xaxis[0].axis_label == "Market Hour (UTC)"
    assert plot.yaxis[0].axis_label == "Transactions"


def test_transaction_time_histogram_converts_to_us_market_hours() -> None:
    """Timezone-aware market data should be counted in US market time."""
    transactions = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=2, freq="h", tz="Asia/Tokyo"),
            "size": [1.0, 1.0],
            "price": [10.0, 10.0],
        }
    )

    plot = transaction_time_histogram(transactions, timezone="America/New_York")
    data = plot.renderers[0].data_source.data
    counts = dict(zip(data["hour"], data["count"], strict=True))

    assert counts["19"] == 1
    assert counts["20"] == 1
    assert counts["9"] == 0


def test_daily_volume_sums_absolute_trade_size() -> None:
    """Buy and sell sizes on the same day should not cancel in volume charts."""
    transactions = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"], utc=True),
            "size": [100.0, -40.0],
            "price": [10.0, 10.0],
        }
    )

    plot = daily_volume(transactions)

    assert list(plot.renderers[0].data_source.data["volume"]) == [140.0]


def test_market_replay_does_not_connect_trade_markers_with_multiline() -> None:
    """OHLC / Trades should not draw diagonal entry-exit trade connector lines."""
    replay = market_replay(_market_data(), fills=_fills(), equity=_series("equity"))
    renderers = _collect_glyph_renderers(replay)

    assert not any(isinstance(renderer.glyph, MultiLine) for renderer in renderers)


def test_market_replay_uses_portfolio_layout_for_multi_asset_inputs() -> None:
    """Multi-asset reports should show allocation and trade activity."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_multi_asset_fills(),
        equity=_series("equity"),
        positions=_portfolio_positions(),
    )

    titles = _collect_plot_titles(replay)

    assert "Allocation" in titles
    assert "Trade Acitivity" in titles
    assert "Holdings / Trades Timeline" not in titles
    assert "OHLC / Trades" not in titles


def test_trade_activity_displays_compact_symbols() -> None:
    """Trade activity axes should stay compact for exchange-prefixed symbols."""
    replay = market_replay(
        {"NASDAQ:AAPL": _market_data(), "NASDAQ:MSFT": _market_data() * 1.5},
        fills=_prefixed_multi_asset_fills(),
        equity=_series("equity"),
    )
    activity = _find_plot(replay, "Trade Acitivity")
    renderers = _collect_glyph_renderers(activity)
    source = next(
        renderer.data_source
        for renderer in renderers
        if "asset" in renderer.data_source.data and "symbol" in renderer.data_source.data
    )

    assert set(activity.y_range.factors) == {"AAPL", "MSFT"}
    assert "NASDAQ:AAPL" in set(source.data["symbol"])
    assert "AAPL" in set(source.data["asset"])


def test_allocation_legend_displays_compact_symbols() -> None:
    """Allocation legends should use report-friendly labels for prefixed symbols."""
    replay = market_replay(
        {"NASDAQ:AAPL": _market_data(), "NASDAQ:MSFT": _market_data() * 1.5},
        fills=_prefixed_multi_asset_fills(),
        equity=_series("equity"),
    )
    allocation = _find_plot(replay, "Allocation")
    labels = [item.label.value for item in allocation.legend[0].items]

    assert "AAPL" in labels
    assert "MSFT" in labels
    assert "NASDAQ:AAPL" not in labels


def test_portfolio_replay_uses_tight_x_padding_for_long_ranges() -> None:
    """Long portfolio replays should not waste chart width on empty side padding."""
    dates = pd.date_range("2024-01-01", periods=201, tz="UTC")
    market_data = {
        "AAA": pd.DataFrame(
            {
                "open": range(201),
                "high": range(1, 202),
                "low": range(201),
                "close": range(1, 202),
                "volume": [100.0] * 201,
            },
            index=dates,
        ),
        "BBB": pd.DataFrame(
            {
                "open": range(201),
                "high": range(1, 202),
                "low": range(201),
                "close": range(1, 202),
                "volume": [100.0] * 201,
            },
            index=dates,
        )
        * 1.5,
    }
    equity = pd.Series(range(1, 202), index=dates)

    replay = market_replay(market_data, fills=_multi_asset_fills(), equity=equity)
    equity_plot = _find_plot(replay, "Equity")

    assert equity_plot.x_range.start == -1.0
    assert equity_plot.x_range.end == 201.0


def test_portfolio_replay_uses_compact_above_legends() -> None:
    """Portfolio replay legends should not narrow the dense chart area."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
        positions=_portfolio_positions(),
    )

    allocation = _find_plot(replay, "Allocation")
    activity = _find_plot(replay, "Trade Acitivity")
    equity = _find_plot(replay, "Equity")

    allocation_legends = [item for item in allocation.above if isinstance(item, Legend)]
    activity_legends = [item for item in activity.above if isinstance(item, Legend)]
    equity_labels = [
        getattr(item.label, "value", None)
        for legend in equity.above
        if isinstance(legend, Legend)
        for item in legend.items
    ]
    assert allocation_legends
    assert activity_legends
    assert "Buy&Hold (Equal Weight)" in equity_labels
    assert "Buy&Hold" not in equity_labels
    assert not [item for item in allocation.right if isinstance(item, Legend)]
    assert not [item for item in activity.right if isinstance(item, Legend)]
    assert all(legend.click_policy == "hide" for legend in allocation_legends + activity_legends)
    assert all(legend.background_fill_alpha >= 0.7 for legend in allocation_legends + activity_legends)
    assert all(legend.location == "top_left" for legend in allocation_legends + activity_legends)
    assert all(legend.margin == 2 for legend in allocation_legends + activity_legends)
    assert all(legend.padding <= 3 for legend in allocation_legends + activity_legends)


def test_allocation_replay_shows_small_selection_legend_and_hides_all_assets() -> None:
    """Allocation shows Top8/Top15 legends and hides the legend for All Assets."""
    symbols = [f"AAA{index}" for index in range(20)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
        positions=_many_asset_positions(symbols),
    )

    allocation = _find_plot(replay, "Allocation")
    legend = next(item for item in allocation.above if isinstance(item, Legend))
    visible_labels = [item.label.value for item in legend.items if item.visible]

    assert not [item for item in allocation.right if isinstance(item, Legend)]
    assert legend.visible is True
    assert len([label for label in visible_labels if label not in {"Others", "Cash"}]) == 8
    assert 'cb_obj.value !== "all"' in _allocation_selector_js()
    assert "allocation_legend.visible = showLegend" in _allocation_selector_js()


def test_market_replay_keeps_single_asset_mapping_on_ohlc_layout() -> None:
    """Single-asset mapping inputs should keep the original OHLC replay layout."""
    replay = market_replay(
        {"AAA": _market_data()},
        fills=_multi_asset_fills(),
        equity=_series("equity"),
        positions=_portfolio_positions(),
    )

    titles = _collect_plot_titles(replay)

    assert "OHLC / Trades" in titles
    assert "Allocation" not in titles
    assert "Holdings / Trades Timeline" not in titles
    assert "Trade Acitivity" not in titles


def test_market_replay_reconstructs_allocation_from_fills_without_positions() -> None:
    """Portfolio replay should still show allocation when stats omit positions."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    titles = _collect_plot_titles(replay)

    assert "Allocation" in titles
    assert "Trade Acitivity" in titles


def test_portfolio_replay_handles_legacy_fills_without_symbol() -> None:
    """Multi-asset replay should not crash on old single-asset fill schemas."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_fills(),
        equity=_series("equity"),
        positions=_portfolio_positions(),
    )

    titles = _collect_plot_titles(replay)

    assert "Allocation" in titles
    assert "Trade Acitivity" in titles


def test_portfolio_replay_draws_trade_activity_by_asset() -> None:
    """Dense portfolio reports should summarize trade activity instead of holdings twice."""
    symbols = [f"AAA{index}" for index in range(10)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
    )

    activity = _find_plot(replay, "Trade Acitivity")
    trade_markers = [
        renderer
        for renderer in activity.renderers
        if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, Scatter)
    ]

    assert trade_markers
    assert len(activity.y_range.factors) == 8
    assert all(renderer.glyph.size == "marker_size" for renderer in trade_markers)
    assert activity.yaxis.axis_label == "Asset"


def test_trade_activity_offsets_sides_without_doubling_asset_axis() -> None:
    """Buy and sell markers should offset inside each asset row."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    activity = _find_plot(replay, "Trade Acitivity")
    factors = list(activity.y_range.factors)

    assert factors == ["BBB", "AAA"]
    assert activity.yaxis.axis_label == "Asset"
    assert all(isinstance(factor, str) for factor in factors)


def test_portfolio_replay_hides_internal_bar_index_axes() -> None:
    """Portfolio replay should expose dates only on the bottom x-axis."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    assert not _find_plot(replay, "Equity").xaxis[0].visible
    assert not _find_plot(replay, "Allocation").xaxis[0].visible
    assert not _find_plot(replay, "Profit / Loss").xaxis[0].visible
    assert _find_plot(replay, "Trade Acitivity").xaxis[0].visible
    assert _find_plot(replay, "Trade Acitivity").xaxis[0].major_label_overrides


def test_portfolio_replay_uses_sparse_numeric_y_ticks() -> None:
    """Numeric replay panels should avoid overly dense y-axis labels."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    for title in ["Equity", "Allocation", "Profit / Loss"]:
        plot = _find_plot(replay, title)
        assert plot.yaxis[0].ticker.desired_num_ticks <= 5

    allocation = _find_plot(replay, "Allocation")
    assert allocation.height >= 155
    assert isinstance(allocation.yaxis[0].ticker, FixedTicker)
    assert allocation.yaxis[0].ticker.ticks == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_trade_activity_uses_one_trade_hover() -> None:
    """Trade activity should avoid duplicate hover boxes while preserving trade dates."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    activity = _find_plot(replay, "Trade Acitivity")
    assert not [
        renderer
        for renderer in activity.renderers
        if isinstance(renderer, GlyphRenderer) and renderer.name == "trade_activity_date_hover"
    ]
    hover_tools = [
        tool
        for tool in activity.tools
        if isinstance(tool, HoverTool)
    ]

    assert len(hover_tools) == 1
    assert ("Date", "@date{%F %T}") in hover_tools[0].tooltips


def test_portfolio_replay_draws_named_benchmark_on_equity_panel() -> None:
    """Portfolio replay equity should show the explicit benchmark curve."""
    benchmark = pd.Series(
        [0.0, 0.02, -0.01],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC"),
        name="HS300",
    )

    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_multi_asset_fills(),
        equity=_series("equity"),
        benchmark=benchmark,
    )
    equity = _find_plot(replay, "Equity")
    labels = [
        getattr(item.label, "value", None)
        for legend in equity.above
        if isinstance(legend, Legend)
        for item in legend.items
    ]
    benchmark_renderer = next(
        renderer
        for renderer in equity.renderers
        if isinstance(renderer, GlyphRenderer)
        and "benchmark_equity" in renderer.data_source.data
    )

    assert "HS300" in labels
    assert list(benchmark_renderer.data_source.data["benchmark_equity"]) == [
        1.0,
        1.02,
        1.0098,
    ]


def test_portfolio_replay_syncs_crosshair_across_panels() -> None:
    """Portfolio panels should share a vertical crosshair for date comparison."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    for title in ["Equity", "Allocation", "Profit / Loss", "Trade Acitivity"]:
        plot = _find_plot(replay, title)
        spans = [
            model
            for model in plot.select({"name": "shared_replay_crosshair"})
            if isinstance(model, Span)
        ]
        assert spans
        assert spans[0].dimension == "height"
        assert spans[0].visible is False


def test_profit_loss_hover_uses_single_renderer() -> None:
    """Profit/Loss hover should not stack duplicate tooltips from background bars."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    plot = _find_plot(replay, "Profit / Loss")
    hover_tools = [tool for tool in plot.tools if isinstance(tool, HoverTool)]

    assert len(hover_tools) == 1
    assert [renderer.name for renderer in hover_tools[0].renderers] == ["avg_pl_bars"]


def test_allocation_hover_shows_holdings_summary() -> None:
    """Allocation hover should summarize the selected holdings count for that date."""
    symbols = [f"AAA{index}" for index in range(10)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
    )

    plot = _find_plot(replay, "Allocation")
    hover_tools = [tool for tool in plot.tools if isinstance(tool, HoverTool)]
    hover_renderer = next(
        renderer
        for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and renderer.name == "allocation_hover_segments"
    )

    assert hover_tools
    assert hover_tools[0].renderers == [hover_renderer]
    assert isinstance(hover_renderer.glyph, VBar)
    assert ("Date", "@date{%F %T}") in hover_tools[0].tooltips
    assert ("Total Invested", "@invested{0.0%}") in hover_tools[0].tooltips
    assert ("Cash", "@cash{0.0%}") in hover_tools[0].tooltips
    assert ("Top Holdings", "@top_holdings{safe}") in hover_tools[0].tooltips
    assert set(hover_renderer.data_source.data["top_count"]) == {8}


def test_allocation_legend_matches_selected_assets_and_keeps_hover_details() -> None:
    """Allocation legend covers small selections while hover keeps detailed weights."""
    symbols = [f"AAA{index}" for index in range(10)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
    )

    plot = _find_plot(replay, "Allocation")
    legend = next(item for item in plot.above if isinstance(item, Legend))
    hover_renderer = next(renderer for renderer in plot.renderers if renderer.name == "allocation_hover_segments")
    visible_labels = [item.label.value for item in legend.items if item.visible]

    assert len([label for label in visible_labels if label not in {"Others", "Cash"}]) == 8
    assert visible_labels[-2:] == ["Others", "Cash"]
    assert "AAA9" in hover_renderer.data_source.data["top_holdings"][0]
    assert "Cash" not in hover_renderer.data_source.data["top_holdings"][0]


def test_trade_activity_draws_rebalance_separators() -> None:
    """Trade activity should mark each rebalance/trade date with subtle separators."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    plot = _find_plot(replay, "Trade Acitivity")
    separator = next(
        renderer
        for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and renderer.name == "trade_rebalance_separators"
    )

    assert isinstance(separator.glyph, Segment)
    assert separator.glyph.line_alpha <= 0.25


def test_trade_activity_separates_asset_rows() -> None:
    """Trade activity rows should be visually separated by asset."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    plot = _find_plot(replay, "Trade Acitivity")

    row_boxes = next(
        renderer
        for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and renderer.name == "trade_asset_row_boxes"
    )

    assert isinstance(row_boxes.glyph, HBar)
    assert row_boxes.glyph.line_alpha >= 0.5
    assert plot.ygrid[0].grid_line_alpha == 0.0
    assert plot.ygrid[0].band_fill_alpha > 0


def test_trade_activity_marker_size_uses_readable_notional_scale() -> None:
    """Trade activity markers should stay readable while preserving relative notional."""
    activity, _symbols = _trade_activity_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, tz="UTC"),
                "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC"],
                "side": ["buy", "sell", "buy", "sell", "buy"],
                "bar_index": [0, 1, 2, 3, 4],
                "size": [1.0, 2.0, 10.0, 100.0, 10000.0],
                "price": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
    )

    sizes = activity.sort_values("notional")["marker_size"].to_list()

    assert sizes[0] >= 7.0
    assert sizes[-1] <= 18.0
    assert sizes == sorted(sizes)
    assert sizes[2] > sizes[0]


def test_portfolio_replay_aggregates_profit_loss() -> None:
    """Dense portfolio P/L should show readable P/L bars and count context."""
    replay = market_replay(
        {"AAA": _market_data(), "BBB": _market_data() * 1.5},
        fills=_closed_trade_fills(),
        equity=_series("equity"),
    )

    profit_loss = _find_plot(replay, "Profit / Loss")
    bars = [
        renderer
        for renderer in profit_loss.renderers
        if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, VBar)
    ]
    trade_markers = [
        renderer
        for renderer in profit_loss.renderers
        if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, Scatter)
    ]

    assert bars
    assert any(renderer.name == "trade_count_background" for renderer in bars)
    assert "trade_count" in profit_loss.extra_y_ranges
    assert not trade_markers


def test_portfolio_replay_trade_activity_has_visibility_selector() -> None:
    """Portfolio replay can expand asset-scoped panels from the default Top 8."""
    symbols = [f"AAA{index}" for index in range(20)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
    )

    selectors = _collect_models(replay, Select)
    selector = next(item for item in selectors if item.title == "Assets")

    assert selector.value == "8"
    assert ("15", "Top 15 by Notional") in selector.options
    assert ("all", "All Assets") in selector.options


def test_trade_activity_height_expands_for_large_asset_sets() -> None:
    """All-assets trade activity views need enough vertical room for asset labels."""
    assert _trade_activity_height(8) < _trade_activity_height(80)
    assert _trade_activity_height(80) > 520
    assert "limit * 18" in _trade_activity_selector_js()
    assert "Math.min(520" not in _trade_activity_selector_js()


def test_portfolio_replay_asset_selector_prepares_allocation_others() -> None:
    """Allocation keeps hidden assets collapsed for the shared asset selector."""
    symbols = [f"AAA{index}" for index in range(20)]
    replay = market_replay(
        {symbol: _market_data() * (index + 1) for index, symbol in enumerate(symbols)},
        fills=_many_asset_fills(symbols),
        equity=_series("equity"),
    )

    allocation = _find_plot(replay, "Allocation")
    sources = [
        renderer.data_source
        for renderer in allocation.renderers
        if isinstance(renderer, GlyphRenderer) and hasattr(renderer, "data_source")
    ]

    assert any("Others" in source.data for source in sources)


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


def _many_asset_positions(symbols: list[str]) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    for date in dates:
        for symbol in symbols:
            rows.append({"date": date, "symbol": symbol, "value": 100.0})
    return pd.DataFrame(rows)


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


def _prefixed_multi_asset_fills() -> pd.DataFrame:
    fills = _multi_asset_fills().copy()
    fills["symbol"] = fills["symbol"].replace(
        {
            "AAA": "NASDAQ:AAPL",
            "BBB": "NASDAQ:MSFT",
        }
    )
    return fills


def _closed_trade_fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
                utc=True,
            ),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "side": ["buy", "sell", "buy", "sell"],
            "size": [10.0, -10.0, 5.0, -5.0],
            "price": [10.0, 11.0, 15.0, 14.0],
        }
    )


def _many_asset_fills(symbols: list[str]) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    for index, symbol in enumerate(symbols):
        rows.append(
            {
                "datetime": dates[0],
                "symbol": symbol,
                "side": "buy",
                "size": float(index + 1),
                "price": 10.0 * (index + 1),
            }
        )
        if index < 8:
            rows.append(
                {
                    "datetime": dates[1],
                    "symbol": symbol,
                    "side": "sell",
                    "size": float(index + 1),
                    "price": 10.5 * (index + 1),
                }
            )
    return pd.DataFrame(rows)


def _collect_plot_titles(layout) -> list[str]:
    titles: list[str] = []
    if hasattr(layout, "title") and getattr(layout.title, "text", None):
        titles.append(layout.title.text)
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        titles.extend(_collect_plot_titles(item))
    return titles


def _collect_plots(layout) -> list:
    plots = []
    if hasattr(layout, "title"):
        plots.append(layout)
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        plots.extend(_collect_plots(item))
    return plots


def _find_plot(layout, title: str):
    if hasattr(layout, "title") and getattr(layout.title, "text", None) == title:
        return layout
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        result = _find_plot(item, title)
        if result is not None:
            return result
    return None


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


def _collect_models(layout, model_type):
    models = []
    if isinstance(layout, model_type):
        models.append(layout)
    for child in getattr(layout, "children", []):
        item = child[0] if isinstance(child, tuple) else child
        models.extend(_collect_models(item, model_type))
    return models


def _fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "side": ["buy", "sell"],
            "price": [10.6, 11.0],
        }
    )
