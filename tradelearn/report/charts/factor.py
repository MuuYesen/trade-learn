"""Factor quantile, IC, turnover, and event distribution Bokeh charts."""

from __future__ import annotations

from statistics import NormalDist

import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    NumeralTickFormatter,
    Span,
)
from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from tradelearn.report.charts._common import (
    MARKET_BLUE,
    MARKET_DOWN,
    MARKET_MUTED,
    MARKET_UP,
    _make_static_chart,
    _normalize_dates,
    _plot_frame,
)


def quantile_returns(returns: pd.DataFrame):
    """Return a factor quantile returns figure."""
    frame = returns.reset_index().rename(columns={returns.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Factor Quantile Returns",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, series_column in enumerate(returns.columns):
        plot.line(
            frame["date"],
            frame[series_column],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=f"Q{series_column}",
        )
    _make_static_chart(plot)
    return plot

def factor_quantile_returns_bar(stats: pd.DataFrame):
    """Return mean forward return by factor quantile."""
    frame = pd.DataFrame(stats).copy()
    if "mean" not in frame:
        frame["mean"] = pd.Series(dtype="float64")
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "quantile"})
    frame["quantile"] = frame["quantile"].astype(str)
    frame["color"] = [MARKET_UP if value >= 0 else MARKET_DOWN for value in frame["mean"]]
    plot = figure(
        title="Mean Return by Quantile",
        x_range=list(frame["quantile"]),
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        plot.vbar("quantile", 0.7, "mean", source=ColumnDataSource(frame), color="color")
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def factor_quantile_returns_violin(forward_returns: pd.DataFrame):
    """Return Alphalens-style forward return distribution by factor quantile."""
    frame = _quantile_forward_frame(forward_returns)
    quantiles = (
        [str(value) for value in sorted(frame["quantile"].unique())]
        if not frame.empty
        else []
    )
    plot = figure(
        title="Quantile Returns Violin",
        x_range=(-0.5, max(0.5, len(quantiles) - 0.5)),
        height=300,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    if not frame.empty:
        for quantile in quantiles:
            quantile_mask = frame["quantile"].astype(str).eq(quantile)
            values = frame.loc[quantile_mask, "forward_return"].dropna()
            if values.empty:
                continue
            bins = min(18, max(4, len(values)))
            counts = pd.cut(values, bins=bins).value_counts(sort=False)
            if counts.empty or counts.max() == 0:
                continue
            centers = [(interval.left + interval.right) / 2 for interval in counts.index]
            widths = [0.42 * count / counts.max() for count in counts.to_numpy()]
            x_center = quantiles.index(quantile)
            xs = [x_center + width for width in widths] + [
                x_center - width for width in reversed(widths)
            ]
            ys = centers + list(reversed(centers))
            plot.patch(xs, ys, fill_color=MARKET_BLUE, fill_alpha=0.28, line_color=MARKET_BLUE)
            plot.scatter([x_center], [float(values.median())], size=7, color="#ff7f0e")
        plot.xaxis.ticker = list(range(len(quantiles)))
        plot.xaxis.major_label_overrides = {
            index: f"Q{value}" for index, value in enumerate(quantiles)
        }
        plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def factor_quantile_spread(spread: pd.Series):
    """Return top-minus-bottom quantile spread time series."""
    frame = _plot_frame(pd.Series(spread).dropna(), "spread")
    plot = figure(
        title="Quantile Spread",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.add_layout(
            Span(location=0, dimension="width", line_color=MARKET_MUTED, line_dash="dashed")
        )
        plot.line(frame["date"], frame["spread"], line_width=2, color=MARKET_BLUE)
    plot.yaxis.formatter = NumeralTickFormatter(format="0.[00]%")
    _make_static_chart(plot)
    return plot

def factor_long_short_returns(returns: pd.DataFrame):
    """Return a factor long-short returns figure."""
    frame = returns.reset_index().rename(columns={returns.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Factor Long-Short Returns",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = {
        "long": "#2ca02c",
        "short": "#d62728",
        "spread": "#1f77b4",
    }
    for series_column in returns.columns:
        plot.line(
            frame["date"],
            frame[series_column],
            line_width=2,
            color=colors.get(str(series_column), "#9467bd"),
            legend_label=str(series_column),
        )
    _make_static_chart(plot)
    return plot

def factor_ic(ic: pd.Series):
    """Return a factor IC figure."""
    frame = _plot_frame(ic, "ic").dropna()
    plot = figure(
        title="Factor IC",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["ic"], line_width=2, color="#1f77b4")
        plot.line(
            frame["date"],
            frame["ic"].expanding(min_periods=1).mean(),
            line_width=2,
            color="#ff7f0e",
            legend_label="Expanding Mean",
        )
    _make_static_chart(plot)
    return plot

def factor_ic_histogram(ic: pd.Series):
    """Return an IC distribution histogram."""
    values = pd.Series(ic).dropna()
    plot = figure(title="IC Histogram", height=300, sizing_mode="stretch_width")
    if not values.empty:
        buckets = min(20, max(1, len(values)))
        counts = pd.cut(values, bins=buckets).value_counts(sort=False)
        frame = pd.DataFrame(
            {
                "left": [interval.left for interval in counts.index],
                "right": [interval.right for interval in counts.index],
                "count": counts.to_numpy(),
            }
        )
        plot.quad(
            top="count",
            bottom=0,
            left="left",
            right="right",
            source=ColumnDataSource(frame),
            fill_color=MARKET_BLUE,
            line_color="white",
            alpha=0.70,
        )
        mean = float(values.mean())
        plot.line([mean, mean], [0, max(1, int(counts.max()))], color="#ff7f0e", line_width=2)
    _make_static_chart(plot)
    return plot

def factor_ic_qq(ic: pd.Series):
    """Return an IC normal QQ chart."""
    values = pd.Series(ic).dropna().sort_values().reset_index(drop=True)
    plot = figure(title="IC QQ", height=300, sizing_mode="stretch_width")
    if not values.empty:
        dist = NormalDist()
        n = len(values)
        theoretical = [dist.inv_cdf((index + 0.5) / n) for index in range(n)]
        frame = pd.DataFrame({"theoretical": theoretical, "sample": values})
        plot.scatter(
            "theoretical",
            "sample",
            source=ColumnDataSource(frame),
            size=7,
            color=MARKET_BLUE,
            alpha=0.75,
        )
        low = min(float(frame["theoretical"].min()), float(frame["sample"].min()))
        high = max(float(frame["theoretical"].max()), float(frame["sample"].max()))
        plot.line([low, high], [low, high], line_color=MARKET_MUTED, line_dash="dashed")
    _make_static_chart(plot)
    return plot

def factor_rank_ic(rank_ic: pd.Series):
    """Return a factor rank IC figure."""
    frame = _plot_frame(rank_ic, "rank_ic").dropna()
    plot = figure(
        title="Factor Rank IC",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rank_ic"], line_width=2, color="#1f77b4")
        plot.line(
            frame["date"],
            frame["rank_ic"].expanding(min_periods=1).mean(),
            line_width=2,
            color="#ff7f0e",
            legend_label="Expanding Mean",
        )
    _make_static_chart(plot)
    return plot

def factor_monthly_ic_heatmap(monthly_ic: pd.DataFrame):
    """Return a monthly IC heatmap figure."""
    values = monthly_ic.dropna(axis=0, how="all").dropna(axis=1, how="all")
    months = [int(column) for column in values.columns]
    years = [str(year) for year in values.index]
    data = {"month": [], "year": [], "ic": []}
    for year in values.index:
        for month in months:
            data["month"].append(str(month))
            data["year"].append(str(year))
            data["ic"].append(values.loc[year, month])
    plot = figure(
        title="Monthly IC Heatmap",
        x_range=[str(month) for month in months],
        y_range=years,
        height=240,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "ic",
        palette=["#d65a5a", "#f7f7f7", "#2ca36c"],
        low=-1.0,
        high=1.0,
    )
    plot.rect(
        "month",
        "year",
        width=0.95,
        height=0.95,
        source=data,
        fill_color=mapper,
        line_color="white",
    )
    _make_static_chart(plot)
    return plot

def quantile_counts(counts: pd.DataFrame):
    """Return per-date factor quantile membership counts."""
    frame = pd.DataFrame(counts).reset_index().rename(
        columns={counts.index.name or "index": "date"}
    )
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Quantile Counts",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, series_column in enumerate(counts.columns):
        plot.line(
            frame["date"],
            frame[series_column],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=f"Q{series_column}",
        )
    _make_static_chart(plot)
    return plot

def factor_turnover(turnover: pd.Series, autocorrelation: pd.Series):
    """Return a factor turnover and autocorrelation figure."""
    turnover_frame = _plot_frame(turnover, "turnover").dropna()
    autocorrelation_frame = _plot_frame(autocorrelation, "autocorrelation").dropna()
    plot = figure(
        title="Factor Turnover",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not turnover_frame.empty:
        plot.line(
            turnover_frame["date"],
            turnover_frame["turnover"],
            line_width=2,
            color="#1f77b4",
            legend_label="Turnover",
        )
    if not autocorrelation_frame.empty:
        plot.line(
            autocorrelation_frame["date"],
            autocorrelation_frame["autocorrelation"],
            line_width=2,
            color="#ff7f0e",
            legend_label="Autocorrelation",
        )
    _make_static_chart(plot)
    return plot

def factor_events_distribution(events: pd.DataFrame | pd.MultiIndex | pd.Series):
    """Return event count distribution over time."""
    frame = _events_frame(events)
    plot = figure(
        title="Events Distribution",
        x_axis_type="datetime",
        height=300,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        counts = frame.groupby(frame["date"].dt.floor("D")).size()
        data = _plot_frame(counts.rename("count"), "count")
        plot.vbar(
            data["date"],
            width=20 * 60 * 60 * 1000,
            top=data["count"],
            color=MARKET_BLUE,
            alpha=0.70,
        )
    _make_static_chart(plot)
    return plot

def _quantile_forward_frame(forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized quantile/forward_return frame."""
    frame = pd.DataFrame(forward_returns).copy()
    if frame.empty:
        return pd.DataFrame(columns=["quantile", "forward_return"])
    lowered = {str(column).lower(): column for column in frame.columns}
    quantile_col = lowered.get("quantile") or lowered.get("factor_quantile")
    return_col = lowered.get("forward_return") or lowered.get("return") or lowered.get("returns")
    if quantile_col is not None and return_col is not None:
        result = pd.DataFrame(
            {
                "quantile": frame[quantile_col],
                "forward_return": pd.to_numeric(frame[return_col], errors="coerce"),
            }
        )
        return result.dropna(subset=["quantile", "forward_return"])
    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels >= 2 and frame.shape[1] >= 1:
        result = frame.iloc[:, 0].reset_index()
        result.columns = [*result.columns[:-1], "forward_return"]
        quantile_name = "factor_quantile" if "factor_quantile" in result else result.columns[-2]
        return result.rename(columns={quantile_name: "quantile"})[["quantile", "forward_return"]]
    return pd.DataFrame(columns=["quantile", "forward_return"])

def _events_frame(events: pd.DataFrame | pd.MultiIndex | pd.Series) -> pd.DataFrame:
    """Return event rows with a normalized date column."""
    if isinstance(events, pd.MultiIndex):
        dates = events.get_level_values(0)
        return pd.DataFrame({"date": _normalize_dates(pd.Series(dates))})
    if isinstance(events, pd.Series):
        if isinstance(events.index, pd.MultiIndex):
            dates = events.index.get_level_values(0)
        else:
            dates = events.index
        return pd.DataFrame({"date": _normalize_dates(pd.Series(dates))})
    frame = pd.DataFrame(events).copy()
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    lowered = {str(column).lower(): column for column in frame.columns}
    date_col = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    return pd.DataFrame({"date": _normalize_dates(frame[date_col])}).dropna()
