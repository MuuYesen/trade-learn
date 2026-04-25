"""HTML tear sheet export."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from bokeh.embed import components
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.transform import linear_cmap


def write_html_report(reporter: Any, path: str | Path) -> Path:
    """Write a single-file HTML tear sheet and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(reporter._get("returns")).copy()
    trades = pd.DataFrame(reporter._get("trades", default=pd.DataFrame())).copy()
    exposure = reporter.exposure()
    summary = reporter.summary()
    plots = [
        _equity_plot(reporter.equity_curve()),
        _drawdown_plot(reporter.drawdown()),
        _monthly_heatmap_plot(reporter.monthly_heatmap()),
        _rolling_sharpe_plot(reporter.rolling_sharpe()),
        _trade_distribution_plot(trades),
    ]
    if _has_multi_asset_exposure(exposure):
        plots.append(_exposure_plot(exposure))
    script, charts = components(
        column(*plots)
    )
    output.write_text(
        _render_html(
            summary=summary,
            charts=charts,
            script=script,
            bokeh_resources=INLINE.render(),
            metadata=_metadata(summary, returns, reporter._get("config", default={}) or {}),
            drawdowns=reporter.top_drawdowns(limit=10),
            exposure=exposure,
            trades=trades,
            returns=returns,
            config=reporter._get("config", default={}) or {},
        )
    )
    return output


def _equity_plot(equity: pd.Series):
    """Return an equity curve figure."""
    frame = _plot_frame(equity, "equity")
    plot = figure(
        title="Equity Curve",
        x_axis_type="datetime",
        height=260,
        sizing_mode="stretch_width",
    )
    plot.line(frame["date"], frame["equity"], line_width=2, color="#1f77b4")
    return plot


def _drawdown_plot(drawdown: pd.Series):
    """Return a drawdown figure."""
    frame = _plot_frame(drawdown, "drawdown")
    plot = figure(
        title="Drawdown",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    plot.varea(frame["date"], y1=0, y2=frame["drawdown"], color="#d62728", alpha=0.35)
    plot.line(frame["date"], frame["drawdown"], line_width=2, color="#d62728")
    return plot


def _monthly_heatmap_plot(monthly: pd.DataFrame):
    """Return a monthly returns heatmap figure."""
    values = monthly.drop(index="month_avg", errors="ignore")
    months = [column for column in range(1, 13) if column in values.columns]
    years = [str(year) for year in values.index]
    data = {
        "month": [],
        "year": [],
        "return": [],
    }
    for year in values.index:
        for month in months:
            data["month"].append(str(month))
            data["year"].append(str(year))
            data["return"].append(values.loc[year, month])
    plot = figure(
        title="Monthly Returns Heatmap",
        x_range=[str(month) for month in months],
        y_range=years,
        height=240,
        sizing_mode="stretch_width",
        toolbar_location=None,
    )
    mapper = linear_cmap(
        "return",
        palette=["#d62728", "#f7f7f7", "#2ca02c"],
        low=-0.05,
        high=0.05,
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
    return plot


def _rolling_sharpe_plot(rolling: pd.Series):
    """Return a rolling Sharpe figure."""
    frame = _plot_frame(rolling, "rolling_sharpe").dropna()
    plot = figure(
        title="Rolling Sharpe",
        x_axis_type="datetime",
        height=220,
        sizing_mode="stretch_width",
    )
    if not frame.empty:
        plot.line(frame["date"], frame["rolling_sharpe"], line_width=2, color="#9467bd")
    return plot


def _trade_distribution_plot(trades: pd.DataFrame):
    """Return a trade PnL histogram figure."""
    plot = figure(title="Trade Distribution", height=220, sizing_mode="stretch_width")
    if "pnl" not in trades or trades.empty:
        return plot
    hist, edges = pd.cut(trades["pnl"], bins=min(20, max(1, len(trades))), retbins=True)
    counts = hist.value_counts(sort=False).to_numpy()
    plot.quad(
        top=counts,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="#2ca02c",
        line_color="white",
        alpha=0.65,
    )
    return plot


def _exposure_plot(exposure: pd.DataFrame):
    """Return a multi-asset exposure figure."""
    frame = exposure.reset_index().rename(columns={exposure.index.name or "index": "date"})
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    plot = figure(
        title="Exposure Chart",
        x_axis_type="datetime",
        height=240,
        sizing_mode="stretch_width",
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, symbol in enumerate(exposure.columns):
        plot.line(
            frame["date"],
            frame[symbol],
            line_width=2,
            color=colors[index % len(colors)],
            legend_label=str(symbol),
        )
    return plot


def _plot_frame(series: pd.Series, name: str) -> pd.DataFrame:
    """Return a timezone-naive plotting frame."""
    frame = series.to_frame(name).reset_index()
    frame.columns = ["date", name]
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    return frame


def _render_html(
    *,
    summary: dict[str, Any],
    charts: str,
    script: str,
    bokeh_resources: str,
    metadata: dict[str, str],
    drawdowns: pd.DataFrame,
    exposure: pd.DataFrame,
    trades: pd.DataFrame,
    returns: pd.Series,
    config: dict[str, Any],
) -> str:
    """Render the report HTML string."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tradelearn Report</title>
  {bokeh_resources}
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <header>
    <h1>{escape(metadata["strategy_name"])}</h1>
    <p>Run ID: {escape(metadata["run_id"])}</p>
    <p>Period: {escape(metadata["start"])} to {escape(metadata["end"])}</p>
    <p>Generated: {escape(metadata["generated_at"])}</p>
    <p>Rows: returns={len(returns)}, trades={len(trades)}</p>
  </header>
  <section>
    <h2>Summary Stats</h2>
    {_summary_table(summary)}
  </section>
  <section>
    <h2>Equity Curve</h2>
    <h2>Drawdown</h2>
    <h2>Top 10 Drawdowns</h2>
    {_frame_table(drawdowns)}
    <h2>Monthly Returns Heatmap</h2>
    <h2>Rolling Sharpe</h2>
    <h2>Trade Distribution</h2>
    {_exposure_section(exposure)}
    {charts}
    {script}
  </section>
  <section>
    <h2>Configuration</h2>
    {_summary_table(config)}
  </section>
  <footer>
    <p>Tradelearn {escape(metadata["version"])} | Generated: {escape(metadata["generated_at"])}</p>
  </footer>
</body>
</html>
"""


def _summary_table(values: dict[str, Any]) -> str:
    """Render a dict as an HTML table."""
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return (
        "<table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _frame_table(frame: pd.DataFrame) -> str:
    """Render a frame as an HTML table."""
    if frame.empty:
        return "<table><tbody></tbody></table>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        + "</tr>"
        for row in frame.itertuples(index=False, name=None)
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _metadata(
    summary: dict[str, Any],
    returns: pd.Series,
    config: dict[str, Any],
) -> dict[str, str]:
    """Return report header and footer metadata."""
    strategy_name = str(
        summary.get("strategy_name") or config.get("strategy") or "Tradelearn Report"
    )
    run_id = str(config.get("run_id") or summary.get("run_id") or "-")
    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "start": _format_date(returns.index.min()),
        "end": _format_date(returns.index.max()),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "version": _package_version(),
    }


def _exposure_section(exposure: pd.DataFrame) -> str:
    """Return the optional exposure section heading."""
    if not _has_multi_asset_exposure(exposure):
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in exposure.columns)
    return f"<h2>Exposure Chart</h2><p>Symbols: {symbols}</p>"


def _has_multi_asset_exposure(exposure: pd.DataFrame) -> bool:
    """Return whether exposure should be rendered as a multi-asset section."""
    return not exposure.empty and len(exposure.columns) > 1


def _format_date(value: Any) -> str:
    """Format report date metadata."""
    if pd.isna(value):
        return "-"
    return str(value)


def _package_version() -> str:
    """Return installed package version for report footer."""
    try:
        return version("trade-learn")
    except PackageNotFoundError:
        return "unknown"


def _format_value(value: Any) -> str:
    """Format scalar values for HTML."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
