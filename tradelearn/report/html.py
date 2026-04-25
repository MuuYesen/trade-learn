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
from bokeh.resources import INLINE

from tradelearn.report import charts


def write_html_report(
    reporter: Any,
    path: str | Path,
    benchmark: pd.Series | None = None,
) -> Path:
    """Write a single-file HTML tear sheet and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(reporter._get("returns")).copy()
    benchmark_returns = None if benchmark is None else pd.Series(benchmark).copy()
    trades = pd.DataFrame(reporter._get("trades", default=pd.DataFrame())).copy()
    exposure = reporter.exposure()
    correlation = reporter.correlation_matrix()
    summary = reporter.summary(benchmark=benchmark_returns)
    plots = [
        charts.equity_curve(reporter.equity_curve(), benchmark_returns),
        charts.drawdown(reporter.drawdown()),
        charts.monthly_heatmap(reporter.monthly_heatmap()),
        charts.rolling_sharpe(reporter.rolling_sharpe()),
        charts.trade_distribution(reporter.trade_distribution()),
    ]
    if _has_multi_asset_exposure(exposure):
        plots.append(charts.correlation_matrix(correlation))
        plots.append(charts.exposure(exposure))
    script, chart_components = components(
        column(*plots)
    )
    output.write_text(
        _render_html(
            summary=summary,
            charts=chart_components,
            script=script,
            bokeh_resources=INLINE.render(),
            metadata=_metadata(summary, returns, reporter._get("config", default={}) or {}),
            drawdowns=reporter.top_drawdowns(limit=10),
            benchmark=benchmark_returns,
            correlation=correlation,
            exposure=exposure,
            trades=trades,
            returns=returns,
            config=reporter._get("config", default={}) or {},
        )
    )
    return output


def _render_html(
    *,
    summary: dict[str, Any],
    charts: str,
    script: str,
    bokeh_resources: str,
    metadata: dict[str, str],
    drawdowns: pd.DataFrame,
    benchmark: pd.Series | None,
    correlation: pd.DataFrame,
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
  {_benchmark_section(benchmark)}
  <section>
    <h2>Equity Curve</h2>
    <h2>Drawdown</h2>
    <h2>Top 10 Drawdowns</h2>
    {_frame_table(drawdowns)}
    <h2>Monthly Returns Heatmap</h2>
    <h2>Rolling Sharpe</h2>
    <h2>Trade Distribution</h2>
    {_correlation_section(correlation)}
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


def _correlation_section(correlation: pd.DataFrame) -> str:
    """Return the optional correlation matrix section heading."""
    if correlation.empty or len(correlation.columns) <= 1:
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in correlation.columns)
    return f"<h2>Correlation Matrix</h2><p>Symbols: {symbols}</p>"


def _benchmark_section(benchmark: pd.Series | None) -> str:
    """Return the optional benchmark section heading."""
    if benchmark is None:
        return ""
    name = escape(str(benchmark.name or "benchmark"))
    return f"<section><h2>Benchmark</h2><p>{name}</p></section>"


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
