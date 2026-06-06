"""HTML tear sheet export."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bokeh.embed import components
from bokeh.resources import INLINE
from jinja2 import Environment, FileSystemLoader, select_autoescape

from tradelearn.report import charts
from tradelearn.report.sections import build_context, render_html_sections

TEMPLATE_DIR = Path(__file__).with_name("templates")
TEMPLATE_NAME = "tear_sheet.html"
MARKET_TIMEZONES = {
    "US": ZoneInfo("America/New_York"),
    "CN": ZoneInfo("Asia/Shanghai"),
    "HK": ZoneInfo("Asia/Hong_Kong"),
    "CRYPTO": ZoneInfo("UTC"),
}


def write_html_report(
    reporter: Any,
    path: str | Path,
    benchmark: pd.Series | None = None,
    sections: list[Any] | tuple[Any, ...] | None = None,
) -> Path:
    """Write a single-file HTML tear sheet and return the output path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.Series(reporter._get("returns")).copy()
    benchmark_returns = None if benchmark is None else pd.Series(benchmark).copy()
    trades = pd.DataFrame(reporter._get("trades", default=pd.DataFrame())).copy()
    rolling_beta = (
        pd.Series(dtype="float64", name="rolling_beta")
        if benchmark_returns is None
        else reporter.rolling_beta(benchmark_returns)
    )
    exposure = reporter.exposure()
    correlation = reporter.correlation_matrix()
    factor_ic = reporter.factor_ic()
    factor_rank_ic = reporter.factor_rank_ic()
    factor_turnover = reporter.factor_turnover()
    factor_autocorrelation = reporter.factor_autocorrelation()
    factor_long_short_returns = reporter.factor_long_short_returns()
    factor_quantile_returns = reporter.factor_quantile_returns()
    factor_quantile_forward_returns = reporter.factor_quantile_forward_returns()
    factor_events_distribution = reporter.factor_events_distribution()
    summary = reporter.summary(benchmark=benchmark_returns)
    config = reporter._get("config", default={}) or {}
    custom_sections = render_html_sections(
        sections,
        build_context(reporter, benchmark=benchmark_returns),
    )
    metadata = _metadata(summary, returns, config)
    top_drawdowns = reporter.top_drawdowns(limit=5)
    if not top_drawdowns.empty:
        # Format max_drawdown as percentage and dates using our clean formatter
        top_drawdowns = top_drawdowns.copy()
        if "max_drawdown" in top_drawdowns.columns:
            top_drawdowns["max_drawdown"] = top_drawdowns["max_drawdown"].apply(lambda x: f"{x:.2%}")
        for col in ["peak", "valley", "recovery"]:
            if col in top_drawdowns.columns:
                top_drawdowns[col] = top_drawdowns[col].apply(_format_date)

    plots = {}
    plots["Equity Curve"] = charts.equity_curve(
        reporter.equity_curve(),
        benchmark_returns,
        drawdowns=top_drawdowns.head(5),
    )
    plots["Drawdown"] = charts.drawdown(reporter.drawdown())
    plots["Annual Returns"] = charts.annual_returns(returns)
    plots["Monthly Returns Heatmap"] = charts.monthly_heatmap(reporter.monthly_heatmap())
    plots["Monthly Returns Distribution"] = charts.monthly_returns_distribution(returns)
    plots["Monthly Returns Timeseries"] = charts.monthly_returns_timeseries(returns)
    plots["Rolling Return (126-Bar)"] = charts.rolling_returns(returns)
    plots["Rolling Sharpe (126-Bar)"] = charts.rolling_sharpe(reporter.rolling_sharpe())
    plots["Rolling Volatility (126-Bar)"] = charts.rolling_volatility(returns)
    plots["Return Quantiles"] = charts.return_quantiles(returns)
    plots["Closed Trade PnL Distribution"] = charts.trade_distribution(
        reporter.trade_distribution()
    )
    price_plot = reporter.price_trades_chart()
    if price_plot is not None:
        plots[_market_replay_title(reporter.market_data)] = price_plot
    if not rolling_beta.empty:
        plots["Rolling Beta"] = charts.rolling_beta(rolling_beta)
    if _has_multi_asset_exposure(exposure):
        plots["Position Weight Correlation"] = charts.correlation_matrix(correlation)
        plots["Exposure Heatmap"] = charts.exposure(exposure)
        positions = reporter._positions_frame(reporter._get("positions", default=pd.DataFrame()))
        fills = reporter._get("fills", default=pd.DataFrame())
        equity_values = reporter._get("equity", default=None)
        plots["Holdings"] = charts.holdings(positions)
        plots["Long/Short Holdings"] = charts.long_short_holdings(positions)
        plots["Gross Leverage"] = charts.gross_leverage(positions, equity=equity_values)
        plots["Position Concentration"] = charts.position_concentration(positions)
        plots["Daily Turnover"] = charts.turnover(fills, positions, equity=equity_values)
        plots["Daily Fill Volume"] = charts.daily_volume(fills)
        plots["Fill Bar Time Histogram"] = charts.transaction_time_histogram(
            fills,
            timezone=_market_timezone(reporter.market_data),
        )
    if not factor_ic.empty:
        plots["Factor IC"] = charts.factor_ic(factor_ic)
        plots["Factor IC Histogram"] = charts.factor_ic_histogram(factor_ic)
        plots["Factor IC QQ"] = charts.factor_ic_qq(factor_ic)
    if not factor_rank_ic.empty:
        plots["Factor Rank IC"] = charts.factor_rank_ic(factor_rank_ic)
    if not factor_turnover.empty or not factor_autocorrelation.empty:
        plots["Factor Turnover"] = charts.factor_turnover(
            factor_turnover,
            factor_autocorrelation,
        )
    if not factor_long_short_returns.empty:
        plots["Factor Long-Short Returns"] = charts.factor_long_short_returns(
            factor_long_short_returns
        )
    if not factor_quantile_returns.empty:
        factor_analyzer = reporter._factor_analyzer()
        if factor_analyzer is not None and hasattr(factor_analyzer, "quantile_stats"):
            plots["Factor Mean Return by Quantile"] = charts.factor_quantile_returns_bar(
                factor_analyzer.quantile_stats()
            )
        if factor_analyzer is not None and hasattr(factor_analyzer, "compute_mean_returns_spread"):
            plots["Factor Quantile Spread"] = charts.factor_quantile_spread(
                factor_analyzer.compute_mean_returns_spread()[0]
            )
        if factor_analyzer is not None and hasattr(factor_analyzer, "quantile_counts"):
            plots["Factor Quantile Counts"] = charts.quantile_counts(
                factor_analyzer.quantile_counts()
            )
        if not factor_quantile_forward_returns.empty:
            plots["Factor Quantile Returns Violin"] = charts.factor_quantile_returns_violin(
                factor_quantile_forward_returns
            )
        if not factor_events_distribution.empty:
            plots["Factor Events Distribution"] = charts.factor_events_distribution(
                factor_events_distribution
            )
        plots["Factor Quantile Returns"] = charts.quantile_returns(factor_quantile_returns)
    plots = {title: plot for title, plot in plots.items() if _has_glyph_renderers(plot)}
    script, chart_components = components(plots)
    output.write_text(
        _render_html(
            summary=summary,
            charts=chart_components,
            script=script,
            bokeh_resources=INLINE.render(),
            metadata=metadata,
            drawdowns=top_drawdowns,
            benchmark=benchmark_returns,
            benchmark_metrics=_benchmark_metrics(summary, benchmark_returns),
            correlation=correlation,
            exposure=exposure,
            factor_ic=factor_ic,
            factor_rank_ic=factor_rank_ic,
            factor_turnover=factor_turnover,
            factor_autocorrelation=factor_autocorrelation,
            factor_long_short_returns=factor_long_short_returns,
            factor_quantile_returns=factor_quantile_returns,
            trades=trades,
            returns=returns,
            config=config,
            custom_sections=custom_sections,
        )
    )
    _write_artifacts(
        directory=output.parent,
        config=config,
        metadata=metadata,
        summary=summary,
    )
    return output


def _write_artifacts(
    *,
    directory: Path,
    config: dict[str, Any],
    metadata: dict[str, str],
    summary: dict[str, Any],
) -> None:
    """Write colocated machine-readable report metadata."""
    (directory / "stats.json").write_text(
        json.dumps(
            {
                "config": _json_safe(config),
                "metadata": metadata,
                "summary": _json_safe(summary),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _has_glyph_renderers(plot: Any) -> bool:
    """Return True when a Bokeh figure has actual plotted glyphs."""
    if getattr(plot, "renderers", ()):
        return True
    children = getattr(plot, "children", ())
    for child in children:
        item = child[0] if isinstance(child, tuple) else child
        if _has_glyph_renderers(item):
            return True
    return False


def _render_html(
    *,
    summary: dict[str, Any],
    charts: str,
    script: str,
    bokeh_resources: str,
    metadata: dict[str, str],
    drawdowns: pd.DataFrame,
    benchmark: pd.Series | None,
    benchmark_metrics: dict[str, Any],
    correlation: pd.DataFrame,
    exposure: pd.DataFrame,
    factor_ic: pd.Series,
    factor_rank_ic: pd.Series,
    factor_turnover: pd.Series,
    factor_autocorrelation: pd.Series,
    factor_long_short_returns: pd.DataFrame,
    factor_quantile_returns: pd.DataFrame,
    trades: pd.DataFrame,
    returns: pd.Series,
    config: dict[str, Any],
    custom_sections: str = "",
) -> str:
    """Render the report HTML string."""
    template = _template_environment().get_template(TEMPLATE_NAME)
    display_config = {key: value for key, value in config.items() if key != "research"}
    return template.render(
        benchmark_section=_benchmark_section(benchmark, benchmark_metrics),
        bokeh_resources=bokeh_resources,
        charts=charts,
        config_table=_summary_table(display_config),
        correlation_section=_correlation_section(correlation),
        drawdowns_table=_frame_table(drawdowns),
        exposure_section=_exposure_section(exposure),
        factor_ic_section=_factor_ic_section(factor_ic),
        factor_rank_ic_section=_factor_rank_ic_section(factor_rank_ic),
        factor_turnover_section=_factor_turnover_section(
            factor_turnover,
            factor_autocorrelation,
        ),
        factor_long_short_section=_factor_long_short_section(factor_long_short_returns),
        factor_section=_factor_section(factor_quantile_returns),
        metadata={key: escape(value) for key, value in metadata.items()},
        research_section=_research_section(config),
        returns_count=len(returns),
        script=script,
        summary_table=_summary_cards(summary),
        closed_trades_count=_closed_trade_count(trades),
        trades_count=len(trades),
        custom_sections=custom_sections,
    )


def _template_environment() -> Environment:
    """Return the Jinja2 environment for report templates."""
    return Environment(
        autoescape=select_autoescape(["html", "xml"]),
        loader=FileSystemLoader(TEMPLATE_DIR),
    )


def _closed_trade_count(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    if "isclosed" not in trades.columns:
        return int(len(trades))
    return int(trades["isclosed"].astype(bool).sum())


def _summary_table(values: dict[str, Any]) -> str:
    """Render a dict as an HTML table."""
    if not values:
        return "<div class=\"table-scroll\"><table><tbody></tbody></table></div>"
    rows = "".join(
        f"<tr><td>{escape(str(key))}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in values.items()
    )
    return (
        "<div class=\"table-scroll\"><table><thead><tr><th>metric</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _summary_cards(values: dict[str, Any]) -> str:
    """Render summary values as responsive KPI cards, sorted by priority."""
    # Keep the KPI grid readable by grouping metrics by how users scan a report:
    # core performance, risk, trade quality, trade behavior, then account context.
    priority = [
        "cumulative_return", "annual_return", "final_value",
        "max_drawdown", "sharpe_ratio", "calmar_ratio", "win_rate",
        "annual_volatility", "sortino_ratio", "avg_drawdown",
        "max_dd_duration", "avg_dd_duration", "exposure_time", "final_margin_used",
        "total_trades", "profit_factor", "expectancy",
        "avg_win", "avg_loss", "best_trade_pct", "worst_trade_pct",
        "avg_trade_pct", "max_trade_duration", "avg_trade_duration",
        "sqn", "kelly_criterion", "total_orders", "total_fills",
        "start", "end", "duration", "bars",
        "peak_value", "final_cash", "final_realized_pnl", "final_unrealized_pnl",
        "turnover",
    ]
    
    # Filter and sort keys
    keys = [
        k
        for k, value in values.items()
        if str(k) != "strategy_name" and not _is_empty_metric_value(value)
    ]
    
    def sort_key(k: str) -> int:
        try:
            return priority.index(k)
        except ValueError:
            return len(priority)

    sorted_keys = sorted(keys, key=sort_key)
    
    cards = "".join(
        "<div class=\"kpi-card\""
        f" data-metric=\"{escape(str(key))}\">"
        f"<div class=\"kpi-label\">{escape(_metric_label(str(key)))}</div>"
        f"<div class=\"kpi-value\">{escape(_format_metric_value(str(key), values[key]))}</div>"
        "</div>"
        for key in sorted_keys
    )
    return f"<div class=\"kpi-grid\">{cards}</div>"


def _research_section(config: dict[str, Any]) -> str:
    """Render research parameters as a readable experiment subsection."""
    research = config.get("research")
    if not isinstance(research, dict) or not research:
        return ""
    rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{escape(_format_value(value))}</td></tr>"
        for key, value in _flatten_display("research", research).items()
    )
    return (
        "<h3>Research Parameters</h3>"
        "<div class=\"table-scroll\"><table><thead><tr><th>parameter</th><th>value</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _flatten_display(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_display(name, value))
        else:
            flattened[name] = value
    return flattened


def _frame_table(frame: pd.DataFrame) -> str:
    """Render a frame as an HTML table."""
    if frame.empty:
        return "<div class=\"table-scroll\"><table><tbody></tbody></table></div>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        + "</tr>"
        for row in frame.itertuples(index=False, name=None)
    )
    return f"<div class=\"table-scroll\"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>"


def _metadata(
    summary: dict[str, Any],
    returns: pd.Series,
    config: dict[str, Any],
) -> dict[str, str]:
    """Return report header and footer metadata."""
    strategy_name = str(
        summary.get("strategy_name") or config.get("strategy") or "TradeLearn Report"
    )
    run_id = str(config.get("run_id") or summary.get("run_id") or "")
    if not run_id or run_id == "-":
        run_id = datetime.now().strftime("RUN-%Y%m%d-%H%M")

    return {
        "strategy_name": strategy_name,
        "run_id": run_id,
        "start": _format_date(returns.index.min()),
        "end": _format_date(returns.index.max()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": _package_version(),
    }


def _exposure_section(exposure: pd.DataFrame) -> str:
    """Return the optional exposure section heading."""
    if not _has_multi_asset_exposure(exposure):
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in exposure.columns)
    return f"<h2>Exposure Heatmap</h2><p>Symbols: {symbols}</p>"


def _correlation_section(correlation: pd.DataFrame) -> str:
    """Return the optional correlation matrix section heading."""
    if correlation.empty or len(correlation.columns) <= 1:
        return ""
    symbols = ", ".join(escape(str(symbol)) for symbol in correlation.columns)
    return f"<h2>Correlation Matrix</h2><p>Symbols: {symbols}</p>"


def _benchmark_metrics(summary: dict[str, Any], benchmark: pd.Series | None) -> dict[str, Any]:
    """Return benchmark-relative metrics for the side panel."""
    if benchmark is None:
        return {}
    keys = ["alpha", "beta", "information_ratio", "active_return", "tracking_error"]
    return {key: summary[key] for key in keys if key in summary}


def _benchmark_section(benchmark: pd.Series | None, metrics: dict[str, Any]) -> str:
    """Return the optional benchmark section heading."""
    if benchmark is None:
        return ""
    name = escape(str(benchmark.name or "benchmark"))
    table = _summary_table(metrics) if metrics else ""
    return (
        "<section><h2>Benchmark-Aware Metrics</h2>"
        f"<p>Benchmark: {name}</p>"
        f"{table}</section>"
    )


def _factor_section(factor_quantile_returns: pd.DataFrame) -> str:
    """Return the optional factor quantile section heading."""
    if factor_quantile_returns.empty:
        return ""
    quantiles = ", ".join(escape(str(column)) for column in factor_quantile_returns.columns)
    return f"<h2>Factor Quantile Returns</h2><p>Quantiles: {quantiles}</p>"


def _factor_ic_section(factor_ic: pd.Series) -> str:
    """Return the optional factor IC section heading."""
    if factor_ic.empty:
        return ""
    return f"<h2>Factor IC</h2><p>Observations: {len(factor_ic)}</p>"


def _factor_rank_ic_section(factor_rank_ic: pd.Series) -> str:
    """Return the optional factor rank IC section heading."""
    if factor_rank_ic.empty:
        return ""
    return f"<h2>Factor Rank IC</h2><p>Observations: {len(factor_rank_ic)}</p>"


def _factor_turnover_section(
    factor_turnover: pd.Series,
    factor_autocorrelation: pd.Series,
) -> str:
    """Return the optional factor turnover section heading."""
    if factor_turnover.empty and factor_autocorrelation.empty:
        return ""
    observations = max(len(factor_turnover), len(factor_autocorrelation))
    return f"<h2>Factor Turnover</h2><p>Observations: {observations}</p>"


def _factor_long_short_section(factor_long_short_returns: pd.DataFrame) -> str:
    """Return the optional factor long-short section heading."""
    if factor_long_short_returns.empty:
        return ""
    columns = ", ".join(escape(str(column)) for column in factor_long_short_returns.columns)
    return f"<h2>Factor Long-Short Returns</h2><p>Series: {columns}</p>"


def _has_multi_asset_exposure(exposure: pd.DataFrame) -> bool:
    """Return whether exposure should be rendered as a multi-asset section."""
    return not exposure.empty and len(exposure.columns) > 1


def _market_replay_title(market_data: Any) -> str:
    """Return the chart section title for the market replay view."""
    if isinstance(market_data, dict):
        valid_feeds = [value for value in market_data.values() if not value.empty]
        if len(valid_feeds) > 1:
            return "Portfolio Replay"
    return "Price / Trades"


def _market_timezone(market_data: Any) -> ZoneInfo:
    """Infer the report timezone from market metadata or symbol prefixes."""
    market = _market_attr(market_data)
    if market in MARKET_TIMEZONES:
        return MARKET_TIMEZONES[market]
    symbols = _market_symbols(market_data)
    if any(symbol.startswith(("NASDAQ:", "NYSE:", "AMEX:")) for symbol in symbols):
        return MARKET_TIMEZONES["US"]
    if any(symbol.startswith(("SH:", "SZ:", "SSE:", "SZSE:")) for symbol in symbols):
        return MARKET_TIMEZONES["CN"]
    if any(symbol.startswith(("HK:", "HKEX:")) for symbol in symbols):
        return MARKET_TIMEZONES["HK"]
    return ZoneInfo("UTC")


def _market_attr(market_data: Any) -> str | None:
    """Return a stable market attr when report market data carries one."""
    if isinstance(market_data, dict):
        markets = {
            str(getattr(frame, "attrs", {}).get("market", "")).upper()
            for frame in market_data.values()
            if getattr(frame, "attrs", {}).get("market")
        }
        markets.discard("GLOBAL")
        if len(markets) == 1:
            return next(iter(markets))
        return None
    market = getattr(market_data, "attrs", {}).get("market") if market_data is not None else None
    market = str(market).upper() if market else None
    return None if market == "GLOBAL" else market


def _market_symbols(market_data: Any) -> list[str]:
    """Return symbols visible in report market data."""
    if isinstance(market_data, dict):
        return [str(symbol) for symbol in market_data.keys()]
    if isinstance(market_data, pd.DataFrame) and isinstance(market_data.index, pd.MultiIndex):
        names = list(market_data.index.names)
        symbol_level = "symbol" if "symbol" in names else names[-1]
        return [str(symbol) for symbol in market_data.index.get_level_values(symbol_level).unique()]
    return []


def _format_date(value: Any) -> str:
    """Format report date metadata."""
    if pd.isna(value):
        return "-"
    if isinstance(value, pd.Timestamp):
        return str(value.tz_localize(None) if value.tzinfo is None else value.tz_convert(None)).split(".")[0]
    return str(value)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable value."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return str(value.replace(microsecond=0).tz_localize(None))
    if isinstance(value, pd.Timedelta):
        return str(value.floor("s"))
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _package_version() -> str:
    """Return installed package version for report footer."""
    try:
        return version("trade-learn")
    except PackageNotFoundError:
        return "unknown"


def _format_value(value: Any) -> str:
    """Format scalar values for HTML."""
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.6f}"
    # Strip timezone suffix from timestamps (e.g. +00:00) for cleaner display.
    if isinstance(value, pd.Timestamp):
        return str(value.tz_convert(None) if value.tzinfo is not None else value).split(".")[0]
    return str(value)


def _metric_label(key: str) -> str:
    """Return a readable label for summary metric keys."""
    aliases = {
        "max_dd_duration": "Max DD Duration",
        "return_pct": "Return",
        "pnlcomm": "PnL After Commission",
    }
    if key in aliases:
        return aliases[key]
    return key.replace("_", " ").title()


def _format_metric_value(key: str, value: Any) -> str:
    """Format KPI values compactly and professionally."""
    if _is_empty_metric_value(value):
        return "-"

    if isinstance(value, pd.Timedelta):
        return str(value.floor("s"))
    if isinstance(value, pd.Timestamp):
        return str(value.replace(microsecond=0).tz_localize(None))

    lowered = key.lower()

    # 1. Handle Count/Integer metrics (Total Trades, Orders, Fills, Bars)
    counts = {"trades", "orders", "fills", "bars"}
    if any(c in lowered for c in counts):
        try:
            return f"{int(float(value)):,}"
        except (ValueError, TypeError):
            return str(value)

    # 2. Handle Percentage metrics (Return, Drawdown, Rate, Turnover)
    if isinstance(value, float):
        # Fix double-scaling: engine provides 25.0 for 25%, but :.2% expects 0.25
        if lowered in {"return_pct", "win_rate_pct", "exposure_time"} or lowered.endswith("_pct"):
            return f"{value / 100.0:.2%}"
            
        if any(
            token in lowered
            for token in ("return", "drawdown", "rate", "turnover", "volatility")
        ):
            # If the value is already large (e.g. > 1.0), it might be pre-scaled (unlikely for QS metrics but safe to check)
            # Usually QuantStats returns decimal (0.05), but native return_pct returns 5.0
            return f"{value:.2%}"
            
        # 3. Handle Ratios (Sharpe, Calmar, Sortino, Alpha, Beta)
        if "ratio" in lowered or lowered in {"alpha", "beta", "sharpe"}:
            return f"{value:.3f}"
            
        # 4. Handle Currency/PnL (Final Cash, Value, PnL)
        if any(token in lowered for token in ("cash", "value", "pnl")):
            return f"{value:,.2f}"
            
        # Default float formatting
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}"
        
    return str(value)


def _is_empty_metric_value(value: Any) -> bool:
    """Return whether a summary metric should be omitted from compact KPI cards."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False
