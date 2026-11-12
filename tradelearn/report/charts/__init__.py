"""Reusable Bokeh report charts."""

__all__ = [
    "annual_returns",
    "drawdown",
    "equity_curve",
    "daily_volume",
    "factor_events_distribution",
    "factor_ic_histogram",
    "factor_ic_qq",
    "factor_ic",
    "factor_long_short_returns",
    "factor_monthly_ic_heatmap",
    "factor_quantile_returns_bar",
    "factor_quantile_returns_violin",
    "factor_quantile_spread",
    "factor_rank_ic",
    "factor_turnover",
    "gross_leverage",
    "holdings",
    "long_short_holdings",
    "market_replay",
    "monthly_returns_distribution",
    "monthly_returns_timeseries",
    "monthly_heatmap",
    "position_concentration",
    "price_trades",
    "quantile_counts",
    "quantile_returns",
    "return_quantiles",
    "rolling_beta",
    "rolling_returns",
    "rolling_sharpe",
    "rolling_volatility",
    "transaction_time_histogram",
    "trade_distribution",
    "turnover",
]


def __getattr__(name):
    """Load Bokeh only when a chart is requested."""
    if name not in __all__:
        raise AttributeError(name)
    from tradelearn.report.charts import core

    value = getattr(core, name)
    globals()[name] = value
    return value
