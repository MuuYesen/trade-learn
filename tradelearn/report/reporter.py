"""Reporter facade backed by tradelearn.metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tradelearn.metrics as metrics
from tradelearn.core import get_logger
from tradelearn.report import charts
from tradelearn.report.analytics import (
    active_return,
    active_returns,
    active_weights,
    annual_returns,
    exposure_correlation,
    exposure_weights,
    monthly_returns_matrix,
    performance_attribution,
    rolling_beta,
    rolling_returns,
    rolling_sharpe,
    rolling_volatility,
    top_drawdowns,
    tracking_error,
    trade_distribution,
)
from tradelearn.report.excel import write_excel_report
from tradelearn.report.explore import explore_trades
from tradelearn.report.html import write_html_report

LOGGER = get_logger("report")


class Reporter:
    """Build report-ready summaries and series from backtest stats."""

    @classmethod
    def from_returns(
        cls,
        *,
        returns: pd.Series,
        positions: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        market_data: pd.DataFrame | None = None,
        periods: int = 252,
        strategy_name: str = "external-returns",
    ) -> Reporter:
        """Create a reporter from external return series and optional holdings."""
        stats = {
            "returns": pd.Series(returns).copy(),
            "trades": cls._trades_from_transactions(transactions),
            "fills": cls._fills_from_transactions(transactions),
            "positions": cls._positions_frame(positions),
            "summary": {"strategy_name": strategy_name},
            "config": {"strategy": strategy_name, "source": "external_returns"},
        }
        reporter = cls(stats, periods=periods, market_data=market_data)
        reporter.benchmark = None if benchmark is None else pd.Series(benchmark).copy()
        return reporter

    def __init__(
        self,
        stats: Any,
        periods: int = 252,
        market_data: pd.DataFrame | None = None,
    ) -> None:
        """Create a reporter from a Stats-like object or mapping."""
        self.stats = stats
        self.periods = periods
        self.market_data = None if market_data is None else pd.DataFrame(market_data).copy()

    def summary(self, benchmark: pd.Series | None = None) -> dict[str, float | int | str]:
        """Return report summary statistics."""
        returns = self._get("returns")
        trades = self._trades()
        summary = dict(self._get("summary", default={}) or {})
        computed: dict[str, float | int] = {
            "annual_return": metrics.annual_return(returns, periods=self.periods),
            "cumulative_return": metrics.cum_returns(returns).iloc[-1],
            "annual_volatility": metrics.volatility(returns, periods=self.periods),
            "sharpe_ratio": metrics.sharpe(returns, periods=self.periods),
            "calmar_ratio": metrics.calmar(returns, periods=self.periods),
            "sortino_ratio": metrics.sortino(returns, periods=self.periods),
            "max_drawdown": metrics.max_drawdown(returns),
            "max_dd_duration": self._max_drawdown_duration(returns),
        }
        if benchmark is not None:
            computed.update(
                {
                    "alpha": metrics.alpha(returns, benchmark, periods=self.periods),
                    "beta": metrics.beta(returns, benchmark),
                    "information_ratio": metrics.information_ratio(
                        returns,
                        benchmark,
                        self.periods,
                    ),
                    "active_return": active_return(returns, benchmark),
                    "tracking_error": tracking_error(returns, benchmark, self.periods),
                }
            )
        computed.update(
            {
                "win_rate": metrics.win_rate(trades),
                "profit_factor": metrics.profit_factor(trades),
                "avg_win": metrics.avg_win(trades),
                "avg_loss": metrics.avg_loss(trades),
                "total_trades": int(len(trades)),
                "turnover": self._turnover(),
            }
        )
        computed.update(self._factor_summary())
        return computed | summary

    def equity_curve(self) -> pd.Series:
        """Return normalized equity curve starting at 1.0."""
        return metrics.cum_returns(self._get("returns"), starting_value=1.0)

    def drawdown(self) -> pd.Series:
        """Return drawdown series."""
        return metrics.drawdown_series(self._get("returns"))

    def monthly_heatmap(self) -> pd.DataFrame:
        """Return monthly returns matrix with yearly and monthly totals."""
        return monthly_returns_matrix(self._get("returns"))

    def annual_returns(self) -> pd.Series:
        """Return annual compounded returns."""
        return annual_returns(self._get("returns"))

    def rolling_returns(self) -> pd.Series:
        """Return cumulative rolling returns."""
        return rolling_returns(self._get("returns"))

    def rolling_volatility(self, window: int = 126) -> pd.Series:
        """Return annualized rolling volatility."""
        return rolling_volatility(self._get("returns"), window=window, periods=self.periods)

    def rolling_sharpe(self, window: int = 126) -> pd.Series:
        """Return rolling Sharpe ratio."""
        return rolling_sharpe(self._get("returns"), window=window, periods=self.periods)

    def rolling_beta(self, benchmark: pd.Series, window: int = 126) -> pd.Series:
        """Return rolling beta to a benchmark."""
        return rolling_beta(self._get("returns"), benchmark=benchmark, window=window)

    def active_returns(self, benchmark: pd.Series) -> pd.Series:
        """Return daily active returns against a benchmark."""
        return active_returns(self._get("returns"), benchmark)

    def tracking_error(self, benchmark: pd.Series) -> float:
        """Return annualized tracking error against a benchmark."""
        return tracking_error(self._get("returns"), benchmark, self.periods)

    def active_weights(
        self,
        benchmark_weights: pd.DataFrame | pd.Series | None = None,
    ) -> pd.DataFrame:
        """Return active exposure weights against optional benchmark weights."""
        return active_weights(
            self._get("positions", default=pd.DataFrame()),
            benchmark_weights=benchmark_weights,
        )

    def performance_attribution(self, benchmark: pd.Series) -> pd.DataFrame:
        """Return benchmark-relative return attribution rows."""
        return performance_attribution(self._get("returns"), benchmark)

    def top_drawdowns(self, limit: int = 10) -> pd.DataFrame:
        """Return the largest drawdown episodes."""
        return top_drawdowns(self._get("returns"), limit=limit)

    def trade_distribution(self, bins: int = 20) -> pd.DataFrame:
        """Return trade PnL histogram bins."""
        return trade_distribution(self._trades(), bins=bins)

    def exposure(self) -> pd.DataFrame:
        """Return daily symbol exposure weights."""
        return exposure_weights(self._get("positions", default=pd.DataFrame()))

    def correlation_matrix(self) -> pd.DataFrame:
        """Return multi-asset exposure correlation matrix."""
        return exposure_correlation(self._get("positions", default=pd.DataFrame()))

    def factor_quantile_returns(self) -> pd.DataFrame:
        """Return factor quantile cumulative returns from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "quantile_cumulative_returns"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.quantile_cumulative_returns()).copy()

    def factor_quantile_stats(self) -> pd.DataFrame:
        """Return factor quantile statistics from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "quantile_stats"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.quantile_stats()).copy()

    def factor_quantile_spread(self) -> pd.Series:
        """Return factor top-minus-bottom quantile spread from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "quantile_spread"):
            return pd.Series(dtype="float64", name="quantile_spread")
        return pd.Series(factor_analyzer.quantile_spread()).copy()

    def factor_quantile_counts(self) -> pd.DataFrame:
        """Return factor quantile membership counts from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "quantile_counts"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.quantile_counts()).copy()

    def factor_quantile_forward_returns(self) -> pd.DataFrame:
        """Return raw forward returns by factor quantile from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "quantile_forward_returns"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.quantile_forward_returns()).copy()

    def factor_long_short_returns(self) -> pd.DataFrame:
        """Return factor long-short cumulative returns from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "long_short_cumulative_returns"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.long_short_cumulative_returns()).copy()

    def factor_ic(self) -> pd.Series:
        """Return factor information coefficient series from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "ic"):
            return pd.Series(dtype="float64", name="ic")
        return pd.Series(factor_analyzer.ic()).copy()

    def factor_rank_ic(self) -> pd.Series:
        """Return factor rank information coefficient series from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "rank_ic"):
            return pd.Series(dtype="float64", name="rank_ic")
        return pd.Series(factor_analyzer.rank_ic()).copy()

    def factor_turnover(self) -> pd.Series:
        """Return factor turnover series from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "turnover"):
            return pd.Series(dtype="float64", name="turnover")
        return pd.Series(factor_analyzer.turnover()).copy()

    def factor_autocorrelation(self) -> pd.Series:
        """Return factor autocorrelation series from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "autocorrelation"):
            return pd.Series(dtype="float64", name="autocorrelation")
        return pd.Series(factor_analyzer.autocorrelation()).copy()

    def factor_events_distribution(self) -> pd.DataFrame:
        """Return factor event rows from analyzers."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "events_distribution"):
            return pd.DataFrame()
        return pd.DataFrame(factor_analyzer.events_distribution()).copy()

    def equity_curve_chart(self, benchmark: pd.Series | None = None):
        """Return a Bokeh equity curve chart."""
        return charts.equity_curve(
            self.equity_curve(),
            benchmark=benchmark,
            drawdowns=self.top_drawdowns(limit=5),
        )

    def drawdown_chart(self):
        """Return a Bokeh drawdown chart."""
        return charts.drawdown(self.drawdown())

    def monthly_heatmap_chart(self):
        """Return a Bokeh monthly heatmap chart."""
        return charts.monthly_heatmap(self.monthly_heatmap())

    def annual_returns_chart(self):
        """Return a Bokeh annual returns chart."""
        return charts.annual_returns(self._get("returns"))

    def monthly_returns_distribution_chart(self):
        """Return a Bokeh monthly returns distribution chart."""
        return charts.monthly_returns_distribution(self._get("returns"))

    def monthly_returns_timeseries_chart(self):
        """Return a Bokeh monthly returns time series chart."""
        return charts.monthly_returns_timeseries(self._get("returns"))

    def rolling_returns_chart(self):
        """Return a Bokeh rolling returns chart."""
        return charts.rolling_returns(self._get("returns"))

    def rolling_volatility_chart(self, window: int = 126):
        """Return a Bokeh rolling volatility chart."""
        return charts.rolling_volatility(self._get("returns"), window=window, periods=self.periods)

    def return_quantiles_chart(self):
        """Return a Bokeh return quantiles chart."""
        return charts.return_quantiles(self._get("returns"))

    def rolling_sharpe_chart(self, window: int = 126):
        """Return a Bokeh rolling Sharpe chart."""
        return charts.rolling_sharpe(self.rolling_sharpe(window=window))

    def rolling_beta_chart(self, benchmark: pd.Series, window: int = 126):
        """Return a Bokeh rolling beta chart."""
        return charts.rolling_beta(self.rolling_beta(benchmark, window=window))

    def trade_distribution_chart(self, bins: int = 20):
        """Return a Bokeh trade distribution chart."""
        return charts.trade_distribution(self.trade_distribution(bins=bins))

    def market_replay_chart(self):
        """Return a market replay chart when market data exists."""
        if self.market_data is None:
            return None
        return charts.market_replay(
            self.market_data,
            self._get("fills", default=pd.DataFrame()),
            self._get("equity", default=pd.Series(dtype="float64")),
        )

    def price_trades_chart(self):
        """Return the market replay chart; kept as an internal compatibility alias."""
        return self.market_replay_chart()

    def exposure_chart(self):
        """Return a Bokeh exposure chart."""
        return charts.exposure(self.exposure())

    def holdings_chart(self):
        """Return a Bokeh holdings chart."""
        return charts.holdings(self._get("positions", default=pd.DataFrame()))

    def long_short_holdings_chart(self):
        """Return a Bokeh long/short holdings chart."""
        return charts.long_short_holdings(self._get("positions", default=pd.DataFrame()))

    def gross_leverage_chart(self):
        """Return a Bokeh gross leverage chart."""
        return charts.gross_leverage(self._get("positions", default=pd.DataFrame()))

    def position_concentration_chart(self):
        """Return a Bokeh position concentration chart."""
        return charts.position_concentration(self._get("positions", default=pd.DataFrame()))

    def turnover_chart(self):
        """Return a Bokeh turnover chart."""
        return charts.turnover(
            self._get("fills", default=pd.DataFrame()),
            self._get("positions", default=pd.DataFrame()),
        )

    def daily_volume_chart(self):
        """Return a Bokeh daily transaction volume chart."""
        return charts.daily_volume(self._get("fills", default=pd.DataFrame()))

    def transaction_time_histogram_chart(self):
        """Return a Bokeh transaction time histogram."""
        return charts.transaction_time_histogram(self._get("fills", default=pd.DataFrame()))

    def correlation_matrix_chart(self):
        """Return a Bokeh correlation matrix chart."""
        return charts.correlation_matrix(self.correlation_matrix())

    def factor_quantile_returns_chart(self):
        """Return a Bokeh factor quantile returns chart."""
        return charts.quantile_returns(self.factor_quantile_returns())

    def factor_quantile_returns_bar_chart(self):
        """Return a Bokeh factor quantile mean returns bar chart."""
        return charts.factor_quantile_returns_bar(self.factor_quantile_stats())

    def factor_quantile_spread_chart(self):
        """Return a Bokeh factor quantile spread chart."""
        return charts.factor_quantile_spread(self.factor_quantile_spread())

    def factor_quantile_counts_chart(self):
        """Return a Bokeh factor quantile counts chart."""
        return charts.quantile_counts(self.factor_quantile_counts())

    def factor_quantile_returns_violin_chart(self):
        """Return a Bokeh factor quantile return distribution chart."""
        return charts.factor_quantile_returns_violin(self.factor_quantile_forward_returns())

    def factor_long_short_returns_chart(self):
        """Return a Bokeh factor long-short returns chart."""
        return charts.factor_long_short_returns(self.factor_long_short_returns())

    def factor_ic_chart(self):
        """Return a Bokeh factor IC chart."""
        return charts.factor_ic(self.factor_ic())

    def factor_ic_histogram_chart(self):
        """Return a Bokeh factor IC histogram chart."""
        return charts.factor_ic_histogram(self.factor_ic())

    def factor_ic_qq_chart(self):
        """Return a Bokeh factor IC QQ chart."""
        return charts.factor_ic_qq(self.factor_ic())

    def factor_rank_ic_chart(self):
        """Return a Bokeh factor rank IC chart."""
        return charts.factor_rank_ic(self.factor_rank_ic())

    def factor_turnover_chart(self):
        """Return a Bokeh factor turnover/autocorrelation chart."""
        return charts.factor_turnover(
            self.factor_turnover(),
            self.factor_autocorrelation(),
        )

    def factor_events_distribution_chart(self):
        """Return a Bokeh factor events distribution chart."""
        return charts.factor_events_distribution(self.factor_events_distribution())

    def excel(
        self,
        path: str,
        benchmark: pd.Series | None = None,
        sections: list[Any] | tuple[Any, ...] | None = None,
    ) -> Any:
        """Write an Excel report."""
        return write_excel_report(self, path, benchmark=benchmark, sections=sections)

    def html(
        self,
        path: str,
        benchmark: pd.Series | None = None,
        sections: list[Any] | tuple[Any, ...] | None = None,
    ) -> Any:
        """Write an HTML report."""
        return write_html_report(self, path, benchmark=benchmark, sections=sections)

    def report(
        self,
        path: str | Path,
        benchmark: pd.Series | None = None,
        format: str | None = None,
        sections: list[Any] | tuple[Any, ...] | None = None,
    ) -> Any:
        """Write a report, dispatching to HTML or Excel from suffix/format."""
        output = Path(path)
        benchmark = benchmark if benchmark is not None else getattr(self, "benchmark", None)
        chosen = (format or output.suffix.lstrip(".") or "html").lower()
        if chosen in {"htm", "html"}:
            if not output.suffix:
                output = output.with_suffix(".html")
            result = self.html(output, benchmark=benchmark, sections=sections)
            LOGGER.info(
                "Report written path=%s format=html benchmark=%s",
                result,
                benchmark is not None,
            )
            return result
        if chosen in {"xls", "xlsx", "excel"}:
            if not output.suffix:
                output = output.with_suffix(".xlsx")
            result = self.excel(output, benchmark=benchmark, sections=sections)
            LOGGER.info(
                "Report written path=%s format=excel benchmark=%s",
                result,
                benchmark is not None,
            )
            return result
        raise ValueError(f"Unsupported report format: {chosen}")

    def explore(self) -> Any:
        """Open an interactive pygwalker explorer for trades."""
        return explore_trades(self._get("trades", default=pd.DataFrame()))

    def _get(self, name: str, default: Any = None) -> Any:
        """Read a field from mapping or object stats."""
        if isinstance(self.stats, Mapping):
            return self.stats.get(name, default)
        return getattr(self.stats, name, default)

    def _trades(self) -> pd.DataFrame:
        """Return trades in a report-safe frame shape."""
        trades = self._get("trades", default=pd.DataFrame())
        if isinstance(trades, pd.DataFrame) and trades.empty and "pnl" not in trades:
            return pd.DataFrame({"pnl": pd.Series(dtype="float64")})
        return trades

    def _turnover(self) -> float:
        """Return explicit turnover summary or NaN until positions are supported."""
        summary = self._get("summary", default={}) or {}
        if "turnover" in summary:
            return float(summary["turnover"])
        return np.nan

    def _factor_analyzer(self) -> Any:
        """Return a configured factor analyzer if present."""
        analyzers = self._get("analyzers", default={}) or {}
        if isinstance(analyzers, Mapping) and "factor" in analyzers:
            return analyzers["factor"]
        return self._get("factor", default=None)

    def _factor_summary(self) -> dict[str, float]:
        """Return prefixed factor analyzer summary metrics."""
        factor_analyzer = self._factor_analyzer()
        if factor_analyzer is None or not hasattr(factor_analyzer, "summary"):
            return {}
        return {
            f"factor_{key}": float(value)
            for key, value in factor_analyzer.summary().items()
        }

    @staticmethod
    def _positions_frame(positions: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize wide positions into report position rows."""
        if positions is None:
            return pd.DataFrame()
        frame = pd.DataFrame(positions).copy()
        if {"date", "symbol", "value"}.issubset(frame.columns):
            return frame
        if {"datetime", "symbol", "value"}.issubset(frame.columns):
            return frame.rename(columns={"datetime": "date"})
        if frame.empty:
            return pd.DataFrame()
        original_index_name = frame.index.name
        long = frame.reset_index()
        index_column = original_index_name if original_index_name in long.columns else "index"
        long = long.rename(columns={index_column: "date"})
        value_columns = [
            column
            for column in long.columns
            if column != "date" and pd.api.types.is_numeric_dtype(long[column])
        ]
        if not value_columns:
            return pd.DataFrame()
        result = long.melt(
            id_vars=["date"],
            value_vars=value_columns,
            var_name="symbol",
            value_name="_position_value",
        )
        return result.rename(columns={"_position_value": "value"})

    @staticmethod
    def _fills_from_transactions(transactions: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize transaction rows into report fills."""
        if transactions is None:
            return pd.DataFrame()
        frame = pd.DataFrame(transactions).copy()
        if frame.empty:
            return pd.DataFrame()
        if "datetime" not in frame:
            frame["datetime"] = frame.index
        if "side" not in frame and "amount" in frame:
            frame["side"] = ["buy" if amount > 0 else "sell" for amount in frame["amount"]]
        if "size" not in frame and "amount" in frame:
            frame["size"] = frame["amount"].abs()
        columns = [
            column
            for column in ["datetime", "symbol", "side", "size", "price", "commission"]
            if column in frame
        ]
        return frame[columns].reset_index(drop=True)

    @staticmethod
    def _trades_from_transactions(transactions: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize transaction rows into report trades."""
        if transactions is None:
            return pd.DataFrame({"pnl": pd.Series(dtype="float64")})
        frame = pd.DataFrame(transactions).copy()
        if frame.empty or "pnl" not in frame:
            return pd.DataFrame({"pnl": pd.Series(dtype="float64")})
        result = frame[["pnl"]].copy()
        if "datetime" in frame:
            result["datetime"] = frame["datetime"]
        else:
            result["datetime"] = frame.index
        if "symbol" in frame:
            result["symbol"] = frame["symbol"]
        return result.reset_index(drop=True)

    @staticmethod
    def _max_drawdown_duration(returns: pd.Series) -> int:
        """Return longest drawdown run length in periods."""
        drawdown = metrics.drawdown_series(returns)
        current = 0
        longest = 0
        for value in drawdown:
            if value < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest
