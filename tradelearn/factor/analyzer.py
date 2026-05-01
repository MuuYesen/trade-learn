"""Alphalens-style factor analysis facade backed by tradelearn.metrics."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd

from tradelearn.metrics import factor as factor_metrics


@dataclass(frozen=True)
class FactorAnalyzer:
    """Analyze cross-sectional factor values and forward returns."""

    factor: pd.Series
    forward_returns: pd.Series | None = None
    prices: pd.Series | None = None
    groups: pd.Series | None = None
    periods: int = 252
    quantiles: int = 5

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        *,
        factor: str,
        target: str = "close",
        period: int | None = None,
        periods: tuple[int, ...] | None = None,
        date: str | None = None,
        symbol: str | None = None,
        groupby: str | pd.Series | None = None,
        quantiles: int = 5,
        annualization_periods: int = 252,
    ) -> FactorAnalyzer | MultiPeriodFactorAnalyzer:
        """Build factor analyzer(s) from a complete factor/price dataset.

        ``data`` can be indexed by ``(date, symbol)`` or contain explicit date
        and symbol columns. ``factor`` names the factor column and ``target``
        names the price column used to derive forward returns.
        """
        if period is not None and periods is not None:
            raise ValueError("use either period or periods, not both")
        selected = _selected_periods(period=period, periods=periods)
        factor_series = _dataset_series(
            data,
            factor,
            date=date,
            symbol=symbol,
            name="factor",
        )
        prices = _dataset_series(
            data,
            target,
            date=date,
            symbol=symbol,
            name="prices",
        )
        groups = _dataset_groupby(data, groupby, date=date, symbol=symbol)
        clean = factor_metrics.clean_factor_and_forward_returns(
            factor_series,
            prices,
            periods=selected,
            quantiles=quantiles,
            groupby=groups,
        )
        analyzers = cls.from_clean_factor_data(
            clean,
            periods=selected,
            annualization_periods=annualization_periods,
            quantiles=quantiles,
        )
        if len(selected) == 1:
            return analyzers[selected[0]]
        return MultiPeriodFactorAnalyzer(analyzers)

    @classmethod
    def from_clean_factor_data(
        cls,
        clean: pd.DataFrame,
        *,
        periods: tuple[int, ...] | None = None,
        annualization_periods: int = 252,
        quantiles: int = 5,
    ) -> dict[int, FactorAnalyzer]:
        """Return one analyzer per ``forward_return_N`` horizon.

        ``clean`` is the frame produced by ``clean_factor_and_forward_returns``.
        """
        if "factor" not in clean:
            raise ValueError("clean factor data must contain a 'factor' column")
        selected = _clean_periods(clean, periods)
        factor = pd.Series(clean["factor"], index=clean.index, name="factor")
        groups = (
            pd.Series(clean["group"], index=clean.index, name="group")
            if "group" in clean
            else None
        )
        return {
            period: cls(
                factor=factor,
                forward_returns=pd.Series(
                    clean[f"forward_return_{period}"],
                    index=clean.index,
                    name="forward_returns",
                ),
                groups=groups,
                periods=annualization_periods,
                quantiles=quantiles,
            )
            for period in selected
        }

    @classmethod
    def multi_period_summary(
        cls,
        clean: pd.DataFrame,
        *,
        periods: tuple[int, ...] | None = None,
        annualization_periods: int = 252,
        quantiles: int = 5,
    ) -> pd.DataFrame:
        """Return summary metrics indexed by prediction horizon."""
        analyzers = cls.from_clean_factor_data(
            clean,
            periods=periods,
            annualization_periods=annualization_periods,
            quantiles=quantiles,
        )
        result = pd.DataFrame(
            {period: analyzer.summary() for period, analyzer in analyzers.items()}
        ).T
        result.index.name = "period"
        return result

    def ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return per-date Pearson information coefficient.

        When ``by_group=True``, returns a date x group frame.
        """
        return factor_metrics.ic(
            self.factor,
            self._forward_returns(),
            groupby=self.groups,
            by_group=by_group,
        )

    def rank_ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return per-date Spearman rank information coefficient.

        When ``by_group=True``, returns a date x group frame.
        """
        return factor_metrics.rank_ic(
            self.factor,
            self._forward_returns(),
            groupby=self.groups,
            by_group=by_group,
        )

    def ic_ir(self) -> float:
        """Return annualized IC information ratio."""
        return factor_metrics.ic_ir(self.ic(), periods=self.periods)

    def quantile_returns(self, group_neutral: bool = False) -> pd.DataFrame:
        """Return mean forward returns by factor quantile."""
        return factor_metrics.quantile_returns(
            self.factor,
            self._forward_returns(),
            quantiles=self.quantiles,
            groupby=self.groups if group_neutral else None,
            group_neutral=group_neutral,
        )

    def quantile_stats(self) -> pd.DataFrame:
        """Return summary statistics by factor quantile."""
        returns = self.quantile_returns()
        result = pd.DataFrame(
            {
                "mean": returns.mean(),
                "std": returns.std(ddof=1),
                "count": returns.count(),
                "cumulative_return": (1.0 + returns).prod() - 1.0,
            }
        )
        result.index.name = "quantile"
        return result

    def quantile_counts(self) -> pd.DataFrame:
        """Return per-date asset counts by factor quantile."""
        aligned, quantiles = self._aligned_quantiles()
        counts = quantiles.groupby([quantiles.index.get_level_values(0), quantiles]).size()
        result = counts.unstack(fill_value=0).sort_index(axis=1)
        result.index.name = self.factor.index.names[0]
        result.columns.name = None
        return result

    def quantile_forward_returns(self) -> pd.DataFrame:
        """Return raw forward return observations with assigned factor quantiles."""
        aligned, quantiles = self._aligned_quantiles()
        return pd.DataFrame(
            {
                "date": aligned.index.get_level_values(0),
                "symbol": aligned.index.get_level_values(1),
                "quantile": quantiles.astype(int).to_numpy(),
                "forward_return": aligned["returns"].to_numpy(),
            },
            index=aligned.index,
        )

    def quantile_decay(self, window: int = 5) -> pd.DataFrame:
        """Return rolling mean returns by factor quantile."""
        if window <= 0:
            raise ValueError("window must be a positive integer")
        return self.quantile_returns().rolling(window, min_periods=1).mean()

    def quantile_cumulative_returns(self) -> pd.DataFrame:
        """Return compounded returns by factor quantile."""
        return (1.0 + self.quantile_returns()).cumprod() - 1.0

    def quantile_spread(self, reverse: bool = False) -> pd.Series:
        """Return top-minus-bottom factor quantile returns."""
        returns = self.quantile_returns()
        bottom = returns.columns.min()
        top = returns.columns.max()
        spread = returns[bottom] - returns[top] if reverse else returns[top] - returns[bottom]
        spread.name = "quantile_spread"
        return spread

    def long_short_returns(self) -> pd.DataFrame:
        """Return long, short, and spread factor returns."""
        returns = self.quantile_returns()
        bottom = returns.columns.min()
        top = returns.columns.max()
        return pd.DataFrame(
            {
                "long": returns[top],
                "short": returns[bottom],
                "spread": self.quantile_spread(),
            }
        )

    def long_short_cumulative_returns(self) -> pd.DataFrame:
        """Return compounded long, short, and spread factor returns."""
        return (1.0 + self.long_short_returns()).cumprod() - 1.0

    def factor_returns(self) -> pd.DataFrame:
        """Return quantile returns derived from configured prices."""
        if self.prices is None:
            return self.quantile_returns()
        return factor_metrics.factor_returns(
            self.factor,
            self.prices,
            quantiles=self.quantiles,
        )

    def monthly_ic_heatmap(self, by_group: bool = False) -> pd.DataFrame:
        """Return mean IC by year and month."""
        return factor_metrics.mean_monthly_ic(self.ic(by_group=by_group))

    @staticmethod
    def event_returns(
        prices: pd.Series,
        events: pd.MultiIndex,
        before: int = 5,
        after: int = 5,
    ) -> pd.DataFrame:
        """Return average event-window returns around ``(date, symbol)`` events."""
        return factor_metrics.event_returns(prices, events, before=before, after=after)

    def turnover(self) -> pd.Series:
        """Return factor rank turnover."""
        return factor_metrics.turnover(self.factor)

    def autocorrelation(self) -> pd.Series:
        """Return factor rank autocorrelation."""
        return factor_metrics.autocorrelation(self.factor)

    def events_distribution(self) -> pd.DataFrame:
        """Return factor observation events for event-count distribution charts."""
        aligned = pd.concat(
            {"factor": self.factor, "returns": self._forward_returns()},
            axis=1,
        ).dropna()
        return pd.DataFrame(
            {
                "date": aligned.index.get_level_values(0),
                "symbol": aligned.index.get_level_values(1),
            },
            index=aligned.index,
        )

    def summary(self) -> dict[str, float]:
        """Return scalar factor diagnostics."""
        ic_values = self.ic()
        rank_ic_values = self.rank_ic()
        turnover_values = self.turnover()
        autocorrelation_values = self.autocorrelation()
        quantile_spread_values = self.quantile_spread()
        return {
            "ic_mean": float(ic_values.mean()),
            "ic_std": float(ic_values.std(ddof=1)),
            "ic_ir": self.ic_ir(),
            "rank_ic_mean": float(rank_ic_values.mean()),
            "quantile_spread_mean": float(quantile_spread_values.mean()),
            "quantile_spread_cumulative_return": float(
                (1.0 + quantile_spread_values).prod() - 1.0
            ),
            "turnover_mean": float(turnover_values.mean()),
            "autocorrelation_mean": float(autocorrelation_values.mean()),
        }

    def monotonicity(self) -> dict[str, float]:
        """Return monotonicity diagnostics for the factor quantile returns.

        Checks whether mean returns increase monotonically from Q1 to Q(n).
        Returns Spearman rank correlation between quantile rank and mean return,
        and a boolean flag indicating perfect monotonicity.
        """
        qr = self.quantile_returns()
        means = qr.mean()
        n = len(means)
        if n < 2:
            return {"spearman_rho": float("nan"), "is_monotone": False}
        ranks = pd.Series(range(1, n + 1), index=means.index, dtype="float64")
        rho = float(means.rank().corr(ranks, method="spearman"))
        diffs = means.diff().dropna()
        is_monotone = bool((diffs > 0).all() or (diffs < 0).all())
        return {"spearman_rho": rho, "is_monotone": is_monotone}

    def plot(self):
        """Return a Bokeh grid of factor diagnostic charts."""
        from bokeh.layouts import column as bk_column

        from tradelearn.report import charts

        items = []
        qs = self.quantile_stats()
        if not qs.empty:
            items.append(charts.factor_quantile_returns_bar(qs))
        qfr = self.quantile_forward_returns()
        if not qfr.empty:
            items.append(charts.factor_quantile_returns_violin(qfr))
        qr = self.quantile_cumulative_returns()
        if not qr.empty:
            items.append(charts.quantile_returns(qr))
        spread = self.quantile_spread()
        if not spread.empty:
            items.append(charts.factor_quantile_spread(spread))
        ls = self.long_short_cumulative_returns()
        if not ls.empty:
            items.append(charts.factor_long_short_returns(ls))
        ic_series = self.ic()
        if not ic_series.empty:
            items.append(charts.factor_ic(ic_series))
            items.append(charts.factor_ic_histogram(ic_series))
            items.append(charts.factor_ic_qq(ic_series))
        ric_series = self.rank_ic()
        if not ric_series.empty:
            items.append(charts.factor_rank_ic(ric_series))
        monthly_ic = self.monthly_ic_heatmap()
        if not monthly_ic.empty:
            items.append(charts.factor_monthly_ic_heatmap(monthly_ic))
        qc = self.quantile_counts()
        if not qc.empty:
            items.append(charts.quantile_counts(qc))
        events = self.events_distribution()
        if not events.empty:
            items.append(charts.factor_events_distribution(events))
        t = self.turnover()
        ac = self.autocorrelation()
        if not t.empty or not ac.empty:
            items.append(charts.factor_turnover(t, ac))
        return bk_column(*items, sizing_mode="stretch_width") if items else None

    def html(self, path: str) -> Path:
        """Write a standalone Alphalens-style factor report and return the output path."""
        from bokeh.embed import components
        from bokeh.resources import INLINE

        grid = self.plot()
        if grid is None:
            raise ValueError("FactorAnalyzer has no data to plot")
        script, chart = components(grid)
        html_content = _render_factor_html(
            title="Tradelearn Factor Analysis",
            summary=self.summary(),
            quantile_stats=self.quantile_stats(),
            quantile_counts=self.quantile_counts(),
            script=script,
            chart=chart,
            bokeh_resources=INLINE.render(),
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content, encoding="utf-8")
        return output

    def report(self, path: str, format: str | None = None) -> Path:
        """Write a factor report using the user-facing report() entrypoint."""
        output = Path(path)
        chosen = (format or output.suffix.lstrip(".") or "html").lower()
        if chosen in {"htm", "html"}:
            if not output.suffix:
                output = output.with_suffix(".html")
            return self.html(str(output))
        raise ValueError(f"Unsupported factor report format: {chosen}")

    def _forward_returns(self) -> pd.Series:
        """Return configured or price-derived forward returns."""
        if self.forward_returns is not None:
            return self.forward_returns
        if self.prices is not None:
            forward = (
                self.prices.groupby(level=1, group_keys=False)
                .pct_change()
                .groupby(level=1)
                .shift(-1)
            )
            forward.name = "forward_returns"
            return forward
        raise ValueError("FactorAnalyzer requires forward_returns or prices")

    def _aligned_quantiles(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return aligned factor/return rows and per-date quantile labels."""
        if self.quantiles <= 0:
            raise ValueError("quantiles must be a positive integer")
        aligned = pd.concat(
            {"factor": self.factor, "returns": self._forward_returns()},
            axis=1,
        ).dropna()

        def _assign_quantiles(frame: pd.Series) -> pd.Series:
            labels = pd.qcut(
                frame.rank(method="first"),
                q=min(self.quantiles, len(frame)),
                labels=False,
            )
            return labels.astype(int) + 1

        quantiles = aligned["factor"].groupby(level=0, group_keys=False).apply(_assign_quantiles)
        return aligned, quantiles


@dataclass(frozen=True)
class MultiPeriodFactorAnalyzer:
    """Thin facade around one ``FactorAnalyzer`` per prediction horizon."""

    analyzers: dict[int, FactorAnalyzer]

    def __post_init__(self) -> None:
        if not self.analyzers:
            raise ValueError("MultiPeriodFactorAnalyzer requires at least one period")
        object.__setattr__(self, "analyzers", dict(sorted(self.analyzers.items())))

    def __getitem__(self, period: int) -> FactorAnalyzer:
        """Return the analyzer for ``period``."""
        return self.analyzers[period]

    def keys(self):
        """Return available prediction horizons."""
        return self.analyzers.keys()

    def values(self):
        """Return analyzers ordered by prediction horizon."""
        return self.analyzers.values()

    def items(self):
        """Return ``(period, analyzer)`` pairs ordered by prediction horizon."""
        return self.analyzers.items()

    def summary(self) -> pd.DataFrame:
        """Return summary metrics indexed by prediction horizon."""
        result = pd.DataFrame(
            {period: analyzer.summary() for period, analyzer in self.analyzers.items()}
        ).T
        result.index.name = "period"
        return result

    def ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return IC series for every prediction horizon."""
        return _period_metric_frame(
            {
                period: analyzer.ic(by_group=by_group)
                for period, analyzer in self.analyzers.items()
            }
        )

    def rank_ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return rank IC series for every prediction horizon."""
        return _period_metric_frame(
            {
                period: analyzer.rank_ic(by_group=by_group)
                for period, analyzer in self.analyzers.items()
            }
        )

    def quantile_returns(self, period: int | None = None) -> pd.DataFrame | dict[int, pd.DataFrame]:
        """Return quantile returns for one or all prediction horizons."""
        if period is not None:
            return self.analyzers[period].quantile_returns()
        return {key: analyzer.quantile_returns() for key, analyzer in self.analyzers.items()}


def _render_factor_html(
    *,
    title: str,
    summary: dict[str, float],
    quantile_stats: pd.DataFrame,
    quantile_counts: pd.DataFrame,
    script: str,
    chart: str,
    bokeh_resources: str,
) -> str:
    """Return a compact Alphalens-style factor report HTML page."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  {bokeh_resources}
  <style>
    :root {{
      --ink: #1f2d33;
      --muted: #65737e;
      --line: #d9e1e6;
      --panel: #f7fafc;
      --accent: #54717b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f3f6f8;
      color: var(--ink);
      font: 13px/1.45 -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    }}
    .shell {{ max-width: 1180px; margin: 28px auto; padding: 0 22px; }}
    header, section {{
      margin: 0 0 24px;
      padding: 22px 26px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: 0 1px 2px rgba(31, 45, 51, .04);
    }}
    header {{ border-top: 4px solid var(--accent); }}
    h1, h2 {{ margin: 0 0 14px; line-height: 1.2; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 24px;
      align-items: start;
    }}
    aside {{ position: sticky; top: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 0; }}
    th, td {{ border: 1px solid var(--line); padding: 7px 9px; text-align: left; }}
    th {{ background: var(--panel); font-weight: 700; }}
    td:last-child, th:last-child {{ text-align: right; }}
    .summary-table th {{ width: 58%; color: var(--muted); }}
    .chart {{ overflow: hidden; }}
    .chart > div, .chart .bk-root, .chart .bk-root > div {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>{escape(title)}</h1>
      <p>Alphalens-style factor diagnostics: quantile returns, long-short spread,
      IC, rank IC, turnover, and monthly IC heatmap.</p>
    </header>
    <div class="grid">
      <div>
        <section class="chart">
          <h2>Information Coefficient and Quantile Charts</h2>
          {chart}
        </section>
      </div>
      <aside>
        <section>
          <h2>Factor Summary</h2>
          {_dict_table(summary, class_name="summary-table")}
        </section>
        <section>
          <h2>Quantile Statistics</h2>
          {_frame_table(quantile_stats)}
        </section>
        <section>
          <h2>Quantile Counts</h2>
          {_frame_table(quantile_counts.tail(10))}
        </section>
      </aside>
    </div>
    {script}
  </main>
</body>
</html>"""


def _dict_table(values: dict[str, float], *, class_name: str = "") -> str:
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(_format_html_value(value))}</td></tr>"
        for key, value in values.items()
    )
    class_attr = f' class="{class_name}"' if class_name else ""
    return f"<table{class_attr}><tbody>{rows}</tbody></table>"


def _frame_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<table><tbody></tbody></table>"
    display = frame.reset_index()
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_html_value(value))}</td>" for value in row)
        + "</tr>"
        for row in display.itertuples(index=False, name=None)
    )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _format_html_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if pd.isna(value):
        return "-"
    return str(value)


def _clean_periods(clean: pd.DataFrame, periods: tuple[int, ...] | None) -> tuple[int, ...]:
    """Return validated periods available in clean factor data."""
    available = sorted(
        int(str(column).removeprefix("forward_return_"))
        for column in clean.columns
        if str(column).startswith("forward_return_")
        and str(column).removeprefix("forward_return_").isdigit()
    )
    if not available:
        raise ValueError("clean factor data must contain forward_return_N columns")
    selected = tuple(available if periods is None else periods)
    missing = [period for period in selected if f"forward_return_{period}" not in clean]
    if missing:
        raise ValueError(f"clean factor data missing forward_return columns for periods: {missing}")
    return selected


def _selected_periods(
    *,
    period: int | None,
    periods: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return validated prediction horizons for dataset-driven analysis."""
    if period is not None:
        selected = (int(period),)
    elif periods is not None:
        selected = tuple(int(item) for item in periods)
    else:
        selected = (1,)
    if not selected or any(item <= 0 for item in selected):
        raise ValueError("periods must contain positive integers")
    return selected


def _dataset_series(
    data: pd.DataFrame,
    column: str,
    *,
    date: str | None,
    symbol: str | None,
    name: str,
) -> pd.Series:
    """Return a ``(date, symbol)`` indexed series from ``data[column]``."""
    if column not in data:
        raise ValueError(f"data must contain column {column!r}")
    index = _dataset_index(data, date=date, symbol=symbol)
    series = pd.Series(data[column].to_numpy(), index=index, name=name)
    return series.sort_index()


def _dataset_groupby(
    data: pd.DataFrame,
    groupby: str | pd.Series | None,
    *,
    date: str | None,
    symbol: str | None,
) -> pd.Series | None:
    """Return optional group labels aligned to dataset index."""
    if groupby is None:
        return None
    index = _dataset_index(data, date=date, symbol=symbol)
    if isinstance(groupby, str):
        if groupby not in data:
            raise ValueError(f"data must contain groupby column {groupby!r}")
        return pd.Series(data[groupby].to_numpy(), index=index, name="group").sort_index()
    return pd.Series(groupby, name="group").sort_index()


def _dataset_index(
    data: pd.DataFrame,
    *,
    date: str | None,
    symbol: str | None,
) -> pd.MultiIndex:
    """Return a normalized ``(date, symbol)`` MultiIndex for factor data."""
    if isinstance(data.index, pd.MultiIndex) and data.index.nlevels >= 2:
        dates = pd.to_datetime(data.index.get_level_values(0))
        symbols = data.index.get_level_values(1)
    else:
        date_col = date or ("date" if "date" in data else None)
        symbol_col = symbol or (
            "symbol" if "symbol" in data else "ticker" if "ticker" in data else None
        )
        if date_col is None or symbol_col is None:
            raise ValueError(
                "data must use a MultiIndex or provide date and symbol/ticker columns"
            )
        dates = pd.to_datetime(data[date_col])
        symbols = data[symbol_col]
    return pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])


def _period_metric_frame(
    values: dict[int, pd.Series | pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-period Series metrics into a period-column frame."""
    if all(isinstance(value, pd.Series) for value in values.values()):
        result = pd.concat(values, axis=1)
        result.columns.name = "period"
        return result
    frames = {
        period: value.stack() if isinstance(value, pd.DataFrame) else value
        for period, value in values.items()
    }
    result = pd.concat(frames, axis=1)
    result.columns.name = "period"
    return result
