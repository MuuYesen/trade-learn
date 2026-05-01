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
        counts = quantiles.groupby([quantiles.index.get_level_values(0), quantiles]).size()
        result = counts.unstack(fill_value=0).sort_index(axis=1)
        result.index.name = self.factor.index.names[0]
        result.columns.name = None
        return result

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
        qr = self.quantile_cumulative_returns()
        if not qr.empty:
            items.append(charts.quantile_returns(qr))
        ls = self.long_short_cumulative_returns()
        if not ls.empty:
            items.append(charts.factor_long_short_returns(ls))
        ic_series = self.ic()
        if not ic_series.empty:
            items.append(charts.factor_ic(ic_series))
        ric_series = self.rank_ic()
        if not ric_series.empty:
            items.append(charts.factor_rank_ic(ric_series))
        monthly_ic = self.monthly_ic_heatmap()
        if not monthly_ic.empty:
            items.append(charts.factor_monthly_ic_heatmap(monthly_ic))
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
