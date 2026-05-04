"""Alphalens-style factor analysis facade backed by tradelearn.metrics."""

from __future__ import annotations

from datetime import datetime
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

    def __post_init__(self) -> None:
        """Promote single-index data to MultiIndex to support single-asset analysis."""
        for attr in ("factor", "forward_returns", "prices", "groups"):
            val = getattr(self, attr)
            if val is not None and not isinstance(val.index, pd.MultiIndex):
                # Add a dummy level to trick multi-index validation
                new_idx = pd.MultiIndex.from_arrays(
                    [val.index, ["ASSET"] * len(val)],
                    names=[val.index.name or "date", "symbol"]
                )
                object.__setattr__(self, attr, val.copy().set_axis(new_idx))

    @classmethod
    def from_clean_factor_data(
        cls,
        clean: pd.DataFrame,
        *,
        periods: tuple[int, ...] | None = None,
        annualization_periods: int = 252,
        quantiles: int = 5,
    ) -> MultiPeriodFactorAnalyzer | MultiFactorAnalyzer:
        """Return an analysis object built from clean factor data.

        ``clean`` is the frame produced by ``clean_factor_and_forward_returns``.
        A ``factor_name`` column produces one multi-period analyzer per factor.
        """
        if "factor_name" in clean:
            analyzers = {
                str(factor_name): cls.from_clean_factor_data(
                    frame.drop(columns=["factor_name"]),
                    periods=periods,
                    annualization_periods=annualization_periods,
                    quantiles=quantiles,
                )
                for factor_name, frame in clean.groupby("factor_name", sort=True)
            }
            return MultiFactorAnalyzer(analyzers)
        if "factor" not in clean:
            raise ValueError("clean factor data must contain a 'factor' column")
        selected = _clean_periods(clean, periods)
        factor = pd.Series(clean["factor"], index=clean.index, name="factor")
        groups = (
            pd.Series(clean["group"], index=clean.index, name="group")
            if "group" in clean
            else None
        )
        analyzers = {
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
        return MultiPeriodFactorAnalyzer(analyzers)

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
        analyzer = cls.from_clean_factor_data(
            clean,
            periods=periods,
            annualization_periods=annualization_periods,
            quantiles=quantiles,
        )
        return analyzer.summary()

    def ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return per-date Pearson information coefficient."""
        # Fallback for single-asset: Use rolling correlation instead of cross-sectional
        if self.factor.index.get_level_values(1).nunique() <= 1:
            # Single asset: calculate rolling 20-day correlation
            res = (
                self.factor
                .rolling(window=20)
                .corr(self._forward_returns())
                .dropna()
            )
            res.index = res.index.get_level_values(0)
            res.name = "ic"
            return res
        
        return factor_metrics.ic(
            self.factor,
            self._forward_returns(),
            groupby=self.groups,
            by_group=by_group,
        )

    def rank_ic(self, by_group: bool = False) -> pd.Series | pd.DataFrame:
        """Return per-date Spearman rank information coefficient."""
        # Fallback for single-asset: Use rolling rank correlation
        if self.factor.index.get_level_values(1).nunique() <= 1:
            y = self._forward_returns()
            res = (
                self.factor
                .rolling(window=20)
                .apply(lambda x: x.rank().corr(y.loc[x.index].rank()))
                .dropna()
            )
            res.index = res.index.get_level_values(0)
            res.name = "rank_ic"
            return res
            
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
            title="TradeLearn Factor Analysis",
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
        print("Factor Analysis Report...", end="", flush=True)
        output = Path(path)
        chosen = (format or output.suffix.lstrip(".") or "html").lower()
        if chosen in {"htm", "html"}:
            if not output.suffix:
                output = output.with_suffix(".html")
            result = self.html(str(output))
            print(f" Done ✓  →  {result}")
            return result
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

    def plot(self, period: int | None = None):
        """Return factor diagnostic charts for a selected prediction horizon."""
        return self._selected(period).plot()

    def html(self, path: str, period: int | None = None) -> Path:
        """Write a standalone factor report for a selected prediction horizon."""
        if period is None and len(self.analyzers) > 1:
            return self._multi_period_html(path)
        return self._selected(period).html(path)

    def report(self, path: str, format: str | None = None, period: int | None = None) -> Path:
        """Write a factor report for a selected prediction horizon."""
        print("Multi-Period Factor Analysis Report...", end="", flush=True)
        output = Path(path)
        chosen = (format or output.suffix.lstrip(".") or "html").lower()
        if chosen in {"htm", "html"}:
            if not output.suffix:
                output = output.with_suffix(".html")
            result = self.html(str(output), period=period)
            print(f" Done ✓  →  {result}")
            return result
        raise ValueError(f"Unsupported factor report format: {chosen}")

    def _selected(self, period: int | None) -> FactorAnalyzer:
        """Return an analyzer by period, defaulting only when the choice is unambiguous."""
        if period is not None:
            return self.analyzers[period]
        if len(self.analyzers) == 1:
            return next(iter(self.analyzers.values()))
        raise ValueError("period must be provided when multiple factor horizons are available")

    def _multi_period_html(self, path: str) -> Path:
        """Write one HTML report containing all configured factor horizons."""
        from bokeh.embed import components
        from bokeh.layouts import column as bk_column
        from bokeh.models import Div
        from bokeh.resources import INLINE

        sections = []
        for period, analyzer in self.analyzers.items():
            grid = analyzer.plot()
            if grid is not None:
                sections.append(
                    bk_column(
                        Div(
                            text=f"<h2>{period}-bar Forward Return</h2>",
                            sizing_mode="stretch_width",
                        ),
                        grid,
                        sizing_mode="stretch_width",
                    )
                )
        chart = bk_column(*sections, sizing_mode="stretch_width") if sections else None
        if chart is None:
            raise ValueError("FactorAnalyzer has no data to plot")
        script, rendered_chart = components(chart)
        html_content = _render_factor_html(
            title="Tradelearn Multi-Period Factor Analysis",
            summary={},
            summary_frame=self.summary(),
            quantile_stats=pd.DataFrame(),
            quantile_counts=pd.DataFrame(),
            script=script,
            chart=rendered_chart,
            bokeh_resources=INLINE.render(),
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content, encoding="utf-8")
        return output


@dataclass(frozen=True)
class MultiFactorAnalyzer:
    """Thin facade around one multi-period analyzer per factor."""

    analyzers: dict[str, MultiPeriodFactorAnalyzer]

    def __post_init__(self) -> None:
        if not self.analyzers:
            raise ValueError("MultiFactorAnalyzer requires at least one factor")
        object.__setattr__(self, "analyzers", dict(sorted(self.analyzers.items())))

    def __getitem__(self, factor: str) -> MultiPeriodFactorAnalyzer:
        """Return the analyzer for ``factor``."""
        return self.analyzers[factor]

    def keys(self):
        """Return available factor names."""
        return self.analyzers.keys()

    def values(self):
        """Return analyzers ordered by factor name."""
        return self.analyzers.values()

    def items(self):
        """Return ``(factor, analyzer)`` pairs ordered by factor name."""
        return self.analyzers.items()

    def summary(self) -> pd.DataFrame:
        """Return summary metrics indexed by ``(factor, period)``."""
        result = pd.concat(
            {factor: analyzer.summary() for factor, analyzer in self.analyzers.items()},
            names=["factor", "period"],
        )
        return result

    def ic(self, by_group: bool = False) -> pd.DataFrame:
        """Return IC series with ``(factor, period)`` columns."""
        result = pd.concat(
            {
                factor: analyzer.ic(by_group=by_group)
                for factor, analyzer in self.analyzers.items()
            },
            axis=1,
        )
        result.columns.names = ["factor", "period"]
        return result

    def rank_ic(self, by_group: bool = False) -> pd.DataFrame:
        """Return rank IC series with ``(factor, period)`` columns."""
        result = pd.concat(
            {
                factor: analyzer.rank_ic(by_group=by_group)
                for factor, analyzer in self.analyzers.items()
            },
            axis=1,
        )
        result.columns.names = ["factor", "period"]
        return result

    def report(self, path: str, format: str | None = None) -> Path:
        """Write a multi-factor report using the user-facing report() entrypoint."""
        print("Multi-Factor Analysis Report...", end="", flush=True)
        output = Path(path)
        chosen = (format or output.suffix.lstrip(".") or "html").lower()
        if chosen in {"htm", "html"}:
            if not output.suffix:
                output = output.with_suffix(".html")
            result = self.html(str(output))
            print(f" Done ✓  →  {result}")
            return result
        raise ValueError(f"Unsupported factor report format: {chosen}")

    def html(self, path: str) -> Path:
        """Write one HTML report containing every factor and prediction horizon."""
        from bokeh.embed import components
        from bokeh.layouts import column as bk_column
        from bokeh.models import Div
        from bokeh.resources import INLINE

        sections = []
        for factor, analyzer in self.analyzers.items():
            for period, period_analyzer in analyzer.items():
                grid = period_analyzer.plot()
                if grid is not None:
                    sections.append(
                        bk_column(
                            Div(
                                text=f"<h2>{escape(factor)} · {period}-bar Forward Return</h2>",
                                sizing_mode="stretch_width",
                            ),
                            grid,
                            sizing_mode="stretch_width",
                        )
                    )
        chart = bk_column(*sections, sizing_mode="stretch_width") if sections else None
        if chart is None:
            raise ValueError("FactorAnalyzer has no data to plot")
        script, rendered_chart = components(chart)
        html_content = _render_factor_html(
            title="Tradelearn Multi-Factor Analysis",
            summary={},
            summary_frame=self.summary(),
            quantile_stats=pd.DataFrame(),
            quantile_counts=pd.DataFrame(),
            script=script,
            chart=rendered_chart,
            bokeh_resources=INLINE.render(),
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_content, encoding="utf-8")
        return output


def _render_factor_html(
    *,
    title: str,
    summary: dict[str, float],
    summary_frame: pd.DataFrame | None = None,
    quantile_stats: pd.DataFrame,
    quantile_counts: pd.DataFrame,
    script: str,
    chart: str,
    bokeh_resources: str,
) -> str:
    """Return a premium Alphalens-style factor report HTML page."""
    summary_title = "Period Summary" if summary_frame is not None else "Factor Summary"
    summary_table = (
        _frame_table(summary_frame)
        if summary_frame is not None
        else _dict_table(summary, class_name="summary-table")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  {bokeh_resources}
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --accent: #334155;
      --row-hover: #f1f5f9;
      --brand: #4f46e5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .shell {{ max-width: 1400px; margin: 40px auto; padding: 0 32px; }}
    header {{
      margin: 0 0 32px;
      padding: 32px 40px;
      background: var(--card);
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      border-top: 6px solid var(--brand);
      background: linear-gradient(to bottom right, #ffffff, #fcfdff);
    }}
    .report-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 32px;
    }}
    .eyebrow {{
      color: var(--brand);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .15em;
      margin: 0 0 8px;
      text-transform: uppercase;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(140px, auto));
      gap: 8px 40px;
      padding-bottom: 4px;
    }}
    .meta-label {{ color: var(--muted); font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 2px; }}
    .meta-value {{ color: var(--accent); font-size: 13px; font-weight: 600; white-space: nowrap; }}
    section {{
      margin: 0 0 32px;
      padding: 32px;
      background: var(--card);
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    h1 {{ margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.025em; color: var(--ink); }}
    h2 {{ margin: 0 0 20px; font-size: 18px; font-weight: 700; color: var(--accent); border-bottom: 2px solid var(--row-hover); padding-bottom: 12px; }}
    p {{ margin: 0; color: var(--muted); font-size: 15px; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 32px;
      align-items: start;
    }}
    aside {{ position: sticky; top: 32px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0; min-width: 600px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--line); }}
    th {{ background: var(--bg); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }}
    tr:hover {{ background: var(--row-hover); }}
    td {{ font-variant-numeric: tabular-nums; }}
    td:last-child, th:last-child {{ text-align: right; }}
    .summary-table {{ min-width: 0; }}
    .summary-table th {{ width: 55%; color: var(--accent); background: none; text-transform: none; font-size: 14px; letter-spacing: normal; padding-left: 0; }}
    .chart {{ overflow: hidden; }}
    .chart > div, .chart .bk-root, .chart .bk-root > div {{
      width: 100% !important;
      max-width: 100% !important;
    }}
    .full-width-tables {{ margin-top: 32px; }}
    .scroll-container {{ overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 8px; border: 1px solid var(--line); }}
    @media (max-width: 1100px) {{
      .grid {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .report-header {{ flex-direction: column; align-items: flex-start; gap: 16px; }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div class="report-header">
        <div>
          <p class="eyebrow">TradeLearn Report</p>
          <h1>{escape(title)}</h1>
        </div>
        <div class="meta-grid">
          <div class="meta-item">
            <div class="meta-label">Generated</div>
            <div class="meta-value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
          </div>
          <div class="meta-item">
            <div class="meta-label">Factor Type</div>
            <div class="meta-value">Alphalens-style</div>
          </div>
        </div>
      </div>
    </header>
    
    <div class="grid">
      <div class="main-content">
        <section class="chart">
          <h2>Information Coefficient and Quantile Charts</h2>
          {chart}
        </section>
      </div>
      <aside>
        <section>
          <h2>{summary_title}</h2>
          {summary_table}
        </section>
      </aside>
    </div>

    <div class="full-width-tables">
      <section>
        <h2>Quantile Statistics</h2>
        <p style="margin-bottom: 16px; font-size: 13px;">Statistical summary of forward returns for each factor quantile. This table shows the performance distribution and risk metrics across groups.</p>
        <div class="scroll-container">
          {_frame_table(quantile_stats)}
        </div>
      </section>
      <section>
        <h2>Quantile Counts</h2>
        <p style="margin-bottom: 16px; font-size: 13px;">The number of assets assigned to each quantile at each point in time. This helps verify data density and ensure balanced quantile allocation across the sample period.</p>
        <div class="scroll-container">
          {_frame_table(quantile_counts.tail(10))}
        </div>
      </section>
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
    import numpy as np
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return "—"
        return f"{value:.4f}"
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
