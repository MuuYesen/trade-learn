"""Replay the Alpha101 Top-5 weights through tradelearn Lite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import tradelearn.lite as tl
from tradelearn.data import TradingViewProvider
from tradelearn.factor import FactorAnalyzer, alpha101, clean_factor_and_forward_returns
from tradelearn.research import ResearchResult
from tradelearn.research.portfolio import topk_equal_weights
from tradelearn.research.preprocess import StandardScaler

START = "2020-01-01"
END = "2025-12-31"
FORWARD_PERIOD = 5
REBALANCE_EVERY = 5
TOP_K = 5

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "alpha101_us_tech"
CASH = 100_000.0
COMMISSION = 0.0003

SYMBOLS = [
    "NASDAQ:AAPL",
    "NASDAQ:MSFT",
    "NASDAQ:NVDA",
    "NASDAQ:AMZN",
    "NASDAQ:GOOGL",
    "NASDAQ:META",
    "NASDAQ:TSLA",
    "NASDAQ:AVGO",
    "NASDAQ:AMD",
    "NASDAQ:NFLX",
    "NYSE:ORCL",
    "NYSE:CRM",
    "NASDAQ:ADBE",
    "NASDAQ:COST",
    "NASDAQ:QCOM",
]


class Alpha101TopKLite(tl.Strategy):
    """Execute precomputed Alpha101 Top-K target weights."""

    def init(self) -> None:
        self.start_on_bar(0)

    def next(self) -> None:
        if not self.research_result.weights.has_current():
            return
        weights = self.research_result.weights[0]
        if weights is None or len(weights) == 0:
            return
        self.target_weights(weights, close_missing=True)


def main() -> None:
    provider = TradingViewProvider(n_bars=2200)
    bars = provider.history_ohlc(
        SYMBOLS,
        start=START,
        end=END,
        freq="1d",
    )
    factors = alpha101(bars).replace([np.inf, -np.inf], np.nan)
    factor_columns = tuple(column for column in factors.columns if column not in {"date", "symbol"})
    clean = clean_factor_and_forward_returns(
        factors,
        factor=factor_columns,
        prices=bars["close"].sort_index(),
        periods=(FORWARD_PERIOD,),
        quantiles=5,
    )
    ranking = rank_factors(clean)
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    selected = ranking.head(TOP_K).copy()

    print(ranking.head(10).to_markdown(index=False, floatfmt=".4f"))

    weights = build_target_weights(factors, selected)
    result = ResearchResult(name="alpha101-us-tech-top5", weights=weights)

    backtest = tl.Backtest(
        bars,
        Alpha101TopKLite,
        cash=CASH,
        commission=COMMISSION,
        trade_on_close=True,
    )
    stats = backtest.run(research_result=result)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = backtest.report(str(OUTPUT_DIR / "alpha101_top5_lite_report.html"))
    print(stats.summary)
    print(f"Report: {report_path}")


def build_target_weights(
    factors: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    gross: float = 1.0,
    rebalance_every: int = REBALANCE_EVERY,
) -> pd.Series:
    scores = build_composite_scores(factors, selected)
    selected_dates = scores.index.get_level_values("date").unique()[::rebalance_every]
    scores = scores[scores.index.get_level_values("date").isin(selected_dates)]
    return topk_equal_weights(scores, k=len(selected), gross=gross)


def build_composite_scores(factors: pd.DataFrame, selected: pd.DataFrame) -> pd.Series:
    direction = pd.Series(selected["direction"].to_numpy(), index=selected["column"])
    scores = factors.set_index(["date", "symbol"])[direction.index].mul(direction, axis=1)
    return StandardScaler(by="date", ddof=0).fit_transform(scores).mean(axis=1).rename("score")


def rank_factors(clean: pd.DataFrame) -> pd.DataFrame:
    analyzer = FactorAnalyzer.from_clean_factor_data(
        clean,
        periods=(FORWARD_PERIOD,),
        annualization_periods=252 // FORWARD_PERIOD,
        quantiles=5,
    )
    ranking = analyzer.summary().reset_index().query("period == @FORWARD_PERIOD").copy()

    print(ranking.head(10).to_markdown(index=False, floatfmt=".4f"))

    ranking["column"] = ranking["factor"]
    ranking["factor"] = ranking["column"].str.replace("_101", "", regex=False)
    ranking["direction"] = np.where(ranking["rank_ic_mean"] >= 0, 1, -1)
    ranking["q5_q1_mean"] = ranking["mean_returns_spread_mean"] * ranking["direction"]
    ranking["q5_q1_annualized"] = (
        ranking["mean_returns_spread_annualized"] * ranking["direction"]
    )
    ranking["q5_q1_cumulative_return"] = ranking["mean_returns_spread_cumulative_return"]
    ranking["monotonicity"] = ranking["monotonicity"] * ranking["direction"]
    ranking["turnover"] = ranking["quantile_turnover_mean"]
    ranking["score"] = factor_score(ranking)
    return ranking


def factor_score(ranking: pd.DataFrame) -> pd.Series:
    rank_ic_mean = ranking["rank_ic_mean"].abs()
    rank_ic_ir = ranking["rank_ic_ir"].fillna(0.0)
    q5_q1 = ranking["q5_q1_annualized"].fillna(0.0)
    monotonicity = ranking["monotonicity"].fillna(0.0)
    turnover = ranking["turnover"].fillna(ranking["turnover"].median())
    return (
        (rank_ic_mean - rank_ic_mean.mean()) / rank_ic_mean.std(ddof=0)
        + (rank_ic_ir - rank_ic_ir.mean()) / rank_ic_ir.std(ddof=0)
        + (q5_q1 - q5_q1.mean()) / q5_q1.std(ddof=0)
        + (monotonicity - monotonicity.mean()) / monotonicity.std(ddof=0)
        - (turnover - turnover.mean()) / turnover.std(ddof=0)
    )


if __name__ == "__main__":
    main()
