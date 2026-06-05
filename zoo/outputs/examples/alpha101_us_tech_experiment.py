"""Run the Alpha101 US tech-stock evaluation used by the zoo article."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradelearn.data import TradingViewProvider
from tradelearn.factor import (
    FactorAnalyzer,
    alpha101,
    clean_factor_and_forward_returns,
)

START = "2020-01-01"
END = "2025-12-31"
FORWARD_PERIOD = 5
REBALANCE_EVERY = 5
TOP_K = 5

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


def main() -> None:
    provider = TradingViewProvider(n_bars=2200)
    bars = provider.history_ohlc(SYMBOLS, start=START, end=END, freq="1d")
    prices = bars["close"].sort_index()

    print("Computing Alpha101 factors...")
    factors = alpha101(bars).replace([np.inf, -np.inf], np.nan)

    print("Ranking factors...")
    factor_columns = tuple(column for column in factors.columns if column not in {"date", "symbol"})
    clean = clean_factor_and_forward_returns(
        factors,
        factor=factor_columns,
        prices=prices,
        periods=(FORWARD_PERIOD,),
        quantiles=5,
    )
    ranking = rank_factors(clean)
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    selected = ranking.head(TOP_K).copy()

    print(f"Selected factors: {', '.join(selected['factor'])}")
    print(
        ranking.head(10)[
            [
                "factor",
                "rank_ic_mean",
                "rank_ic_ir",
                "q5_q1_annualized",
                "monotonicity",
                "turnover",
                "score",
            ]
        ].to_markdown(index=False, floatfmt=".4f")
    )


def rank_factors(clean: pd.DataFrame) -> pd.DataFrame:
    analyzer = FactorAnalyzer.from_clean_factor_data(
        clean,
        periods=(FORWARD_PERIOD,),
        annualization_periods=252 // FORWARD_PERIOD,
        quantiles=5,
    )
    ranking = analyzer.summary().reset_index().query("period == @FORWARD_PERIOD").copy()
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
