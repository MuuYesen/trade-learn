"""Public API inventory tests for metrics."""

import tradelearn.metrics as metrics


def test_metrics_public_api_matches_documented_inventory() -> None:
    """The metrics facade exposes the documented returns/risk/factor/trade groups."""
    expected_returns = {
        "annual_return",
        "cum_returns",
        "excess_returns",
        "log_to_simple",
        "simple_returns",
    }
    expected_risk = {
        "alpha",
        "beta",
        "calmar",
        "cvar",
        "downside_risk",
        "drawdown_series",
        "information_ratio",
        "max_drawdown",
        "omega",
        "sharpe",
        "sortino",
        "tail_ratio",
        "var",
        "volatility",
    }
    expected_factor = {
        "autocorrelation",
        "factor_returns",
        "ic",
        "ic_ir",
        "quantile_returns",
        "rank_ic",
        "turnover",
    }
    expected_trade = {
        "avg_loss",
        "avg_win",
        "expectancy",
        "max_consecutive_losses",
        "max_consecutive_wins",
        "profit_factor",
        "win_rate",
    }

    expected = expected_returns | expected_risk | expected_factor | expected_trade

    assert set(metrics.__all__) == expected
    assert len(metrics.__all__) == 33
