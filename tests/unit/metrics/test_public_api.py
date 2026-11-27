"""Public API inventory tests for metrics."""

import tradelearn.metrics as metrics


def test_metrics_public_api_matches_documented_inventory() -> None:
    """The metrics facade exposes the documented returns/risk/factor/trade groups."""
    expected_returns = {
        "cash_flow_return",
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
        "quantile_turnover",
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

    expected_inference = {
        "sharpe_se",
        "hac_mean_se",
        "paired_sharpe_diff",
        "probabilistic_sharpe",
        "deflated_sharpe",
        "minimum_track_record_length",
    }
    expected = expected_returns | expected_risk | expected_factor | expected_trade
    expected |= expected_inference

    assert set(metrics.__all__) == expected
    assert len(metrics.__all__) == len(expected)
