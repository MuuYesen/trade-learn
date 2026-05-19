from __future__ import annotations

import pandas as pd
import pytest

from tradelearn.factor import FactorAnalyzer, alpha101, clean_factor_and_forward_returns
from tradelearn.metrics.factor import quantile_turnover


def test_factor_analyzer_matches_alphalens_core_metrics() -> None:
    alphalens_perf = _alphalens_performance()
    clean = _clean_factor_data()
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1,), quantiles=2)[1]
    alphalens_clean = _alphalens_clean(clean)

    expected_rank_ic = alphalens_perf.factor_information_coefficient(alphalens_clean)["1D"]
    expected_rank_ic.name = "factor_information_coefficient"
    pd.testing.assert_series_equal(analyzer.factor_information_coefficient(), expected_rank_ic, check_freq=False)

    expected_mean_returns, expected_std_error = alphalens_perf.mean_return_by_quantile(
        alphalens_clean,
        by_date=True,
        demeaned=False,
        group_adjust=False,
    )
    expected_quantile_returns = expected_mean_returns["1D"].unstack("factor_quantile")
    expected_quantile_returns.columns.name = None
    pd.testing.assert_frame_equal(
        analyzer.mean_return_by_quantile(),
        expected_quantile_returns,
        check_freq=False,
    )

    expected_spread, expected_spread_error = alphalens_perf.compute_mean_returns_spread(
        expected_mean_returns,
        upper_quant=2,
        lower_quant=1,
        std_err=expected_std_error,
    )
    expected_spread = expected_spread["1D"]
    expected_spread.name = "mean_returns_spread"
    pd.testing.assert_series_equal(analyzer.compute_mean_returns_spread()[0], expected_spread, check_freq=False)
    assert expected_spread_error["1D"].notna().all()

    expected_autocorrelation = alphalens_perf.factor_rank_autocorrelation(
        alphalens_clean,
        period=1,
    ).dropna()
    expected_autocorrelation.name = "factor_rank_autocorrelation"
    pd.testing.assert_series_equal(
        analyzer.factor_rank_autocorrelation(),
        expected_autocorrelation,
        check_freq=False,
    )

    expected_turnover = alphalens_perf.quantile_turnover(
        alphalens_clean["factor_quantile"],
        quantile=2,
        period=1,
    ).dropna()
    pd.testing.assert_series_equal(
        analyzer.quantile_turnover(quantile=2),
        expected_turnover,
        check_freq=False,
    )


def test_factor_analyzer_matches_alphalens_extended_metrics() -> None:
    alphalens_perf = _alphalens_performance()
    clean = _clean_factor_data()
    analyzer = FactorAnalyzer.from_clean_factor_data(clean, periods=(1,), quantiles=2)[1]
    alphalens_clean = _alphalens_clean(clean)

    expected_mean_ic = alphalens_perf.mean_information_coefficient(
        alphalens_clean,
        by_time="M",
    )
    pd.testing.assert_frame_equal(
        analyzer.mean_information_coefficient(by_time="M"),
        expected_mean_ic,
        check_freq=False,
    )

    expected_mean_returns, expected_std_error = alphalens_perf.mean_return_by_quantile(
        alphalens_clean,
        by_date=True,
        demeaned=False,
        group_adjust=False,
    )
    expected_std_error = expected_std_error["1D"].unstack("factor_quantile")
    expected_std_error.columns.name = None
    pd.testing.assert_frame_equal(
        analyzer.mean_return_by_quantile_std_error(),
        expected_std_error,
        check_freq=False,
    )

    expected_spread, expected_spread_error = alphalens_perf.compute_mean_returns_spread(
        expected_mean_returns,
        upper_quant=2,
        lower_quant=1,
        std_err=alphalens_perf.mean_return_by_quantile(
            alphalens_clean,
            by_date=True,
            demeaned=False,
            group_adjust=False,
        )[1],
    )
    expected_spread = expected_spread["1D"]
    expected_spread.name = "mean_returns_spread"
    expected_spread_error = expected_spread_error["1D"]
    expected_spread_error.name = "mean_returns_spread_std_error"

    spread, spread_error = analyzer.compute_mean_returns_spread()
    pd.testing.assert_series_equal(spread, expected_spread, check_freq=False)
    pd.testing.assert_series_equal(spread_error, expected_spread_error, check_freq=False)

    expected_factor_returns = alphalens_perf.factor_returns(
        alphalens_clean,
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    )
    pd.testing.assert_frame_equal(
        analyzer.factor_returns(),
        expected_factor_returns,
        check_freq=False,
    )

    expected_cumulative = alphalens_perf.factor_cumulative_returns(
        alphalens_clean,
        period="1D",
        long_short=True,
        group_neutral=False,
        equal_weight=False,
    )
    expected_cumulative.name = "factor_cumulative_returns"
    pd.testing.assert_series_equal(
        analyzer.factor_cumulative_returns(),
        expected_cumulative,
        check_freq=False,
    )

    expected_average = alphalens_perf.average_cumulative_return_by_quantile(
        alphalens_clean,
        _returns_frame(),
        periods_before=1,
        periods_after=1,
        demeaned=True,
    )
    pd.testing.assert_frame_equal(
        analyzer.average_cumulative_return_by_quantile(
            _returns_frame(),
            periods_before=1,
            periods_after=1,
            demeaned=True,
        ),
        expected_average,
        check_freq=False,
    )


def test_alpha101_clean_data_matches_alphalens_extended_metrics() -> None:
    alphalens_perf = _alphalens_performance()
    clean, returns = _alpha101_clean_factor_data()
    analyzer = FactorAnalyzer.from_clean_factor_data(
        clean,
        periods=(1, 5, 10),
        quantiles=5,
    )
    alphalens_clean = _alphalens_clean_multi_period(clean)

    expected_ic = alphalens_perf.factor_information_coefficient(alphalens_clean)
    actual_ic = analyzer.factor_information_coefficient().rename(columns={1: "1D", 5: "5D", 10: "10D"})
    actual_ic.columns.name = None
    pd.testing.assert_frame_equal(actual_ic, expected_ic, check_freq=False)

    expected_mean_ic = alphalens_perf.mean_information_coefficient(
        alphalens_clean,
        by_time="M",
    )
    actual_mean_ic = actual_ic.groupby(pd.Grouper(freq="M")).mean()
    pd.testing.assert_frame_equal(actual_mean_ic, expected_mean_ic, check_freq=False)

    period_analyzer = analyzer[1]
    expected_mean_returns, expected_std_error = alphalens_perf.mean_return_by_quantile(
        alphalens_clean[["factor", "factor_quantile", "1D"]],
        by_date=True,
        demeaned=False,
        group_adjust=False,
    )
    expected_std_error = expected_std_error["1D"].unstack("factor_quantile")
    expected_std_error.columns.name = None
    pd.testing.assert_frame_equal(
        period_analyzer.mean_return_by_quantile_std_error(),
        expected_std_error,
        check_freq=False,
    )

    expected_spread, expected_spread_error = alphalens_perf.compute_mean_returns_spread(
        expected_mean_returns,
        upper_quant=5,
        lower_quant=1,
        std_err=alphalens_perf.mean_return_by_quantile(
            alphalens_clean[["factor", "factor_quantile", "1D"]],
            by_date=True,
            demeaned=False,
            group_adjust=False,
        )[1],
    )
    expected_spread = expected_spread["1D"]
    expected_spread.name = "mean_returns_spread"
    expected_spread_error = expected_spread_error["1D"]
    expected_spread_error.name = "mean_returns_spread_std_error"
    spread, spread_error = period_analyzer.compute_mean_returns_spread()
    pd.testing.assert_series_equal(spread, expected_spread, check_freq=False)
    pd.testing.assert_series_equal(spread_error, expected_spread_error, check_freq=False)

    expected_turnover = alphalens_perf.quantile_turnover(
        alphalens_clean["factor_quantile"],
        quantile=5,
        period=1,
    ).dropna()
    pd.testing.assert_series_equal(
        period_analyzer.quantile_turnover(quantile=5),
        expected_turnover,
        check_freq=False,
    )

    expected_autocorr = alphalens_perf.factor_rank_autocorrelation(
        alphalens_clean,
        period=1,
    ).dropna()
    expected_autocorr.name = "factor_rank_autocorrelation"
    pd.testing.assert_series_equal(
        period_analyzer.factor_rank_autocorrelation(),
        expected_autocorr,
        check_freq=False,
    )

    expected_factor_returns = alphalens_perf.factor_returns(
        alphalens_clean[["factor", "factor_quantile", "1D"]],
        demeaned=True,
        group_adjust=False,
        equal_weight=False,
    )
    pd.testing.assert_frame_equal(
        period_analyzer.factor_returns(),
        expected_factor_returns,
        check_freq=False,
    )

    expected_cumulative = alphalens_perf.factor_cumulative_returns(
        alphalens_clean[["factor", "factor_quantile", "1D"]],
        period="1D",
        long_short=True,
        group_neutral=False,
        equal_weight=False,
    )
    expected_cumulative.name = "factor_cumulative_returns"
    pd.testing.assert_series_equal(
        period_analyzer.factor_cumulative_returns(),
        expected_cumulative,
        check_freq=False,
    )

    expected_average = alphalens_perf.average_cumulative_return_by_quantile(
        alphalens_clean[["factor", "factor_quantile", "1D"]],
        returns,
        periods_before=2,
        periods_after=2,
        demeaned=True,
    )
    pd.testing.assert_frame_equal(
        period_analyzer.average_cumulative_return_by_quantile(
            returns,
            periods_before=2,
            periods_after=2,
            demeaned=True,
        ),
        expected_average,
        check_freq=False,
    )


def test_quantile_turnover_matches_alphalens() -> None:
    alphalens_perf = _alphalens_performance()
    clean = _alphalens_clean(_clean_factor_data())

    result = quantile_turnover(clean["factor_quantile"], quantile=1)
    expected = alphalens_perf.quantile_turnover(
        clean["factor_quantile"],
        quantile=1,
    ).dropna()

    pd.testing.assert_series_equal(result, expected, check_freq=False)


def _clean_factor_data() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4, freq="D"), ["AAA", "BBB", "CCC", "DDD"]],
        names=["date", "symbol"],
    )
    return pd.DataFrame(
        {
            "factor": [
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                3.0,
                2.0,
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                2.0,
                1.0,
                4.0,
                3.0,
            ],
            "forward_return_1": [
                0.01,
                0.02,
                0.03,
                0.04,
                0.04,
                0.03,
                0.02,
                0.01,
                0.00,
                0.01,
                0.02,
                0.03,
                0.02,
                0.01,
                0.04,
                0.03,
            ],
            "factor_quantile": [1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2],
        },
        index=index,
    )


def _returns_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.00, 0.01],
            "BBB": [0.00, 0.01, 0.02, 0.01],
            "CCC": [0.02, 0.00, 0.01, 0.03],
            "DDD": [0.01, 0.03, 0.00, 0.02],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )


def _alpha101_clean_factor_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    symbols = [f"S{idx:02d}" for idx in range(8)]
    rows = []
    for date_idx, date in enumerate(dates):
        for symbol_idx, symbol in enumerate(symbols):
            base = 20.0 + symbol_idx * 3.0 + date_idx * (0.15 + symbol_idx * 0.01)
            wave = ((date_idx + 1) * (symbol_idx + 2)) % 11 / 50.0
            open_ = base * (1.0 + wave / 20.0)
            close = base * (1.0 + ((symbol_idx % 3) - 1) * 0.003 + wave / 30.0)
            high = max(open_, close) * 1.01
            low = min(open_, close) * 0.99
            volume = 10_000.0 + date_idx * 100.0 + symbol_idx * 1_000.0
            rows.append(
                {
                    "date": date,
                    "code": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "vwap": (open_ + high + low + close) / 4.0,
                }
            )
    stock = pd.DataFrame(rows)
    factors = alpha101(stock, names=["alpha010"]).rename(columns={"code": "symbol"})
    factors = factors.set_index(["date", "symbol"]).rename(columns={"alpha010_101": "alpha010"})
    prices = stock.pivot(index="date", columns="code", values="close")
    price_series = prices.stack().rename("close")
    price_series.index.names = ["date", "symbol"]
    clean = clean_factor_and_forward_returns(
        factors[["alpha010"]],
        factor="alpha010",
        prices=price_series,
        periods=(1, 5, 10),
        quantiles=5,
    )
    returns = prices.pct_change().fillna(0.0)
    return clean, returns


def _alphalens_clean(clean: pd.DataFrame) -> pd.DataFrame:
    result = clean.rename(columns={"forward_return_1": "1D"}).copy()
    result.index = result.index.set_names(["date", "asset"])
    return result


def _alphalens_clean_multi_period(clean: pd.DataFrame) -> pd.DataFrame:
    result = clean.rename(
        columns={
            "forward_return_1": "1D",
            "forward_return_5": "5D",
            "forward_return_10": "10D",
        }
    ).copy()
    result.index = result.index.set_names(["date", "asset"])
    return result


def _alphalens_performance():
    pytest.importorskip("alphalens")
    from alphalens import performance

    return performance
