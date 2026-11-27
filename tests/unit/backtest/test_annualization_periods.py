"""Annualization periods are resolved once and reused by summary, Reporter and artifacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradelearn import metrics

ANNUALIZED = ("annual_return", "volatility", "sharpe", "sortino", "calmar")


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, len(index)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1e6},
        index=index,
    )


def _lite_strategy():
    from tradelearn.lite import Strategy

    class Hold(Strategy):
        size = 10

        def next(self):
            if not self.position():
                self.buy(size=self.size)

    return Hold


def _daily_stock() -> pd.DataFrame:
    return _frame(pd.bdate_range("2024-01-01", periods=120))


def _daily_24x7() -> pd.DataFrame:
    return _frame(pd.date_range("2024-01-01", periods=120, freq="D"))


def _thirty_minute_continuous() -> pd.DataFrame:
    return _frame(pd.date_range("2024-01-01", periods=400, freq="30min"))


def _assert_summary_uses(stats, periods: int, source: str) -> None:
    returns = stats.returns
    assert stats.summary["sharpe"] == pytest.approx(metrics.sharpe(returns, periods=periods))
    assert stats.summary["annual_return"] == pytest.approx(
        metrics.annual_return(returns, periods=periods)
    )
    assert stats.summary["volatility"] == pytest.approx(metrics.volatility(returns, periods=periods))
    for name in ANNUALIZED:
        assert stats.metric_metadata[name]["periods"] == periods
        assert stats.metric_metadata[name]["period_source"] == source
    assert stats.config["annualization"] == {"periods": periods, "period_source": source}


def _assert_reporter_agrees(stats, periods: int) -> None:
    from tradelearn.report import Reporter

    reporter = Reporter(stats)
    assert reporter.periods == periods
    summary = reporter.summary()
    assert summary["sharpe_ratio"] == pytest.approx(stats.summary["sharpe"])
    assert summary["annual_return"] == pytest.approx(stats.summary["annual_return"])
    assert summary["annual_volatility"] == pytest.approx(stats.summary["volatility"])
    facts = reporter.metric_facts().set_index("name")
    assert (facts.loc[["sharpe_ratio", "annual_return"], "periods"] == periods).all()


def test_daily_stock_default_keeps_252_trading_days():
    from tradelearn.lite import Backtest

    stats = Backtest(_daily_stock(), _lite_strategy(), cash=100_000).run()
    _assert_summary_uses(stats, 252, "default_per_bar_profile")
    _assert_reporter_agrees(stats, 252)


def test_daily_24x7_is_not_guessed_as_365_without_explicit_periods():
    from tradelearn.lite import Backtest

    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000).run()
    _assert_summary_uses(stats, 252, "default_per_bar_profile")


def test_daily_24x7_explicit_365_reaches_summary_reporter_and_metadata():
    from tradelearn.lite import Backtest

    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365).run()
    _assert_summary_uses(stats, 365, "explicit")
    _assert_reporter_agrees(stats, 365)


def test_thirty_minute_continuous_summary_matches_reporter_return_frequency():
    from tradelearn.lite import Backtest

    stats = Backtest(_thirty_minute_continuous(), _lite_strategy(), cash=100_000).run()
    _assert_summary_uses(stats, 17520, "inferred_return_frequency")
    _assert_reporter_agrees(stats, 17520)


def test_lazy_stats_mode_uses_same_periods():
    from tradelearn.lite import Backtest

    stats = Backtest(
        _thirty_minute_continuous(), _lite_strategy(), cash=100_000, stats_mode="lazy"
    ).run()
    _assert_summary_uses(stats.stats, 17520, "inferred_return_frequency")


def test_engine_explicit_periods_via_constructor_and_run():
    from tradelearn.engine import Cerebro, Strategy
    from tradelearn.engine.feeds import PandasData

    class Hold(Strategy):
        def next(self):
            if not self.position:
                self.buy(size=10)

    def run(**options):
        cerebro = Cerebro(**options.get("init", {}))
        cerebro.adddata(PandasData(dataname=_daily_24x7()))
        cerebro.addstrategy(Hold)
        cerebro.broker.setcash(100_000)
        return cerebro.run(**options.get("run", {}))[0].stats

    _assert_summary_uses(run(init={"periods": 365}), 365, "explicit")
    _assert_reporter_agrees(run(init={"periods": 365}), 365)
    _assert_summary_uses(run(run={"periods": 365}), 365, "explicit")
    _assert_summary_uses(run(), 252, "default_per_bar_profile")


@pytest.mark.parametrize("bad", [0, -1, 365.5, float("nan"), float("inf"), True, np.bool_(True), "365"])
def test_invalid_explicit_periods_are_rejected(bad):
    from tradelearn.lite import Backtest
    from tradelearn.engine import Cerebro
    from tradelearn.report import Reporter

    with pytest.raises((TypeError, ValueError)):
        Backtest(_daily_stock(), _lite_strategy(), periods=bad)
    with pytest.raises((TypeError, ValueError)):
        Cerebro(periods=bad)
    with pytest.raises((TypeError, ValueError)):
        Reporter({"returns": pd.Series(dtype=float)}, periods=bad)


def test_search_config_and_trial_results_record_periods():
    from tradelearn.lite import Backtest

    bt = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365)
    bt.optimize(size=[5, 10])
    assert len(bt.trial_records) == 2
    for record in bt.trial_records:
        assert record["config"]["execution"]["periods"] == 365
        assert record["status"] == "completed"


def test_reporter_explicit_argument_overrides_stats_metadata():
    from tradelearn.lite import Backtest
    from tradelearn.report import Reporter

    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365).run()
    assert Reporter(stats, periods=252).periods == 252


def test_historical_default_metadata_keeps_reporter_inference():
    """Pre-existing artifacts labelled 252/default keep the Reporter's own inference."""
    from tradelearn.report import Reporter

    returns = pd.Series(
        np.linspace(-0.001, 0.001, 400),
        index=pd.date_range("2024-01-01", periods=400, freq="30min"),
    )
    legacy = {
        "returns": returns,
        "summary": {"sharpe": 1.0},
        "metric_metadata": {
            "sharpe": {"unit": "dimensionless", "periods": 252, "period_source": "default_per_bar_profile"}
        },
    }
    assert Reporter(legacy).periods == 17520
    assert Reporter({"returns": returns}).periods == 17520
    daily = {"returns": pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2024-01-01", periods=3))}
    assert Reporter(daily).periods == 252


def test_serialized_explicit_metadata_is_honoured_by_reporter():
    from tradelearn.report import Reporter

    returns = pd.Series([0.01, -0.02, 0.015], index=pd.date_range("2024-01-01", periods=3))
    payload = {
        "returns": returns,
        "metric_metadata": {
            name: {"periods": 365, "period_source": "explicit"} for name in ANNUALIZED
        },
    }
    assert Reporter(payload).periods == 365


def test_artifact_bundle_and_mlflow_tags_use_actual_periods(tmp_path):
    from tradelearn.lite import Backtest
    from tradelearn.report.artifacts import write_artifact_bundle

    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365).run()
    write_artifact_bundle(stats.stats, tmp_path)
    payload = json.loads((tmp_path / "stats.json").read_text())
    facts = {row["name"]: row for row in payload["metric_facts"]}
    assert facts["sharpe_ratio"]["periods"] == 365
    assert facts["sharpe_ratio"]["value"] == pytest.approx(stats.summary["sharpe"])
    assert payload["config"]["annualization"] == {"periods": 365, "period_source": "explicit"}


def test_mlflow_records_actual_periods(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    import os

    from tradelearn.lite import Backtest

    previous = mlflow.get_tracking_uri()
    previous_env = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    mlflow.set_experiment("Default")
    try:
        bt = Backtest(_thirty_minute_continuous(), _lite_strategy(), cash=100_000)
        bt.run()
        result = bt.log_mlflow()
        tags = mlflow.get_run(result.run_id).data.tags
        params = mlflow.get_run(result.run_id).data.params
        assert tags["tradelearn.metric.sharpe.periods"] == "17520"
        assert tags["tradelearn.metric.sharpe.period_source"] == "inferred_return_frequency"
        assert "tradelearn.metric.max_drawdown.periods" not in tags
        assert params["config.annualization.periods"] == "17520"
    finally:
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.set_tracking_uri(previous)
        if previous_env is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_env


def test_engine_search_records_periods():
    from tradelearn.engine import Cerebro, Strategy
    from tradelearn.engine.feeds import PandasData

    class Hold(Strategy):
        params = (("size", 10),)

        def next(self):
            if not self.position:
                self.buy(size=self.p.size)

    cerebro = Cerebro(periods=365)
    cerebro.adddata(PandasData(dataname=_daily_24x7()))
    cerebro.optstrategy(Hold, size=[5, 10])
    cerebro.broker.setcash(100_000)
    cerebro.run(optreturn=False)
    assert [r["config"]["execution"]["periods"] for r in cerebro.trial_records] == [365, 365]


def test_direct_stats_construction_keeps_legacy_signature_and_default_label():
    from tradelearn.backtest.models import Stats

    stats = Stats(returns=pd.Series([0.0, 0.01]), equity=pd.Series([100.0, 101.0]),
                  trades=pd.DataFrame(), positions=pd.DataFrame(), orders=pd.DataFrame(),
                  fills=pd.DataFrame(), config={}, analyzers={},
                  summary={"sharpe": 1.0, "max_drawdown": -0.1, "profit_factor": 2.0})
    assert stats.metric_metadata["sharpe"]["periods"] == 252
    assert stats.metric_metadata["sharpe"]["period_source"] == "default_per_bar_profile"
    assert "periods" not in stats.metric_metadata["max_drawdown"]
    assert stats.metric_metadata["profit_factor"]["pnl_basis"] == "pnlcomm"


def test_reporter_infer_helper_remains_importable():
    from tradelearn.report.reporter import _infer_periods

    index = pd.date_range("2024-01-01", periods=400, freq="30min")
    assert _infer_periods(pd.Series(0.0, index=index)) == 17520
    short = pd.date_range("2024-01-01", periods=10, freq="30min")
    assert _infer_periods(pd.Series(0.0, index=short)) == 252
    assert _infer_periods(pd.Series(0.0, index=pd.date_range("2024-01-01", periods=10))) == 252


def test_integral_float_periods_are_normalised_to_int():
    from tradelearn.core.annualization import validate_periods
    from tradelearn.lite import Backtest

    assert validate_periods(365.0) == 365 and type(validate_periods(365.0)) is int
    assert validate_periods(np.int64(252)) == 252
    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365.0).run()
    assert stats.metric_metadata["sharpe"]["periods"] == 365
    assert type(stats.config["annualization"]["periods"]) is int


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 00:30", "2024-01-01 01:00"]),
        pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 00:30", "2024-01-01 00:00", "2024-01-01 01:30"]),
        pd.DatetimeIndex(["2024-01-01 00:00", pd.NaT, "2024-01-01 01:00", "2024-01-01 01:30"]),
    ],
    ids=["duplicate", "not-increasing", "nat"],
)
def test_unmeasurable_index_is_not_inferred(index):
    from tradelearn.core.annualization import resolve_periods

    returns = pd.Series([0.0, 0.01, -0.01, 0.0], index=index)
    assert resolve_periods(returns) == (252, "default_per_bar_profile")
    assert resolve_periods(returns, 17520) == (17520, "explicit")


def test_artifact_tables_trace_explicit_periods(tmp_path):
    from tradelearn.lite import Backtest
    from tradelearn.report.artifacts import write_artifact_bundle

    stats = Backtest(_daily_24x7(), _lite_strategy(), cash=100_000, periods=365).run()
    write_artifact_bundle(stats.stats, tmp_path, log_report=False)
    metadata = pd.read_csv(tmp_path / "metadata.csv").set_index("key")["value"]
    assert metadata["annualization_periods"] == "365"
    assert metadata["annualization_period_source"] == "explicit"


def _one_session_minutes() -> pd.DataFrame:
    return _frame(pd.date_range("2024-01-02 09:30", periods=390, freq="1min"))


def test_single_session_minute_bars_are_not_extrapolated_to_525600():
    from tradelearn.core.annualization import resolve_periods
    from tradelearn.lite import Backtest
    from tradelearn.report import Reporter

    stats = Backtest(_one_session_minutes(), _lite_strategy(), cash=100_000).run()
    _assert_summary_uses(stats, 252, "insufficient_intraday_coverage")
    reporter = Reporter(stats)
    assert reporter.periods == 252
    assert reporter.period_source == "insufficient_intraday_coverage"
    assert resolve_periods(stats.returns, 98280) == (98280, "explicit")
    explicit = Backtest(_one_session_minutes(), _lite_strategy(), cash=100_000, periods=98280).run()
    _assert_summary_uses(explicit, 98280, "explicit")
    _assert_reporter_agrees(explicit, 98280)


def test_week_of_session_minute_bars_is_measured_including_gaps():
    """A full week of 390-minute sessions infers from observed density, gaps included."""
    from tradelearn.core.annualization import resolve_periods

    days = pd.bdate_range("2024-01-01", periods=10)
    index = pd.DatetimeIndex(
        [ts for day in days for ts in pd.date_range(day + pd.Timedelta("9h30min"), periods=390, freq="1min")]
    )
    periods, source = resolve_periods(pd.Series(0.0, index=index))
    assert source == "inferred_return_frequency"
    span = (index[-1] - index[0]).total_seconds()
    assert periods == round((len(index) - 1) * 365 * 86400 / span)
    assert periods < 525600


def test_daily_index_with_weekends_is_not_inferred_as_365():
    from tradelearn.core.annualization import resolve_periods

    returns = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=400, freq="D"))
    assert resolve_periods(returns) == (252, "default_per_bar_profile")
