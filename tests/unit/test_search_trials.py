"""Search shares historical trial execution, without changing facade returns."""

import pandas as pd
import pytest
from tradelearn import engine as bt
from tradelearn.lite import Backtest, Strategy
from tradelearn.research import ResearchRun


def bars():
    return pd.DataFrame(
        {k: [10.0, 11.0, 12.0, 13.0] for k in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2026-01-01", periods=4, tz="UTC"),
    )


class EngineTrial(bt.Strategy):
    params = (("amount", 1),)

    def next(self):
        if len(self) == 1:
            self.buy(size=self.p.amount)


class LiteTrial(Strategy):
    amount = 1

    def next(self):
        if len(self) == 1:
            self.buy(size=self.amount)


def test_lite_records_full_config_and_keeps_best_return():
    owner = Backtest(bars(), LiteTrial, cash=500, commission=0.01, trade_on_close=True)
    with ResearchRun("search") as run:
        best = owner.optimize(amount=[1, 2])
    assert best.strategy.amount == 2
    assert [r["status"] for r in owner.trial_records] == ["completed", "completed"]
    assert [r["params"] for r in owner.trial_records] == [{"amount": 1}, {"amount": 2}]
    assert owner.trial_records[0]["config"]["broker"]["cash"] == 500
    assert owner.trial_records[0]["config"]["broker"]["commission"] == 0.01
    assert owner.trial_records[0]["config"]["data"][0]["sha256"]
    assert (
        owner.trial_records[0]["evaluation"] == best["return_pct"]
        or owner.trial_records[1]["evaluation"] == best["return_pct"]
    )
    assert len([step for step in run.steps if step.category == "search.trial"]) == 2


def test_engine_isolates_mutated_broker_and_preserves_returns():
    class Mutates(bt.Strategy):
        params = (("n", 1),)

        def start(self):
            self.initial = self.broker.getcash()
            self.broker.setcash(888)

    c = bt.Cerebro(stdstats=False)
    c.adddata(bt.feeds.PandasData(dataname=bars()))
    c.broker.setcash(777)
    c.optstrategy(Mutates, n=[1, 2])
    out = c.run(optreturn=False)
    assert [x[0].initial for x in out] == [777, 777]
    assert out[0][0].broker is not out[1][0].broker
    assert c.broker._cash == 777
    assert [r["config"]["broker"]["commission"] for r in c.trial_records] == [0, 0]
    assert len(c.strats) == 2


def test_failure_recorded_and_original_exception_preserved():
    sentinel = ValueError("deliberate")

    class Broken(Strategy):
        n = 1

        def next(self):
            raise sentinel

    owner = Backtest(bars(), Broken)
    with pytest.raises(ValueError) as caught:
        owner.optimize(n=[1, 2])
    assert caught.value is sentinel
    assert [r["status"] for r in owner.trial_records] == ["failed", "cancelled"]
    assert owner.trial_records[0]["error_type"] == "ValueError"
    assert owner.trial_records[1]["reason"] == "search_aborted"


def test_grid_uses_configured_isolated_cerebro_and_records():
    from tradelearn.integrations.search import grid_search

    def factory():
        c = bt.Cerebro(stdstats=False)
        c.broker.setcash(500)
        c.broker.setcommission(0.01)
        return c

    results = grid_search(EngineTrial, bars(), {"amount": [1, 2]}, cerebro_factory=factory)
    assert type(results) is list
    assert [r.trial_record["config"]["broker"]["cash"] for r in results] == [500, 500]
    assert all(r.trial_record["status"] == "completed" for r in results)
    assert results[0].strategy.broker is not results[1].strategy.broker


def test_grid_failure_has_prior_and_cancelled_records_without_changing_exception():
    from tradelearn.integrations.search import grid_search

    sentinel = RuntimeError("deliberate grid")

    class Broken(EngineTrial):
        def next(self):
            if self.p.amount == 2:
                raise sentinel

    with pytest.raises(RuntimeError) as caught:
        grid_search(Broken, bars(), {"amount": [1, 2, 3]})
    assert caught.value is sentinel
    assert [r["status"] for r in caught.value.trial_records] == ["completed", "failed", "cancelled"]
    assert caught.value.trial_records[2]["config"]["strategies"][0]["params"]["amount"] == 3


def test_lite_cancellation_retains_original_and_records_remaining():
    sentinel = KeyboardInterrupt()

    class Cancelled(Strategy):
        n = 1

        def next(self):
            raise sentinel

    owner = Backtest(bars(), Cancelled)
    with pytest.raises(KeyboardInterrupt) as caught:
        owner.optimize(n=[1, 2])
    assert caught.value is sentinel
    assert [r["status"] for r in owner.trial_records] == ["cancelled", "cancelled"]


def test_optuna_records_pruned_failed_and_success_without_forced_engine_metric():
    import optuna
    from tradelearn.optimize import OptunaSearch

    def objective(trial):
        if trial.number == 0:
            raise optuna.TrialPruned()
        if trial.number == 1:
            raise ValueError("deliberate")
        if trial.number == 2:
            return float("nan")
        return 3.0

    search = OptunaSearch(objective, n_trials=4)
    study = search.run(catch=(ValueError,))
    assert isinstance(study, optuna.Study)
    assert [r["status"] for r in search.trial_records] == [
        "pruned",
        "failed",
        "failed",
        "completed",
    ]
    assert search.trial_records[-1]["evaluation"] == 3.0


def test_optuna_backtest_uses_full_config_and_keeps_factory_owner_pristine():
    from tradelearn.optimize import OptunaBacktestSearch

    owner = Backtest(bars(), LiteTrial, cash=500, commission=0.01)
    search = OptunaBacktestSearch(
        backtest_factory=lambda params: owner,
        param_space={"amount": [1, 2]},
        metric="return_pct",
        n_trials=2,
    )
    study = search.run()
    assert len(study.trials) == 2
    assert all(r["config"]["broker"]["cash"] == 500 for r in search.trial_records)
    assert owner.broker._engine is None


def test_trial_preserves_custom_costs_matches_single_and_does_not_consume_models():
    from tradelearn.core.costs import FixedCommission, FixedSlippage
    import json

    owner = Backtest(bars(), LiteTrial, cash=500)
    owner.broker.configure_matching(commission=FixedCommission(0.7), slippage=FixedSlippage(0.25))
    expected = owner.run(amount=2)
    result = owner.optimize(amount=[2, 2])
    pd.testing.assert_frame_equal(result.fills, expected.fills)
    assert all(
        r["config"]["broker"]["slippage_model"]["params"]["amount"] == 0.25
        for r in owner.trial_records
    )
    json.dumps(owner.trial_records, allow_nan=False)


def test_engine_optimization_can_repeat_after_failure_without_replacing_specs():
    class Broken(bt.Strategy):
        params = (("n", 1),)

        def next(self):
            raise ValueError("deliberate")

    owner = bt.Cerebro(stdstats=False)
    owner.adddata(bt.feeds.PandasData(dataname=bars()))
    owner.optstrategy(Broken, n=[1, 2])
    specs = owner.strats
    for _ in range(2):
        with pytest.raises(ValueError):
            owner.run()
        assert owner.strats is specs
        assert [r["status"] for r in owner.trial_records] == ["failed", "cancelled"]
        assert owner.broker._engine is None


def test_live_search_fails_before_any_connect():
    owner = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=bars())
    data.is_live = True
    owner.adddata(data)
    owner.optstrategy(EngineTrial, amount=[1, 2])
    with pytest.raises(ValueError, match="historical"):
        owner.run()
    assert owner.broker._engine is None


def test_lite_effective_broker_cash_and_commission_are_single_run_authority():
    owner = Backtest(bars(), LiteTrial)
    owner.broker.setcash(777)
    owner.broker.setcommission(0.02)
    result = owner.run(amount=1)
    assert result.strategy.broker._engine.get_cash() == pytest.approx(777 - 11 - 11 * 0.02)
    optimized = owner.optimize(amount=[1])
    pd.testing.assert_frame_equal(optimized.fills, result.fills)
    assert owner.trial_records[0]["config"]["broker"]["cash"] == 777
    assert owner.trial_records[0]["config"]["broker"]["commission"] == 0.02


def test_optuna_records_survive_study_storage_reload(tmp_path):
    import optuna
    from tradelearn.optimize import OptunaSearch

    url = "sqlite:///" + str(tmp_path / "study.db")
    search = OptunaSearch(
        lambda trial: trial.suggest_int("n", 1, 2), n_trials=2, storage=url, study_name="local"
    )
    search.run()
    restored = optuna.load_study(study_name="local", storage=url)
    assert restored.user_attrs["tradelearn.trials"] == search.trial_records
    assert [r["status"] for r in search.trial_records] == ["completed", "completed"]


def test_optuna_objectives_remain_optional_engine_free_and_thread_scoped():
    from tradelearn.optimize import OptunaSearch

    search = OptunaSearch(lambda trial: float(trial.suggest_int("n", 1, 2)), n_trials=4)
    study = search.run(n_jobs=2)
    assert sorted(r["trial_number"] for r in search.trial_records) == [
        t.number for t in study.trials
    ]
    assert all(r["status"] == "completed" for r in search.trial_records)


def test_optuna_resume_preserves_prior_records_without_republishing_them(tmp_path):
    from tradelearn.optimize import OptunaSearch

    url = "sqlite:///" + str(tmp_path / "resume.db")
    first = OptunaSearch(lambda t: 1.0, n_trials=1, study_name="resume", storage=url)
    first.run()
    second = OptunaSearch(
        lambda t: 2.0, n_trials=1, study_name="resume", storage=url, load_if_exists=True
    )
    with ResearchRun("resumed") as run:
        study = second.run()
    assert len(study.user_attrs["tradelearn.trials"]) == 2
    assert study.user_attrs["tradelearn.trials"][0] == first.trial_records[0]
    assert len([s for s in run.steps if s.category == "search.trial"]) == 1


def test_optuna_tracking_failure_does_not_mask_original_trial_error(monkeypatch):
    import sys, types
    from tradelearn.optimize import OptunaSearch

    sentinel = ValueError("primary")

    class Study:
        def optimize(self, objective, **kwargs):
            objective(types.SimpleNamespace())

        def set_user_attr(self, *args):
            raise RuntimeError("secondary")

    monkeypatch.setitem(
        sys.modules, "optuna", types.SimpleNamespace(create_study=lambda **k: Study())
    )
    search = OptunaSearch(lambda t: (_ for _ in ()).throw(sentinel), n_trials=1)
    with pytest.raises(ValueError) as caught:
        search.run()
    assert caught.value is sentinel
    assert search.trial_records[0]["status"] == "failed"


def test_engine_trials_do_not_share_nested_selected_parameter_state():
    class Mutates(bt.Strategy):
        params = (("config", None), ("n", 0))

        def start(self):
            self.before = list(self.p.config["items"])
            self.p.config["items"].append(self.p.n)

    config = {"items": []}
    owner = bt.Cerebro(stdstats=False)
    owner.adddata(bt.feeds.PandasData(dataname=bars()))
    owner.optstrategy(Mutates, config=[config], n=[1, 2])
    out = owner.run(optreturn=False)
    assert [x[0].before for x in out] == [[], []]
    assert config == {"items": []}


def test_lite_optimize_no_parameters_does_not_reuse_last_run_parameters():
    owner = Backtest(bars(), LiteTrial)
    owner.run(amount=2)
    assert owner.optimize().strategy.amount == 1
