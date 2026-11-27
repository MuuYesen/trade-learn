"""Contract tests for the public Chinese research-workflow Skill.

These tests guard statements that change research conclusions (gates,
standard-error conventions, deflation with a single trial, local evidence
defaults) and the reference helpers shipped in ``scripts/backtest_checks.py``.
They only use the standard library and numpy; they do not import tradelearn.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SKILL = Path(__file__).resolve().parents[1] / ".agents" / "skills" / "tradelearn-research"
REFS = SKILL / "references"


def _load_checks():
    path = SKILL / "scripts" / "backtest_checks.py"
    spec = importlib.util.spec_from_file_location("tl_skill_backtest_checks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checks = _load_checks()


def _text(name: str) -> str:
    path = SKILL / name if name == "SKILL.md" else REFS / name
    return path.read_text(encoding="utf-8")


def _all_docs() -> dict[str, str]:
    docs = {"SKILL.md": _text("SKILL.md")}
    for path in sorted(REFS.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


# --------------------------------------------------------------------------
# Reference helpers: known answers and small counterexamples
# --------------------------------------------------------------------------


def test_annual_sharpe_se_known_answer_and_first_round_bug():
    assert math.isclose(checks.annual_sharpe_se(1.0, 1460, 365), 0.500342348545, abs_tol=1e-9)
    # First-round report forgot to divide S^2 by q: sqrt((1 + S^2/2) / T_years).
    assert not math.isclose(checks.annual_sharpe_se(1.0, 1460, 365), 0.6123724, abs_tol=1e-4)


def test_script_has_no_second_statistics_implementation():
    for name in ("sharpe_se", "paired_sharpe_diff", "probabilistic_sharpe", "deflated_sharpe",
                 "minimum_track_record_length", "expected_max_sharpe", "variance_inflation",
                 "stable_region"):
        assert not hasattr(checks, name), name


class _FakeResult(SimpleNamespace):
    pass


def _fake_metrics(*, flip_paired=False, accept_single_trial=False):
    """A stand-in for tradelearn.metrics answering the frozen inputs correctly."""
    import pandas as pd

    def sharpe(r, periods):
        return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))

    def sharpe_se(r, periods, *, method="iid", bandwidth=None):
        if list(r) == checks.HAC_INPUT and method == "hac":
            return checks.HAC_BANDWIDTH_1 if bandwidth == 1 else checks.HAC_BANDWIDTH_1 / 2
        raise AssertionError("unexpected input")

    def paired(left, right, periods, *, block_length, draws=4000, seed=0, confidence=.95):
        est = sharpe(right, periods) - sharpe(left, periods)
        if flip_paired:
            est = -est
        return _FakeResult(estimate=est, standard_error=.1, lower=est - 1, upper=est + 1,
                           observations=len(left), valid_draws=draws)

    def psr(r, *, benchmark=0.0):
        return 0.9 if benchmark == 0 else 0.6

    def dsr(r, *, n_trials, variance_trials):
        if n_trials < 2 and not accept_single_trial:
            raise ValueError("n_trials must be an integer of at least two")
        return 0.6

    def mintrl(r, *, benchmark=0.0, confidence=.95):
        return 14 if r.mean() > 0 else math.inf

    del pd
    return SimpleNamespace(sharpe=sharpe, sharpe_se=sharpe_se, paired_sharpe_diff=paired,
                           probabilistic_sharpe=psr, deflated_sharpe=dsr,
                           minimum_track_record_length=mintrl)


def test_native_known_answers_pass_block_and_catch_wrong_direction():
    pytest.importorskip("pandas")
    good = checks.native_known_answers(_fake_metrics())
    assert set(good.values()) == {"通过"}, good
    assert checks.native_known_answers(SimpleNamespace())["sharpe_se"] == "受阻"
    assert checks.native_known_answers(_fake_metrics(flip_paired=True))["paired_sharpe_diff"] == "不过"
    assert checks.native_known_answers(_fake_metrics(accept_single_trial=True))["deflated_sharpe"] == "不过"


def test_native_known_answers_on_installed_tradelearn_if_present():
    metrics = pytest.importorskip("tradelearn.metrics")
    status = checks.native_known_answers(metrics)
    assert "不过" not in status.values(), status


def test_evidence_direction_does_not_let_negative_interval_pass():
    assert checks.evidence_direction(-1.2, -0.3) == "负"
    assert checks.evidence_direction(-0.706, 1.710) == "不显著"
    assert checks.evidence_direction(0.1, 0.9) == "正"
    with pytest.raises(ValueError):
        checks.evidence_direction(1.0, 0.0)


def test_positive_return_and_paired_improvement_are_independent():
    # Both candidates lose money, yet one loses significantly less.
    result = checks.comparison_verdict(
        candidate_interval=(-2.0, -0.4), baseline_interval=(-2.5, -0.9), paired_interval=(0.1, 0.8)
    )
    assert result == {"候选绝对收益": "负", "基准绝对收益": "负", "配对相对改善": "正"}


def test_periods_per_year_follows_market_clock():
    assert checks.periods_per_year("crypto", "1D") == 365
    assert checks.periods_per_year("crypto", "30min") == 17520
    assert checks.periods_per_year("cn_equity", "1D") == 252
    assert checks.periods_per_year("cn_equity", "30min") == 252 * 8
    assert checks.periods_per_year("us_equity", "1h") == 252 * 6.5
    with pytest.raises(ValueError):
        checks.periods_per_year("moon", "1D")


def test_metric_period_check_flags_per_bar_default():
    meta = {"sharpe": {"periods": 252, "period_source": "default_per_bar_profile"}}
    assert checks.metric_periods_ok(meta, "sharpe", 17520) is False
    assert checks.metric_periods_ok(meta, "sharpe", 252) is True
    assert checks.metric_periods_ok({}, "sharpe", 252) is False
    meta["sharpe"]["period_source"] = "insufficient_intraday_coverage"
    assert checks.metric_periods_ok(meta, "sharpe", 252) is False
    assert checks.metric_periods_ok({"sharpe": {"periods": "unknown"}}, "sharpe", 252) is False


def test_capabilities_detect_without_calling():
    called = []

    def sharpe_se(*args, **kwargs):  # pragma: no cover - must not be called
        called.append(1)

    fake = SimpleNamespace(metrics=SimpleNamespace(sharpe_se=sharpe_se))  # noqa: F841
    caps = checks.capabilities(fake)
    assert caps["metrics.sharpe_se"] == "可用"
    assert caps["metrics.paired_sharpe_diff"] == "缺失"
    assert caps["sentinel"] == "缺失"
    assert called == []


def test_save_local_evidence_writes_json_without_mlflow(tmp_path):
    sys.modules.pop("mlflow", None)
    records = [{"format": "tradelearn.search.trial.v1", "index": 0, "status": "completed",
                "params": {"fast": 10}}]
    research = SimpleNamespace(to_dict=lambda: {"name": "probe", "steps": []})
    files = checks.save_local_evidence(
        tmp_path / "run1", trial_records=records, research_result=research,
        manifest={"data_sha256": "abc", "periods": 365, "se_method": "iid"},
    )
    names = sorted(p.name for p in files)
    assert names == ["manifest.json", "research_result.json", "trial_records.json"]
    assert json.loads((tmp_path / "run1" / "manifest.json").read_text())["periods"] == 365
    assert "mlflow" not in sys.modules
    with pytest.raises(FileExistsError):
        checks.save_local_evidence(tmp_path / "run1", trial_records=records)


def test_needs_fill_review():
    assert checks.needs_fill_review(position_change_expected=True, fill_count=0)
    assert not checks.needs_fill_review(position_change_expected=False, fill_count=0)


# --------------------------------------------------------------------------
# Document contract
# --------------------------------------------------------------------------

ROUTES = ["读取/复现", "测量/诊断", "比较与稳健性", "搜索", "因子与模型", "组合及执行", "只读复盘"]
ROUTE_FIELDS = ["运行入口", "最少证据", "关卡", "缺证状态"]


def test_skill_routes_every_question_type():
    skill = _text("SKILL.md")
    routes = _text("routes.md")
    for route in ROUTES:
        assert route in skill, route
        assert f"## {route}" in routes or re.search(rf"^## .*{re.escape(route)}", routes, re.M), route
    sections = re.split(r"^## ", routes, flags=re.M)
    for route in ROUTES:
        body = next(s for s in sections if s.split("\n", 1)[0].find(route) >= 0)
        for field in ROUTE_FIELDS:
            assert field in body, (route, field)


def test_no_undirected_interval_gate():
    gates = _text("gates.md")
    assert "| 不含 0 |" not in gates
    assert "下限 > 0" in gates
    assert "配对" in gates


def test_ranking_tie_uses_paired_difference():
    gates = _text("gates.md")
    section = gates.split("## 五、", 1)[1].split("## 六、", 1)[0]
    assert "配对差" in section
    assert "各自的标准误" in section  # explicitly rejected


def test_single_trial_reports_psr_not_dsr():
    validation = _text("validation.md")
    assert "N = 1" in validation and "PSR" in validation
    assert "通常是 1 或对比过的方案数" not in validation


def test_standard_error_method_is_declared():
    for name in ("validation.md", "self-check.md", "report.md"):
        text = _text(name)
        assert "IID" in text and "HAC" in text and "块自助" in text, name


def test_final_holdout_not_banned_from_significance():
    validation = _text("validation.md")
    assert "不再做显著性检验" not in validation


def test_method_status_vocabulary_defined_once():
    gates = _text("gates.md")
    for word in ("不适用", "受阻", "未检验"):
        assert word in gates
    assert "方法状态" in gates


def test_report_does_not_cite_summary_sharpe_blindly():
    skill = _text("SKILL.md")
    assert 'summary["sharpe"]' in skill
    assert "period_source" in skill
    assert "口径表" in _text("report.md")


def test_mlflow_requires_explicit_user_target():
    docs = _all_docs()
    assert "每次运行都记进实验记录——Lite 回测用" not in docs["SKILL.md"]
    for name, text in docs.items():
        for line in text.splitlines():
            if "log_mlflow" in line or "integrations.mlflow" in line:
                assert "uri" in line or "显式" in line, (name, line)
    for name in ("SKILL.md", "experiment-log.md"):
        assert "save_local_evidence" in docs[name] or "write_artifact_bundle" in docs[name], name


def test_lite_optimize_not_used_for_selection():
    routes = _text("routes.md")
    assert "Backtest.optimize" in routes
    assert "return_pct" in routes


def test_sentinel_is_not_claimed_available():
    live = _text("live-review.md")
    assert "没有 `tradelearn.sentinel`" in live
    assert "暂停" in live
    assert "当前版本为 `leafquant_monitor.codec.decode_fact`" not in live


def test_fill_and_fractional_quantity_checks_kept():
    self_check = _text("self-check.md")
    assert "小数" in self_check and "零成交" in self_check
    assert "数据" in _text("data-audit.md") and "sha256" in _text("routes.md").lower()


def test_relative_links_resolve():
    for name, text in _all_docs().items():
        base = SKILL if name == "SKILL.md" else REFS
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if target.startswith("http"):
                continue
            assert (base / target).exists(), (name, target)


def test_skill_stays_chinese():
    for name, text in _all_docs().items():
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        prose = re.sub(r"`[^`]*`", "", prose)
        cjk = len(re.findall(r"[一-鿿]", prose))
        latin = len(re.findall(r"[A-Za-z]", prose))
        assert cjk > latin, name


def _route(name: str) -> str:
    sections = re.split(r"^## ", _text("routes.md"), flags=re.M)
    return next(s for s in sections if s.split("\n", 1)[0].find(name) >= 0)


def test_comparison_route_requires_out_of_sample_and_frozen_costs():
    body = _route("比较与稳健性")
    for word in ("样本外", "预热", "跨段", "已查看", "邻域", "不是搜索"):
        assert word in body, word
    assert "left=基准" in body.replace(" ", "") and "right=候选" in body.replace(" ", "")
    assert "一概" in body or "不能标" in body


def test_paired_direction_is_consistent_everywhere():
    for name, text in _all_docs().items():
        for line in text.splitlines():
            if "paired_sharpe_diff(" in line:
                assert "基准" in line and "候选" in line, (name, line)


def test_search_and_factor_routes_use_native_functions():
    assert "stable_region" in _route("搜索")
    assert "standard_error" in _route("搜索")
    factor = _route("因子与模型")
    assert "hac_mean_se" in factor and "回归" in factor


def test_route_examples_run_and_read_back(tmp_path):
    pytest.importorskip("pandas")
    validation = pytest.importorskip("tradelearn.research.validation")
    del validation
    spec = importlib.util.spec_from_file_location("route_examples", SKILL / "scripts" / "route_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.self_test(tmp_path)
    assert report["read_back"] is True, report
    assert report["comparison_direction"] == "候选 − 基准", report


def test_local_evidence_records_input_that_native_reader_can_verify(tmp_path):
    import pandas as pd
    from tradelearn.research import validation

    bars = pd.DataFrame({"close": [1.0, 1.2]})
    destination = tmp_path / "single"
    checks.save_local_evidence(destination, data=bars)
    bundle = validation.load_bundle(destination)
    assert bundle.manifest["input_fingerprint"]["rows"] == 2
    assert validation.verify_inputs(bundle, data=bars) == []
    assert validation.verify_inputs(bundle, data=bars + 1)


def test_failed_local_evidence_write_does_not_leave_partial_bundle(tmp_path, monkeypatch):
    from tradelearn.report import artifacts

    def broken_write(stats, directory, **kwargs):
        (directory / "partial.csv").write_text("incomplete")
        raise OSError("simulated exporter failure")

    monkeypatch.setattr(artifacts, "write_artifact_bundle", broken_write)
    destination = tmp_path / "single"
    with pytest.raises(OSError, match="simulated exporter"):
        checks.save_local_evidence(destination, stats=object())
    assert not destination.exists()
    assert checks.save_local_evidence(destination) == [destination / "manifest.json"]


def _run_lite(data, settings):
    from tradelearn.lite import Backtest, Strategy

    class Hold(Strategy):
        def next(self):
            if not self.position():
                self.buy(size=1)

    return Backtest(data, Hold, **settings).run()


def _run_engine(data, settings):
    from tradelearn.engine import Cerebro, Strategy
    from tradelearn.engine.feeds import PandasData

    class Hold(Strategy):
        def next(self):
            if not self.position:
                self.buy(size=1)

    engine = Cerebro(periods=settings["periods"])
    engine.broker.setcash(settings["cash"])
    engine.broker.setcommission(commission=settings["commission"])
    engine.adddata(PandasData(dataname=data))
    engine.addstrategy(Hold)
    return engine.run()[0].stats


@pytest.mark.parametrize("evaluate", [_run_lite, _run_engine], ids=["lite", "engine"])
def test_follow_up_reuses_real_run_and_preserves_source(
    monkeypatch, tmp_path, evaluate,
):
    import pandas as pd
    from tradelearn.research.validation import load_bundle

    monkeypatch.syspath_prepend(str(SKILL / "scripts"))
    from route_examples import record_backtest, rerun_backtest

    frame = pd.DataFrame(
        {**{name: [100.0, 101.0, 104.0, 103.0, 105.0, 106.0]
            for name in ("open", "high", "low", "close")}, "volume": 100.0},
        index=pd.date_range("2024-01-01", periods=6, tz="UTC"),
    )
    baseline = tmp_path / "original"
    changed = tmp_path / "new"
    settings = {"cash": 1000, "commission": 0.001, "periods": 365}
    changed_setting = "commission"
    expected = evaluate(frame, settings)
    record_backtest(
        frame, settings=settings, evaluate=evaluate, command="python research.py",
        directory=baseline, timeframe="1D",
    )
    original_files = {path.name: path.read_bytes() for path in baseline.iterdir()}

    rerun_backtest(baseline, frame, evaluate=evaluate, setting=changed_setting,
                   value=0.002, directory=changed, command="python research.py --scenario doubled")

    before, after = load_bundle(baseline), load_bundle(changed)
    assert float(before.summary["return_pct"]) == pytest.approx(expected.summary["return_pct"])
    assert float(after.summary["return_pct"]) < float(before.summary["return_pct"])
    assert len(before.tables["fills"]) == len(after.tables["fills"]) > 0
    before_fees = before.tables["fills"]["commission"].sum()
    after_fees = after.tables["fills"]["commission"].sum()
    assert before_fees > 0
    assert after_fees == pytest.approx(2 * before_fees)
    assert after.manifest["backtest_run"]["settings"] == {**settings, changed_setting: 0.002}
    assert after.manifest["comparison"]["change"] == {
        "setting": changed_setting, "before": 0.001, "after": 0.002,
    }
    assert after.manifest["comparison"]["baseline"] == str(baseline.resolve())
    assert {path.name: path.read_bytes() for path in baseline.iterdir()} == original_files

    altered = frame.copy()
    altered.iloc[0] += 1
    for supplied_data, callback, setting, error in [
        (altered, evaluate, changed_setting, "sha256"),
        (frame, lambda *_: None, changed_setting, "entrypoint"),
        (frame, evaluate, "unrecorded", "unrecorded"),
    ]:
        with pytest.raises(ValueError, match=error):
            rerun_backtest(baseline, supplied_data, evaluate=callback, setting=setting,
                           value=0.003, directory=tmp_path / "rejected", command="python research.py")
    assert not (tmp_path / "rejected").exists()
    with pytest.raises(FileExistsError):
        rerun_backtest(baseline, frame, evaluate=evaluate, setting=changed_setting,
                       value=0.003, directory=baseline, command="python research.py")
