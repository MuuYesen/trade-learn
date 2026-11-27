from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tradelearn.research import ResearchRun, validation as v
from tradelearn.research.temporal import forward_return


def _bars(n=120, tz="UTC"):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz=tz, name="timestamp")
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": close}, index=idx)


def _momentum(params, frame, fold, segment):
    """Short synthetic strategy: long when close above its rolling mean."""
    window = params.get("window", 5)
    close = frame["close"]
    signal = (close > close.rolling(window).mean()).astype(float).shift(1)
    return (close.pct_change() * signal).fillna(0.0)


# --------------------------------------------------------------------- folds


def test_folds_are_deterministic_non_overlapping_and_respect_holdout():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20, embargo=2, holdout_start="2024-04-20")
    again = v.walk_forward_folds(bars, train=40, test=20, embargo=2, holdout_start="2024-04-20")
    assert [f.to_dict() for f in folds] == [f.to_dict() for f in again]
    holdout = pd.Timestamp("2024-04-20", tz="UTC")
    for prev, cur in zip(folds, folds[1:]):
        assert prev.test_end < cur.test_start
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert (fold.test_start - fold.train_end).days == 3  # two embargo days skipped
        assert fold.test_end < holdout
    assert len(folds) == 3  # 110 pre-holdout days: 42 + 3*20, partial 4th dropped


def test_folds_refuse_overlapping_step_and_support_expanding():
    bars = _bars()
    with pytest.raises(ValueError, match="overlap"):
        v.walk_forward_folds(bars, train=40, test=20, step=10)
    folds = v.walk_forward_folds(bars, train=40, test=20, expanding=True)
    assert all(f.train_start == bars.index[0] for f in folds)


# -------------------------------------------------------------- walk-forward


def test_fixed_walk_forward_stitches_only_test_windows_with_warmup_clipped():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)
    seen = []

    def evaluate(params, frame, fold, segment):
        seen.append((segment, frame.index[0], frame.index[-1]))
        return _momentum(params, frame, fold, segment)

    result = v.walk_forward(bars, folds, evaluate=evaluate, params={"window": 5}, warmup=10)
    assert result.complete and result.payload["mode"] == "fixed"
    assert all(segment == "test" for segment, *_ in seen)
    for (_, start, end), fold in zip(seen, folds):
        assert start < fold.test_start and end == fold.test_end  # warmup history only
    assert result.oos_returns.index.is_unique
    assert result.oos_returns.index[0] == folds[0].test_start
    assert len(result.oos_returns) == 20 * len(folds)
    assert all(t["returns_outside_window"] == 10 for t in result.trials)
    assert result.payload["state_policy"] == "independent_run_per_call"


def test_insufficient_warmup_fold_is_incomplete_not_dropped():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=5, test=20)
    result = v.walk_forward(bars, folds, evaluate=_momentum, warmup=10)
    assert result.folds[0]["status"] == "incomplete"
    assert result.folds[0]["reason"] == "insufficient_warmup"
    assert not result.complete
    assert result.oos_returns.index[0] == folds[1].test_start


def test_search_selects_on_train_only_and_never_sees_test_data():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)
    calls = []

    def evaluate(params, frame, fold, segment):
        calls.append((segment, frame.index.max(), fold))
        return _momentum(params, frame, fold, segment)

    with ResearchRun("wf") as run:
        result = v.walk_forward(
            bars, folds, evaluate=evaluate,
            candidates=[{"window": 3}, {"window": 8}],
            objective=lambda r: float(r.mean() / r.std()) if r.std() > 0 else float("nan"),
        )
    assert result.complete
    for segment, last, fold in calls:
        if segment == "train":
            assert last <= fold.train_end < fold.test_start
    assert all("selected_candidate" in f for f in result.folds)
    assert len(result.trials) == len(folds) * 3
    assert {t["format"] for t in result.trials} == {"tradelearn.search.trial.v1"}
    categories = [s.category for s in run.steps]
    assert categories.count("search.trial") == len(result.trials)
    assert "validation.walk_forward" in categories


def test_search_requires_explicit_objective():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)
    with pytest.raises(ValueError, match="objective"):
        v.walk_forward(bars, folds, evaluate=_momentum, candidates=[{"window": 3}])


def test_failed_candidate_fails_the_fold_and_is_kept_in_evidence():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)

    def evaluate(params, frame, fold, segment):
        if params["window"] == 8 and fold.number == 1:
            raise RuntimeError("boom")
        return _momentum(params, frame, fold, segment)

    result = v.walk_forward(bars, folds, evaluate=evaluate,
                            candidates=[{"window": 3}, {"window": 8}], objective=lambda r: r.sum())
    fold = result.folds[1]
    assert fold["status"] == "failed" and fold["reason"] == "candidate_failed"
    assert fold["failed_candidates"] == [1]
    assert not result.complete
    assert result.payload["fold_status"]["failed"] == 1
    failed = [t for t in result.trials if t["status"] == "failed"]
    assert failed and failed[0]["error_type"] == "RuntimeError"
    window = (result.oos_returns.index >= folds[1].test_start) & (
        result.oos_returns.index <= folds[1].test_end)
    assert not window.any()


def test_labels_purge_training_samples_crossing_into_test():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"), ["A"]],
        names=["timestamp", "symbol"])
    bars = pd.DataFrame({"close": np.linspace(100, 160, 60)}, index=idx)
    labels = forward_return(bars, horizon=5)
    folds = v.walk_forward_folds(bars, train=30, test=10)
    seen = {}

    def evaluate(params, frame, fold, segment):
        stamps = frame.index.get_level_values("timestamp")
        seen.setdefault((fold.number, segment), stamps.max())
        return pd.Series(0.0, index=pd.DatetimeIndex(stamps.unique()))

    v.walk_forward(bars, folds, evaluate=evaluate, candidates=[{}], objective=lambda r: 0.0,
                   labels=labels)
    for fold in folds:
        # last train sample's 5-day label must end before the test window
        assert seen[(fold.number, "train")] + pd.Timedelta(days=5) < fold.test_start


def test_hand_built_overlapping_folds_are_refused():
    bars = _bars()
    a, b = v.walk_forward_folds(bars, train=40, test=20)[:2]
    bad = dataclasses.replace(b, train_end=a.train_end, test_start=a.test_end)
    with pytest.raises(ValueError, match="overlap"):
        v.walk_forward(bars, [a, bad], evaluate=_momentum)


# --------------------------------------------------------- scenarios/compare


def test_scenarios_keep_failures():
    bars = _bars()

    def evaluate(scenario):
        if scenario["cost"] > 0.01:
            raise ValueError("unsupported")
        r = _momentum({"window": 5}, bars, None, "test")
        return r - scenario["cost"] * (r != 0)

    result = v.run_scenarios(evaluate, {"base": {"cost": 0.0}, "2x": {"cost": 0.001},
                                        "broken": {"cost": 0.5}}, periods=252)
    table = result.table()
    assert list(table.index) == ["base", "2x", "broken"]
    assert table.loc["broken", "status"] == "failed"
    assert table.loc["2x", "total_return"] < table.loc["base", "total_return"]
    assert not result.payload["complete"]


@dataclasses.dataclass(frozen=True)
class _Diff:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    observations: int
    valid_draws: int


def test_compare_reports_alignment_and_direction(monkeypatch):
    from tradelearn import metrics

    calls = []

    def fake(left, right, periods, *, block_length, draws=4000, seed=0, confidence=0.95):
        assert left.index.equals(right.index) and left.index.is_unique
        calls.append((left.name, right.name))
        return _Diff(float(right.mean() - left.mean()), 0.1, -1, 1, len(left), draws)

    monkeypatch.setattr(metrics, "paired_sharpe_diff", fake, raising=False)
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    base = pd.Series(0.0, index=idx)
    cand = pd.Series(0.01, index=idx[2:])
    out = v.compare_candidates({"base": base, "cand": cand}, baseline="base", periods=252,
                               block_length=2, n_trials=12)
    assert calls == [("base", "cand")]
    assert out["comparisons"]["cand"]["estimate"] > 0
    assert out["alignment"]["base"]["dropped"] == 2 and out["misaligned"]
    assert out["aligned_start"] == idx[2].isoformat()
    assert out["n_trials"] == 12
    with pytest.raises(ValueError, match="not aligned"):
        v.compare_candidates({"base": base, "cand": cand}, baseline="base", periods=252,
                             block_length=2, on_misaligned="raise")


# --------------------------------------------------------------- CPCV / PBO


def test_cpcv_splits_and_paths():
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    splits = v.cpcv_splits(idx, n_groups=6, n_test_groups=2, embargo=1)
    assert len(splits) == 15
    for split in splits:
        assert not set(split["train"]) & set(split["test"])
    paths = v.cpcv_paths(6, 2)
    assert len(paths) == 5
    for path in paths:
        assert [g for g, _ in path] == list(range(6))
        for g, s in path:
            assert g in splits[s]["test_groups"]


def test_cpcv_purges_labels_overlapping_test():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"), ["A"]],
        names=["timestamp", "symbol"])
    labels = forward_return(pd.DataFrame({"close": np.linspace(1, 2, 60)}, index=idx), horizon=3)
    split = v.cpcv_splits(None, n_groups=6, n_test_groups=1, labels=labels)[2]  # test days 20..29
    train_stamps = split["train"].get_level_values("timestamp")
    test_start = pd.Timestamp("2024-01-21", tz="UTC")
    assert not ((train_stamps < test_start) & (train_stamps + pd.Timedelta(days=3) >= test_start)).any()


def test_pbo_detects_noise_selection_and_rewards_true_edge():
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(0, 0.01, (400, 20)))
    noisy = v.pbo(noise, n_splits=8)
    assert noisy["n_combinations"] == 70
    assert noisy["pbo"] > 0.3
    edge = noise.copy()
    edge[0] += 0.004
    good = v.pbo(edge, n_splits=8)
    assert good["pbo"] < 0.1
    with pytest.raises(ValueError, match="even"):
        v.pbo(noise, n_splits=7)


# ----------------------------------------------------------------- readback


def _bundle(tmp_path, version):
    pd.DataFrame([{"key": "artifact_schema_version", "value": "1.0"},
                  {"key": "artifact_kind", "value": "backtest"},
                  {"key": "tradelearn_version", "value": version}]).to_csv(
        tmp_path / "metadata.csv", index=False)
    pd.DataFrame([{"key": "sharpe", "value": 1.5}]).to_csv(tmp_path / "summary.csv", index=False)


def test_write_and_load_roundtrip_with_version_and_data_checks(tmp_path):
    from tradelearn import __version__

    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)
    result = v.walk_forward(bars, folds, evaluate=_momentum, params={"window": 5})
    _bundle(tmp_path, __version__)
    path = result.write(tmp_path)
    assert json.loads(path.read_text())["format"] == v.VALIDATION_FORMAT
    with pytest.raises(ValueError, match="already exists"):
        result.write(tmp_path)

    bundle = v.load_bundle(tmp_path, expect_version=__version__)
    assert bundle.summary == {"sharpe": 1.5}
    pd.testing.assert_series_equal(bundle.oos_returns(), result.oos_returns, check_freq=False)
    assert bundle.entry("walk_forward")["complete"] is True
    assert v.verify_inputs(bundle, data=bars) == []
    changed = bars.copy()
    changed.iloc[0, 0] += 1
    assert v.verify_inputs(bundle, data=changed)
    assert v.verify_inputs(result, config={"warmup": 0}) == []
    assert v.verify_inputs(result, config={"warmup": 3}) == ["config 'warmup' differs"]
    with pytest.raises(ValueError, match="expected"):
        v.load_bundle(tmp_path, expect_version="0.0.0")


def test_load_bundle_refuses_schema_and_mixed_versions(tmp_path):
    _bundle(tmp_path, "9.9.9")
    meta = pd.read_csv(tmp_path / "metadata.csv", dtype=str)
    (tmp_path / VALIDATION).write_text(json.dumps(
        {"format": v.VALIDATION_FORMAT, "tradelearn_version": "1.0.0", "entries": {}}))
    with pytest.raises(ValueError, match="mixes"):
        v.load_bundle(tmp_path)
    meta.loc[0, "value"] = "2.0"
    meta.to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="schema"):
        v.load_bundle(tmp_path)


VALIDATION = "validation.json"


def test_verify_engine_trial_record_data_fingerprint():
    from tradelearn.core.trial_records import _describe

    bars = _bars()
    record = {"format": "tradelearn.search.trial.v1", "config": {"data": [_describe(bars)]}}
    assert v.verify_inputs(record, data=bars) == []
    assert v.verify_inputs(record, data=bars.iloc[1:])


# ----------------------------------------------------- scoring clock/coverage


def test_single_in_window_return_is_not_complete():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)

    def evaluate(params, frame, fold, segment):
        return pd.Series([0.01], index=[fold.test_start])

    result = v.walk_forward(bars, folds, evaluate=evaluate)
    assert not result.complete
    assert {f["reason"] for f in result.folds} == {"insufficient_coverage"}
    assert result.folds[0]["scoring"]["missing"] == 19
    assert result.oos_returns.empty


def test_non_finite_return_is_not_complete():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)

    def evaluate(params, frame, fold, segment):
        r = _momentum({"window": 5}, frame, fold, segment)
        if fold.number == 0:
            r.iloc[-1] = np.inf
        return r

    result = v.walk_forward(bars, folds, evaluate=evaluate)
    assert result.folds[0]["status"] == "incomplete"
    assert result.folds[0]["scoring"]["non_finite"] == 1
    assert result.folds[1]["status"] == "completed"


def test_lower_frequency_needs_explicit_clock():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=21)
    weekly = bars.index[bars.index.day_of_week == 0]

    def evaluate(params, frame, fold, segment):
        stamps = frame.index[frame.index.isin(weekly)]
        return pd.Series(0.001, index=stamps)

    implicit = v.walk_forward(bars, folds, evaluate=evaluate)
    assert not implicit.complete
    explicit = v.walk_forward(bars, folds, evaluate=evaluate, clock=weekly)
    assert explicit.complete
    assert explicit.folds[0]["scoring"]["expected"] == 3
    off = v.walk_forward(bars, folds, evaluate=lambda *a: pd.Series(
        0.0, index=a[1].index), clock=weekly)
    assert {f["reason"] for f in off.folds} == {"insufficient_coverage"}
    assert off.folds[0]["scoring"]["off_clock"] > 0


def test_incomplete_train_candidate_fails_fold():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)

    def evaluate(params, frame, fold, segment):
        r = _momentum(params, frame, fold, segment)
        if segment == "train" and params["window"] == 8:
            r.iloc[3] = np.nan
        return r

    result = v.walk_forward(bars, folds, evaluate=evaluate,
                            candidates=[{"window": 3}, {"window": 8}], objective=lambda r: r.sum())
    assert {f["reason"] for f in result.folds} == {"candidate_incomplete"}
    assert result.oos_returns.empty and not result.complete


# ------------------------------------------------------------ custom select


def test_select_sees_only_training_evidence_and_can_return_region_config():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)
    windows = [{"window": w} for w in (3, 4, 5, 6, 8)]
    tested = []

    def evaluate(params, frame, fold, segment):
        if segment == "test":
            tested.append(params)
            parts = [_momentum(p, frame, fold, segment) for p in params["members"]]
            return sum(parts) / len(parts)
        return _momentum(params, frame, fold, segment)

    def select(evidence, window):
        assert set(window) == {"fold", "train_start", "train_end"}
        for item in evidence:
            assert item["returns"].index.max() <= window["train_end"]
        scores = [item["score"] for item in evidence]
        centre = int(np.argmax(np.convolve(scores, np.ones(3) / 3, mode="same")))
        chosen = [i for i in (centre - 1, centre, centre + 1) if 0 <= i < len(evidence)]
        return v.Selection({"members": [evidence[i]["params"] for i in chosen]}, tuple(chosen),
                           {"rule": "stable_region", "smoothed_centre": centre})

    result = v.walk_forward(bars, folds, evaluate=evaluate, candidates=windows,
                            objective=lambda r: float(r.sum()), select=select)
    assert result.complete
    assert len(tested) == len(folds) and all("members" in p for p in tested)
    sel = result.folds[0]["selection"]
    assert sel["evidence"]["rule"] == "stable_region" and len(sel["candidates"]) >= 2
    assert "selected_candidate" not in result.folds[0]
    assert result.payload["config"]["selection"] == "custom"


def test_select_failure_fails_fold_and_select_without_objective_allowed():
    bars = _bars()
    folds = v.walk_forward_folds(bars, train=40, test=20)

    def boom(evidence, window):
        raise KeyError("x")

    failed = v.walk_forward(bars, folds, evaluate=_momentum, candidates=[{"window": 3}],
                            select=boom)
    assert {f["reason"] for f in failed.folds} == {"select_failed"}
    ok = v.walk_forward(bars, folds, evaluate=_momentum, candidates=[{"window": 3}],
                        select=lambda ev, w: ev[0]["params"])
    assert ok.complete and ok.folds[0]["params"] == {"window": 3}
    with pytest.raises(ValueError, match="search mode"):
        v.walk_forward(bars, folds, evaluate=_momentum, objective=lambda r: 0.0)


def test_scenario_non_finite_is_incomplete():
    result = v.run_scenarios(lambda s: pd.Series([0.01, np.nan],
                             index=pd.date_range("2024-01-01", periods=2)),
                             {"base": {}}, periods=252)
    assert result.table().loc["base", "status"] == "incomplete"


def test_compare_with_real_paired_sharpe_diff():
    from tradelearn import metrics

    if "paired_sharpe_diff" not in getattr(metrics, "__all__", []):
        pytest.skip("paired_sharpe_diff not available in this checkout")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    base = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    cand = base + 0.002
    out = v.compare_candidates({"base": base, "cand": cand}, baseline="base", periods=252,
                               block_length=5, draws=500)
    diff = out["comparisons"]["cand"]
    assert diff["estimate"] > 0 and diff["observations"] == 300
    assert not out["misaligned"]


# ------------------------------------------------------ review boundaries


def test_cpcv_purges_by_test_label_information_interval():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"), ["A"]],
        names=["timestamp", "symbol"])
    labels = forward_return(pd.DataFrame({"close": np.linspace(1, 2, 60)}, index=idx), horizon=4)
    split = v.cpcv_splits(None, n_groups=6, n_test_groups=1, labels=labels)[2]  # days 20..29
    lo, hi = split["test_intervals"][0]
    assert pd.Timestamp(hi) == pd.Timestamp("2024-02-03", tz="UTC")  # day 29 + 4 days
    train_stamps = split["train"].get_level_values("timestamp")
    # samples starting within 4 days after the test block overlap its labels
    after = (train_stamps > pd.Timestamp("2024-01-30", tz="UTC")) & (
        train_stamps <= pd.Timestamp(hi))
    assert not after.any()
    assert pd.Timestamp("2024-02-04", tz="UTC") in train_stamps


def test_pbo_keeps_undefined_splits_and_blocks_when_none_valid():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(0, 0.01, (80, 3)))
    frame.iloc[:40, 1] = 0.0  # zero variance in the first half
    partial = v.pbo(frame, n_splits=4)
    assert partial["status"] == "partial" and not partial["complete"]
    assert partial["invalid_combinations"] and partial["pbo"] is not None
    assert partial["n_valid_combinations"] + len(partial["invalid_combinations"]) == 6
    flat = pd.DataFrame(0.0, index=range(40), columns=["a", "b"])
    blocked = v.pbo(flat, n_splits=4)
    assert blocked["status"] == "blocked" and blocked["pbo"] is None


def test_write_validation_single_writer_and_no_version_mixing(tmp_path):
    entry = v.Evidence("pbo", "x", {"pbo": 0.1})
    (tmp_path / "validation.json.lock").write_text("123")
    with pytest.raises(FileExistsError, match="another writer"):
        entry.write(tmp_path)
    assert not (tmp_path / "validation.json").exists()
    (tmp_path / "validation.json.lock").unlink()
    entry.write(tmp_path)
    v.Evidence("pbo", "y", {"pbo": 0.2}).write(tmp_path)
    doc = json.loads((tmp_path / "validation.json").read_text())
    assert set(doc["entries"]) == {"x", "y"}
    assert not list(tmp_path.glob("*.tmp")) and not (tmp_path / "validation.json.lock").exists()
    doc["tradelearn_version"] = "0.0.1"
    (tmp_path / "validation.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="mixing versions"):
        v.Evidence("pbo", "z", {}).write(tmp_path, overwrite=True)
    assert not (tmp_path / "validation.json.lock").exists()


# ------------------------------------------------------ PBO custom selector


def _pbo_frame():
    rng = np.random.default_rng(5)
    return pd.DataFrame(rng.normal(0, 0.01, (160, 4)), columns=["a", "b", "c", "d"])


def test_pbo_selector_sees_only_in_sample_rows_and_can_pick_other_candidates():
    frame = _pbo_frame()
    seen = []

    def lowest_vol(train):
        seen.append(train.index)
        return train.std().idxmin()

    custom = v.pbo(frame, n_splits=4, select=lowest_vol)
    default = v.pbo(frame, n_splits=4)
    assert len(seen) == 6
    blocks = np.array_split(np.arange(len(frame)), 4)
    for index, item in zip(seen, custom["selections"]):
        assert len(index) == 80
        oos = np.concatenate([blocks[b] for b in range(4) if b not in item["is_blocks"]])
        assert not set(index) & set(oos)
    assert custom["rule"] == "custom" and default["rule"] == "max_sharpe"
    assert [s["selected"] for s in custom["selections"]] != [
        s["selected"] for s in default["selections"]]
    for item in default["selections"]:
        assert item["selected"] in frame.columns


def test_pbo_invalid_selector_marks_combinations_invalid():
    frame = _pbo_frame()
    bad = v.pbo(frame, n_splits=4, select=lambda train: "not-a-candidate")
    assert bad["status"] == "blocked" and bad["pbo"] is None
    assert {i["reason"] for i in bad["invalid_combinations"]} == {"selector_invalid"}

    def boom(train):
        raise RuntimeError("x")

    raised = v.pbo(frame, n_splits=4, select=boom)
    assert {i["error_type"] for i in raised["invalid_combinations"]} == {"RuntimeError"}
    mixed = v.pbo(frame, n_splits=4,
                  select=lambda train: "a" if train.index[0] == 0 else ["a", "b"])
    assert mixed["status"] == "partial"


def test_pbo_requires_chronological_rows_and_unique_candidate_labels():
    data = pd.DataFrame({"a": [1., 2., 3., 4.], "b": [4., 3., 2., 1.]})
    with pytest.raises(ValueError, match="time order"):
        v.pbo(data.iloc[::-1], n_splits=2)
    with pytest.raises(ValueError, match="candidate labels"):
        v.pbo(data.set_axis(["a", "a"], axis=1), n_splits=2)


def test_pbo_can_select_a_preregistered_tuple_configuration():
    data = pd.DataFrame({(3, 11): [1., 2., 3., 4.], (5, 15): [4., 3., 2., 1.]})
    result = v.pbo(data, n_splits=2, select=lambda train: (5, 15))
    assert result["rule"] == "custom"
    assert result["n_valid_combinations"] == 2
    assert [row["selected"] for row in result["selections"]] == [(5, 15), (5, 15)]


def test_reads_local_research_manifest_and_trials_without_a_backtest(tmp_path):
    from tradelearn import __version__
    from tradelearn.core.trial_records import _describe

    data = _bars()
    trial = {"format": "tradelearn.search.trial.v1", "config": {"data": _describe(data)}}
    (tmp_path / "manifest.json").write_text(json.dumps({"trade_learn_version": __version__}))
    (tmp_path / "research_result.json").write_text(json.dumps({"name": "offline", "steps": []}))
    (tmp_path / "trial_records.json").write_text(json.dumps([trial]))
    bundle = v.load_bundle(tmp_path, expect_version=__version__)
    assert bundle.research["name"] == "offline"
    assert bundle.manifest["trade_learn_version"] == __version__
    assert bundle.trial_records == [trial]
    assert v.verify_inputs(bundle, data=data) == []


def test_input_verification_checks_market_metadata_even_when_data_values_match():
    from tradelearn.core.trial_records import _describe

    data = _bars()
    data.attrs["market"] = "continuous"
    old = {"config": {"data": _describe(data)}}
    modified = data.copy()
    modified.attrs["market"] = "exchange"
    assert _describe(data)["sha256"] == _describe(modified)["sha256"]
    assert v.verify_inputs(old, data=modified) == ["recorded data metadata differs"]
