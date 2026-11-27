"""Deterministic research validation orchestration and local evidence readback.

Thin layer over existing pieces: data is sliced with the same datetime rules as
``time_split``, labels follow ``purged_splits`` (``label_end`` before the next
section), each evaluation is recorded with the shared ``trial.v1`` record, and
local readback reads the files ``write_artifact_bundle`` already produces.
Nothing here runs a backtest itself: callers pass an ``evaluate`` callback that
uses the normal Engine/Lite API.

Evidence rules
--------------
* Fold windows never overlap out of sample; a holdout never enters any fold.
* Training data ends before the test window (minus embargo); with labels, a
  training sample is kept only if its label ended before the test window.
* Every test call is a fresh, independent run: no strategy/account state is
  carried across folds. ``warmup`` only prepends history rows to the test call;
  returns outside the test window are discarded and counted.
* Failed or incomplete folds and candidates stay in the evidence. Only fully
  completed folds contribute OOS returns, and ``complete`` is ``False`` if any
  fold or candidate did not complete.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradelearn.research.run import current_run
from tradelearn.research.split import _datetime_values

VALIDATION_FORMAT = "tradelearn.research.validation.v1"
VALIDATION_FILE = "validation.json"
_FOLD_STATUSES = ("completed", "failed", "incomplete")


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Fold:
    """One walk-forward fold; all bounds are inclusive timestamps."""

    number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def _stamps(index: Any) -> pd.DatetimeIndex:
    if isinstance(index, pd.DataFrame | pd.Series):
        index = index.index
    values = _datetime_values(pd.Index(index) if not isinstance(index, pd.Index) else index,
                              level=None)
    return pd.DatetimeIndex(values.unique()).sort_values()


def _align(value: Any, stamps: pd.DatetimeIndex) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    tz = stamps.tz
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def walk_forward_folds(
    index: Any,
    *,
    train: int,
    test: int,
    step: int | None = None,
    embargo: int = 0,
    expanding: bool = False,
    holdout_start: Any | None = None,
) -> list[Fold]:
    """Build deterministic walk-forward folds over unique sorted timestamps.

    Sizes count unique timestamps. ``step`` defaults to ``test`` and may not be
    smaller (overlapping OOS windows are refused). ``embargo`` timestamps are
    skipped between train and test. Only complete test windows are emitted.
    Timestamps on or after ``holdout_start`` are excluded from every fold.
    """
    train = _positive_int("train", train)
    test = _positive_int("test", test)
    step = test if step is None else _positive_int("step", step)
    embargo = _positive_int("embargo", embargo, minimum=0)
    if step < test:
        raise ValueError("step < test would overlap out-of-sample windows")
    stamps = _stamps(index)
    if holdout_start is not None:
        stamps = stamps[stamps < _align(holdout_start, stamps)]
    folds: list[Fold] = []
    test_lo = train + embargo
    while test_lo + test <= len(stamps):
        train_hi = test_lo - embargo - 1
        train_lo = 0 if expanding else train_hi - train + 1
        folds.append(
            Fold(
                number=len(folds),
                train_start=stamps[train_lo],
                train_end=stamps[train_hi],
                test_start=stamps[test_lo],
                test_end=stamps[test_lo + test - 1],
            )
        )
        test_lo += step
    if not folds:
        raise ValueError("not enough timestamps for one train/test fold")
    return folds


def _check_folds(folds: Sequence[Fold], holdout: pd.Timestamp | None) -> None:
    if not folds:
        raise ValueError("folds must be nonempty")
    previous_end = None
    for fold in folds:
        if not fold.train_start <= fold.train_end < fold.test_start <= fold.test_end:
            raise ValueError(f"fold {fold.number}: train must end before test starts")
        if previous_end is not None and fold.test_start <= previous_end:
            raise ValueError(f"fold {fold.number}: out-of-sample windows overlap or are unordered")
        if holdout is not None and fold.test_end >= holdout:
            raise ValueError(f"fold {fold.number}: reaches into the holdout")
        previous_end = fold.test_end


# ---------------------------------------------------------------------------
# Evidence containers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Evidence:
    """A named, JSON-serializable validation payload."""

    kind: str
    name: str
    payload: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, **_jsonable(self.payload)}

    def write(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Merge this entry into ``<directory>/validation.json``."""
        return write_validation(directory, self, overwrite=overwrite)


@dataclasses.dataclass
class WalkForwardResult(Evidence):
    folds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    trials: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    oos_returns: pd.Series = dataclasses.field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def complete(self) -> bool:
        return bool(self.payload["complete"])

    def summary(self) -> dict[str, Any]:
        """Small payload suitable for ``ResearchRun.finish(artifacts=...)``."""
        keys = ("mode", "complete", "fold_status", "oos_observations", "oos_start", "oos_end")
        return {key: self.payload[key] for key in keys}

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "folds": _jsonable(self.folds),
            "trials": _jsonable(self.trials),
            "oos_returns": _series_payload(self.oos_returns),
        }


@dataclasses.dataclass
class ScenarioResult(Evidence):
    returns: dict[str, pd.Series] = dataclasses.field(default_factory=dict)
    trials: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.payload["rows"]).set_index("scenario")

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "trials": _jsonable(self.trials)}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def _extract_returns(result: Any) -> pd.Series:
    if isinstance(result, pd.Series):
        returns = result
    elif isinstance(result, Mapping) and "returns" in result:
        returns = result["returns"]
    elif hasattr(result, "returns"):
        returns = result.returns
    else:
        raise TypeError("evaluate must return a returns Series or an object with .returns")
    returns = pd.Series(returns, dtype="float64")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must be indexed by a DatetimeIndex")
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be unique and sorted")
    return returns


def _clip(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, int]:
    inside = (returns.index >= start) & (returns.index <= end)
    return returns[inside], int((~inside).sum())


def _window(values: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.asarray((values >= start) & (values <= end))


@dataclasses.dataclass(frozen=True)
class Selection:
    """What a custom ``select`` hands to the test call, plus its own evidence.

    ``params`` may be one candidate's config or a merged/region config; it is
    passed to ``evaluate`` unchanged. ``candidates`` lists the candidate
    indexes the selection is built from.
    """

    params: Any
    candidates: tuple[int, ...] = ()
    evidence: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _expected_clock(clock: Any, stamps: pd.DatetimeIndex, start, end, fold) -> pd.DatetimeIndex:
    if clock is None:
        expected = stamps
    elif callable(clock):
        expected = pd.DatetimeIndex(clock(fold))
    else:
        expected = pd.DatetimeIndex(clock)
    expected = expected.unique().sort_values()
    return expected[(expected >= start) & (expected <= end)]


def _coverage(returns: pd.Series, expected: pd.DatetimeIndex) -> dict[str, Any]:
    on_clock = returns.index.isin(expected)
    finite = np.isfinite(returns.to_numpy())
    covered = int((on_clock & finite).sum())
    return {
        "expected": len(expected),
        "observed": int(on_clock.sum()),
        "off_clock": int((~on_clock).sum()),
        "non_finite": int((on_clock & ~finite).sum()),
        "missing": len(expected) - covered,
        "coverage": covered / len(expected) if len(expected) else 0.0,
    }


def walk_forward(
    data: pd.DataFrame | pd.Series,
    folds: Sequence[Fold],
    *,
    evaluate: Callable[[Any, Any, Fold, str], Any],
    params: Mapping[str, Any] | None = None,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    objective: Callable[[pd.Series], float] | None = None,
    select: Callable[[list[dict[str, Any]], dict[str, Any]], Any] | None = None,
    labels: pd.DataFrame | None = None,
    warmup: int = 0,
    clock: Any | None = None,
    min_coverage: float = 1.0,
    holdout_start: Any | None = None,
    name: str = "walk_forward",
) -> WalkForwardResult:
    """Run fixed or searched walk-forward validation through a callback.

    ``evaluate(params, data_slice, fold, segment)`` runs one independent
    historical backtest and returns periodic returns (``segment`` is
    ``"train"`` or ``"test"``).

    Scoring clock: returns are scored against an explicit expected index per
    window -- ``clock=None`` uses the timestamps of the data slice passed (after
    purging), a DatetimeIndex
    (e.g. weekly rebalance dates) or ``clock(fold) -> index`` declares a lower
    frequency. A return per clock stamp is required (0 when flat); this does
    not assume continuous holding. Off-clock stamps, non-finite values or
    coverage below ``min_coverage`` leave the fold/candidate ``incomplete``.

    * Fixed: ``candidates=None``; ``params`` (default ``{}``) is evaluated on
      each test window only.
    * Search: each candidate is evaluated on the fold's training data. Default
      selection is the best finite ``objective`` (ties -> lowest index). A
      custom ``select(train_evidence, train_window)`` sees only training
      evidence (params, score, train returns) and the train bounds, and returns
      a params mapping or a :class:`Selection` (e.g. a stable-region config).
      If any candidate fails or is incomplete, the fold fails: a selection over
      a partial set is not evidence.
    """
    from tradelearn.core.trial_records import _describe, execute_trial

    if candidates is not None:
        if objective is None and select is None:
            raise ValueError("search walk-forward requires an explicit objective or select")
        candidates = [dict(item) for item in candidates]
        if not candidates:
            raise ValueError("candidates must be nonempty")
        if params is not None:
            raise ValueError("pass either fixed params or candidates, not both")
    elif select is not None or objective is not None:
        raise ValueError("objective/select apply to search mode only")
    if not 0 < float(min_coverage) <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    warmup = _positive_int("warmup", warmup, minimum=0)
    values = pd.DatetimeIndex(_datetime_values(data.index, level=None))
    stamps = pd.DatetimeIndex(values.unique()).sort_values()
    holdout = None if holdout_start is None else _align(holdout_start, stamps)
    _check_folds(folds, holdout)
    label_end = None
    if labels is not None:
        if "label_end" not in labels:
            raise ValueError("labels need label_end")
        if data.index.has_duplicates:
            raise ValueError("data index must be unique when labels are supplied")
        label_end = pd.to_datetime(labels["label_end"].reindex(data.index), utc=True)

    mode = "fixed" if candidates is None else "search"
    trials: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    oos_parts: list[pd.Series] = []

    def run(trial_params, segment, fold, frame, start, end, extra):
        config = {"validation": name, "segment": segment, "fold": fold.to_dict(), **extra}
        frame_stamps = pd.DatetimeIndex(_datetime_values(frame.index, level=None))
        expected = _expected_clock(clock, frame_stamps, start, end, fold)

        def call():
            returns = _extract_returns(evaluate(deepcopy(trial_params), frame, fold, segment))
            clipped, outside = _clip(returns, start, end)
            record = trials[-1]
            record["returns_outside_window"] = outside
            record["scoring"] = _coverage(clipped, expected)
            return clipped

        def score(clipped):
            scoring = trials[-1]["scoring"]
            if not _covered(scoring, min_coverage) or objective is None:
                return None
            return objective(clipped)

        try:
            returns = execute_trial(trials, trial_params, config, call,
                                    score if segment == "train" else None)
        except Exception:
            return None, trials[-1]
        trial = trials[-1]
        trial["covered"] = _covered(trial["scoring"], min_coverage)
        return returns, trial

    for fold in folds:
        record: dict[str, Any] = {**fold.to_dict(), "status": "completed", "reason": None}
        fold_records.append(record)
        test_lo = int(stamps.searchsorted(fold.test_start))
        if test_lo < warmup:
            record.update(status="incomplete", reason="insufficient_warmup",
                          warmup_available=test_lo)
            continue
        if mode == "fixed":
            chosen: Any = dict(params or {})
        else:
            train_mask = _window(values, fold.train_start, fold.train_end)
            if label_end is not None:
                train_mask &= np.asarray(label_end.notna() & (label_end < _utc(fold.test_start)))
            train_frame = data.loc[train_mask].copy()
            record["train_rows"] = int(train_mask.sum())
            evidence = []
            for number, candidate in enumerate(candidates):
                returns, trial = run(candidate, "train", fold, train_frame,
                                     fold.train_start, fold.train_end, {"candidate": number})
                score = trial.get("evaluation")
                evidence.append({
                    "candidate": number,
                    "params": deepcopy(candidate),
                    "status": trial["status"] if trial.get("covered", True) else "incomplete",
                    "score": score if _finite(score) else None,
                    "scoring": trial.get("scoring"),
                    "returns": returns,
                })
                trial["eligible"] = evidence[-1]["status"] == "completed" and (
                    objective is None or _finite(score))
            record["candidate_scores"] = [item["score"] for item in evidence]
            failed = [e["candidate"] for e in evidence if e["status"] == "failed"]
            incomplete = [e["candidate"] for e in evidence if e["status"] == "incomplete"]
            if failed or incomplete:
                record.update(status="failed", reason="candidate_failed" if failed
                              else "candidate_incomplete",
                              failed_candidates=failed, incomplete_candidates=incomplete)
                continue
            if select is None:
                eligible = [e["candidate"] for e in evidence if e["score"] is not None]
                if not eligible:
                    record.update(status="failed", reason="no_finite_objective")
                    continue
                best = max(eligible, key=lambda i: (evidence[i]["score"], -i))
                selection = Selection(candidates[best], (best,), {"rule": "best_objective"})
            else:
                window = {"fold": fold.number, "train_start": fold.train_start,
                          "train_end": fold.train_end}
                try:
                    picked = select(deepcopy(evidence), window)
                except Exception as exc:
                    record.update(status="failed", reason="select_failed",
                                  error_type=type(exc).__name__)
                    continue
                selection = picked if isinstance(picked, Selection) else Selection(picked)
            record["selection"] = {
                "candidates": list(selection.candidates),
                "evidence": _describe(dict(selection.evidence)),
            }
            if len(selection.candidates) == 1:
                record["selected_candidate"] = selection.candidates[0]
            chosen = selection.params
        record["params"] = _describe(chosen)
        start_stamp = stamps[test_lo - warmup]
        test_mask = _window(values, start_stamp, fold.test_end)
        returns, trial = run(chosen, "test", fold, data.loc[test_mask].copy(),
                             fold.test_start, fold.test_end, {"warmup": warmup})
        record["test_trial"] = trial["index"]
        record["scoring"] = trial.get("scoring")
        if returns is None:
            record.update(status="failed", reason="test_failed", error_type=trial.get("error_type"))
            continue
        if not trial["covered"]:
            record.update(status="incomplete", reason="insufficient_coverage")
            continue
        record["oos_observations"] = len(returns)
        oos_parts.append(returns)

    oos = (pd.concat(oos_parts) if oos_parts else pd.Series(dtype="float64")).rename("returns")
    if oos.index.has_duplicates or not oos.index.is_monotonic_increasing:
        raise RuntimeError("out-of-sample windows overlap")
    status_counts = {key: sum(r["status"] == key for r in fold_records) for key in _FOLD_STATUSES}
    payload = {
        "mode": mode,
        "complete": status_counts["completed"] == len(fold_records),
        "fold_status": status_counts,
        "state_policy": "independent_run_per_call",
        "config": {
            "warmup": warmup,
            "clock": "data" if clock is None else ("callable" if callable(clock)
                                                   else _describe(pd.Series(pd.DatetimeIndex(clock)))),
            "min_coverage": float(min_coverage),
            "holdout_start": None if holdout is None else holdout.isoformat(),
            "labels_purged": label_end is not None,
            "selection": None if mode == "fixed" else ("custom" if select else "best_objective"),
            "params": None if params is None else _describe(dict(params)),
            "candidates": None if candidates is None else _describe(candidates),
            "data": _describe(data),
        },
        "oos_observations": len(oos),
        "oos_start": oos.index[0].isoformat() if len(oos) else None,
        "oos_end": oos.index[-1].isoformat() if len(oos) else None,
    }
    result = WalkForwardResult("walk_forward", name, payload, fold_records, trials, oos)
    run_ = current_run()
    if run_ is not None:
        run_.record_step(name, "validation.walk_forward", _jsonable(result.summary()))
    return result


def _covered(scoring: Mapping[str, Any] | None, minimum: float) -> bool:
    return bool(scoring) and scoring["expected"] > 0 and not scoring["off_clock"] \
        and not scoring["non_finite"] and scoring["coverage"] >= minimum


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Scenarios / costs
# ---------------------------------------------------------------------------


def run_scenarios(
    evaluate: Callable[[dict[str, Any]], Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    periods: int,
    name: str = "scenarios",
) -> ScenarioResult:
    """Evaluate the same strategy under named cost/fill/data scenarios.

    ``evaluate(scenario)`` runs one independent backtest and returns returns.
    Failures stay as rows with ``status="failed"``; nothing is dropped.
    """
    from tradelearn.core.trial_records import execute_trial
    from tradelearn.metrics import max_drawdown, sharpe

    if not scenarios:
        raise ValueError("scenarios must be nonempty")
    periods = _positive_int("periods", periods)
    trials: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    returns_map: dict[str, pd.Series] = {}
    for scenario_name, scenario in scenarios.items():
        scenario = dict(scenario)
        row: dict[str, Any] = {"scenario": str(scenario_name), "status": "completed"}
        try:
            returns = execute_trial(trials, scenario, {"validation": name,
                                    "scenario": str(scenario_name)},
                                    lambda s=scenario: _extract_returns(evaluate(deepcopy(s))))
        except Exception:
            row.update(status="failed", error_type=trials[-1].get("error_type"))
            rows.append(row)
            continue
        returns_map[str(scenario_name)] = returns
        non_finite = int((~np.isfinite(returns.to_numpy())).sum())
        if returns.empty or non_finite:
            row.update(status="incomplete", observations=len(returns), non_finite=non_finite)
            rows.append(row)
            continue
        row.update(_return_stats(returns, periods, sharpe, max_drawdown))
        rows.append(row)
    payload = {
        "periods": periods,
        "rows": rows,
        "complete": all(row["status"] == "completed" for row in rows),
    }
    result = ScenarioResult("scenarios", name, payload, returns_map, trials)
    run_ = current_run()
    if run_ is not None:
        run_.record_step(name, "validation.scenarios",
                         {"scenarios": [r["scenario"] for r in rows], "complete": payload["complete"]})
    return result


def _return_stats(returns: pd.Series, periods: int, sharpe, max_drawdown) -> dict[str, Any]:
    clean = returns.dropna()
    stats = {
        "observations": len(clean),
        "start": clean.index[0].isoformat() if len(clean) else None,
        "end": clean.index[-1].isoformat() if len(clean) else None,
        "total_return": float((1 + clean).prod() - 1) if len(clean) else None,
        "sharpe": None,
        "max_drawdown": None,
    }
    if len(clean) >= 2 and clean.std(ddof=1) > 0:
        stats["sharpe"] = float(sharpe(clean, periods))
    if len(clean):
        stats["max_drawdown"] = float(max_drawdown(clean))
    return stats


# ---------------------------------------------------------------------------
# Candidate comparison
# ---------------------------------------------------------------------------


def compare_candidates(
    returns: Mapping[str, pd.Series],
    *,
    baseline: str,
    periods: int,
    block_length: int,
    draws: int = 4000,
    seed: int = 0,
    confidence: float = 0.95,
    n_trials: int | None = None,
    on_misaligned: str = "report",
    name: str = "comparison",
) -> Evidence:
    """Paired Sharpe differences of each candidate against ``baseline``.

    Series are intersected on common timestamps; rows dropped per candidate and
    the retained span are reported (``on_misaligned="raise"`` refuses instead).
    ``estimate`` is ``candidate - baseline`` (``paired_sharpe_diff`` is called
    with ``left=baseline, right=candidate``). ``n_trials`` records how many
    configurations were tried in total, for multiple-testing review.
    """
    from tradelearn import metrics

    if baseline not in returns:
        raise ValueError(f"baseline {baseline!r} not in returns")
    if len(returns) < 2:
        raise ValueError("need at least one candidate besides the baseline")
    if on_misaligned not in {"report", "raise"}:
        raise ValueError("on_misaligned must be report or raise")
    try:
        paired = metrics.paired_sharpe_diff
    except AttributeError as exc:  # pragma: no cover - depends on metrics owner
        raise ImportError("tradelearn.metrics.paired_sharpe_diff is required") from exc
    series = {}
    for key, values in returns.items():
        values = pd.Series(values, dtype="float64")
        if values.index.has_duplicates:
            raise ValueError(f"{key}: returns index must be unique")
        series[str(key)] = values.sort_index()
    frame = pd.concat(series, axis=1, join="outer")
    aligned = frame.dropna()
    if aligned.empty:
        raise ValueError("candidates share no common timestamps")
    alignment = {
        key: {
            "observations": int(values.notna().sum()),
            "dropped": int(values.notna().sum() - len(aligned)),
            "start": _iso(values.dropna().index.min()),
            "end": _iso(values.dropna().index.max()),
        }
        for key, values in frame.items()
    }
    misaligned = any(item["dropped"] for item in alignment.values())
    if misaligned and on_misaligned == "raise":
        raise ValueError(f"returns are not aligned: {alignment}")
    comparisons = {}
    for key in series:
        if key == str(baseline):
            continue
        result = paired(aligned[str(baseline)], aligned[key], periods,
                        block_length=block_length, draws=draws, seed=seed, confidence=confidence)
        comparisons[key] = (dataclasses.asdict(result) if dataclasses.is_dataclass(result)
                            else dict(result))
    payload = {
        "baseline": str(baseline),
        "direction": "candidate_minus_baseline",
        "periods": periods,
        "block_length": block_length,
        "draws": draws,
        "seed": seed,
        "confidence": confidence,
        "n_candidates": len(series) - 1,
        "n_trials": n_trials,
        "aligned_observations": len(aligned),
        "aligned_start": _iso(aligned.index[0]),
        "aligned_end": _iso(aligned.index[-1]),
        "misaligned": misaligned,
        "alignment": alignment,
        "comparisons": comparisons,
    }
    run_ = current_run()
    if run_ is not None:
        run_.record_step(name, "validation.compare", {"baseline": str(baseline),
                         "n_candidates": payload["n_candidates"], "misaligned": misaligned})
    return Evidence("comparison", name, payload)


def _iso(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


# ---------------------------------------------------------------------------
# CPCV / PBO
# ---------------------------------------------------------------------------


def cpcv_splits(
    index: Any,
    *,
    n_groups: int,
    n_test_groups: int,
    embargo: int = 0,
    labels: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Combinatorial purged cross-validation splits (Lopez de Prado).

    Unique timestamps are cut into ``n_groups`` contiguous groups; each
    combination of ``n_test_groups`` groups is a test set. ``embargo``
    timestamps after each test block leave the training set. With ``labels``
    (``label_end``), training samples whose label interval touches a test block
    are purged and train/test are returned as label index entries; otherwise as
    timestamps. Splits are in lexicographic group order.
    """
    n_groups = _positive_int("n_groups", n_groups, minimum=2)
    n_test_groups = _positive_int("n_test_groups", n_test_groups)
    embargo = _positive_int("embargo", embargo, minimum=0)
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be smaller than n_groups")
    stamps = _stamps(labels if labels is not None else index)
    if len(stamps) < n_groups:
        raise ValueError("fewer timestamps than groups")
    groups = np.array_split(np.arange(len(stamps)), n_groups)
    if labels is not None:
        if "label_end" not in labels:
            raise ValueError("labels need label_end")
        sample_start = pd.DatetimeIndex(_datetime_values(labels.index, level=None))
        sample_end = pd.DatetimeIndex(pd.to_datetime(labels["label_end"], utc=True))
        if sample_start.tz is None:
            sample_end = sample_end.tz_localize(None)
    splits = []
    for number, chosen in enumerate(combinations(range(n_groups), n_test_groups)):
        test_pos = np.concatenate([groups[g] for g in chosen])
        blocked = np.zeros(len(stamps), dtype=bool)
        blocked[test_pos] = True
        blocks = _blocks(sorted(chosen))
        for _, last in blocks:
            end = groups[last][-1]
            blocked[end + 1 : end + 1 + embargo] = True
        train_stamps = stamps[~blocked]
        test_stamps = stamps[test_pos]
        split = {"split": number, "test_groups": list(chosen)}
        if labels is None:
            split.update(train=train_stamps, test=test_stamps)
        else:
            valid = np.asarray(sample_end.notna())
            in_test = np.asarray(sample_start.isin(test_stamps)) & valid
            in_train = np.asarray(sample_start.isin(train_stamps)) & valid
            intervals = []
            for first, last in blocks:
                lo, hi = stamps[groups[first][0]], stamps[groups[last][-1]]
                block = in_test & np.asarray((sample_start >= lo) & (sample_start <= hi))
                # The block's information reaches as far as its test labels do.
                info_end = max(hi, sample_end[block].max()) if block.any() else hi
                intervals.append((lo, info_end))
                in_train &= ~np.asarray((sample_start <= info_end) & (sample_end >= lo))
            split.update(
                train=labels.index[in_train],
                test=labels.index[in_test],
                test_intervals=[(lo.isoformat(), hi.isoformat()) for lo, hi in intervals],
            )
        splits.append(split)
    return splits


def _blocks(groups: list[int]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for g in groups:
        if blocks and blocks[-1][1] == g - 1:
            blocks[-1] = (blocks[-1][0], g)
        else:
            blocks.append((g, g))
    return blocks


def cpcv_paths(n_groups: int, n_test_groups: int) -> list[list[tuple[int, int]]]:
    """Assemble CPCV backtest paths as ``[(group, split), ...]`` per path.

    There are ``C(n-1, k-1)`` paths; path ``j`` uses, for every group, the
    ``j``-th split (in ``cpcv_splits`` order) that tests that group.
    """
    owners: dict[int, list[int]] = {g: [] for g in range(n_groups)}
    for number, chosen in enumerate(combinations(range(n_groups), n_test_groups)):
        for g in chosen:
            owners[g].append(number)
    n_paths = math.comb(n_groups - 1, n_test_groups - 1)
    return [[(g, owners[g][j]) for g in range(n_groups)] for j in range(n_paths)]


def pbo(
    performance: pd.DataFrame,
    *,
    n_splits: int = 16,
    select: Callable[[pd.DataFrame], Any] | None = None,
    name: str = "pbo",
) -> Evidence:
    """Probability of backtest overfitting via CSCV (Bailey et al. 2017).

    ``select(train_frame) -> column label`` replaces the default max-Sharpe
    rule with the real selection method; it receives only the in-sample rows
    and must return one existing column (an actually backtested candidate or a
    combination listed as its own column beforehand). Any other return value
    or an exception marks that combination invalid -- returns of different
    columns are never blended into a synthetic holding. Every combination's
    selection is kept in ``selections``.

    ``performance`` holds one column of periodic returns per tried candidate,
    rows in time order, **all** tried candidates included. Rows are cut into
    ``n_splits`` contiguous blocks; for each half/half combination the best
    in-sample (per-period Sharpe) candidate's out-of-sample relative rank is
    turned into a logit. PBO is the share of logits <= 0.
    """
    frame = pd.DataFrame(performance).astype("float64")
    if frame.shape[1] < 2:
        raise ValueError("pbo needs at least two candidates")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("performance rows must be unique and in time order")
    if not frame.columns.is_unique:
        raise ValueError("performance requires unique candidate labels")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("performance must not contain missing or infinite values; align first")
    n_splits = _positive_int("n_splits", n_splits, minimum=2)
    if n_splits % 2:
        raise ValueError("n_splits must be even")
    if len(frame) < 2 * n_splits:
        raise ValueError("need at least two rows per split")
    blocks = np.array_split(np.arange(len(frame)), n_splits)
    values = frame.to_numpy()
    n = values.shape[1]
    columns = list(frame.columns)
    positions = {label: index for index, label in enumerate(columns)}
    logits, is_best, oos_best, invalid, selections = [], [], [], [], []
    for number, chosen in enumerate(combinations(range(n_splits), n_splits // 2)):
        is_rows = np.concatenate([blocks[b] for b in chosen])
        oos_rows = np.concatenate([blocks[b] for b in range(n_splits) if b not in chosen])
        is_metric = _per_period_sharpe(values[is_rows])
        oos_metric = _per_period_sharpe(values[oos_rows])
        base = {"combination": number, "is_blocks": list(chosen)}
        # Ranking needs every candidate's OOS metric; the default rule also
        # needs every IS metric (e.g. zero variance leaves it undefined).
        if np.isnan(oos_metric).any() or (select is None and np.isnan(is_metric).any()):
            invalid.append({**base, "reason": "undefined_metric",
                            "undefined_is": np.flatnonzero(np.isnan(is_metric)).tolist(),
                            "undefined_oos": np.flatnonzero(np.isnan(oos_metric)).tolist()})
            continue
        if select is None:
            best = int(np.argmax(is_metric))
        else:
            try:
                label = select(frame.iloc[is_rows].copy())
            except Exception as exc:
                invalid.append({**base, "reason": "selector_failed",
                                "error_type": type(exc).__name__})
                continue
            try:
                best = positions[label]
            except (KeyError, TypeError):
                invalid.append({**base, "reason": "selector_invalid",
                                "returned": _describe_label(label)})
                continue
        omega = float(pd.Series(oos_metric).rank(method="average").iloc[best]) / (n + 1)
        logit = math.log(omega / (1 - omega))
        selections.append({**base, "selected": columns[best], "is_metric": _num(is_metric[best]),
                           "oos_metric": _num(oos_metric[best]), "logit": logit})
        logits.append(logit)
        is_best.append(float(is_metric[best]))
        oos_best.append(float(oos_metric[best]))
    total = math.comb(n_splits, n_splits // 2)
    logits_arr, is_arr, oos_arr = np.asarray(logits), np.asarray(is_best), np.asarray(oos_best)
    usable = len(logits) > 0
    fit = np.isfinite(is_arr)
    slope = (float(np.polyfit(is_arr[fit], oos_arr[fit], 1)[0])
             if fit.sum() >= 2 and np.ptp(is_arr[fit]) > 0 else None)
    payload = {
        "status": "complete" if not invalid else ("partial" if usable else "blocked"),
        "complete": not invalid,
        "pbo": float((logits_arr <= 0).mean()) if usable else None,
        "n_candidates": n,
        "n_splits": n_splits,
        "n_combinations": total,
        "n_valid_combinations": len(logits),
        "invalid_combinations": invalid,
        "observations": len(frame),
        "block_rows": [len(b) for b in blocks],
        "metric": "per_period_sharpe",
        "rule": "max_sharpe" if select is None else "custom",
        "selector": None if select is None else getattr(select, "__qualname__", type(select).__name__),
        "selections": selections,
        "logit_median": float(np.median(logits_arr)) if usable else None,
        "prob_oos_loss": float((oos_arr < 0).mean()) if usable else None,
        "degradation_slope": slope,
        "logits": logits,
        "candidates": [str(c) for c in frame.columns],
    }
    run_ = current_run()
    if run_ is not None:
        run_.record_step(name, "validation.pbo", {k: payload[k] for k in
                         ("status", "rule", "pbo", "n_candidates", "n_splits", "n_combinations",
                          "n_valid_combinations")})
    return Evidence("pbo", name, payload)


def _num(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _describe_label(label: Any) -> Any:
    if label is None or isinstance(label, str | int | float | bool):
        return label
    return type(label).__name__


def _per_period_sharpe(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std, np.nan)


# ---------------------------------------------------------------------------
# Local persistence and readback
# ---------------------------------------------------------------------------


def write_validation(directory: str | Path, *entries: Evidence, overwrite: bool = False) -> Path:
    """Merge evidence entries into ``<directory>/validation.json``.

    Single-writer contract: a ``validation.json.lock`` file is created
    exclusively for the read-merge-replace; a concurrent writer gets
    ``FileExistsError`` instead of silently losing entries (a stale lock left
    by a crashed process must be removed by hand). Each write uses a unique
    temporary file and an atomic replace. Files from another tradelearn
    version are never merged; ``overwrite`` only replaces same-name entries.
    """
    import os
    import tempfile

    from tradelearn import __version__

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / VALIDATION_FILE
    lock = root / f"{VALIDATION_FILE}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"{lock} exists: another writer is active or crashed") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as marker:
            marker.write(str(os.getpid()))
        document = {"format": VALIDATION_FORMAT, "tradelearn_version": __version__, "entries": {}}
        if path.exists():
            existing = _read_validation(path)
            if existing["tradelearn_version"] != __version__:
                raise ValueError(
                    f"{path} was written by tradelearn {existing['tradelearn_version']}; "
                    "write to a new directory instead of mixing versions"
                )
            document["entries"] = existing["entries"]
        for entry in entries:
            if entry.name in document["entries"] and not overwrite:
                raise ValueError(f"validation entry {entry.name!r} already exists")
            document["entries"][entry.name] = entry.to_dict()
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
        handle = tempfile.NamedTemporaryFile(
            "w", dir=root, prefix=f".{VALIDATION_FILE}.", suffix=".tmp", delete=False
        )
        try:
            with handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
    finally:
        lock.unlink(missing_ok=True)
    return path


def _read_validation(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("format") != VALIDATION_FORMAT:
        raise ValueError(f"unsupported validation format {document.get('format')!r}")
    return document


@dataclasses.dataclass
class Bundle:
    """Read-only view of a local artifact bundle directory."""

    directory: Path
    metadata: dict[str, str]
    summary: dict[str, Any]
    research: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    validation: dict[str, Any] | None
    manifest: dict[str, Any] = dataclasses.field(default_factory=dict)
    trial_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)

    def entry(self, name: str) -> dict[str, Any]:
        if not self.validation or name not in self.validation["entries"]:
            raise KeyError(f"no validation entry {name!r}")
        return self.validation["entries"][name]

    def oos_returns(self, name: str = "walk_forward") -> pd.Series:
        return _series_from_payload(self.entry(name)["oos_returns"])

    def trials(self, name: str) -> list[dict[str, Any]]:
        return list(self.entry(name).get("trials", []))


def load_bundle(
    directory: str | Path,
    *,
    expect_version: str | None = None,
    expect_kind: str | None = "backtest",
) -> Bundle:
    """Read the existing backtest, trial, validation and research output files.

    Refuses unknown artifact schema versions, a different artifact kind, or a
    tradelearn_version other than expect_version when given.
    """
    from tradelearn.report.artifacts import ARTIFACT_SCHEMA_VERSION

    root = Path(directory)
    meta_path = root / "metadata.csv"
    validation_path = root / VALIDATION_FILE
    paths = [meta_path, validation_path] + [
        root / name for name in ("manifest.json", "trial_records.json", "research_result.json", "stats.json")
    ]
    if not any(path.is_file() for path in paths):
        raise FileNotFoundError(f"{root} has no supported research artifacts")
    metadata: dict[str, str] = {}
    tables: dict[str, pd.DataFrame] = {}
    if meta_path.is_file():
        meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        metadata = dict(zip(meta["key"], meta["value"], strict=True))
        if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema {metadata.get('artifact_schema_version')!r} "
                f"!= supported {ARTIFACT_SCHEMA_VERSION!r}"
            )
        if expect_kind is not None and metadata.get("artifact_kind") != expect_kind:
            raise ValueError(f"artifact kind {metadata.get('artifact_kind')!r} != {expect_kind!r}")
        for csv in sorted(root.glob("*.csv")):
            if csv.name != "metadata.csv":
                tables[csv.stem] = pd.read_csv(csv)
    validation = _read_validation(validation_path) if validation_path.is_file() else None
    manifest = _read_optional_json(root / "manifest.json", dict)
    trials = _read_optional_json(root / "trial_records.json", list)
    if not all(isinstance(record, dict) for record in trials):
        raise ValueError("trial_records.json must contain trial objects")
    research = _read_optional_json(root / "research_result.json", dict)
    stats = _read_optional_json(root / "stats.json", dict)
    versions = {
        metadata.get("tradelearn_version"),
        None if validation is None else validation["tradelearn_version"],
        manifest.get("tradelearn_version"),
        manifest.get("trade_learn_version"),
    } - {None, "", "未知"}
    if len(versions) > 1:
        raise ValueError(f"bundle mixes tradelearn versions {sorted(versions)}")
    if expect_version is not None and versions != {expect_version}:
        raise ValueError(f"bundle version {sorted(versions)} != expected {expect_version!r}")
    return Bundle(
        directory=root,
        metadata=metadata,
        summary=_key_values(tables.get("summary")),
        research=research or _key_values(tables.get("research")),
        tables=tables,
        validation=validation,
        manifest=manifest,
        trial_records=trials,
        stats=stats,
    )


def _read_optional_json(path: Path, container: type):
    if not path.is_file():
        return container()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, container):
        raise ValueError(f"{path.name} must contain a JSON {container.__name__}")
    return value


def verify_inputs(
    source: Any,
    *,
    data: pd.DataFrame | pd.Series | None = None,
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Compare current data/config with what was recorded; return mismatches.

    ``source`` may be a ``trial.v1`` record, an ``Evidence``/``to_dict()``
    payload, or a ``Bundle`` (all validation entries are checked). Data is
    fingerprinted with the same ``_describe`` used by trial records.
    """
    from tradelearn.core.trial_records import _describe

    if isinstance(source, Bundle):
        entries = list(source.validation["entries"].items()) if source.validation else []
        entries.extend((f"trial {i}", item) for i, item in enumerate(source.trial_records))
        fingerprint = source.manifest.get("input_fingerprint")
        if fingerprint is not None:
            entries.append(("manifest", {"config": {"data": fingerprint}}))
        if not entries:
            return ["bundle has no recorded validation or input fingerprints"]
        return [f"{name}: {problem}" for name, entry in entries
                for problem in verify_inputs(entry, data=data, config=config)]
    if isinstance(source, Evidence):
        source = source.to_dict()
    problems: list[str] = []
    if data is not None:
        current = _describe(data)
        recorded = _recorded_data(source)
        matched = [
            r for r in recorded if isinstance(r, Mapping) and r.get("sha256") == current["sha256"]
        ]
        if not recorded:
            problems.append("no recorded data fingerprint")
        elif not matched:
            problems.append(f"data sha256 {current['sha256'][:12]} not among recorded inputs")
        elif not any(all(_jsonable(value) == _jsonable(r.get(key))
                         for key, value in current.items()) for r in matched):
            problems.append("recorded data metadata differs")
    if config is not None:
        recorded_config = source.get("config", {})
        for key, value in _jsonable(_describe(dict(config))).items():
            if key not in recorded_config:
                problems.append(f"config {key!r} not recorded")
            elif _jsonable(recorded_config[key]) != value:
                problems.append(f"config {key!r} differs")
    return problems


def _recorded_data(source: Mapping[str, Any]) -> list[Any]:
    data = source.get("config", {}).get("data")
    if isinstance(data, list):
        return data
    return [] if data is None else [data]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _key_values(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or not {"key", "value"}.issubset(frame.columns):
        return {}
    return {str(k): _scalar(v) for k, v in zip(frame["key"], frame["value"], strict=True)}


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _series_payload(series: pd.Series) -> dict[str, Any]:
    tz = getattr(series.index, "tz", None)
    return {
        "tz": None if tz is None else str(tz),
        "index_name": series.index.name,
        "index": [ts.isoformat() for ts in series.index],
        "values": [None if pd.isna(v) else float(v) for v in series.to_numpy()],
    }


def _series_from_payload(payload: Mapping[str, Any]) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(payload["index"], utc=payload.get("tz") is not None))
    if payload.get("tz"):
        index = index.tz_convert(payload["tz"])
    index.name = payload.get("index_name")
    return pd.Series(payload["values"], index=index, dtype="float64", name="returns")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Index):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "Bundle",
    "Evidence",
    "Fold",
    "ScenarioResult",
    "Selection",
    "VALIDATION_FORMAT",
    "WalkForwardResult",
    "compare_candidates",
    "cpcv_paths",
    "cpcv_splits",
    "load_bundle",
    "pbo",
    "run_scenarios",
    "verify_inputs",
    "walk_forward",
    "walk_forward_folds",
    "write_validation",
]
