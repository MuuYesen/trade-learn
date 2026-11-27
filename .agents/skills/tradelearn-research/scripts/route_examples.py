"""七条研究路径中可在本地运行、可读回的接线示例。

只编排已安装 TradeLearn 的接口，不实现任何统计方法。每个函数先做能力核验，
缺接口时返回 ``{"状态": "受阻", "缺失": [...]}``，不改调别的接口。
``evaluate(params, frame, fold, segment)`` 由研究者提供：跑一次独立回测并返回
逐期**扣成本**收益（与 ``tradelearn.research.validation.walk_forward`` 的约定一致）。

运行 ``self_test(目录)`` 在合成数据上走一遍（只验证接线与读回，不是研究证据）。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _need(*dotted: str) -> list[str]:
    missing = []
    for name in dotted:
        module, _, attr = name.rpartition(".")
        try:
            if not hasattr(importlib.import_module(module), attr):
                missing.append(name)
        except ImportError:
            missing.append(name)
    return missing


def _blocked(missing: list[str]) -> dict[str, Any]:
    return {"状态": "受阻", "缺失": missing}


# ---------------------------------------------------------------------------
# 读取/复现
# ---------------------------------------------------------------------------


def read_back(directory: str | Path, *, expect_version: str | None = None,
              data: Any = None) -> dict[str, Any]:
    """读回本地研究目录（write_artifact_bundle 产物与 validation.json），并核对输入。"""
    missing = _need("tradelearn.research.validation.load_bundle",
                    "tradelearn.research.validation.verify_inputs")
    if missing:
        return _blocked(missing)
    from tradelearn.research import validation as v

    bundle = v.load_bundle(directory, expect_version=expect_version)
    mismatches = v.verify_inputs(bundle, data=data) if data is not None else []
    return {"状态": "已读回", "bundle": bundle, "不一致": mismatches}


# ---------------------------------------------------------------------------
# 比较与稳健性
# ---------------------------------------------------------------------------


def compare_fixed(
    data: Any,
    *,
    evaluate: Callable[..., Any],
    baseline: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    periods: int,
    block_length: int,
    train: int,
    test: int,
    warmup: int = 0,
    directory: str | Path,
    seed: int = 0,
) -> dict[str, Any]:
    """事先固定的候选：每个候选在同一组样本外窗口上跑，拼接后与基准配对比较。

    配对差由 compare_candidates 调 metrics.paired_sharpe_diff(left=基准, right=候选)，
    estimate 为候选 − 基准。各候选的绝对证据另用 metrics.sharpe_se 给出。
    """
    missing = _need("tradelearn.research.validation.walk_forward",
                    "tradelearn.research.validation.compare_candidates",
                    "tradelearn.metrics.sharpe_se")
    if missing:
        return _blocked(missing)
    from tradelearn import metrics
    from tradelearn.research import validation as v

    folds = v.walk_forward_folds(data, train=train, test=test)
    runs = {"baseline": v.walk_forward(data, folds, evaluate=evaluate, params=dict(baseline),
                                       warmup=warmup, name="wf_baseline")}
    for name, params in candidates.items():
        runs[name] = v.walk_forward(data, folds, evaluate=evaluate, params=dict(params),
                                    warmup=warmup, name=f"wf_{name}")
    incomplete = [name for name, run in runs.items() if not run.complete]
    if incomplete:
        v.write_validation(directory, *runs.values())
        return {"状态": "无法判断", "不完整": incomplete}
    returns = {name: run.oos_returns for name, run in runs.items()}
    comparison = v.compare_candidates(returns, baseline="baseline", periods=periods,
                                      block_length=block_length, seed=seed,
                                      n_trials=len(candidates) + 1)
    path = v.write_validation(directory, *runs.values(), comparison)
    absolute = {}
    for name, r in returns.items():
        se = metrics.sharpe_se(r, periods, method="hac")
        sharpe = metrics.sharpe(r, periods)
        absolute[name] = {"sharpe": sharpe, "se_hac": se,
                          "lower": sharpe - 1.96 * se, "upper": sharpe + 1.96 * se}
    return {"状态": "已检验", "validation": path, "绝对": absolute,
            "配对": comparison["comparisons"], "方向": "候选 − 基准"}


def cost_scenarios(evaluate_scenario: Callable[[Mapping[str, Any]], Any],
                   scenarios: Mapping[str, Mapping[str, Any]], *, periods: int,
                   directory: str | Path) -> dict[str, Any]:
    """已冻结成本假设的压力情景（成本 ×2 等）；情景不是搜索，不计入尝试数。"""
    missing = _need("tradelearn.research.validation.run_scenarios")
    if missing:
        return _blocked(missing)
    from tradelearn.research import validation as v

    result = v.run_scenarios(evaluate_scenario, scenarios, periods=periods)
    return {"状态": "已检验", "validation": result.write(directory), "表": result.table()}


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------


def search_with_stable_region(
    data: Any,
    *,
    evaluate: Callable[..., Any],
    axes: Mapping[str, Sequence[Any]],
    periods: int,
    train: int,
    test: int,
    se_method: str = "hac",
    directory: str | Path,
) -> dict[str, Any]:
    """真搜索：每折只在训练段扫网格，用 stable_region 诊断稳定区域并取平滑最优格。

    评分口径固定为训练段年化夏普、标准误口径由 se_method 事先指定。
    缺失格由 stable_region 计入 unavailable_cells，不当通过。
    这里只把平滑最优格带到测试段；区域内多组合的仓位合并需要研究者在
    evaluate 里显式实现，本函数不做。
    """
    missing = _need("tradelearn.research.validation.walk_forward",
                    "tradelearn.research.parameter_surface.stable_region",
                    "tradelearn.metrics.sharpe_se")
    if missing:
        return _blocked(missing)
    from itertools import product

    import pandas as pd

    from tradelearn import metrics
    from tradelearn.research import validation as v
    from tradelearn.research.parameter_surface import stable_region

    names = list(axes)
    grid = [dict(zip(names, cell, strict=True)) for cell in product(*axes.values())]

    def objective(r: Any) -> float:
        return float(metrics.sharpe(r, periods))

    def select(evidence: list[dict[str, Any]], window: Mapping[str, Any]) -> Any:
        index = pd.MultiIndex.from_tuples([tuple(e["params"][n] for n in names) for e in evidence],
                                          names=names)
        scores = pd.Series([e["score"] for e in evidence], index=index, dtype="float64")
        best = int(scores.fillna(float("-inf")).to_numpy().argmax())
        se = metrics.sharpe_se(evidence[best]["returns"], periods, method=se_method)
        region = stable_region(scores, axes, standard_error=se)
        cell = dict(zip(names, region.best_cell, strict=True))
        chosen = next(i for i, e in enumerate(evidence) if e["params"] == cell)
        return v.Selection(cell, (chosen,), {
            "rule": "stable_region.best_cell", "se_method": se_method, "standard_error": se,
            "coverage": region.coverage, "components": len(region.components),
            "boundary_cells": len(region.boundary_cells),
            "unavailable_cells": region.unavailable_cells,
        })

    folds = v.walk_forward_folds(data, train=train, test=test)
    result = v.walk_forward(data, folds, evaluate=evaluate, candidates=grid,
                            objective=objective, select=select, name="wf_search")
    return {"状态": "已检验" if result.complete else "无法判断",
            "validation": result.write(directory), "尝试数": len(grid) * len(folds),
            "folds": result.folds}


# ---------------------------------------------------------------------------
# 因子与模型
# ---------------------------------------------------------------------------


def ic_mean_with_hac(ic: Any, *, bandwidth: int | None = None) -> dict[str, Any]:
    """逐期 IC（按规则时钟对齐）的均值与 HAC 标准误；不用于回归系数或 alpha。"""
    missing = _need("tradelearn.metrics.hac_mean_se")
    if missing:
        return _blocked(missing)
    from tradelearn import metrics

    se = metrics.hac_mean_se(ic, bandwidth=bandwidth)
    mean = float(ic.mean())
    return {"状态": "已检验", "mean": mean, "se": se, "t": mean / se if se else float("nan")}


# ---------------------------------------------------------------------------
# 合成数据自测（接线与读回，不是研究证据）
# ---------------------------------------------------------------------------


def self_test(directory: str | Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=240, freq="D", tz="UTC", name="timestamp")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0, 0.01, len(idx))))
    bars = pd.DataFrame({"close": close}, index=idx)

    def evaluate(params, frame, fold, segment):
        c = frame["close"]
        signal = (c > c.rolling(params["window"]).mean()).astype(float).shift(1)
        return (c.pct_change() * signal - 0.0005 * signal.diff().abs()).fillna(0.0)

    root = Path(directory)
    report: dict[str, Any] = {}
    comp = compare_fixed(bars, evaluate=evaluate, baseline={"window": 10},
                         candidates={"w20": {"window": 20}}, periods=365, block_length=7,
                         train=60, test=30, warmup=20, directory=root / "compare")
    report["compare"] = comp["状态"]
    if comp["状态"] == "已检验":
        back = read_back(root / "compare")
        bundle = back["bundle"]
        stored = bundle.entry("comparison")["comparisons"]["w20"]["estimate"]
        report["read_back"] = abs(stored - comp["配对"]["w20"]["estimate"]) < 1e-12
        report["comparison_direction"] = comp["方向"]
    search = search_with_stable_region(bars, evaluate=evaluate, axes={"window": [5, 10, 15, 20]},
                                       periods=365, train=60, test=30, directory=root / "search")
    report["search"] = search["状态"]
    ic = pd.Series(np.random.default_rng(3).normal(0.02, 0.05, 120),
                   index=pd.date_range("2024-01-01", periods=120, freq="D"))
    report["ic"] = ic_mean_with_hac(ic)["状态"]
    return report


if __name__ == "__main__":
    import sys
    import tempfile

    print(self_test(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()))
