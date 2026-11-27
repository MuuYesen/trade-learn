"""研究出数前的薄层检查：成交、年化口径、已知答案、能力核验、本地存证。

统计计算只有一处：已安装 TradeLearn 的 ``tradelearn.metrics`` 与
``tradelearn.research``。这里**不**重复实现 HAC、配对重抽、PSR、DSR、MinTRL、
稳定区域；只用冻结输入和冻结答案核对已安装的实现，并在旧版本缺这些接口时
给出"受阻"状态。唯一保留的公式是 IID 夏普标准误（``annual_sharpe_se``）：它是
已知答案本身，也是旧版本上唯一可用的标准误口径。
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import re
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# ---------------------------------------------------------------------------
# 已知答案
# ---------------------------------------------------------------------------


def annual_sharpe_se(annual_sharpe: float, observations: float, periods_per_year: float) -> float:
    """IID 近似的年化夏普标准误；observations 与年化所用的期数同频。"""
    if not isfinite(annual_sharpe):
        raise ValueError("annual_sharpe must be finite")
    if not isfinite(observations) or observations <= 1:
        raise ValueError("observations must be finite and greater than one")
    if not isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    years = observations / periods_per_year
    return sqrt((1 + annual_sharpe**2 / (2 * periods_per_year)) / years)


# 冻结输入与答案（取自 TradeLearn 的 tests/unit/metrics/test_sharpe_inference.py）。
HAC_INPUT = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
HAC_BANDWIDTH_1 = sqrt((7 / 8) * (1 + 3 / 8) / 8)
PSR_INPUT = [-0.01, 0.03] * 5
PSR_MIN_TRL = 14


def native(name: str, module: Any = None) -> Any:
    """取已安装 ``tradelearn.metrics`` 里的函数；没有时返回 None（不报错、不调用）。"""
    if module is None:
        try:
            module = importlib.import_module("tradelearn.metrics")
        except ImportError:
            return None
    try:
        return getattr(module, name)
    except AttributeError:
        return None


def native_known_answers(module: Any = None) -> dict[str, str]:
    """用冻结输入核对已安装的推断接口；结果为 通过 / 不过 / 受阻（接口缺失）。"""
    import numpy as np
    import pandas as pd

    status: dict[str, str] = {}

    def run(name: str, check) -> None:
        func = native(name, module)
        if func is None:
            status[name] = "受阻"
            return
        try:
            status[name] = "通过" if check(func) else "不过"
        except Exception:  # noqa: BLE001 - any failure on frozen input is a failed check
            status[name] = "不过"

    hac = pd.Series(HAC_INPUT)
    run("sharpe_se", lambda f: math.isclose(
        f(hac, 1, method="hac", bandwidth=1), HAC_BANDWIDTH_1, rel_tol=1e-9)
        and f(hac, 1, method="hac", bandwidth=0) < HAC_BANDWIDTH_1)

    base = pd.Series(np.sin(np.arange(100)) / 100)

    def paired(f) -> bool:
        sharpe = native("sharpe", module)
        result = f(base, base + 0.002, 252, block_length=5, draws=200, seed=42)
        expected = sharpe(base + 0.002, 252) - sharpe(base, 252)  # 候选 − 基准
        same = f(base, 2 * base, 365, block_length=7, draws=128)
        return (
            expected > 0
            and math.isclose(result.estimate, expected, rel_tol=1e-9)
            and result.lower < expected < result.upper
            and math.isclose(same.estimate, 0, abs_tol=1e-12)
            and result.observations == 100
        )

    run("paired_sharpe_diff", paired)
    psr = pd.Series(PSR_INPUT)
    run("minimum_track_record_length", lambda f: f(psr) == PSR_MIN_TRL and f(-psr) == math.inf)
    run("probabilistic_sharpe", lambda f: 0.5 < f(psr) < 1)

    def dsr(f) -> bool:
        try:
            f(psr, n_trials=1, variance_trials=0.02)
        except ValueError:
            pass
        else:
            return False  # N = 1 必须拒绝，改报 PSR
        return f(psr, n_trials=5, variance_trials=0.02) < native("probabilistic_sharpe", module)(psr)

    run("deflated_sharpe", dsr)
    return status


# ---------------------------------------------------------------------------
# 证据方向（判读规则，不是统计计算）
# ---------------------------------------------------------------------------


def evidence_direction(low: float, high: float) -> str:
    """按区间给出证据方向：下限 > 0 为"正"，上限 < 0 为"负"，否则"不显著"。"""
    if not (isfinite(low) and isfinite(high)) or low > high:
        raise ValueError("interval must be finite with low <= high")
    if low > 0:
        return "正"
    if high < 0:
        return "负"
    return "不显著"


def comparison_verdict(
    *,
    candidate_interval: Sequence[float],
    baseline_interval: Sequence[float],
    paired_interval: Sequence[float],
) -> dict[str, str]:
    """绝对收益与配对相对改善分开判，互不推出；paired_interval 为候选 − 基准。"""
    return {
        "候选绝对收益": evidence_direction(*candidate_interval),
        "基准绝对收益": evidence_direction(*baseline_interval),
        "配对相对改善": evidence_direction(*paired_interval),
    }


# ---------------------------------------------------------------------------
# 市场时钟与年化期数
# ---------------------------------------------------------------------------

# 每年交易日数、每个交易日的连续交易时长（小时）。期货夜盘、港股午休等
# 时段不规则的市场不在表里：用实际 bar 数 / 年数自行计算并写明依据。
MARKET_CLOCKS: dict[str, tuple[int, float]] = {
    "crypto": (365, 24.0),
    "cn_equity": (252, 4.0),
    "us_equity": (252, 6.5),
}


def periods_per_year(market: str, bar: str) -> float:
    """按市场时钟给出年化期数。bar 例：1D、1W、1M、30min、15m、1h。"""
    if market not in MARKET_CLOCKS:
        raise ValueError(f"unknown market clock {market!r}; count bars per year explicitly")
    days, hours = MARKET_CLOCKS[market]
    text = bar.strip()
    if text.upper() in {"1D", "D"}:
        return days
    if text.upper() in {"1W", "W"}:
        return 52
    if text in {"1M", "M", "1mo"}:
        return 12
    match = re.fullmatch(r"(\d+)\s*(min|m|h)", text)
    if not match:
        raise ValueError(f"unsupported bar {bar!r}")
    minutes = int(match.group(1)) * (60 if match.group(2) == "h" else 1)
    value = days * hours * 60 / minutes
    return int(value) if float(value).is_integer() else value


def metric_periods_ok(metric_metadata: Mapping[str, Any], key: str, expected: float) -> bool:
    """框架指标的年化期数是否与市场时钟一致；缺元数据视为不一致。"""
    entry = metric_metadata.get(key) if metric_metadata else None
    if not isinstance(entry, Mapping) or entry.get("period_source") == "insufficient_intraday_coverage":
        return False
    try:
        return math.isclose(float(entry["periods"]), float(expected), rel_tol=1e-9)
    except (KeyError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 成交
# ---------------------------------------------------------------------------


def needs_fill_review(position_change_expected: bool, fill_count: int) -> bool:
    """有改变仓位的意图却零成交：先查原因，结果不进入评价。"""
    return position_change_expected and fill_count == 0


# ---------------------------------------------------------------------------
# 能力核验与本地存证
# ---------------------------------------------------------------------------

CAPABILITIES = (
    "metrics.sharpe",
    "metrics.sharpe_se",
    "metrics.paired_sharpe_diff",
    "metrics.probabilistic_sharpe",
    "metrics.deflated_sharpe",
    "metrics.minimum_track_record_length",
    "metrics.hac_mean_se",
    "research.ResearchRun",
    "research.time_split",
    "research.temporal.purged_splits",
    "research.parameter_surface.stable_region",
    "research.validation.walk_forward_folds",
    "research.validation.walk_forward",
    "research.validation.run_scenarios",
    "research.validation.compare_candidates",
    "research.validation.cpcv_splits",
    "research.validation.pbo",
    "research.validation.load_bundle",
    "research.validation.verify_inputs",
    "integrations.search.grid_search",
    "report.artifacts.write_artifact_bundle",
    "ml.save_bundle",
    "sentinel",
)


def _resolve(root: Any, dotted: str) -> bool:
    current = root
    parts = dotted.split(".")
    for index, part in enumerate(parts):
        try:
            current = getattr(current, part)
            continue
        except AttributeError:
            pass
        name = getattr(root, "__name__", None)
        if not isinstance(name, str):
            return False
        try:
            current = importlib.import_module(".".join([name, *parts[: index + 1]]))
        except ImportError:
            return False
    return True


def capabilities(module: Any = None, names: Sequence[str] = CAPABILITIES) -> dict[str, str]:
    """只查已安装 TradeLearn 有没有这些名字，不调用它们。"""
    if module is None:
        try:
            module = importlib.import_module("tradelearn")
        except ImportError:
            return {name: "缺失" for name in names}
    return {name: ("可用" if _resolve(module, name) else "缺失") for name in names}


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def save_local_evidence(
    directory: str | Path,
    *,
    stats: Any = None,
    data: Any = None,
    trial_records: Sequence[Mapping[str, Any]] | None = None,
    research_result: Any = None,
    manifest: Mapping[str, Any] | None = None,
    strategy: Any = None,
    timeframe: str | None = None,
) -> list[Path]:
    """完整写出后发布新目录。现有目录保留，失败后可重试。

    stats 交给 TradeLearn 的 write_artifact_bundle。data 是实际回测输入，
    支持时以与试验相同的方式记录结构和内容指纹。旧版本仍接受调用者的原始
    数据哈希。验证结果可由 TradeLearn 的 write_validation 写入同一目录。
    """
    out = Path(directory)
    if out.exists():
        raise FileExistsError(f"{out} already exists; write a new directory")
    info = dict(manifest or {})
    info.setdefault("python", platform.python_version())
    try:
        runtime = importlib.import_module("tradelearn")
    except ImportError:
        info.setdefault("trade_learn_version", "未知")
    else:
        info.setdefault("trade_learn_version", getattr(runtime, "__version__", "未知"))
    if data is not None:
        try:
            from tradelearn.core.trial_records import _describe
        except ImportError:
            pass
        else:
            info["input_fingerprint"] = _describe(data)

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=f".{out.name}-") as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        files: list[Path] = []
        if stats is not None:
            from tradelearn.report.artifacts import write_artifact_bundle

            files.extend(write_artifact_bundle(
                stats, staging, strategy=strategy, timeframe=timeframe,
            ))

        def dump(name: str, payload: Any) -> None:
            path = staging / name
            path.write_text(
                json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            files.append(path)

        if trial_records is not None:
            dump("trial_records.json", list(trial_records))
        if research_result is not None:
            dump("research_result.json", research_result)
        dump("manifest.json", info)
        staging.rename(out)
        return sorted(out / path.relative_to(staging) for path in files)


if __name__ == "__main__":  # 已知答案冒烟
    assert math.isclose(annual_sharpe_se(1.0, 1460, 365), 0.500342348545, abs_tol=1e-9)
    assert periods_per_year("crypto", "30min") == 17520
    print(native_known_answers())
