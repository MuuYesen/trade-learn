"""Point-in-time labels and single-symbol factor assessment.

The public functions accept a UTC timestamp/symbol panel. Horizons can count
observations or elapsed seconds; an unavailable exact-time label stays missing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _panel(values: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(values.index, pd.MultiIndex) or values.index.names != ["timestamp", "symbol"]:
        raise ValueError("expected MultiIndex(timestamp, symbol)")
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError("timestamp/symbol index must be unique and sorted")
    times = values.index.get_level_values("timestamp")
    if not isinstance(times, pd.DatetimeIndex) or times.tz is None or str(times.tz) != "UTC":
        raise ValueError("timestamps must be UTC-aware")
    return values


def forward_return(
    bars: pd.DataFrame,
    *,
    horizon: int,
    unit: str = "observations",
    price: str = "close",
) -> pd.DataFrame:
    """Return label value and exact ending timestamp for each signal row.

    ``seconds`` requires an observation at exactly t+h; gaps stay NA. This is a
    predictive price label, not an executable trade return.
    """
    _panel(bars)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if unit not in {"observations", "seconds"}:
        raise ValueError("unit must be observations or seconds")
    if price not in bars:
        raise ValueError(f"missing price column: {price}")
    result = pd.DataFrame(index=bars.index, columns=["forward_return", "label_end"], dtype=object)
    for _, group in bars.groupby(level="symbol", sort=False):
        stamp = pd.DatetimeIndex(group.index.get_level_values("timestamp"))
        current = pd.to_numeric(group[price], errors="coerce").astype(float)
        if unit == "observations":
            future = current.shift(-horizon)
            end = pd.Series(stamp, index=group.index).shift(-horizon)
        else:
            lookup = pd.Series(current.to_numpy(), index=stamp)
            target = stamp + pd.Timedelta(seconds=horizon)
            future = pd.Series(lookup.reindex(target).to_numpy(), index=group.index)
            end = pd.Series(target.where(target.isin(stamp), pd.NaT), index=group.index)
        result.loc[group.index, "forward_return"] = (future / current.where(current > 0) - 1).replace([np.inf, -np.inf], np.nan)
        result.loc[group.index, "label_end"] = end
    result["forward_return"] = pd.to_numeric(result["forward_return"], errors="coerce")
    result["label_end"] = pd.to_datetime(result["label_end"], utc=True)
    return result


def assessment(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factors: Sequence[str],
    block: str = "1D",
    min_pairs: int = 30,
) -> dict:
    """Evaluate each symbol through time, never across symbols.

    Missing or constant samples are reported as unavailable, never as zero.
    """
    _panel(values)
    _panel(labels)
    if not values.index.equals(labels.index):
        raise ValueError("factor and label indexes must match exactly")
    if "forward_return" not in labels:
        raise ValueError("labels need forward_return")
    if not factors or len(set(factors)) != len(factors) or any(name not in values for name in factors):
        raise ValueError("factors must be unique existing columns")
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least 3")
    if block not in {"1D", "1H", "5min"}:
        raise ValueError("block must be 1D, 1H, or 5min")
    pandas_block = "1h" if block == "1H" else block
    output = {}
    for name in factors:
        by_symbol = {}
        for symbol, part in values.groupby(level="symbol", sort=False):
            paired = pd.DataFrame({"factor": part[name], "return": labels.loc[part.index, "forward_return"]}).replace([np.inf, -np.inf], np.nan).dropna()
            result = {"observations": len(paired), "coverage": len(paired) / len(part), "IC": None, "rankIC": None, "ICIR": None, "rankICIR": None, "blocks": [], "reason": None}
            if len(paired) < min_pairs:
                result["reason"] = "insufficient_pairs"
            elif paired.factor.nunique() < 2 or paired["return"].nunique() < 2:
                result["reason"] = "constant_values"
            else:
                result["IC"] = float(paired.factor.corr(paired["return"]))
                result["rankIC"] = float(paired.factor.corr(paired["return"], method="spearman"))
                stamps = paired.index.get_level_values("timestamp")
                for period, section in paired.groupby(stamps.floor(pandas_block)):
                    if len(section) < min_pairs or section.factor.nunique() < 2 or section["return"].nunique() < 2:
                        continue
                    result["blocks"].append({"timestamp": period.isoformat(), "observations": len(section), "IC": float(section.factor.corr(section["return"])), "rankIC": float(section.factor.corr(section["return"], method="spearman"))})
                for metric, ratio in (("IC", "ICIR"), ("rankIC", "rankICIR")):
                    series = pd.Series([item[metric] for item in result["blocks"]])
                    if len(series) >= 2 and series.std(ddof=1) > 0:
                        result[ratio] = float(series.mean() / series.std(ddof=1))
            by_symbol[str(symbol)] = result
        output[name] = by_symbol
    return {"mode": "time_series", "factors": output, "block": block, "minPairs": min_pairs}


def purged_splits(
    labels: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> dict[str, pd.Index]:
    """Chronological train/validation/test indexes with crossing labels purged.

    Boundaries are exclusive for the preceding section and inclusive for the
    following section. A sample is retained only if its label ended before the
    next section begins. The caller fits transforms only on the train index.
    """
    _panel(labels)
    if "label_end" not in labels:
        raise ValueError("labels need label_end")
    first = pd.Timestamp(train_end)
    second = pd.Timestamp(validation_end)
    first = first.tz_localize("UTC") if first.tzinfo is None else first.tz_convert("UTC")
    second = second.tz_localize("UTC") if second.tzinfo is None else second.tz_convert("UTC")
    if first >= second:
        raise ValueError("train_end must precede validation_end")
    stamps = labels.index.get_level_values("timestamp")
    ends = pd.to_datetime(labels.label_end, utc=True)
    valid = labels.forward_return.notna()
    return {
        "train": labels.index[valid & (stamps < first) & (ends < first)],
        "validation": labels.index[valid & (stamps >= first) & (stamps < second) & (ends < second)],
        "test": labels.index[valid & (stamps >= second) & ends.notna()],
    }
