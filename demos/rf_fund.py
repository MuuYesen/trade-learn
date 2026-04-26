"""Random-forest rotation demo using offline data and the new API."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import tradelearn.compat.backtrader as bt


@dataclass(frozen=True)
class RotationPlan:
    """Model output consumed by the backtrader-style strategy."""

    selected_symbol: str
    probabilities: dict[str, float]
    train_rows: int


class RandomForestRotation(bt.Strategy):
    """Buy the deterministic top-ranked symbol from the RF model."""

    params = (("selected_symbol", "asset_a"), ("size", 5))

    def __init__(self) -> None:
        self._entered = False

    def next(self) -> None:
        if self._entered:
            return
        for data in self.datas:
            if data._name == self.p.selected_symbol:
                self.buy(data=data, size=self.p.size)
                self._entered = True
                return


def run_demo() -> dict[str, Any]:
    """Train a deterministic RF signal and run a compact rotation backtest."""
    panels = sample_panel()
    plan = build_rotation_plan(panels)

    cerebro = bt.Cerebro(trade_on_close=True)
    for symbol, bars in panels.items():
        cerebro.adddata(bt.feeds.PandasData(dataname=bars, name=symbol))
    cerebro.addstrategy(RandomForestRotation, selected_symbol=plan.selected_symbol)

    [strategy] = cerebro.run()
    if strategy.stats is None:
        raise RuntimeError("RF fund demo did not produce stats")

    return {
        "strategy": RandomForestRotation.__name__,
        "symbols": list(panels),
        "selected_symbol": plan.selected_symbol,
        "probabilities": plan.probabilities,
        "train_rows": plan.train_rows,
        "fills": len(strategy.stats.fills),
        "final_value": float(strategy.stats.summary["final_value"]),
    }


def build_rotation_plan(panels: dict[str, pd.DataFrame]) -> RotationPlan:
    """Fit the RF model on lagged return features and rank the latest bar."""
    features = _feature_frame(panels)
    latest_timestamp = features.index.max()
    train = features.loc[features.index < latest_timestamp]
    latest = features.loc[features.index == latest_timestamp]

    feature_columns = ["return_1", "return_3", "volatility_3"]
    model = RandomForestClassifier(
        n_estimators=32,
        max_depth=3,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[feature_columns], train["label"])

    probabilities = model.predict_proba(latest[feature_columns])[:, 1]
    ranked = pd.Series(probabilities, index=latest["symbol"]).sort_values(ascending=False)
    return RotationPlan(
        selected_symbol=str(ranked.index[0]),
        probabilities={str(symbol): float(value) for symbol, value in ranked.items()},
        train_rows=len(train),
    )


def sample_panel() -> dict[str, pd.DataFrame]:
    """Return deterministic multi-asset bars for the offline RF demo."""
    index = pd.date_range("2026-01-01", periods=48, freq="D", tz="UTC")
    return {
        "asset_a": _bars(index, base=30.0, slope=0.18, cycle=0.35),
        "asset_b": _bars(index, base=22.0, slope=0.10, cycle=0.55),
        "asset_c": _bars(index, base=16.0, slope=0.15, cycle=-0.40),
    }


def _bars(
    index: pd.DatetimeIndex,
    *,
    base: float,
    slope: float,
    cycle: float,
) -> pd.DataFrame:
    rows = []
    for offset, _timestamp in enumerate(index):
        seasonal = ((offset % 6) - 2.5) * cycle
        close = base + offset * slope + seasonal
        rows.append(
            {
                "open": close - 0.15,
                "high": close + 0.45,
                "low": close - 0.50,
                "close": close,
                "volume": 3000.0 + offset * 40.0 + base,
            }
        )
    return pd.DataFrame(rows, index=index)


def _feature_frame(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, bars in panels.items():
        close = bars["close"]
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "return_1": close.pct_change(),
                "return_3": close.pct_change(3),
                "volatility_3": close.pct_change().rolling(3).std(),
                "label": (close.pct_change().shift(-1) > 0).astype(int),
            },
            index=bars.index,
        )
        frames.append(frame)
    return pd.concat(frames).dropna().sort_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RF rotation demo.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    result = run_demo()
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "RandomForestRotation "
            f"selected={result['selected_symbol']} fills={result['fills']} "
            f"final_value={result['final_value']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
