from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import tradelearn.engine as bt
from tradelearn.research import ResearchResult

ROOT = Path(__file__).resolve().parents[3]
STRATEGY = ROOT / "zoo" / "tushare_sw_hs300" / "hs300_alpha101_strategy.py"
BACKTEST = ROOT / "zoo" / "tushare_sw_hs300" / "alpha101_hs300_backtest.py"
LIVE = ROOT / "zoo" / "tushare_sw_hs300" / "alpha101_hs300_live.py"


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location("hs300_alpha101_strategy", STRATEGY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hs300_strategy_classes_live_in_strategy_file() -> None:
    module = _load_strategy_module()

    assert module.HS300RebalanceEngineStrategy.__name__ == "HS300RebalanceEngineStrategy"


def test_hs300_runners_import_strategy_classes_instead_of_defining_them() -> None:
    backtest_source = BACKTEST.read_text(encoding="utf-8")
    live_source = LIVE.read_text(encoding="utf-8")

    assert (
        "from hs300_alpha101_strategy import HS300RebalanceEngineStrategy"
        in backtest_source
    )
    assert "class HS300RebalanceEngineStrategy" not in backtest_source
    assert "tradelearn.lite" not in backtest_source
    assert "bt.Cerebro" in backtest_source
    assert "BACKTEST_BUY_LOT_SIZE = 100" in backtest_source
    assert "buy_lot_size=BACKTEST_BUY_LOT_SIZE" in backtest_source
    assert (
        "from hs300_alpha101_strategy import HS300RebalanceEngineStrategy"
        in live_source
    )
    assert "class HS300RebalanceEngineStrategy" not in live_source
    assert "buy_lot_size=100" in live_source


def test_engine_strategy_uses_explicit_buy_sell_orders() -> None:
    source = STRATEGY.read_text(encoding="utf-8")
    engine_source = source.split("class HS300RebalanceEngineStrategy", 1)[1]

    assert "has_current()" in engine_source
    assert "weights[0]" in engine_source
    assert "if self.submitted:" not in engine_source
    assert "as_weight_dict()" not in engine_source
    assert "self.buy(" in engine_source
    assert "self.sell(" in engine_source
    assert "self.buy_lot_size" in engine_source
    assert "enforce_trade_constraints" in engine_source
    assert "can_trade(" in engine_source
    assert "target_weights(" not in engine_source
    assert "RebalanceIntent" not in source
    assert "target_weights_for_date" not in source
    assert "target_date" not in engine_source
    assert "_positions_by_symbol" not in engine_source
    assert "refresh_positions" not in engine_source
    assert "store.get_bars(" not in engine_source
    assert "self.store" not in engine_source
    assert ".close[0]" in engine_source
    assert "getdatabyname(" in engine_source
    assert "data_by_symbol" not in engine_source
    assert "broker.proxy" not in engine_source
    assert "current_positions" not in engine_source
    assert "getposition(" in engine_source
    assert "rebalance_plan" not in engine_source


def test_engine_strategy_skips_buy_at_up_limit_when_constraints_enabled() -> None:
    module = _load_strategy_module()
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    bars = pd.DataFrame(
        {
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1_000_000.0, 1_000_000.0],
            "tradestatus": [1, 1],
            "up_limit": [10.0, 10.0],
            "down_limit": [9.0, 9.0],
        },
        index=dates,
    )
    weights = pd.Series(
        [0.5],
        index=pd.MultiIndex.from_tuples(
            [(dates[0], "AAA")],
            names=["date", "symbol"],
        ),
        name="weight",
    )

    cerebro = bt.Cerebro(trade_on_close=False)
    cerebro.setcash(1_000_000.0)
    cerebro.adddata(bars, name="AAA")
    cerebro.addstrategy(
        module.HS300RebalanceEngineStrategy,
        research_result=ResearchResult(name="test", weights=weights),
        execute=True,
        enforce_trade_constraints=True,
    )
    [strategy] = cerebro.run()

    assert strategy.orders == []
