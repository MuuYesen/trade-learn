from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

from tradelearn.live import QMTLiveBroker, QMTProxyBroker, QMTProxyConfig

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "zoo" / "tushare_sw_hs300" / "alpha101_hs300_live.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("alpha101_hs300_live", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_main_uses_single_research_builder() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "pd.read_csv(" in source
    assert "DEFAULT_WEIGHTS_PATH" in source
    assert "parse_args" not in source
    assert "argparse" not in source
    assert "args.weights_path" not in source
    assert "--weights-path" not in source
    assert "--target-date" not in source
    assert "--base-url" not in source
    assert "--account-id" not in source
    assert "--execute" not in source
    assert "build_live_weights" not in source
    assert "weights_by_symbol" not in source
    assert "generate_latest_target_weights" not in source
    assert ".to_csv(" not in source
    assert "latest_weight_slice(" not in source
    assert "scale_weight_series(" not in source
    assert "LiveResearchResult" not in source
    assert "current_position_symbols(" not in source
    assert "bar_frame(" not in source
    assert "selected" not in source
    assert "HS300_BENCHMARK_PATH" not in source
    assert "def run_live_strategy(" not in source
    assert "def target_data_frames(" not in source
    assert "def print_plan_header(" not in source
    assert "ENFORCE_TRADE_CONSTRAINTS = True" in source
    assert "enforce_trade_constraints=ENFORCE_TRADE_CONSTRAINTS" in source


def test_live_main_fetches_bars_and_submits_orders_through_broker(tmp_path, monkeypatch) -> None:
    module = _load_module()
    broker_calls: list[tuple[str, str, dict | None]] = []
    data_calls: list[tuple[str, str, dict | None]] = []

    def broker_transport(method: str, path: str, payload: dict | None):
        broker_calls.append((method, path, payload))
        if path == "/api/v1/trading/sessions":
            return {"session_id": "S1"}
        if method == "DELETE" and path == "/api/v1/trading/sessions/S1":
            return {"success": True}
        if path.endswith("/asset"):
            return {"data": {"available_cash": 1000.0, "total_asset": 100000.0}}
        if path.endswith("/positions"):
            return {
                "data": [
                    {
                        "stock_code": "300750.SZ",
                        "current_amount": 100,
                        "cost_price": 20.0,
                    }
                ]
            }
        if path.endswith("/orders"):
            order_count = len(
                [call for call in broker_calls if call[1].endswith("/orders")]
            )
            return {
                "order_id": f"O{order_count}"
            }
        if path.endswith("/trades"):
            return {"data": []}
        raise AssertionError((method, path, payload))

    def market_data_transport(method: str, path: str, payload: dict | None):
        data_calls.append((method, path, payload))
        assert path == "/api/v1/data/kline-history"
        return {
            "data": {
                "items": [
                    {
                        "symbol": "600000.SH",
                        "bars": [
                            {
                                "time": "20260102",
                                "open": 10.0,
                                "high": 10.0,
                                "low": 10.0,
                                "close": 10.0,
                                "volume": 1000.0,
                            }
                        ],
                    },
                    {
                        "symbol": "000001.SZ",
                        "bars": [
                            {
                                "time": "20260102",
                                "open": 8.0,
                                "high": 8.0,
                                "low": 8.0,
                                "close": 8.0,
                                "volume": 1000.0,
                            }
                        ],
                    },
                    {
                        "symbol": "300750.SZ",
                        "bars": [
                            {
                                "time": "20260102",
                                "open": 20.0,
                                "high": 20.0,
                                "low": 20.0,
                                "close": 20.0,
                                "volume": 1000.0,
                            }
                        ],
                    },
                ]
            }
        }

    weights_path = tmp_path / "weights.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02"],
            "symbol": ["600000.SH", "000001.SZ"],
            "weight": [0.20, 0.10],
        }
    ).to_csv(weights_path, index=False)

    stores = []
    captured_strategy = {}
    original_store = module.QMTStore
    original_strategy = module.HS300RebalanceEngineStrategy

    class CapturingStrategy(original_strategy):
        def next(self):
            super().next()
            captured_strategy["strategy"] = self

    def fake_store(*_args, **_kwargs):
        store = original_store(
            config=QMTProxyConfig(account_id="A1"),
            broker=QMTLiveBroker(
                QMTProxyBroker(QMTProxyConfig(account_id="A1"), transport=broker_transport)
            ),
            market_data_transport=market_data_transport,
        )
        stores.append(store)
        return store

    monkeypatch.setattr(module, "DEFAULT_WEIGHTS_PATH", weights_path)
    monkeypatch.setattr(module, "TARGET_DATE", "2026-01-02")
    monkeypatch.setattr(module, "QMT_ACCOUNT_ID", "A1")
    monkeypatch.setattr(module, "EXECUTE_ORDERS", True)
    monkeypatch.setattr(module, "QMTStore", fake_store)
    monkeypatch.setattr(module, "HS300RebalanceEngineStrategy", CapturingStrategy)

    module.main()

    submitted = [
        call[2]
        for call in broker_calls
        if call[0] == "POST" and call[1].endswith("/orders")
    ]
    assert captured_strategy["strategy"].submitted is True
    assert stores[0].getbroker().is_connected() is False
    assert data_calls[0][2]["symbols"] == ["000001.SZ", "300750.SZ", "600000.SH"]
    assert broker_calls[-1] == ("DELETE", "/api/v1/trading/sessions/S1", None)
    assert submitted == [
        {
            "stock_code": "300750.SZ",
            "side": "sell",
            "price_type": 5,
            "volume": 100,
            "price": 0.0,
            "strategy_name": "tradelearn",
            "order_remark": "",
        },
        {
            "stock_code": "000001.SZ",
            "side": "buy",
            "price_type": 5,
            "volume": 1200,
            "price": 0.0,
            "strategy_name": "tradelearn",
            "order_remark": "",
        },
        {
            "stock_code": "600000.SH",
            "side": "buy",
            "price_type": 5,
            "volume": 1900,
            "price": 0.0,
            "strategy_name": "tradelearn",
            "order_remark": "",
        },
    ]
