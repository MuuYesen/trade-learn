from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from tradelearn.live import QMTLiveBroker, QMTProxyBroker, QMTProxyConfig

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "zoo"
    / "deploy"
    / "strategy"
    / "alpha101-hs300-index-enhance-v1.0-qmt-stock-1d"
    / "alpha101_hs300_live.py"
)
DEPLOY_DIR = SCRIPT.parent

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(),
    reason="ignored HS300 deploy example is not present",
)


def _load_module():
    sys.path.insert(0, str(DEPLOY_DIR))
    spec = importlib.util.spec_from_file_location("alpha101_hs300_live", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(DEPLOY_DIR))
        except ValueError:
            pass


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
    assert "CONFIG_PATH" in source
    assert "load_deploy_config" in source
    assert "os.environ" not in source
    assert "download_live_weights_from_s3(DEFAULT_WEIGHTS_PATH, s3_trade_date)" in source
    assert "weights_s3_key" in source
    assert "Cerebro" in source
    assert "ResearchResult" in source
    assert "HS300RebalanceEngineStrategy" in source
    assert "from qmt_live_adapter.qmt import QMTProxyConfig, QMTStore" in source


def test_live_deploy_config_files_are_present_and_secret_free() -> None:
    config_files = [
        DEPLOY_DIR / "requirements.txt",
        DEPLOY_DIR / "config" / "config.yaml",
        DEPLOY_DIR / "config" / "trading_node" / "trading_node.yaml",
        DEPLOY_DIR / "config" / "strategy" / "strategy.yaml",
        DEPLOY_DIR / "config" / "qmt" / "qmt.yaml",
        DEPLOY_DIR / "config" / "s3" / "_group_.yaml",
        DEPLOY_DIR / "config" / "s3" / "credentials.yaml",
        DEPLOY_DIR / "config" / "s3" / "settings.yaml",
    ]

    for path in config_files:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        if path.name == "config.yaml":
            assert "defaults:" in text
            assert "trading_node: trading_node" in text
        if path.name not in {"qmt.yaml", "credentials.yaml"}:
            assert "dev-api-key" not in text
            assert "40705983" not in text
        assert "jtDP" not in text
        assert "_env:" not in text

    s3_settings = (DEPLOY_DIR / "config" / "s3" / "settings.yaml").read_text(
        encoding="utf-8"
    )
    assert "high-frequency" not in s3_settings
    assert (
        "prefix: \"Index-Enhancement/alpha101-hs300-index-enhance-v1.0-qmt-stock-1d\""
        in s3_settings
    )
    assert (
        "weights_prefix: "
        "\"Index-Enhancement/alpha101-hs300-index-enhance-weights-v1.0-tushare-1d\""
        in s3_settings
    )
    assert "weights_key_template: \"alpha101_hs300_live_weights_{date}.csv\"" in s3_settings
    assert "trade_date_timezone: \"Asia/Shanghai\"" in s3_settings


def test_live_weights_s3_key_uses_trade_date() -> None:
    module = _load_module()
    key = module.weights_s3_key(pd.Timestamp("2026-06-30"))

    assert key == (
        "Index-Enhancement/alpha101-hs300-index-enhance-weights-v1.0-tushare-1d/"
        "alpha101_hs300_live_weights_20260630.csv"
    )


def test_live_trade_log_s3_key_does_not_add_date_folder(tmp_path, monkeypatch) -> None:
    module = _load_module()
    uploaded: list[tuple[str, str, str]] = []

    class FakeS3:
        def upload_file(self, path: str, bucket: str, key: str) -> None:
            uploaded.append((path, bucket, key))

    class FakeBoto3:
        @staticmethod
        def client(_name: str, **_kwargs):
            return FakeS3()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3)
    logger = module.TextTradeLogger(tmp_path, "strategy-id")
    uri = logger.upload_s3(bucket="bucket", prefix="Index-Enhancement/strategy-id")

    assert len(uploaded) == 1
    assert uploaded[0][1] == "bucket"
    assert uploaded[0][2] == f"Index-Enhancement/strategy-id/{logger.log_path.name}"
    assert uri == [f"s3://bucket/Index-Enhancement/strategy-id/{logger.log_path.name}"]


def test_live_deploy_is_self_contained() -> None:
    dockerfile = (DEPLOY_DIR / "dockerfile").read_text(encoding="utf-8")
    startup = (DEPLOY_DIR / "startup.sh").read_text(encoding="utf-8")
    requirements = (DEPLOY_DIR / "requirements.txt").read_text(encoding="utf-8")
    strategy_config = (
        DEPLOY_DIR / "config" / "strategy" / "strategy.yaml"
    ).read_text(encoding="utf-8")
    trading_node_config = (
        DEPLOY_DIR / "config" / "trading_node" / "trading_node.yaml"
    ).read_text(encoding="utf-8")

    assert "trade-learn==0.2.4" in requirements
    assert "pynesys-pynecore==6.4.2" not in requirements
    assert (
        ROOT / "zoo" / "deploy" / "qmt" / "qmt_live_adapter" / "qmt.py"
    ).exists()
    assert (
        ROOT / "zoo" / "deploy" / "qmt" / "qmt_live_adapter" / "__init__.py"
    ).exists()
    assert not (DEPLOY_DIR / "tradelearn").exists()
    assert (
        ROOT
        / "zoo"
        / "deploy"
        / "strategy"
        / "alpha101-hs300-index-enhance-common"
        / "hs300_alpha101_strategy.py"
    ).exists()
    assert "COPY tradelearn ./tradelearn" in dockerfile
    assert "COPY zoo/deploy/strategy/alpha101-hs300-index-enhance-common" in dockerfile
    assert "COPY zoo/deploy/qmt/qmt_live_adapter" in dockerfile
    assert "COPY rust" not in dockerfile
    assert "pyproject.toml" not in dockerfile
    assert 'BUILD_CONTEXT="$(cd "${DEPLOY_DIR}/../../../.." && pwd)"' in startup
    assert "REPO_ROOT" not in startup
    assert "alpha101-hs300-index-enhance-v1.0-qmt-stock-1d" in strategy_config
    assert "./res/alpha101_hs300_live_weights.csv" in strategy_config
    assert "log_directory: ./logs" in trading_node_config


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
            if method == "GET":
                return {
                    "data": {
                        "items": [
                            {
                                "order_id": "OLD",
                                "stock_code": "999999.SH",
                                "order_status_code": 57,
                                "status_msg": "historical order",
                                "order_volume": 999,
                                "traded_volume": 0,
                                "traded_price": 0.0,
                            },
                            {
                                "order_id": "O1",
                                "stock_code": "300750.SZ",
                                "order_status_code": 56,
                                "status_msg": "",
                                "order_volume": 100,
                                "traded_volume": 100,
                                "traded_price": 20.0,
                            },
                            {
                                "order_id": "O2",
                                "stock_code": "000001.SZ",
                                "order_status_code": 50,
                                "status_msg": "",
                                "order_volume": 1200,
                                "traded_volume": 0,
                                "traded_price": 0.0,
                            },
                            {
                                "order_id": "O3",
                                "stock_code": "600000.SH",
                                "order_status_code": 57,
                                "status_msg": "[COUNTER][260200][可撤]",
                                "order_volume": 1900,
                                "traded_volume": 0,
                                "traded_price": 0.0,
                            },
                        ]
                    }
                }
            order_count = len(
                [
                    call
                    for call in broker_calls
                    if call[0] == "POST" and call[1].endswith("/orders")
                ]
            )
            return {"order_id": f"O{order_count}"}
        if path.endswith("/trades"):
            return {
                "data": {
                    "items": [
                        {
                            "trade_id": "T1",
                            "order_id": "O1",
                            "stock_code": "300750.SZ",
                            "volume": 100,
                            "price": 20.0,
                        }
                    ]
                }
            }
        raise AssertionError((method, path, payload))

    def market_data_transport(method: str, path: str, payload: dict | None):
        data_calls.append((method, path, payload))
        assert path == "/api/v1/data/latest-bars"
        return {
            "data": {
                "items": [
                    {
                        "symbol": "600000.SH",
                        "time": "20260102",
                        "open": 10.0,
                        "high": 10.0,
                        "low": 10.0,
                        "close": 10.0,
                        "volume": 1000.0,
                    },
                    {
                        "symbol": "000001.SZ",
                        "time": "20260102",
                        "open": 8.0,
                        "high": 8.0,
                        "low": 8.0,
                        "close": 8.0,
                        "volume": 1000.0,
                    },
                    {
                        "symbol": "300750.SZ",
                        "time": "20260102",
                        "open": 20.0,
                        "high": 20.0,
                        "low": 20.0,
                        "close": 20.0,
                        "volume": 1000.0,
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
    original_store = module.QMTStore

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
    monkeypatch.setattr(module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(module, "S3_BUCKET", "")
    monkeypatch.setattr(module, "QMT_ACCOUNT_ID", "A1")
    monkeypatch.setattr(module, "EXECUTE_ORDERS", True)
    monkeypatch.setattr(module, "POST_ORDER_POLL_SECONDS", 0.01)
    monkeypatch.setattr(module, "POST_ORDER_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(module, "QMTStore", fake_store)

    module.main()

    submitted = [
        call[2]
        for call in broker_calls
        if call[0] == "POST" and call[1].endswith("/orders")
    ]
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

    logs = list(
        (tmp_path / "logs").glob(
            "alpha101-hs300-index-enhance-v1.0-qmt-stock-1d_*.log"
        )
    )
    assert len(logs) == 1
    log_file = logs[0]
    log_text = log_file.read_text(encoding="utf-8")
    assert ">>> [SIGNAL_START]" in log_text
    assert ">>> [ORDER_SEND]" in log_text
    assert "|-- [ACTION]" in log_text
    assert "|-- [ACCOUNT]" in log_text
    assert log_text.count("|-- [ORDER_FEEDBACK]") == 3
    assert "|-- [ORDER_STATUS]" in log_text
    assert "|-- [FILL]" in log_text
    assert "|-- [EXECUTION_TIMING]" in log_text
    assert "First_Fill_Count: 1" in log_text
    assert "No_Fill_Count: 2" in log_text
    assert "|-- [POSITION]" in log_text
    assert "|-- [LATENCY]" in log_text
    assert "|-- [RESULT]" in log_text
    assert "Status: SUCCESS" in log_text
    assert "Orders_Submitted: 3" in log_text
    assert "Order_ID: OLD" not in log_text
    assert not (tmp_path / "logs" / "latest_summary.json").exists()
