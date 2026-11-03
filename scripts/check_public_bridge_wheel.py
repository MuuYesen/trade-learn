"""Installed official-Na bridge construction without SDK configuration or accounts."""

import json
import tempfile

from nautilus_trader.model import AccountId, StrategyId
from nautilus_trader.trading import StrategyConfig

from tradelearn.engine import Strategy
from tradelearn_runtime.execution import create_execution_host


class Business(Strategy):
    def next(self):
        pass


with tempfile.TemporaryDirectory() as directory:
    host = create_execution_host(
        StrategyConfig(strategy_id=StrategyId("PUBLIC-001")),
        business_cls=Business,
        feeds={},
        instruments={},
        account_id=AccountId("PUBLIC-001"),
        intent_identity="public/synthetic/PUBLIC-001",
        intent_path=directory + "/intent.sqlite",
    )
    assert not hasattr(host, "logger"), "disabled logging must create no SDK writer"
    assert host.frontend is None
    print(json.dumps({"official_na_bridge_constructed": True, "sdk_configuration_required": False}))
