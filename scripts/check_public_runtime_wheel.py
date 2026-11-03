"""Installed distribution check: no source checkout and no account/network use."""

import importlib.util
import json
import tempfile
from pathlib import Path

import tradelearn
from leafquant_monitor import MonitorSettings, TradeLogger

assert importlib.util.find_spec("nautilus_trader") is None, "base environment must have no Nautilus"
with tempfile.TemporaryDirectory() as directory:
    settings = MonitorSettings(
        "LOCAL", "installed-check", "synthetic", directory, account_source="external"
    )
    logger = TradeLogger(settings)
    fact = logger.record("heartbeat", {"expected_running": True})
    assert fact["source"] == "LOCAL"
    assert logger.flush()["error"] is None
    assert logger.close()["error"] is None
    records = [
        json.loads(line)
        for path in (Path(directory) / "segments").glob("*.log")
        for line in path.read_text().splitlines()
    ]
    assert len(records) == 1 and records[0]["kind"] == "heartbeat"
print(
    json.dumps(
        {
            "tradelearn": tradelearn.__version__,
            "installed_root": str(Path(tradelearn.__file__).parent),
            "native_record_flush_close": "passed",
            "nautilus_installed": False,
        }
    )
)
