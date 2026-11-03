"""Installed distribution check: no source checkout and no account/network use."""

import importlib.util
import json
import tempfile
from importlib import metadata, resources
from pathlib import Path

import tradelearn
from leafquant_monitor import MonitorSettings, TradeLogger

pinned = json.loads(
    resources.files("tradelearn_runtime").joinpath("official_adapters.json").read_text()
)
assert pinned["nautilus_version"] == "2.0.0rc7"
assert len(pinned["adapters"]) == 19
assert not any(row["package"] == "ctrader" for row in pinned["adapters"])

distribution = metadata.distribution("trade-learn")
license_files = [
    entry for entry in distribution.files or () if "THIRD_PARTY_LICENSES" in Path(str(entry)).parts
]
inventories = [entry for entry in license_files if Path(str(entry)).name == "inventory.json"]
assert len(inventories) == 1, "installed native dependency license inventory required"
inventory = json.loads(distribution.locate_file(inventories[0]).read_text())
for package in inventory["packages"]:
    for notice in package["included_notices"]:
        suffix = f"{package['name']}-{package['version']}/{notice}"
        assert any(str(entry).replace("\\", "/").endswith(suffix) for entry in license_files), (
            suffix
        )

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
    recording = "passed"

print(
    json.dumps(
        {
            "tradelearn": tradelearn.__version__,
            "installed_root": str(Path(tradelearn.__file__).parent),
            "native_record_flush_close": recording,
            "nautilus_installed": False,
        }
    )
)
