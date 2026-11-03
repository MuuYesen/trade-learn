"""Inventory uses package metadata, never broker imports or a guessed list."""

from types import SimpleNamespace

import pytest

from tradelearn_runtime.support import adapter_inventory


def test_discovery_is_only_installed_adapter_directories():
    dist = SimpleNamespace(
        version="2.0.0rc7",
        files=[
            "nautilus_trader/adapters/binance/__init__.py",
            "nautilus_trader/adapters/ib/__init__.py",
            "nautilus_trader/adapters/__init__.py",
            "nautilus_trader/adapters/_template/__init__.py",
            "leafquant_ctrader/__init__.py",
        ],
    )
    result = adapter_inventory(dist)
    assert result["adapters"] == ["binance", "ib"]
    assert result["scope"] == "installed_official_adapter_discovery_only"
    assert "not evidence" in result["acceptance"]


def test_wrong_version_and_missing_files_rejected():
    for dist in (
        SimpleNamespace(version="2.0.0rc6", files=[]),
        SimpleNamespace(version="2.0.0rc7", files=None),
    ):
        with pytest.raises(ValueError):
            adapter_inventory(dist)
