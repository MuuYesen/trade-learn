"""Build-only installed Linux consumer gate; never grants release approval."""

import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import zipfile

import pytest
import tradelearn._rust

import tradelearn

ROOT = pathlib.Path(__file__).resolve().parents[1]
WHEEL = next((ROOT / "consumer-wheels").glob("*.whl"))
SITE = pathlib.Path(tradelearn.__file__).parent.parent
assert not pathlib.Path(tradelearn.__file__).is_relative_to(ROOT)
with zipfile.ZipFile(WHEEL) as archive:
    expected = {
        name: hashlib.sha256(archive.read(name)).hexdigest()
        for name in archive.namelist()
        if name.startswith(("tradelearn/", "tradelearn_runtime/", "leafquant_monitor/"))
    }


def verify():
    """Ensure the same installed package bytes remain authoritative."""
    for name, digest in expected.items():
        assert hashlib.sha256((SITE / name).read_bytes()).hexdigest() == digest, name


verify()
code = """import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('nautilus_trader', 'tl_store', 'boto3', 'botocore', 'tradelearn_runtime')
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from tradelearn import engine as bt
assert 'NautilusStore' in bt.stores.__all__
"""
tests = types.ModuleType("tests")
tests.__path__ = [str(ROOT / "tests")]
sys.modules["tests"] = tests
with tempfile.TemporaryDirectory() as temporary:
    os.chdir(temporary)
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": ""})
    config = pathlib.Path(temporary) / "pytest.ini"
    config.write_text("[pytest]\n")
    result = pytest.main(
        [
            "-q",
            "--import-mode=importlib",
            "--confcutdir",
            temporary,
            "-c",
            str(config),
            "-o",
            "pythonpath=",
            str(ROOT / "tests/unit/engine/test_public_store_entry.py"),
            str(ROOT / "tests/unit/engine/test_nautilus_store.py"),
            "-k",
            "(public_store_entry and not namespace_does_not_load) or "
            "official_bar_replay_uses_one_runtime",
            "-W",
            "error::pytest.PytestUnraisableExceptionWarning",
        ]
    )
verify()
for name, module in tuple(sys.modules.items()):
    if name.split(".")[0] in ("tradelearn", "tradelearn_runtime", "leafquant_monitor") and getattr(
        module, "__file__", None
    ):
        assert pathlib.Path(module.__file__).is_relative_to(SITE), (name, module.__file__)
receipt = {
    "exit_code": int(result),
    "installed_package_members_before_after": len(expected),
    "wheel_sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(
        pathlib.Path(tradelearn._rust.__file__).read_bytes()
    ).hexdigest(),
    "platform": sys.platform,
    "python": sys.version,
    "versions": {
        name: metadata.version(name) for name in ("trade-learn", "tl-store", "nautilus_trader")
    },
    "template_sha256": hashlib.sha256(
        (ROOT / "examples/live/ctrader_monitor.py").read_bytes()
    ).hexdigest(),
    "source_overlay": False,
    "public_pypi_release": False,
    "real_monitor_sender_archive": False,
    "all_C1_C8_closed": False,
}
(ROOT / "consumer-qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
raise SystemExit(result)
