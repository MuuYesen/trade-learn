"""Installed CP311 owner ABI gate for the same frozen SDK; no NA/live secrets."""

import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import zipfile

import tradelearn._rust

import leafquant_monitor
import tradelearn

ROOT = pathlib.Path(__file__).resolve().parents[1]
WHEEL = next((ROOT / "consumer-wheels").glob("*cp311*.whl"))
SITE = pathlib.Path(tradelearn.__file__).parent.parent
assert sys.version_info[:2] == (3, 11)
assert not pathlib.Path(tradelearn.__file__).is_relative_to(ROOT)
assert pathlib.Path(leafquant_monitor.__file__).is_relative_to(SITE)
with zipfile.ZipFile(WHEEL) as archive:
    expected = {
        name: hashlib.sha256(archive.read(name)).hexdigest()
        for name in archive.namelist()
        if name.startswith(("tradelearn/", "tradelearn_runtime/", "leafquant_monitor/"))
    }
    sdk = {
        name: digest
        for name, digest in expected.items()
        if name.startswith("leafquant_monitor/") and name.endswith(".py")
    }


def verify():
    """Check installed bytes and the exact frozen SDK source."""
    for name, digest in expected.items():
        assert hashlib.sha256((SITE / name).read_bytes()).hexdigest() == digest, name
    for name, digest in sdk.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


verify()
assert len(sdk) == 19
with tempfile.TemporaryDirectory() as temporary:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/native_recorder_smoke.py"), tradelearn._rust.__file__],
        cwd=temporary,
        check=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
verify()
out = {
    "python": sys.version,
    "platform": sys.platform,
    "wheel_sha256": hashlib.sha256(WHEEL.read_bytes()).hexdigest(),
    "native_sha256": hashlib.sha256(
        pathlib.Path(tradelearn._rust.__file__).read_bytes()
    ).hexdigest(),
    "sdk_hashes": sdk,
    "installed_members_before_after_equal": len(expected),
    "source_overlay": False,
    "native_recorder_smoke": True,
    "boto3": metadata.version("boto3"),
    "botocore": metadata.version("botocore"),
    "NA_execution_tested": False,
    "public_two_field_store_entry_tested": False,
    "real_sender_archive_roundtrip": False,
    "public_pypi_release": False,
}
(ROOT / "consumer-qualification-311.json").write_text(json.dumps(out, indent=2) + "\n")
