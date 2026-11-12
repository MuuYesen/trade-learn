"""Release metadata cleanup preserves native bytes and verifies wheel RECORD."""

import base64
import csv
import hashlib
import io
import json
from zipfile import ZipFile

import pytest


def test_wheel_finalizer_normalizes_sbom_paths_only_and_rebuilds_record(tmp_path):
    from scripts.finalize_wheel import finalize

    name = "trade_learn-0.2.9-cp312-cp312-macosx_11_0_arm64.whl"
    source = tmp_path / name
    destination = tmp_path / "final" / name
    native = b"native bytes must stay exact"
    prefix = "trade_learn-0.2.9.dist-info"
    sbom = {
        "components": [
            {
                "bom-ref": "path+file:///private/builder/native#crate@1",
                "licenses": ["Apache-2.0"],
                "url": "https://github.com/public/repo",
            }
        ]
    }
    with ZipFile(source, "w") as wheel:
        wheel.writestr("tradelearn/_rust.cpython-312-darwin.so", native)
        wheel.writestr(prefix + "/sboms/native.cyclonedx.json", json.dumps(sbom))
        wheel.writestr(prefix + "/WHEEL", "Tag: cp312-cp312-macosx_11_0_arm64\n")
        wheel.writestr(prefix + "/RECORD", "")
    finalize(source, destination)
    with ZipFile(destination) as wheel:
        assert wheel.read("tradelearn/_rust.cpython-312-darwin.so") == native
        parsed = json.loads(wheel.read(prefix + "/sboms/native.cyclonedx.json"))
        assert "/private/builder" not in repr(parsed)
        assert parsed["components"][0]["licenses"] == ["Apache-2.0"]
        assert parsed["components"][0]["url"] == "https://github.com/public/repo"
        for path, digest, size in csv.reader(io.StringIO(wheel.read(prefix + "/RECORD").decode())):
            if path.endswith("/RECORD"):
                assert digest == size == ""
            else:
                body = wheel.read(path)
                expected = (
                    base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
                )
                assert digest == "sha256=" + expected and int(size) == len(body)
    with pytest.raises(ValueError, match="name"):
        finalize(source, tmp_path / "wrong-tag.whl")


def test_release_flags_remap_paths_and_preserve_existing_compiler_flags(tmp_path):
    from scripts.build_release import build_environment

    env = build_environment(tmp_path, {"RUSTFLAGS": "-C debuginfo=1"})
    flags = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[:2] == ["-C", "debuginfo=1"]
    assert f"--remap-path-prefix={tmp_path.resolve()}=/source/tradelearn" in flags
