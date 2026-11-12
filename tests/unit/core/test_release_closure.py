"""Frozen-byte gates reject stale source, omitted inputs and altered wheels."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest


def archives(tmp_path, *, omit_example=False, stale_wheel=False, ship_checker=False):
    from scripts.finalize_wheel import finalize

    root = tmp_path / "source"
    sources = {
        "tradelearn/example.py": b"VALUE = 1\n",
        "examples/research/helper.py": b"HELPER = 1\n",
    }
    if ship_checker:
        from scripts import check_release_closure

        sources["scripts/check_release_closure.py"] = Path(
            check_release_closure.__file__
        ).read_bytes()
    for name, body in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    manifest = tmp_path / "SOURCE-READY.json"
    manifest.write_text(
        json.dumps(
            {"files": {name: hashlib.sha256(body).hexdigest() for name, body in sources.items()}}
        )
    )
    wheel = tmp_path / "trade_learn-0.2.9-cp312-cp312-macosx_11_0_arm64.whl"
    with ZipFile(wheel, "w") as z:
        z.writestr(
            "tradelearn/example.py",
            b"STALE = 1\n" if stale_wheel else sources["tradelearn/example.py"],
        )
        z.writestr("trade_learn-0.2.9.dist-info/RECORD", "")
    finalize(wheel, wheel)
    sdist = tmp_path / "trade_learn-0.2.9.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, body in sources.items():
            if omit_example and name.startswith("examples/"):
                continue
            info = tarfile.TarInfo("trade_learn-0.2.9/" + name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return root, wheel, sdist, [manifest]


def test_closed_source_and_both_artifacts_have_exact_frozen_bytes(tmp_path):
    from scripts.check_release_closure import verify

    receipt = verify(*archives(tmp_path))
    assert receipt["frozen_files"] == 2 and receipt["wheel_frozen_files"] == 1
    assert receipt["runtime_verified"] is False


@pytest.mark.parametrize("failure", ["source", "wheel", "sdist", "record"])
def test_actual_stale_or_missing_input_cannot_pass_closure_gate(tmp_path, failure):
    from scripts.check_release_closure import verify

    arguments = archives(tmp_path, omit_example=failure == "sdist", stale_wheel=failure == "wheel")
    root, wheel, _, _ = arguments
    if failure == "source":
        (root / "tradelearn/example.py").write_text("CHANGED = 1\n")
    if failure == "record":
        with ZipFile(wheel) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["tradelearn/example.py"] = b"TAMPERED = 1\n"
        with ZipFile(wheel, "w") as archive:
            for name, body in members.items():
                archive.writestr(name, body)
    with pytest.raises(ValueError, match=failure):
        verify(*arguments)


def test_failed_replay_replaces_a_previous_success_receipt(tmp_path):
    from scripts.check_release_closure import main

    root, wheel, sdist, manifests = archives(tmp_path)
    wheel.unlink()
    output = tmp_path / "receipt.json"
    output.write_text(json.dumps({"artifact_closure_verified": True}))
    code = main(
        [
            "--root",
            str(root),
            "--wheel",
            str(wheel),
            "--sdist",
            str(sdist),
            "--manifest",
            str(manifests[0]),
            "--out",
            str(output),
        ]
    )
    assert code == 1
    assert json.loads(output.read_text())["artifact_closure_verified"] is False


def test_checker_can_be_shipped_in_the_sdist_it_checks(tmp_path):
    from scripts.check_release_closure import verify

    assert verify(*archives(tmp_path, ship_checker=True))["artifact_closure_verified"]
