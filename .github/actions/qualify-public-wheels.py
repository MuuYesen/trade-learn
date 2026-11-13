"""Qualify immutable public or explicitly selected CI wheels; never build or publish."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TESTS = [
    "test_store.py",
    "tests/test_sdk_tl_version_boundary.py",
    "tests/test_live_sizers.py",
    "tests/test_live_trailing_limit.py",
    "tests/test_live_drawdown_analyzer.py",
    "tests/unit/engine/test_bt_lifecycle_calls.py",
    "tests/unit/engine/test_bt_scalar_calls.py",
    "tests/unit/backtest/test_trailing_matching.py",
    "tests/unit/backtest/test_bt_trail_projection.py",
]
TESTS_032 = [
    "tests/unit/engine/test_bt_analyzer_param_inheritance.py",
    "tests/unit/engine/test_bt_component_construction.py",
    "tests/unit/engine/test_bt_returns_actual.py",
    "tests/unit/engine/test_bt_sharpe_actual.py",
    "tests/unit/engine/test_bt_stream_datetime.py",
    "tests/unit/engine/test_native_only_broker_controls.py",
    "tests/test_live_components.py",
    "tests/test_live_feed_metadata.py",
    "tests/test_live_returns_analyzer.py",
    "tests/test_live_post_only.py",
]
TESTS_033 = [
    "tests/unit/engine/test_bt_line_get_return.py",
    "tests/unit/engine/test_bt_stream_lifecycle_clock.py",
    "tests/unit/engine/test_bt_stream_master_datetime.py",
    "tests/unit/engine/test_bt_analyzer_master_clock.py",
    "tests/unit/engine/test_bt_datetime_algebra.py",
    "tests/unit/engine/test_nautilus_store.py",
    "tests/test_live_engine_transparent.py",
]
TESTS_034 = [
    "tests/test_live_close_routes.py",
    "tests/unit/engine/test_bt_validity_actual.py",
    "tests/unit/engine/test_bt_observer_lines.py",
    "tests/test_live_engine_market_validity.py",
    "tests/test_live_native_owner_release.py",
    "tests/test_live_sdk_owner_release.py",
]
PACKAGES = {"tradelearn", "tradelearn_runtime", "leafquant_monitor"}


def selected_tests(version):
    """Keep historical 031 evidence stable; require new tests on new releases."""
    release = tuple(map(int, re.match(r"(\d+)\.(\d+)\.(\d+)", version).groups()))
    tests = TESTS + (TESTS_032 if release >= (0, 3, 2) else [])
    if release >= (0, 3, 3):
        tests += TESTS_033
    if release >= (0, 3, 4):
        tests += TESTS_034
    return tests


def digest(path):
    """Hash the actual downloaded or installed file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed(args):
    """Verify installed public bytes and run the existing offline boundary tests."""
    work = args.work.resolve()
    expected = json.loads((work / "public-files.json").read_text())
    attempts = []

    def audit(event, _arguments):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            attempts.append(event)
            raise RuntimeError("network forbidden during installed qualification")

    sys.addaudithook(audit)
    sys.path.insert(0, str(work))
    import tl_store.ctrader._native
    import tl_store.ctrader.factories

    import tradelearn
    from tradelearn import _rust

    assert metadata.version("trade-learn") == args.tl
    assert tradelearn.__version__ == _rust.tradelearn_rust_version() == args.tl
    assert metadata.version("tl-store") == args.store
    assert metadata.version("nautilus-trader") == "2.0.0rc7"
    assert importlib.util.find_spec("lq_adapter_ctrader") is None
    site = Path(tradelearn.__file__).parent.parent
    sources = json.loads((work / "source-inventory.json").read_text())
    for name, sha in sources.items():
        assert digest(site / name) == sha, name
    records = {}
    for project in ["trade-learn", "tl-store", "nautilus-trader"]:
        distribution = metadata.distribution(project)
        count = 0
        for name, encoded, size in csv.reader(distribution.read_text("RECORD").splitlines()):
            if not encoded:
                continue
            path = Path(distribution.locate_file(name))
            algorithm, want = encoded.split("=", 1)
            assert (
                base64.urlsafe_b64encode(hashlib.new(algorithm, path.read_bytes()).digest())
                .rstrip(b"=")
                .decode()
                == want
            ), name
            assert path.stat().st_size == int(size), name
            count += 1
        with zipfile.ZipFile(work / "wheels" / expected[project]["filename"]) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if name.endswith("/RECORD"):
                    continue
                assert ".data/" not in name, "explicit wheel installation mapping required"
                assert Path(distribution.locate_file(name)).read_bytes() == archive.read(name), name
        records[project] = count
    native = {}
    for project, module in [("trade-learn", _rust), ("tl-store", tl_store.ctrader._native)]:
        path = Path(module.__file__)
        assert path.is_relative_to(work / "venv"), path
        native[project] = digest(path)
        with zipfile.ZipFile(work / "wheels" / expected[project]["filename"]) as archive:
            matches = [
                n
                for n in archive.namelist()
                if n.endswith(".so") and ("/_rust." in n or "/_native." in n)
            ]
            assert len(matches) == 1
            assert hashlib.sha256(archive.read(matches[0])).hexdigest() == native[project]
    store_wheel = work / "wheels" / expected["tl-store"]["filename"]
    with zipfile.ZipFile(store_wheel) as archive:
        members = {n: archive.read(n) for n in archive.namelist() if "/corresponding-source/" in n}
    tool_name = next(n for n in members if n.endswith("/tools/finalize_wheel.py"))
    tool = work / "recipient-verifier.py"
    tool.write_bytes(members[tool_name])
    spec = importlib.util.spec_from_file_location("recipient_verifier", tool)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    source_receipt = verifier.verify_corresponding_source(members)
    import pytest

    code = pytest.main(
        [
            "-c",
            str(work / "pytest.ini"),
            "-v",
            "-W",
            "error::pytest.PytestUnraisableExceptionWarning",
            "--junitxml=" + str(work / "tests.xml"),
        ]
        + [str(work / name) for name in selected_tests(args.tl)]
    )
    suites = ET.parse(work / "tests.xml").getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    assert counts["tests"] > 0 and counts["skipped"] == 0, counts
    assert not attempts
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in PACKAGES | {"tl_store"} and getattr(module, "__file__", None):
            assert Path(module.__file__).is_relative_to(work / "venv"), (name, module.__file__)
    result = {
        "source_sha": args.sha,
        "versions": {p: metadata.version(p) for p in expected},
        "Python": sys.version,
        "platform": sys.platform,
        "frozen_package_files": len(sources),
        "native_sha256": native,
        "RECORD_files": records,
        "recipient_source": source_receipt,
        "network_attempts": attempts,
        "pytest_exit": code,
        "test_counts": counts,
        "test_paths": selected_tests(args.tl),
        "own_packages_are_public_final_wheels": not bool(args.candidate_wheel),
        "candidate_wheel": args.candidate_wheel,
    }
    (work / "installed-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return code


def prepare(args):
    """Download immutable public files and create a fresh isolated Linux environment."""
    assert sys.version_info[:2] == (3, 13)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", args.tl)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", args.store)
    assert re.fullmatch(r"[0-9a-f]{40}", args.sha)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
    ).strip()
    assert actual == args.sha
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    wheels = work / "wheels"
    wheels.mkdir()
    candidate = Path(args.candidate_wheel).resolve() if args.candidate_wheel else None
    if candidate:
        assert candidate.is_file() and candidate.name.startswith("trade_learn-" + args.tl + "-")
        shutil.copyfile(candidate, wheels / candidate.name)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "download",
            "--index-url",
            "https://pypi.org/simple",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(wheels),
            *([] if candidate else ["trade-learn==" + args.tl]),
            "tl-store==" + args.store,
            "nautilus-trader==2.0.0rc7",
        ],
        check=True,
        timeout=300,
    )
    expected = {}
    for project, version, prefix in [
        ("trade-learn", args.tl, "trade_learn-"),
        ("tl-store", args.store, "tl_store-"),
        ("nautilus-trader", "2.0.0rc7", "nautilus_trader-"),
    ]:
        wheel = next(wheels.glob(prefix + "*.whl"))
        if project == "trade-learn" and candidate:
            expected[project] = {
                "version": version,
                "filename": wheel.name,
                "sha256": digest(wheel),
                "url": None,
                "candidate_not_public": True,
            }
            continue
        with urllib.request.urlopen(
            "https://pypi.org/pypi/" + project + "/" + version + "/json", timeout=45
        ) as response:
            release = json.load(response)
        published = next(x for x in release["urls"] if x["filename"] == wheel.name)
        assert digest(wheel) == published["digests"]["sha256"]
        expected[project] = {
            "version": version,
            "filename": wheel.name,
            "sha256": digest(wheel),
            "url": published["url"],
        }
    (work / "public-files.json").write_text(json.dumps(expected, indent=2) + "\n")
    inventory = {}
    for package in PACKAGES:
        for path in (args.source / package).rglob("*"):
            if path.is_file() and path.suffix in {
                ".py",
                ".json",
                ".html",
                ".hpp",
                ".jinja2",
                ".md",
            }:
                inventory[path.relative_to(args.source).as_posix()] = digest(path)
    assert inventory
    (work / "source-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    paths = [name for name in selected_tests(args.tl) if name != "test_store.py"] + [
        "tests/test_live_order_mapping.py",
        "tests/windows_private_directory.py",
    ]
    if "tests/unit/engine/test_nautilus_store.py" in selected_tests(args.tl):
        paths += ["tests/unit/backtest/_engine_config.py", "docs/guides/live.md"]
    for name in paths:
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / name, destination)
    for name in [
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/engine/__init__.py",
        "tests/unit/backtest/__init__.py",
    ]:
        (work / name).write_text("")
    shutil.copyfile(
        Path(__file__).with_name("public_store_binding_test.py"), work / "test_store.py"
    )
    (work / "pytest.ini").write_text("[pytest]\naddopts = -ra\n")
    subprocess.run([sys.executable, "-m", "venv", str(work / "venv")], check=True)
    python = work / "venv/bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--index-url",
            "https://pypi.org/simple",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--find-links",
            str(wheels),
            "trade-learn[nautilus,tl-store]==" + args.tl,
            "tl-store==" + args.store,
            "backtrader==1.9.78.123",
            "pytest",
        ],
        check=True,
        timeout=300,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            str(python),
            str(Path(__file__).resolve()),
            "--installed",
            "--source",
            str(args.source.resolve()),
            "--work",
            str(work),
            "--tl",
            args.tl,
            "--store",
            args.store,
            "--sha",
            args.sha,
            *(["--candidate-wheel", str(candidate)] if candidate else []),
        ],
        cwd=work,
        env=env,
        check=False,
        timeout=300,
    ).returncode


def main():
    """Run preparation or the fresh installed interpreter phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--tl", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument(
        "--candidate-wheel", help="Exact CI wheel path, never labelled public final"
    )
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    return installed(args) if args.installed else prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
