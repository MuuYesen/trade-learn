"""Exercise the built extension, not a Python recorder substitute.

Run with the target CPython:
    python native_smoke.py /absolute/path/to/tradelearn/_rust.cpython-313-darwin.so
The binary must have been built for that exact Python/platform ABI.
"""

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


def load(binary):
    spec = importlib.util.spec_from_loader(
        "_rust", importlib.machinery.ExtensionFileLoader("_rust", binary)
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


def main(binary):
    native = load(binary)
    with tempfile.TemporaryDirectory() as state_dir:
        recorder = native.Recorder(state_dir, capacity=1024, segment_bytes=4096)
        for n in range(100):
            assert recorder.append(json.dumps({"n": n, "price": "1.000000000001"}))
        status = json.loads(recorder.flush())
        assert status["accepted"] == status["durable"] == 100, status
        assert status["error"] is None, status
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(json.loads(recorder.close())))
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive(), "close did not complete"
        assert len(results) == 4 and all(s["closed"] and s["durable"] == 100 for s in results)
        committed = {p.name: p.read_bytes() for p in Path(state_dir).glob("*.log")}
        assert sum(len(data.splitlines()) for data in committed.values()) == 100
        child = subprocess.run(
            [sys.executable, __file__, binary, "--crash", state_dir], check=False
        )
        assert child.returncode == 73, child.returncode
        recovered = native.Recorder(state_dir)
        assert [p.name for p in Path(state_dir).glob("*.pending")] == ["unknown-history.pending"]
        assert (Path(state_dir) / "unknown-history.pending").read_bytes() == b"unknown\xff"
        for name, data in committed.items():
            assert (Path(state_dir) / name).read_bytes() == data
        for path in Path(state_dir).glob("*.log"):
            raw = path.read_bytes()
            for line in raw.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    assert line == b'{"incomplete":\xce'
                    continue
                assert isinstance(json.loads(line), dict)
            anchor = json.loads(path.with_suffix(".native-time.json").read_bytes())
            assert anchor["start_ts_ns"] > 0
        recovered.close()
    with tempfile.TemporaryDirectory() as invalid_dir:
        pending = Path(invalid_dir) / ("a" * 32 + "-00000000000000000000.pending")
        pending.write_bytes(b"unknown\xff")
        pending.with_suffix(".native-time.json").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    start_ts_ns=1760000000000000000,
                    source_schema="native-log-file.v1",
                    unexpected=True,
                )
            )
        )
        try:
            invalid = native.Recorder(invalid_dir)
        except ValueError:
            pass
        else:
            invalid.close()
            raise AssertionError("non-exact native anchor accepted")
        assert pending.read_bytes() == b"unknown\xff" and not pending.with_suffix(".log").exists()
    print(
        json.dumps(
            {
                "records": 100,
                "concurrent_close": 4,
                "crash_recovery": "passed",
                "python": sys.version.split()[0],
            }
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--crash":
        native = load(sys.argv[1])
        recorder = native.Recorder(sys.argv[3])
        assert recorder.append('{"committed_before_crash":true}')
        assert json.loads(recorder.flush())["durable"] == 1
        # Simulate an interrupted source write after a committed segment.
        own_run = json.loads(recorder.status())["run_id"]
        own = next(Path(sys.argv[3]).glob(own_run + "-*.log"))
        pending = own.with_suffix(".pending")
        own.rename(pending)
        (Path(sys.argv[3]) / "unknown-history.pending").write_bytes(b"unknown\xff")
        with open(pending, "ab", buffering=0) as tail:
            tail.write(b'{"incomplete":\xce')
            os.fsync(tail.fileno())
        os._exit(73)
    main(sys.argv[1])
