"""Compile and qualify one explicit rolling-mean graph, never arbitrary lambdas.

This is a deployment-preparation check, not a report or runtime operation.
It does not certify the full Alpha101/158 families or another OS/ISA.
Caller owns the heavy slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def main(argv=None) -> int:
    """Build batch/STREAM modules and verify state, numerical facts and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    from KunQuant.Driver import KunCompilerConfig
    from KunQuant.jit.cfake import compileit
    from KunQuant.Op import Builder, Input, Output
    from KunQuant.ops import WindowedAvg
    from KunQuant.runner import KunRunner as kr
    from KunQuant.Stage import Function

    from tradelearn.indicators._kunquant import Artifact, KunQuantStream
    from tradelearn.indicators._kunquant_compile import compiler_config

    graph = Builder()
    with graph:
        close = Input("close")
        Output(WindowedAvg(close, 3), "mean3")
    function = Function(graph.ops)
    compiler = compiler_config()
    machine = platform.machine()
    # STREAM buffer alignment is four even for double-precision ARM vectors.
    width = 4
    compiled_width = 2 if machine in {"arm64", "aarch64"} else 4
    common = dict(dtype="double", blocking_len=compiled_width, input_layout="TS")
    library_path = compileit(
        [
            ("mean3_stream", function, KunCompilerConfig(**common, output_layout="STREAM")),
            ("mean3_batch", function, KunCompilerConfig(**common, output_layout="TS")),
        ],
        "mean3_acceptance",
        compiler,
        tempdir=str(output),
        keep_files=True,
        load=False,
    )
    path = Path(library_path)
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    compiler_version = subprocess.check_output(
        [compiler.compiler, "--version"], text=True
    ).splitlines()[0]
    spec = Artifact(
        source_revision=f"KunQuant=={version('KunQuant')};check_script_sha256={script_digest}",
        expression_sha256=hashlib.sha256(b"WindowedAvg(Input(close),3)->mean3;float64").hexdigest(),
        library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runtime_version=version("KunQuant"),
        system=platform.system(),
        machine=machine,
        compiler=compiler_version,
        isa=("ASIMD",) if machine in {"arm64", "aarch64"} else ("AVX2", "FMA3"),
        dtype="float64",
        module="mean3_stream",
        inputs=("close",),
        outputs=("mean3",),
        symbols=tuple(f"ASSET{i}" for i in range(width)),
        warmup=3,
        simd_width=compiled_width,
    )
    values = np.arange(12 * width, dtype=np.float64).reshape(12, width) + 1.0
    library = kr.Library.load(str(path))
    executor = kr.createSingleThreadExecutor()
    batch = kr.runGraph(
        executor, library.getModule("mean3_batch"), {"close": values}, 0, len(values)
    )["mean3"]
    identity = ("synthetic-mean-check", "synthetic-bars", "one-minute")
    stream = KunQuantStream(path, spec, identity=identity)
    outputs = []
    for timestamp, row in enumerate(values):
        outputs.append(
            stream.update(
                symbols=spec.symbols,
                asof=timestamp,
                inputs={"close": row},
                input_times=np.full(width, timestamp),
            )["mean3"]
        )
        if timestamp == 5:
            checkpoint = stream.snapshot()
    actual = np.asarray(outputs)
    expected = pd.DataFrame(values).rolling(3).mean().to_numpy()[2:]
    np.testing.assert_allclose(actual[2:], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, batch, equal_nan=True, rtol=0, atol=1e-12)
    restored = KunQuantStream(path, spec, identity=identity)
    restored.restore(checkpoint)
    for timestamp in range(6, len(values)):
        got = restored.update(
            symbols=spec.symbols,
            asof=timestamp,
            inputs={"close": values[timestamp]},
            input_times=np.full(width, timestamp),
        )["mean3"]
        np.testing.assert_array_equal(got, actual[timestamp])
    # An identical event is reused and returned arrays remain caller-owned.
    duplicate = stream.update(
        symbols=spec.symbols, asof=11, inputs={"close": values[-1]}, input_times=np.full(width, 11)
    )["mean3"]
    np.testing.assert_array_equal(duplicate, actual[-1])
    duplicate[:] = -1
    np.testing.assert_array_equal(
        stream.update(
            symbols=spec.symbols,
            asof=11,
            inputs={"close": values[-1]},
            input_times=np.full(width, 11),
        )["mean3"],
        actual[-1],
    )
    try:
        stream.update(
            symbols=spec.symbols,
            asof=11,
            inputs={"close": values[-1] + 1},
            input_times=np.full(width, 11),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting revision accepted")
    # Real identity/order/time counterexamples; no padding or partial cross section.
    for symbols, timestamp, times in [
        (spec.symbols[::-1], 12, np.full(width, 12)),
        (spec.symbols, 10, np.full(width, 10)),
        (spec.symbols, 12, np.full(width, 11)),
    ]:
        try:
            stream.update(
                symbols=symbols, asof=timestamp, inputs={"close": values[-1]}, input_times=times
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid cross section accepted")
    payload = {
        "artifact": asdict(spec),
        "library": path.name,
        "batch_stream_equal": True,
        "pandas_oracle_equal": True,
        "restore_exact": True,
        "counterexamples": 4,
        "duplicate_reused_with_owned_output": True,
        "full_provided_graphs_qualified": False,
    }
    (output / "acceptance.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
