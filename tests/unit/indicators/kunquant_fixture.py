"""Explicit real KunQuant TS/STREAM build under the shared heavy slot."""

import hashlib
import json
import os
import platform
import resource
import subprocess
from pathlib import Path

from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.ops import WindowedAvg, WindowedCorrelation
from KunQuant.Stage import Function

from tradelearn.indicators._kunquant_compile import compile_modules, compiler_config
from tradelearn.indicators._kunquant_graph import ALPHA101_NAMES, build_alpha101, build_alpha158


def basic_graph():
    builder = Builder()
    with builder:
        close = Input("close")
        op = Input("open")
        Output(WindowedAvg(close, 5), "sma")
        Output(WindowedCorrelation(close, 5, op), "correlation")
        Output(Rank(close), "rank")
    return Function(builder.ops), dict(
        inputs=("close", "open"), profile="native primitive reference"
    )


def verify_build_owner(
    *,
    github_actions=False,
    owner_path=Path("/private/tmp/leafquant-20261112-heavy.lock/owner"),
):
    if github_actions:
        if (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
        ):
            raise RuntimeError("CI scope requires an isolated GitHub-hosted job")
        return "github-hosted-isolated-job"
    owner = json.loads(owner_path.read_text())
    if owner.get("task") not in {"TL2", "TL-INTEGRATION"} or not owner.get("nonce"):
        raise RuntimeError("TL must own the nonce-bound heavy slot before compiling")
    return "local-heavy"


def build(output, *, github_actions=False):
    resource_scope = verify_build_owner(github_actions=github_actions)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    width = 2 if platform.machine() in {"arm64", "aarch64"} else 4
    modules = []
    metadata = {}
    expressions = {}
    for label in ["basic", "alpha101", "alpha158"]:
        for layout in ["TS", "STREAM"]:
            # Compile passes mutate their Function; create a fresh definition.
            if label == "basic":
                function, description = basic_graph()
            elif label == "alpha101":
                function, description = build_alpha101(ALPHA101_NAMES)
            else:
                function, description = build_alpha158()
            metadata[label] = description
            expressions[label] = hashlib.sha256(str(function).encode()).hexdigest()
            config = KunCompilerConfig(
                dtype="double",
                blocking_len=width,
                input_layout="TS",
                output_layout=layout,
                partition_factor=8,
                split_source=8,
                allow_unaligned=False,
                options={
                    "opt_reduce": False,
                    "no_fast_stat": True,
                    "fast_log": False,
                    "no_skip_list": False,
                },
            )
            modules.append((f"tl2_{label}_{layout.lower()}", function, config))
    paths = compile_modules(modules, output)
    from numpy._core._multiarray_umath import __cpu_features__

    result = dict(
        resource_scope=resource_scope,
        libraries=paths,
        minimum_macos="14.0" if platform.system() == "Darwin" else None,
        max_child_rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        isa=tuple(name for name, enabled in __cpu_features__.items() if enabled),
        metadata=metadata,
        expressions=expressions,
        compiler=subprocess.check_output(
            [compiler_config().compiler, "--version"], text=True
        ).splitlines()[0],
        compiler_executable=compiler_config().compiler,
        max_child_rss_unit="bytes" if platform.system() == "Darwin" else "KiB",
        max_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        modules=[item[0] for item in modules],
        blocking_len=width,
        dtype="float64",
        options={
            "no_skip_list": False,
            "opt_reduce": False,
            "no_fast_stat": True,
            "fast_log": False,
        },
        lowering_profile="provided-window-extrema-v1",
        compiled=True,
        qualified=False,
    )
    (output / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"metadata"}}, indent=2
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--github-actions", action="store_true")
    args = parser.parse_args()
    build(args.output, github_actions=args.github_actions)
