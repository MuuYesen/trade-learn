"""Deployment-preparation compiler config for fixed KunQuant 0.1.12.

The split-source module metadata references SkipListState without the header
or namespace it requires. Force the provided declaration header into compiler
translation units; leave the upstream library, algorithms and linker intact.
Importing this module never launches a compiler. Explicit compile_modules
callers own resource budgets and the heavy slot.
"""

import platform
from pathlib import Path


def compiler_config(*, opt_level=2):
    from importlib.metadata import version

    if version("KunQuant") != "0.1.12":
        raise ImportError("compiler compatibility is pinned to KunQuant==0.1.12")
    from KunQuant.jit import cfake

    header = Path(__file__).with_name("native") / "kunquant_split_state.hpp"

    class HeaderDeclarations(cfake.GCCCommandLineBuilder):
        @staticmethod
        def build_compile_options(config, source, output):
            return [
                *cfake.GCCCommandLineBuilder.build_compile_options(config, source, output),
                "-include",
                str(header),
            ]

    if platform.system() not in {"Darwin", "Linux"}:
        raise NotImplementedError("qualified compiler profile currently requires Darwin/Linux")
    flags = ("-mmacosx-version-min=14.0",) if platform.system() == "Darwin" else ()
    result = cfake.CppCompilerConfig(
        opt_level=opt_level, for_each=cfake.single_thread_compile, other_flags=flags
    )
    result.builder = HeaderDeclarations
    return result


def prepare_causal_extrema(function):
    """Expand only fixed-source argmin/argmax using provided window reductions.

    The source skip-list index uses invocation-local time. Quantiles and other
    operators retain their normal source paths; no global option or upstream
    class/function is patched. Like upstream compilation, this mutates Function.
    """
    from KunQuant.ops import TsArgMax, TsArgMin
    from KunQuant.ops.CompOp import WindowedReduce

    replacements = {}
    result = []
    for op in function.ops:
        op.replace_inputs(replacements)
        if isinstance(op, (TsArgMax, TsArgMin)):
            expanded = WindowedReduce.decompose(op, {})
            result.extend(expanded)
            replacements[op] = expanded[-1]
        else:
            result.append(op)
    function.set_ops(result)
    return function


def compile_modules(modules, output):
    """Compile a fresh isolated library per module using the safe source profile.

    Fresh objects ensure graph and compiler options belong to the manifest;
    generated objects from a different profile are never silently reused.
    The caller owns resource budgets and the heavy slot.
    """
    from KunQuant.jit import cfake

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    compiler = compiler_config()
    result = {}
    for name, function, configuration in modules:
        if configuration.options.get("no_skip_list", False):
            raise ValueError("global no_skip_list is incompatible with provided quantiles")
        prepare_causal_extrema(function)
        result[name] = cfake.compileit(
            [(name, function, configuration)],
            name,
            compiler,
            tempdir=str(output),
            keep_files=True,
            load=False,
        )
    return result
