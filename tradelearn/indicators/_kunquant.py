"""Internal artifact adapter for explicitly qualified KunQuant101/158 graphs.

Compilation belongs to deployment preparation. This module never compiles,
installs packages, pads a stock universe or converts arbitrary Python lambdas.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Artifact:
    source_revision: str
    expression_sha256: str
    library_sha256: str
    runtime_version: str
    system: str
    machine: str
    compiler: str
    isa: tuple[str, ...]
    dtype: str
    module: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    symbols: tuple[str, ...]
    warmup: int
    simd_width: int

    def __post_init__(self):
        for key in ("isa", "inputs", "outputs", "symbols"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be nonempty, ordered and unique")
        if self.simd_width < 1 or len(self.symbols) % self.simd_width:
            raise ValueError("stock count must match the compiled SIMD width; padding is forbidden")
        if not isinstance(self.warmup, int) or self.warmup < 1:
            raise ValueError("warmup must be positive")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("unsupported dtype")
        for names in (self.inputs, self.outputs):
            if not names or len(set(names)) != len(names):
                raise ValueError("duplicate or empty schema")
        for digest in (self.expression_sha256, self.library_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("artifact requires SHA256 digests")
        if not all((self.source_revision, self.runtime_version, self.compiler, self.module)):
            raise ValueError("artifact provenance must be explicit")

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_cross_section(spec, symbols, asof, inputs, input_times):
    """Validate the real universe and decision-time inputs before any pushData."""
    if tuple(symbols) != spec.symbols:
        raise ValueError("symbol identity/order mismatch")
    times = np.asarray(input_times)
    if times.shape != (len(symbols),) or not np.all(times == asof):
        raise ValueError("cross-section requires every real asset at the same decision time")
    if set(inputs) != set(spec.inputs):
        raise ValueError("input schema mismatch")
    result = {}
    for name, value in inputs.items():
        data = np.array(value, dtype=spec.dtype, copy=True, order="C")
        if data.shape != (len(symbols),) or not np.isfinite(data).all():
            raise ValueError("input shape/nonfinite data not qualified")
        result[name] = data
    return result


class _Kernel:
    """Internal loader for a separately built and verified STREAM module.

    Native runtime tests are mandatory before publishing any graph backed by
    this loader. Contract tests alone do not establish numerical qualification.
    """

    def __init__(self, path, spec: Artifact):
        if (platform.system(), platform.machine()) != (spec.system, spec.machine):
            raise ValueError("incompatible artifact OS/architecture")
        from importlib.metadata import version

        if version("KunQuant") != spec.runtime_version:
            raise ValueError("incompatible KunQuant version")
        from numpy._core._multiarray_umath import __cpu_features__

        if any(not __cpu_features__.get(flag.upper(), False) for flag in spec.isa):
            raise ValueError("incompatible CPU instruction set")
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec.library_sha256:
            raise ValueError("artifact library digest mismatch")
        from KunQuant.runner import KunRunner as kr

        self._kr, self.spec = kr, spec
        self._library = kr.Library.load(str(path))
        self._module = self._library.getModule(spec.module)
        if self._module.blocking_len != spec.simd_width:
            raise ValueError("artifact SIMD metadata differs from compiled module")
        if set(self._module.getOutputNames()) != set(spec.outputs):
            raise ValueError("artifact output schema differs from compiled module")
        self._unreliable = self._module.getOutputUnreliableCount()
        if spec.warmup < max(self._unreliable.values(), default=0) + 1:
            raise ValueError("artifact understates native warmup")
        self._available = {name: False for name in spec.outputs}
        self._executor = kr.createSingleThreadExecutor()


class KunQuantStream(_Kernel):
    """One persistent, explicitly identified same-time universe per event."""

    def __init__(self, path, spec, *, identity):
        if (
            not isinstance(identity, tuple)
            or len(identity) != 3
            or any(not isinstance(value, str) or not value for value in identity)
        ):
            raise ValueError("identity must be (strategy, feed, timeframe) nonempty strings")
        self._identity = identity
        super().__init__(path, spec)
        if self._module.output_layout != "STREAM":
            raise ValueError("module is not STREAM layout")
        self.reset()

    def _cache_handles(self):
        self._handles = {
            name: self._stream.queryBufferHandle(name)
            for name in (*self.spec.inputs, *self.spec.outputs)
        }

    @property
    def ready(self):
        return self.count >= self.spec.warmup and all(self._available.values()) and not self._failed

    def update(self, *, symbols, asof, inputs, input_times, sequence=None):
        if self._failed:
            raise RuntimeError("KunQuant state failed; reset/replay required")
        sequence = asof if sequence is None else sequence
        data = validate_cross_section(self.spec, symbols, asof, inputs, input_times)
        if self._sequence is not None:
            if sequence == self._sequence:
                if asof != self._asof or any(
                    not np.array_equal(data[key], self._last_inputs[key], equal_nan=True)
                    for key in data
                ):
                    raise ValueError("unsupported event revision or identity conflict")
                return {name: value.copy() for name, value in self._last_outputs.items()}
            if sequence < self._sequence or asof < self._asof:
                raise ValueError("out-of-order event")
        try:
            for name, array in data.items():
                self._stream.pushData(self._handles[name], array)
            self._stream.run()
            outputs = {
                name: np.array(self._stream.getCurrentBuffer(self._handles[name]), copy=True)
                for name in self.spec.outputs
            }
            for name, value in outputs.items():
                if value.shape != (len(self.spec.symbols),):
                    raise ValueError("native output shape differs from real asset set")
                if self.count + 1 <= self._unreliable[name]:
                    value.fill(np.nan)
                self._available[name] = bool(np.isfinite(value).all())
        except Exception:
            self._failed = True
            raise
        self._asof = asof
        self._sequence = sequence
        self.count += 1
        self._last_inputs = data
        self._last_outputs = {name: value.copy() for name, value in outputs.items()}
        return outputs

    def snapshot(self):
        if self._failed or not self.count:
            raise RuntimeError("snapshot requires a successful run")
        state = bytes(self._stream.serializeStates())
        return dict(
            version=4,
            identity=self._identity,
            fingerprint=self.spec.fingerprint(),
            asof=self._asof,
            count=self.count,
            state=state,
            state_sha256=hashlib.sha256(state).hexdigest(),
            available=dict(self._available),
            sequence=self._sequence,
            inputs={name: value.copy() for name, value in self._last_inputs.items()},
            outputs={name: value.copy() for name, value in self._last_outputs.items()},
        )

    def restore(self, snapshot):
        if snapshot.get("identity") != self._identity:
            raise ValueError("incompatible KunQuant snapshot identity")
        if snapshot.get("version") != 4 or snapshot.get("fingerprint") != self.spec.fingerprint():
            raise ValueError("incompatible KunQuant snapshot")
        if (
            set(snapshot.get("inputs", {})) != set(self.spec.inputs)
            or set(snapshot.get("outputs", {})) != set(self.spec.outputs)
            or type(snapshot.get("count")) is not int
            or snapshot["count"] < 1
        ):
            raise ValueError("incompatible snapshot schema/count")
        if any(
            np.asarray(value).shape != (len(self.spec.symbols),)
            for value in (*snapshot["inputs"].values(), *snapshot["outputs"].values())
        ):
            raise ValueError("incompatible snapshot array shape")
        state = snapshot.get("state")
        if not isinstance(state, bytes) or hashlib.sha256(state).hexdigest() != snapshot.get(
            "state_sha256"
        ):
            raise ValueError("incompatible snapshot native state checksum")
        stream = self._kr.StreamContext(
            self._executor, self._module, len(self.spec.symbols), snapshot["state"]
        )
        self._stream = stream
        self._cache_handles()
        self._sequence = snapshot["sequence"]
        self._last_inputs = {name: value.copy() for name, value in snapshot["inputs"].items()}
        self._last_outputs = {name: value.copy() for name, value in snapshot["outputs"].items()}
        self._available = {
            name: bool(np.isfinite(value).all()) for name, value in self._last_outputs.items()
        }
        self._asof = snapshot["asof"]
        self.count = snapshot["count"]
        self._failed = False

    def reset(self):
        self._stream = self._kr.StreamContext(self._executor, self._module, len(self.spec.symbols))
        self._cache_handles()
        self._asof = None
        self._sequence = None
        self._last_inputs = {}
        self._last_outputs = {}
        self.count = 0
        self._available = {name: False for name in self.spec.outputs}
        self._failed = False

    def close(self):
        """Release working native state; reset can warm a new context."""
        self._stream = None
        self._last_inputs.clear()
        self._last_outputs.clear()
        self._failed = True
        self._available = {name: False for name in self.spec.outputs}


class KunQuantBatch(_Kernel):
    """One native TS matrix pass with the same artifact/source warmup policy."""

    def __init__(self, path, spec):
        super().__init__(path, spec)
        if self._module.output_layout != "TS":
            raise ValueError("batch module is not TS layout")

    def compute(self, *, symbols, times, inputs, input_times):
        if tuple(symbols) != self.spec.symbols:
            raise ValueError("symbol identity/order mismatch")
        times = np.asarray(times)
        if times.ndim != 1 or not (
            np.issubdtype(times.dtype, np.integer) or np.issubdtype(times.dtype, np.datetime64)
        ):
            raise ValueError("times require explicit integer clock units or normalized datetime64")
        if np.any(times != times) or np.any(times[1:] < times[:-1]):
            raise ValueError("missing/out-of-order decision times")
        clocks = np.asarray(input_times)
        if clocks.shape != (len(times), len(symbols)) or not np.all(clocks == times[:, None]):
            raise ValueError("every real asset must share its decision time")
        if set(inputs) != set(self.spec.inputs):
            raise ValueError("input schema mismatch")
        data = {}
        for name, value in inputs.items():
            array = np.array(value, dtype=self.spec.dtype, copy=True, order="C")
            if array.shape != (len(times), len(symbols)) or not np.isfinite(array).all():
                raise ValueError("input shape/nonfinite data not qualified")
            data[name] = array
        if not len(times):
            return {
                name: np.empty((0, len(symbols)), dtype=self.spec.dtype)
                for name in self.spec.outputs
            }
        raw = self._kr.runGraph(
            self._executor, self._module, data, 0, len(times), num_stocks=len(symbols)
        )
        result = {name: np.array(raw[name], copy=True) for name in self.spec.outputs}
        for name, value in result.items():
            value[: self._unreliable[name]] = np.nan
        return result

    def compute_frame(self, bars):
        """Preserve the existing Pandas Bars/features contract, no new dataset DTO."""
        import pandas as pd

        from tradelearn.research.features import _PanelView

        panel = _PanelView(bars)
        first = panel.field(self.spec.inputs[0])
        if set(first.columns) != set(self.spec.symbols):
            raise ValueError("Bars universe differs from artifact")
        clock = first.index.as_unit("ns").asi8
        inputs = {
            name: panel.field(name)
            .reindex(index=first.index, columns=self.spec.symbols)
            .to_numpy(dtype=self.spec.dtype)
            for name in self.spec.inputs
        }
        output = self.compute(
            symbols=self.spec.symbols,
            times=clock,
            inputs=inputs,
            input_times=np.repeat(clock[:, None], len(self.spec.symbols), axis=1),
        )
        index = pd.MultiIndex.from_product(
            [first.index, self.spec.symbols], names=["timestamp", "symbol"]
        )
        frame = pd.DataFrame(
            {name: value.reshape(-1) for name, value in output.items()}, index=index
        ).sort_index()
        frame.attrs["backend"] = self.get_params()
        return frame

    def fit(self, data, y=None):
        """No training or hidden computation; supports the existing Pipeline protocol."""
        return self

    def transform(self, data):
        return self.compute_frame(data)

    def fit_transform(self, data, y=None):
        return self.fit(data, y).transform(data)

    def get_params(self, deep=True):
        """Serializable source metadata for explicit ResearchRun registration."""
        from tradelearn.core.contracts import REQUIRED_BAR_COLUMNS

        return {
            "type": type(self).__name__,
            "bars_columns": tuple(dict.fromkeys((*REQUIRED_BAR_COLUMNS, *self.spec.inputs))),
            "source_revision": self.spec.source_revision,
            "runtime_version": self.spec.runtime_version,
            "module": self.spec.module,
            "expression_sha256": self.spec.expression_sha256,
            "library_sha256": self.spec.library_sha256,
            "inputs": self.spec.inputs,
            "outputs": self.spec.outputs,
            "symbols": self.spec.symbols,
            "dtype": self.spec.dtype,
            "warmup": self.spec.warmup,
        }
