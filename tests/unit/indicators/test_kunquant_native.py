"""Real KunQuant native qualification; compile fixture in a separate heavy step."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("KunQuant") is None, reason="real KunQuant dependency absent"
)


def test_real_provided_builders_and_scope():
    from tradelearn.indicators._kunquant_graph import ALPHA101_NAMES, build_alpha101, build_alpha158

    function, metadata = build_alpha101(ALPHA101_NAMES)
    assert len(ALPHA101_NAMES) == 82 and len(function.ops) > 82
    assert metadata["profile"] == "KunQuant-0.1.12-upstream-Alpha101"
    assert not metadata["qualified"]
    function, metadata = build_alpha158()
    assert len(metadata["outputs"]) == 158 and len(set(metadata["outputs"])) == 158
    assert not metadata["qualified"]
    with pytest.raises(ValueError):
        build_alpha101(("alpha191",))


@pytest.fixture
def native_fixture():
    path = Path(
        os.environ.get("TL2_KUNQUANT_FIXTURE", "/private/tmp/tl2-kunquant-upstream/fixture.json")
    )
    if not path.exists():
        pytest.skip("explicit real native fixture has not been compiled")
    result = json.loads(path.read_text())
    assert all(Path(value).exists() for value in result["libraries"].values())
    assert set(result["metadata"]) == {"basic", "alpha101", "alpha158"}
    return result


def matrix_inputs(kind="normal", count=640):
    rng = np.random.default_rng(481)
    close = 30 + np.cumsum(rng.normal(0, 0.1, (count, 8)), axis=0)
    op = close + rng.normal(0, 0.05, close.shape)
    high = np.maximum(close, op) + 0.5
    low = np.minimum(close, op) - 0.5
    volume = rng.uniform(1000.0, 2000.0, close.shape)
    if kind == "constant":
        close[:] = 30.0
        op[:] = 29.9
        high[:] = 31.0
        low[:] = 29.0
        volume[:] = 1000.0
    values = dict(close=close, open=op, high=high, low=low, volume=volume, amount=close * volume)
    for value in values.values():
        value[:, 1] = value[:, 0]  # actual tied assets
    return values


def binary(fixture, label, layout="stream"):
    return fixture["libraries"][f"tl2_{label}_{layout}"]


def artifact(fixture, label, module, layout="stream"):
    import hashlib
    import importlib.metadata as metadata
    import platform

    from tradelearn.indicators._kunquant import Artifact

    path = Path(binary(fixture, label, layout))
    description = fixture["metadata"][label]
    return Artifact(
        source_revision=description.get("source_revision", "KunQuant==0.1.12"),
        expression_sha256=fixture["expressions"][label],
        library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        runtime_version=metadata.version("KunQuant"),
        system=platform.system(),
        machine=platform.machine(),
        compiler=fixture["compiler"],
        isa=tuple(fixture["isa"]),
        dtype="float64",
        module=f"tl2_{label}_{layout}",
        inputs=tuple(description["inputs"]),
        outputs=tuple(module.getOutputNames()),
        symbols=tuple(f"S{i}" for i in range(8)),
        warmup=max(module.getOutputUnreliableCount().values(), default=0) + 1,
        simd_width=module.blocking_len,
    )


def warmup_mask(values, count):
    values = np.array(values, copy=True)
    values[:count] = np.nan
    return values


@pytest.mark.parametrize("label", ["basic", "alpha101", "alpha158"])
@pytest.mark.parametrize("kind", ["normal", "constant"])
def test_real_ts_stream_restart_ties_reference_and_ownership(native_fixture, label, kind):
    from dataclasses import replace

    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantBatch, KunQuantStream

    values = matrix_inputs(kind)
    count = len(values["close"])
    data = {
        key: np.ascontiguousarray(values[key])
        for key in native_fixture["metadata"][label]["inputs"]
    }
    library = kr.Library.load(binary(native_fixture, label))
    stream_module = library.getModule(f"tl2_{label}_stream")
    batch_library = kr.Library.load(binary(native_fixture, label, "ts"))
    batch_module = batch_library.getModule(f"tl2_{label}_ts")
    executor = kr.createSingleThreadExecutor()
    batch = kr.runGraph(executor, batch_module, data, 0, count, num_stocks=8)
    unreliable = stream_module.getOutputUnreliableCount()
    spec = artifact(native_fixture, label, stream_module)
    batch_adapter = KunQuantBatch(
        binary(native_fixture, label, "ts"), artifact(native_fixture, label, batch_module, "ts")
    )
    times = np.arange(1, count + 1, dtype=np.int64)
    adapted_batch = batch_adapter.compute(
        symbols=spec.symbols,
        times=times,
        inputs=data,
        input_times=np.repeat(times[:, None], 8, axis=1),
    )
    stream = KunQuantStream(
        binary(native_fixture, label),
        spec,
        identity=("test-strategy", "synthetic-feed", "bar-sequence"),
    )
    outputs = {name: [] for name in spec.outputs}
    resumed = None
    saved = None
    for i in range(count):
        row = {key: value[i] for key, value in data.items()}
        result = stream.update(
            symbols=spec.symbols, asof=i + 1, inputs=row, input_times=np.full(8, i + 1)
        )
        for name in outputs:
            outputs[name].append(result[name])
        if i == 319:
            snapshot = stream.snapshot()
            resumed = KunQuantStream(
                binary(native_fixture, label),
                spec,
                identity=("test-strategy", "synthetic-feed", "bar-sequence"),
            )
            resumed.restore(snapshot)
            saved = {name: value.copy() for name, value in result.items()}
            incompatible = KunQuantStream(
                binary(native_fixture, label),
                replace(spec, symbols=spec.symbols[::-1]),
                identity=("test-strategy", "synthetic-feed", "bar-sequence"),
            )
            with pytest.raises(ValueError, match="incompatible"):
                incompatible.restore(snapshot)
        elif i > 319:
            restored = resumed.update(
                symbols=spec.symbols, asof=i + 1, inputs=row, input_times=np.full(8, i + 1)
            )
            for name in outputs:
                np.testing.assert_allclose(
                    restored[name], result[name], equal_nan=True, rtol=2e-8, atol=2e-8
                )
    errors = []
    for name, rows in outputs.items():
        expected = warmup_mask(batch[name], unreliable[name])
        np.testing.assert_allclose(
            adapted_batch[name], expected, equal_nan=True, rtol=2e-8, atol=2e-8
        )
        actual = np.asarray(rows)
        if not np.allclose(actual, expected, equal_nan=True, rtol=2e-8, atol=2e-8):
            errors.append(name)
        assert np.array_equal(rows[319], saved[name], equal_nan=True)
    assert not errors, f"TS/STREAM mismatches: {errors}"
    close = pd.DataFrame(values["close"])
    op = pd.DataFrame(values["open"])
    if label == "basic":
        expected = {
            "sma": close.rolling(5).mean().to_numpy(),
            "correlation": close.rolling(5).corr(op).to_numpy(),
            "rank": close.rank(axis=1, pct=True, method="average").to_numpy(),
        }
    elif label == "alpha101":
        # Provided upstream profile, intentionally independent of old TL rank=min.
        returns = close.pct_change(fill_method=None)
        inner = close.copy()
        inner[returns < 0] = returns.rolling(20).std()
        argmax = (inner**2).rolling(5).apply(np.argmax, raw=True) + 1
        expected = {
            "alpha001": argmax.rank(axis=1, pct=True, method="average").to_numpy(),
            "alpha101": (values["close"] - values["open"])
            / (values["high"] - values["low"] + 0.001),
        }
    else:
        expected = {
            "KMID": (values["close"] - values["open"]) / values["open"],
            "MA5": close.rolling(5).mean().to_numpy() / values["close"],
            "ROC5": close.shift(5).to_numpy() / values["close"],
            "STD5": close.rolling(5).std().to_numpy() / values["close"],
        }
    for name, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(outputs[name]),
            warmup_mask(value, unreliable[name]),
            equal_nan=True,
            rtol=2e-8,
            atol=2e-8,
        )


def test_native_manifest_cannot_understate_seed_or_change_module_width(native_fixture):
    from dataclasses import replace

    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantStream

    library = kr.Library.load(binary(native_fixture, "basic"))
    module = library.getModule("tl2_basic_stream")
    spec = artifact(native_fixture, "basic", module)
    with pytest.raises(ValueError, match="warmup"):
        KunQuantStream(
            binary(native_fixture, "basic"),
            replace(spec, warmup=1),
            identity=("test-strategy", "synthetic-feed", "bar-sequence"),
        )
    with pytest.raises(ValueError, match="SIMD"):
        KunQuantStream(
            binary(native_fixture, "basic"),
            replace(spec, simd_width=4 if spec.simd_width == 2 else 2),
            identity=("test-strategy", "synthetic-feed", "bar-sequence"),
        )


def test_native_same_timestamp_identity_duplicate_and_restart(native_fixture):
    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantStream

    library = kr.Library.load(binary(native_fixture, "basic"))
    module = library.getModule("tl2_basic_stream")
    spec = artifact(native_fixture, "basic", module)
    stream = KunQuantStream(
        binary(native_fixture, "basic"),
        spec,
        identity=("test-strategy", "synthetic-feed", "bar-sequence"),
    )
    inputs = {"close": np.arange(8, dtype=float) + 20, "open": np.arange(8, dtype=float) + 19}
    first = stream.update(
        symbols=spec.symbols, asof=10, sequence=1, inputs=inputs, input_times=np.full(8, 10)
    )
    duplicate = stream.update(
        symbols=spec.symbols, asof=10, sequence=1, inputs=inputs, input_times=np.full(8, 10)
    )
    assert stream.count == 1
    for name in first:
        np.testing.assert_equal(first[name], duplicate[name])
    second = {key: value + 0.1 for key, value in inputs.items()}
    stream.update(
        symbols=spec.symbols, asof=10, sequence=2, inputs=second, input_times=np.full(8, 10)
    )
    assert stream.count == 2
    restored = KunQuantStream(
        binary(native_fixture, "basic"),
        spec,
        identity=("test-strategy", "synthetic-feed", "bar-sequence"),
    )
    restored.restore(stream.snapshot())
    restored.update(
        symbols=spec.symbols, asof=10, sequence=2, inputs=second, input_times=np.full(8, 10)
    )
    assert restored.count == 2
    with pytest.raises(ValueError, match="revision"):
        restored.update(
            symbols=spec.symbols, asof=10, sequence=2, inputs=inputs, input_times=np.full(8, 10)
        )


def test_corrupt_controller_snapshot_rejects_before_mutation(native_fixture):
    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantStream

    library = kr.Library.load(binary(native_fixture, "basic"))
    spec = artifact(native_fixture, "basic", library.getModule("tl2_basic_stream"))
    stream = KunQuantStream(
        binary(native_fixture, "basic"),
        spec,
        identity=("test-strategy", "synthetic-feed", "bar-sequence"),
    )
    inputs = {"close": np.arange(8, dtype=float) + 20, "open": np.arange(8, dtype=float) + 19}
    stream.update(symbols=spec.symbols, asof=1, inputs=inputs, input_times=np.ones(8, dtype=int))
    snapshot = stream.snapshot()
    snapshot["inputs"].pop("close")
    before = stream._stream
    with pytest.raises(ValueError, match="snapshot"):
        stream.restore(snapshot)
    assert stream._stream is before
    snapshot = stream.snapshot()
    snapshot["state"] = snapshot["state"][:-1]
    with pytest.raises(ValueError, match="checksum"):
        stream.restore(snapshot)
    assert stream._stream is before


def test_provided_config_never_prepares_future_or_zero_rolling_windows():
    from tradelearn.indicators._kunquant_graph import build_alpha158

    for configuration in [
        {"price": {"windows": [-1], "feature": ["OPEN"]}},
        {"volume": {"windows": [-1]}},
        {"rolling": {"windows": [0], "include": ["MA"]}},
        {"rolling": {"windows": [2.5], "include": ["MA"]}},
    ]:
        with pytest.raises(ValueError, match="window"):
            build_alpha158(configuration)


def test_native_batch_keeps_existing_pandas_features_contract(native_fixture):
    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantBatch
    from tradelearn.research.features import FeatureSet

    values = matrix_inputs(count=20)
    times = pd.date_range("2026-01-01", periods=20, tz="UTC")
    symbols = tuple(f"S{i}" for i in range(8))
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    frame = pd.DataFrame({key: value.reshape(-1) for key, value in values.items()}, index=index)
    library = kr.Library.load(binary(native_fixture, "basic", "ts"))
    module = library.getModule("tl2_basic_ts")
    backend = KunQuantBatch(
        binary(native_fixture, "basic", "ts"), artifact(native_fixture, "basic", module, "ts")
    )
    transformed = backend.fit_transform(frame)
    assert transformed.index.equals(frame.index)
    assert set(transformed) == {"sma", "correlation", "rank"}
    features = FeatureSet({"rank": transformed["rank"]}).fit_transform(frame)
    pd.testing.assert_series_equal(features["rank"], transformed["rank"])
    assert backend.get_params()["expression_sha256"] == backend.spec.expression_sha256
    broken = frame.drop(index=(times[3], symbols[2]))
    with pytest.raises(ValueError, match="nonfinite"):
        backend.transform(broken)


@pytest.mark.parametrize("label", ["basic", "alpha101", "alpha158"])
def test_real_asset_permutation_and_smaller_real_universe(native_fixture, label):
    from dataclasses import replace

    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantBatch

    library = kr.Library.load(binary(native_fixture, label, "ts"))
    module = library.getModule(f"tl2_{label}_ts")
    spec = artifact(native_fixture, label, module, "ts")
    backend = KunQuantBatch(binary(native_fixture, label, "ts"), spec)
    values = matrix_inputs(count=320)
    times = np.arange(len(values["close"]))
    inputs = {key: value for key, value in values.items() if key in spec.inputs}
    expected = backend.compute(
        symbols=spec.symbols,
        times=times,
        inputs=inputs,
        input_times=np.repeat(times[:, None], 8, axis=1),
    )
    permutation = [3, 1, 0, 7, 4, 6, 2, 5]
    inverse = np.argsort(permutation)
    changed = replace(spec, symbols=tuple(spec.symbols[i] for i in permutation))
    alternate = KunQuantBatch(binary(native_fixture, label, "ts"), changed)
    outputs = alternate.compute(
        symbols=changed.symbols,
        times=times,
        inputs={key: value[:, permutation] for key, value in inputs.items()},
        input_times=np.repeat(times[:, None], 8, axis=1),
    )
    for name in outputs:
        np.testing.assert_allclose(
            outputs[name][:, inverse], expected[name], equal_nan=True, rtol=2e-8, atol=2e-8
        )
    small = replace(spec, symbols=spec.symbols[:4])
    subset = KunQuantBatch(binary(native_fixture, label, "ts"), small)
    result = subset.compute(
        symbols=small.symbols,
        times=times,
        inputs={key: value[:, :4] for key, value in inputs.items()},
        input_times=np.repeat(times[:, None], 4, axis=1),
    )
    assert all(value.shape == (len(times), 4) for value in result.values())
    if label == "basic":
        np.testing.assert_allclose(
            result["rank"], pd.DataFrame(inputs["close"][:, :4]).rank(axis=1, pct=True).to_numpy()
        )


def test_restart_cannot_cross_strategy_feed_or_timeframe_identity(native_fixture):
    from KunQuant.runner import KunRunner as kr

    from tradelearn.indicators._kunquant import KunQuantStream

    library = kr.Library.load(binary(native_fixture, "basic"))
    spec = artifact(native_fixture, "basic", library.getModule("tl2_basic_stream"))
    first = KunQuantStream(binary(native_fixture, "basic"), spec, identity=("A", "feed", "1m"))
    first.update(
        symbols=spec.symbols,
        asof=1,
        inputs={"close": np.arange(8, dtype=float) + 20, "open": np.arange(8, dtype=float) + 19},
        input_times=np.ones(8, dtype=int),
    )
    snapshot = first.snapshot()
    for identity in [("B", "feed", "1m"), ("A", "other-feed", "1m"), ("A", "feed", "5m")]:
        other = KunQuantStream(binary(native_fixture, "basic"), spec, identity=identity)
        with pytest.raises(ValueError, match="identity"):
            other.restore(snapshot)


@pytest.mark.parametrize("label", ["alpha101", "alpha158"])
def test_native_all_output_prefixes_and_bounded_working_state(native_fixture, label):
    from KunQuant.runner import KunRunner as kr

    values = matrix_inputs(count=640)
    data = {
        name: np.ascontiguousarray(values[name])
        for name in native_fixture["metadata"][label]["inputs"]
    }
    library = kr.Library.load(binary(native_fixture, label, "ts"))
    module = library.getModule(f"tl2_{label}_ts")
    executor = kr.createSingleThreadExecutor()
    full = kr.runGraph(executor, module, data, 0, 640, num_stocks=8)
    unreliable = module.getOutputUnreliableCount()
    for count in (65, 129, 320):
        prefix = kr.runGraph(
            executor,
            module,
            {name: np.ascontiguousarray(value[:count]) for name, value in data.items()},
            0,
            count,
            num_stocks=8,
        )
        for name in full:
            np.testing.assert_allclose(
                warmup_mask(prefix[name], unreliable[name]),
                warmup_mask(full[name][:count], unreliable[name]),
                equal_nan=True,
                rtol=2e-8,
                atol=2e-8,
            )
    stream_library = kr.Library.load(binary(native_fixture, label))
    stream_module = stream_library.getModule(f"tl2_{label}_stream")
    stream = kr.StreamContext(executor, stream_module, 8)
    handles = {name: stream.queryBufferHandle(name) for name in data}
    seed_size = None
    for index in range(12000):
        for name, value in data.items():
            stream.pushData(handles[name], value[index % 640])
        stream.run()
        if index == 5999:
            seed_size = len(bytes(stream.serializeStates()))
    assert len(bytes(stream.serializeStates())) == seed_size


def test_long_extremum_source_profile_avoids_invocation_local_skiplist():
    from KunQuant.Driver import compileit

    from tradelearn.indicators._kunquant_compile import prepare_causal_extrema
    from tradelearn.indicators._kunquant_graph import build_alpha158

    width = 2 if __import__("platform").machine() in {"arm64", "aarch64"} else 4
    config = {"rolling": {"windows": [60], "include": ["IMAX", "IMIN", "IMXD"]}}
    for layout in ("TS", "STREAM"):
        function, metadata = build_alpha158(config)
        prepare_causal_extrema(function)
        source = "\n".join(
            compileit(
                function,
                "extremum_profile",
                dtype="double",
                blocking_len=width,
                input_layout="TS",
                output_layout=layout,
                allow_unaligned=False,
                options={"no_skip_list": False, "opt_reduce": False},
            )
        )
        assert "SkipListArgMin" not in source
        assert f"ReduceArgMax<double, {width}>" in source
        assert f"ReduceArgMin<double, {width}>" in source
        assert set(metadata["outputs"]) == {"IMAX60", "IMIN60", "IMXD60"}


@pytest.mark.parametrize("label", ["alpha101", "alpha158"])
@pytest.mark.parametrize("layout", ["TS", "STREAM"])
def test_full_source_lowering_retains_quantiles_and_removes_only_extremum_skiplist(label, layout):
    from KunQuant.Driver import compileit
    from KunQuant.ops import TsArgMax, TsArgMin

    from tradelearn.indicators._kunquant_compile import prepare_causal_extrema
    from tradelearn.indicators._kunquant_graph import ALPHA101_NAMES, build_alpha101, build_alpha158

    width = 2 if __import__("platform").machine() in {"arm64", "aarch64"} else 4
    function, metadata = build_alpha101(ALPHA101_NAMES) if label == "alpha101" else build_alpha158()
    assert any(isinstance(op, (TsArgMax, TsArgMin)) for op in function.ops)
    prepare_causal_extrema(function)
    assert not any(isinstance(op, (TsArgMax, TsArgMin)) for op in function.ops)
    source = "\n".join(
        compileit(
            function,
            "full_profile",
            dtype="double",
            blocking_len=width,
            input_layout="TS",
            output_layout=layout,
            allow_unaligned=False,
            options={"no_skip_list": False, "opt_reduce": False, "no_fast_stat": True},
        )
    )
    assert "SkipListArgMin" not in source
    if label == "alpha158":
        assert "SkipListQuantile" in source
    assert len(metadata.get("factors", metadata.get("outputs"))) == (
        82 if label == "alpha101" else 158
    )
