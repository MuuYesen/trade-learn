"""Artifact and cross-section rejection tests; these do not qualify a native graph."""

import platform

import numpy as np
import pytest

from tradelearn.indicators._kunquant import Artifact, validate_cross_section


def artifact(**kwargs):
    data = dict(
        source_revision="abc123",
        expression_sha256="a" * 64,
        library_sha256="b" * 64,
        runtime_version="fixed",
        system=platform.system(),
        machine=platform.machine(),
        compiler="clang-fixed",
        isa=(),
        dtype="float64",
        module="f",
        inputs=("close",),
        outputs=("rank",),
        symbols=("A", "B"),
        warmup=5,
        simd_width=2,
    )
    return Artifact(**{**data, **kwargs})


def test_no_padding_or_ambiguous_identity_and_snapshot_fingerprint():
    with pytest.raises(ValueError, match="symbols"):
        artifact(symbols=("A", "A"))
    with pytest.raises(ValueError, match="SIMD"):
        artifact(symbols=("A", "B", "C"))
    with pytest.raises(ValueError, match="warmup"):
        artifact(warmup=0)
    assert artifact().fingerprint() != artifact(symbols=("B", "A")).fingerprint()
    assert artifact().fingerprint() != artifact(compiler="other").fingerprint()


def test_cross_section_requires_exact_order_time_schema_and_owned_arrays():
    spec = artifact()
    x = np.array([3.0, 5.0])
    result = validate_cross_section(spec, ("A", "B"), 10, {"close": x}, np.array([10, 10]))
    x[0] = 999
    assert result["close"][0] == 3
    for symbols, time, data, times in [
        (("B", "A"), 10, {"close": x}, [10, 10]),
        (("A", "B"), 10, {"close": x}, [9, 10]),
        (("A", "B"), 10, {"close": x}, [11, 10]),
        (("A", "B"), 10, {"close": x, "extra": x}, [10, 10]),
        (("A", "B"), 10, {"close": np.array([np.nan, 2])}, [10, 10]),
    ]:
        with pytest.raises(ValueError):
            validate_cross_section(spec, symbols, time, data, times)


def test_fixed_factor_selection_excludes_missing_alpha101_and_all_alpha191():
    from tradelearn.indicators._kunquant_graph import ALPHA101_NAMES, validate_names

    assert len(ALPHA101_NAMES) == 82
    assert validate_names(["alpha001", "alpha101"]) == ("alpha001", "alpha101")
    for names in [[], ["alpha001", "alpha001"], ["alpha048"], ["alpha191"], [lambda x: x]]:
        with pytest.raises((ValueError, TypeError)):
            validate_names(names)
