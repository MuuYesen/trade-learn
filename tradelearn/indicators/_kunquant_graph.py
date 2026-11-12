"""Internal fixed-upstream Alpha101 expression preparation, without compilation.

No graph produced here is automatically numerically qualified or published.
Arbitrary Python callables and Alpha191 are deliberately not accepted.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

SOURCE_REVISION = "921e5a26ac6e12ffeea812f8a38adaec4458c6ad"
SOURCE_SHA256 = "14667961f60963f312165a58e045cb9d562672c651b00af3e9c2b39bb3c72c91"
ALPHA101_NAMES = (
    "alpha001",
    "alpha002",
    "alpha003",
    "alpha004",
    "alpha005",
    "alpha006",
    "alpha007",
    "alpha008",
    "alpha009",
    "alpha010",
    "alpha011",
    "alpha012",
    "alpha013",
    "alpha014",
    "alpha015",
    "alpha016",
    "alpha017",
    "alpha018",
    "alpha019",
    "alpha020",
    "alpha021",
    "alpha022",
    "alpha023",
    "alpha024",
    "alpha025",
    "alpha026",
    "alpha027",
    "alpha028",
    "alpha029",
    "alpha030",
    "alpha031",
    "alpha032",
    "alpha033",
    "alpha034",
    "alpha035",
    "alpha036",
    "alpha037",
    "alpha038",
    "alpha039",
    "alpha040",
    "alpha041",
    "alpha042",
    "alpha043",
    "alpha044",
    "alpha045",
    "alpha046",
    "alpha047",
    "alpha049",
    "alpha050",
    "alpha051",
    "alpha052",
    "alpha053",
    "alpha054",
    "alpha055",
    "alpha057",
    "alpha060",
    "alpha061",
    "alpha062",
    "alpha064",
    "alpha065",
    "alpha066",
    "alpha068",
    "alpha071",
    "alpha072",
    "alpha073",
    "alpha074",
    "alpha075",
    "alpha077",
    "alpha078",
    "alpha081",
    "alpha083",
    "alpha084",
    "alpha085",
    "alpha086",
    "alpha088",
    "alpha092",
    "alpha094",
    "alpha095",
    "alpha096",
    "alpha098",
    "alpha099",
    "alpha101",
)


def _fixed_source_file(path, canonical_sha256, windows_sha256):
    """Accept exactly the two verified official 0.1.12 source byte variants.

    The Windows wheel uses CRLF. Its full raw SHA AND LF-normalized SHA must
    match; other byte or formula edits remain rejected. Native Windows build
    capability is independent of this source-identity check.
    """
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual == canonical_sha256:
        return actual
    if (
        actual == windows_sha256
        and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == canonical_sha256
    ):
        return actual
    raise ValueError("KunQuant source differs from fixed qualification scope")


def validate_names(names):
    names = tuple(names)
    if any(not isinstance(name, str) for name in names):
        raise ValueError("factor names must be explicit strings")
    if not names or len(names) != len(set(names)):
        raise ValueError("factor names must be explicit, nonempty and unique")
    unsupported = set(names) - set(ALPHA101_NAMES)
    if unsupported:
        raise ValueError(f"fixed KunQuant source has no factors: {sorted(unsupported)}")
    return names


def build_alpha101(names):
    """Return a real KunQuant Function and provenance for explicit named factors.

    The caller compiles TS/STREAM variants during deployment preparation, then
    performs prefix, universe/tie, missing-data and restart qualification. This
    function neither compiles nor marks an artifact ready.
    """
    names = validate_names(names)
    from importlib.metadata import version

    if version("KunQuant") != "0.1.12":
        raise ImportError("graph qualification requires KunQuant==0.1.12")
    from KunQuant.Op import Builder, Input, Output
    from KunQuant.predefined import Alpha101
    from KunQuant.Stage import Function

    source = Path(inspect.getfile(Alpha101))
    source_file_sha256 = _fixed_source_file(
        source,
        SOURCE_SHA256,
        "6b974d42b4289ea6a8af8c7b26f752ee777a1caa697e190cd199fdda4537c5f3",
    )
    builder = Builder()
    schema = ("open", "close", "high", "low", "volume", "amount")
    with builder:
        inputs = {name: Input(name) for name in schema}
        all_data = Alpha101.AllData(**inputs)
        for name in names:
            value = getattr(Alpha101, name)(all_data)
            Output(value, name)
    function = Function(builder.ops)
    schema = tuple(op.attrs["name"] for op in function.ops if isinstance(op, Input))
    expression = dict(
        source_revision=SOURCE_REVISION,
        source_sha256=SOURCE_SHA256,
        factors=names,
        inputs=schema,
        profile="KunQuant-0.1.12-upstream-Alpha101",
    )
    metadata = {
        **expression,
        "source_file_sha256": source_file_sha256,
        "expression_sha256": hashlib.sha256(
            json.dumps(expression, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "qualified": False,
        "compiled": False,
        "alpha191_supported": False,
    }
    return function, metadata


ALPHA158_SOURCE_SHA256 = "bae7d8b1bea46c4b8426667227ff5a69dc6fb64d6b164a5f0c8a0f8301a535d9"
DEFAULT_ALPHA158_CONFIG = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {"windows": [5, 10, 20, 30, 60]},
}


def build_alpha158(config=None):
    """Prepare the provided KunQuant Alpha158 feature graph; no TL formulas."""
    from copy import deepcopy
    from importlib.metadata import version

    if version("KunQuant") != "0.1.12":
        raise ImportError("graph qualification requires KunQuant==0.1.12")
    from KunQuant.Op import Builder, Input, Output
    from KunQuant.predefined import Alpha158
    from KunQuant.Stage import Function

    source_file_sha256 = _fixed_source_file(
        Path(inspect.getfile(Alpha158)),
        ALPHA158_SOURCE_SHA256,
        "5afe30b56fbbe4fe5431a232153ed1c77dc25571b8786d4bae35be07a27a70b4",
    )
    configuration = deepcopy(DEFAULT_ALPHA158_CONFIG if config is None else config)
    # Metadata must remain ordinary serializable source configuration.
    json.dumps(configuration, sort_keys=True)
    if set(configuration) - {"kbar", "price", "volume", "rolling"}:
        raise ValueError("unknown Alpha158 configuration")
    for section in ("price", "volume", "rolling"):
        if section not in configuration:
            continue
        windows = configuration[section].get(
            "windows", [5, 10, 20, 30, 60] if section == "rolling" else list(range(5))
        )
        if not windows or any(
            type(window) is not int or window < (1 if section == "rolling" else 0)
            for window in windows
        ):
            raise ValueError("Alpha158 windows require causal integer periods")
    builder = Builder()
    with builder:
        inputs = {
            name: Input(name) for name in ("open", "close", "high", "low", "volume", "amount")
        }
        data = Alpha158.AllData(**inputs)
        native_config = deepcopy(configuration)
        if "price" in native_config and "feature" in native_config["price"]:
            fields = {
                "OPEN": data.open,
                "HIGH": data.high,
                "LOW": data.low,
                "CLOSE": data.close,
                "VWAP": data.vwap,
            }
            native_config["price"]["feature"] = [
                (name, fields[name]) for name in native_config["price"]["feature"]
            ]
        features, names = data.build(native_config)
        if not names or len(names) != len(set(names)):
            raise ValueError("empty/duplicate Alpha158 output names")
        for name, value in zip(names, features, strict=True):
            Output(value, name)
    function = Function(builder.ops)
    schema = tuple(op.attrs["name"] for op in function.ops if isinstance(op, Input))
    expression = dict(
        profile="KunQuant-0.1.12-upstream-Alpha158",
        config=configuration,
        inputs=schema,
        outputs=names,
        source_sha256=ALPHA158_SOURCE_SHA256,
    )
    return function, {
        **expression,
        "source_file_sha256": source_file_sha256,
        "expression_sha256": hashlib.sha256(
            json.dumps(expression, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_revision": "KunQuant==0.1.12-Alpha158",
        "qualified": False,
        "compiled": False,
    }
