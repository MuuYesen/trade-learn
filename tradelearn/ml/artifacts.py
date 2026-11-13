"""Explicit local delivery of fitted research state; never publishes models."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

_REQUIRED = {"feature_set", "pipeline", "model", "scorer", "allocator", "metadata", "input_schema"}


def _versions():
    import tradelearn

    # Source checkouts may contain historical egg-info before the active wheel
    # metadata on sys.path. Record the module actually supplying runtime code.
    result = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "trade-learn": tradelearn.__version__,
    }
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        result[package] = importlib.metadata.version(package)
    return result


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        raise ValueError("code paths must be relative Python module paths")
    return relative


def save_bundle(
    bundle: dict[str, Any],
    path: str | Path,
    *,
    code_paths: dict[str, str | Path] | None = None,
    dependencies: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Write fitted components plus schema/label/version manifest locally.

    The directory must be empty. Definitions must be serializable and their
    Python modules available at load time; no code/data/secret discovery occurs.
    """
    missing = _REQUIRED - set(bundle)
    if missing:
        raise ValueError(f"research bundle missing components: {sorted(missing)}")
    if not bundle["input_schema"] or not bundle["metadata"].get("label"):
        raise ValueError("research bundle missing input schema or label/purge metadata")
    root = Path(path)
    if root.exists() and any(root.iterdir()):
        raise ValueError("research bundle destination must be empty")
    root.mkdir(parents=True, exist_ok=True)
    definitions = {
        name: {"type": type(bundle[name]).__name__, "module": type(bundle[name]).__module__}
        for name in sorted(_REQUIRED - {"metadata", "input_schema"})
    }
    manifest = {
        "format": "tradelearn.research.bundle.v1",
        "versions": {
            **_versions(),
            **{name: importlib.metadata.version(name) for name in dependencies},
        },
        "definitions": definitions,
        "input_schema": bundle["input_schema"],
        "metadata": bundle["metadata"],
        "state": "state.joblib",
    }
    code = {
        str(_code_relative(name)): Path(source).read_bytes()
        for name, source in (code_paths or {}).items()
    }
    manifest["code"] = {name: hashlib.sha256(content).hexdigest() for name, content in code.items()}
    # Reject non-JSON metadata before serializing executable state.
    json.dumps(manifest, allow_nan=False)
    import joblib

    temporary = root / f".state-{uuid.uuid4().hex}.joblib"
    try:
        for name, content in code.items():
            destination = root / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        joblib.dump(bundle, temporary)
        manifest["sha256"] = _digest(temporary)
        os.replace(temporary, root / "state.joblib")
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def load_bundle(path: str | Path, *, trusted: bool = False) -> dict[str, Any]:
    """Check integrity/versions, then explicitly load trusted executable state.

    Hash integrity is not authenticity. Only set trusted=True for a producer
    and code you trust. This does not register, publish, deploy or fit anything.
    """
    if not trusted:
        raise ValueError("loading joblib requires explicit trusted=True")
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "tradelearn.research.bundle.v1"
        or manifest.get("state") != "state.joblib"
    ):
        raise ValueError("unsupported research bundle manifest")
    if _digest(root / "state.joblib") != manifest.get("sha256"):
        raise ValueError("research bundle digest mismatch")
    current = _versions()
    for package, version in manifest["versions"].items():
        installed = current.get(package)
        if installed is None:
            installed = importlib.metadata.version(package)
        if installed != version:
            raise ValueError(f"research bundle version mismatch: {package}")
    code_root = (root / "code").resolve()
    for name, digest in manifest.get("code", {}).items():
        source = (code_root / _code_relative(name)).resolve()
        if not source.is_relative_to(code_root) or _digest(source) != digest:
            raise ValueError("research bundle code digest mismatch")
        relative = _code_relative(name)
        if relative.name != "__init__.py":
            module_name = ".".join(relative.with_suffix("").parts)
            loaded = sys.modules.get(module_name)
            if loaded is not None:
                origin = getattr(loaded, "__file__", None)
                if origin is None or not Path(origin).is_file() or _digest(Path(origin)) != digest:
                    raise ValueError(
                        f"loaded module code conflict: {module_name}; use a fresh process"
                    )
    if manifest.get("code"):
        # Explicit trusted code stays importable for later lazy inference imports.
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        importlib.invalidate_caches()
    import joblib

    bundle = joblib.load(root / "state.joblib")
    if not isinstance(bundle, dict) or _REQUIRED - set(bundle):
        raise ValueError("research bundle missing components")
    if (
        bundle["input_schema"] != manifest["input_schema"]
        or bundle["metadata"] != manifest["metadata"]
    ):
        raise ValueError("research bundle metadata mismatch")
    return bundle


__all__ = ["save_bundle", "load_bundle", "predict_bundle"]


def predict_bundle(bundle: dict[str, Any], bars):
    """Apply the same fitted local bundle to a research or strategy window.

    ``load_bundle(..., trusted=True)`` remains the explicit executable-state
    trust boundary. This function never loads, fits, fetches or caches scores.
    An empty valid window returns no scores, never a previous prediction.
    """
    import pandas as pd
    from tradelearn.core.contracts import validate_bars

    missing = _REQUIRED - set(bundle)
    if missing:
        raise ValueError(f'research bundle missing components: {sorted(missing)}')
    schema = bundle['input_schema']
    if not isinstance(schema, dict) or not schema:
        raise ValueError('research bundle input schema must be a nonempty mapping')
    frame = pd.DataFrame(bars)
    for name, dtype in schema.items():
        if name not in frame.columns:
            raise ValueError(f'bundle input schema missing field: {name}')
        if str(frame[name].dtype) != dtype:
            raise ValueError(f'bundle input dtype differs for field: {name}')
    validate_bars(frame)
    scorer = bundle['scorer']
    if getattr(scorer, 'model', None) is not bundle['model']:
        raise ValueError('bundle scorer must reference the delivered model')
    if frame.empty:
        return pd.Series(dtype=float, name='score')
    features = bundle['feature_set'].transform(frame)
    transformed = bundle['pipeline'].transform(features)
    if not transformed.empty:
        if not isinstance(transformed.index, pd.MultiIndex) or not transformed.index.isin(frame.index).all():
            raise ValueError('bundle transformed features must belong to the supplied window')
        if getattr(scorer, 'current', False) and transformed.index.get_level_values(0).max() != frame.index.get_level_values(0).max():
            raise ValueError('bundle transformed features lack the current observation')
    scores = scorer.predict(transformed)
    if not isinstance(scores, pd.Series):
        raise TypeError('bundle scorer must produce a Series')
    if isinstance(scores.index, pd.MultiIndex):
        valid_index = scores.index.isin(frame.index).all()
    else:
        latest = frame.index.get_level_values(0).max()
        symbols = frame.xs(latest, level=0).index
        valid_index = bool(getattr(scorer, 'current', False)) and scores.index.isin(symbols).all()
    if not scores.index.is_unique or not valid_index:
        raise ValueError('bundle scores must belong to the supplied window')
    import numpy as np
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError('bundle scorer produced nonfinite scores')
    return scores
