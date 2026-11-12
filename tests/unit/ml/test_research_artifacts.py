import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from tradelearn.ml.registry import load_bundle, save_bundle
from tradelearn.research import Allocator, EqualWeight, FeatureSet, ModelScorer, Pipeline, TopK


def feature(panel):
    return panel.close


def test_complete_bundle_integrity_and_explicit_trust(tmp_path):
    model = Ridge().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), [2.0, 4.0, 6.0])
    bundle = {
        "feature_set": FeatureSet({"x": feature}),
        "pipeline": Pipeline([]),
        "model": model,
        "scorer": ModelScorer(model, features=["x"]),
        "allocator": Allocator(select=TopK(1), weight=EqualWeight()),
        "metadata": {"label": {"horizon": 1, "split": "2026-01-01", "purged_rows": 1}},
        "input_schema": {"close": "float64"},
    }
    root = tmp_path / "bundle"
    manifest = save_bundle(bundle, root)
    assert manifest["versions"]["scikit-learn"]
    with pytest.raises(ValueError, match="trusted"):
        load_bundle(root)
    restored = load_bundle(root, trusted=True)
    assert restored["scorer"].predict(pd.DataFrame({"x": [4.0]})).iloc[0] > 6
    with (root / "state.joblib").open("ab") as output:
        output.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        load_bundle(root, trusted=True)


def test_incomplete_bundle_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        save_bundle({"model": object()}, tmp_path)


def test_bundle_restores_explicit_feature_code_without_original_source(tmp_path, monkeypatch):
    import importlib
    import sys

    source = tmp_path / "external_factors.py"
    source.write_text("def close(panel):\n    return panel.close\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("external_factors")
    model = Ridge().fit(pd.DataFrame({"x": [1.0, 2.0]}), [2.0, 4.0])
    bundle = {
        "feature_set": FeatureSet({"x": module.close}),
        "pipeline": Pipeline([]),
        "model": model,
        "scorer": ModelScorer(model, features=["x"]),
        "allocator": Allocator(select=TopK(1), weight=EqualWeight()),
        "metadata": {"label": {"horizon": 1}},
        "input_schema": {"close": "float64"},
    }
    root = tmp_path / "delivery"
    manifest = save_bundle(bundle, root, code_paths={"external_factors.py": source})
    assert manifest["code"]["external_factors.py"]
    source.unlink()
    del sys.modules["external_factors"]
    restored = load_bundle(root, trusted=True)
    assert restored["feature_set"].features["x"].__module__ == "external_factors"
    (root / "code" / "external_factors.py").write_text("tampered")
    with pytest.raises(ValueError, match="code digest"):
        load_bundle(root, trusted=True)


def test_manifest_records_loaded_tradelearn_version_not_stale_distribution():
    import tradelearn
    from tradelearn.ml.artifacts import _versions

    assert _versions()["trade-learn"] == tradelearn.__version__


def test_bundle_rejects_already_loaded_conflicting_feature_module(tmp_path, monkeypatch):
    import importlib
    import sys

    source = tmp_path / "versioned_features.py"
    source.write_text("def close(panel):\n    return panel.close\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("versioned_features")
    model = Ridge().fit(pd.DataFrame({"x": [1.0, 2.0]}), [2.0, 4.0])
    bundle = {
        "feature_set": FeatureSet({"x": module.close}),
        "pipeline": Pipeline([]),
        "model": model,
        "scorer": ModelScorer(model, features=["x"]),
        "allocator": Allocator(select=TopK(1), weight=EqualWeight()),
        "metadata": {"label": {"horizon": 1}},
        "input_schema": {"close": "float64"},
    }
    root = tmp_path / "versioned-bundle"
    save_bundle(bundle, root, code_paths={"versioned_features.py": source})
    source.write_text("def close(panel):\n    return panel.close * 100\n")
    del sys.modules["versioned_features"]
    importlib.invalidate_caches()
    importlib.import_module("versioned_features")
    with pytest.raises(ValueError, match="loaded.*code.*conflict"):
        load_bundle(root, trusted=True)
