from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[3]


def load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_maturin_build_backend_points_to_tradelearn_rust_extension() -> None:
    pyproject = load_toml(ROOT / "pyproject.toml")

    assert pyproject["build-system"] == {
        "requires": ["maturin>=1.7,<2"],
        "build-backend": "maturin",
    }
    maturin = pyproject["tool"]["maturin"]
    assert maturin["manifest-path"] == "rust/tradelearn-rust/Cargo.toml"
    assert maturin["module-name"] == "tradelearn._rust"
    assert maturin["python-source"] == "."
    assert maturin["features"] == ["extension-module"]


def test_cargo_workspace_declares_tradelearn_rust_member() -> None:
    cargo = load_toml(ROOT / "Cargo.toml")

    assert cargo["workspace"]["members"] == ["rust/tradelearn-rust"]
    assert cargo["workspace"]["resolver"] == "2"


def test_tradelearn_rust_crate_exposes_pyo3_module_entrypoint() -> None:
    crate = load_toml(ROOT / "rust" / "tradelearn-rust" / "Cargo.toml")
    lib_rs = (ROOT / "rust" / "tradelearn-rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )

    assert crate["package"]["name"] == "tradelearn-rust"
    assert crate["lib"]["name"] == "_rust"
    assert crate["lib"]["crate-type"] == ["cdylib", "rlib"]
    assert "pyo3" in crate["dependencies"]
    assert "#[pymodule]" in lib_rs
    assert "fn _rust" in lib_rs
    assert "tradelearn_rust_version" in lib_rs
