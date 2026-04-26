use pyo3::prelude::*;

#[pyfunction]
fn tradelearn_rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    Ok(())
}
