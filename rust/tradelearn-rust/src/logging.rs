//! Python binding only. Persistence is owned by the shared Rust recorder.
use leafquant_external_log_core::recorder::{Recorder as DurableRecorder, RecorderError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[pyclass]
pub struct Recorder {
    inner: DurableRecorder,
}

fn python_error(error: RecorderError) -> PyErr {
    match error {
        RecorderError::Invalid(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[pymethods]
impl Recorder {
    #[new]
    #[pyo3(signature = (state_dir, capacity=4096, segment_bytes=8388608, max_disk_bytes=1073741824, archive_window_seconds=None))]
    fn new(state_dir: String, capacity: usize, segment_bytes: usize, max_disk_bytes: u64, archive_window_seconds: Option<u64>) -> PyResult<Self> {
        match archive_window_seconds { Some(seconds) => DurableRecorder::new_windowed(state_dir, capacity, segment_bytes, max_disk_bytes, std::time::Duration::from_secs(seconds)), None => DurableRecorder::new(state_dir, capacity, segment_bytes, max_disk_bytes) }
            .map(|inner| Self { inner }).map_err(python_error)
    }
    fn append(&self, record_json: String) -> bool { self.inner.append(record_json) }
    fn status(&self) -> String { self.inner.status() }
    fn flush(&self, py: Python<'_>) -> String { py.detach(|| self.inner.flush()) }
    fn close(&self, py: Python<'_>) -> String { py.detach(|| self.inner.close()) }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Recorder>() }
