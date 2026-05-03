use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type OhlcvColumns = (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn resample_ohlcv(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    period_seconds: i64,
) -> PyResult<OhlcvColumns> {
    if period_seconds <= 0 {
        return Err(PyValueError::new_err("period_seconds must be positive"));
    }
    let len = timestamps.len();
    if opens.len() != len
        || highs.len() != len
        || lows.len() != len
        || closes.len() != len
        || volumes.len() != len
    {
        return Err(PyValueError::new_err(
            "timestamps and OHLCV arrays must have the same length",
        ));
    }
    if len == 0 {
        return Ok((
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ));
    }

    let mut out_ts = Vec::new();
    let mut out_open = Vec::new();
    let mut out_high = Vec::new();
    let mut out_low = Vec::new();
    let mut out_close = Vec::new();
    let mut out_volume = Vec::new();

    let mut current_label = right_closed_label(timestamps[0], period_seconds);
    let mut current_open = opens[0];
    let mut current_high = highs[0];
    let mut current_low = lows[0];
    let mut current_close = closes[0];
    let mut current_volume = volumes[0];

    for index in 1..len {
        let label = right_closed_label(timestamps[index], period_seconds);
        if label != current_label {
            out_ts.push(current_label);
            out_open.push(current_open);
            out_high.push(current_high);
            out_low.push(current_low);
            out_close.push(current_close);
            out_volume.push(current_volume);

            current_label = label;
            current_open = opens[index];
            current_high = highs[index];
            current_low = lows[index];
            current_close = closes[index];
            current_volume = volumes[index];
            continue;
        }
        current_high = current_high.max(highs[index]);
        current_low = current_low.min(lows[index]);
        current_close = closes[index];
        current_volume += volumes[index];
    }

    out_ts.push(current_label);
    out_open.push(current_open);
    out_high.push(current_high);
    out_low.push(current_low);
    out_close.push(current_close);
    out_volume.push(current_volume);

    Ok((out_ts, out_open, out_high, out_low, out_close, out_volume))
}

fn right_closed_label(timestamp: i64, period_seconds: i64) -> i64 {
    if timestamp >= 0 {
        ((timestamp + period_seconds - 1) / period_seconds) * period_seconds
    } else {
        (timestamp.div_euclid(period_seconds)) * period_seconds
    }
}

pub(crate) fn register_pyfunctions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resample_ohlcv, m)?)?;
    Ok(())
}
