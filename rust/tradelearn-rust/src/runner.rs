use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::types::*;
use crate::{parse_order_side, parse_order_type, RustBacktestEngine};

impl MultiDataFeed {
    pub fn new() -> Self {
        Self { feeds: Vec::new() }
    }

    pub fn add_feed(&mut self, name: impl Into<String>, bars: Vec<BarEvent>) -> usize {
        let feed_index = self.feeds.len();
        self.feeds.push(BarDataFeed {
            name: name.into(),
            bars,
            cursor: 0,
        });
        feed_index
    }

    pub fn next_bar(&mut self) -> Option<DataFeedBar> {
        let feed_index = self.next_feed_index()?;
        let feed = &mut self.feeds[feed_index];
        let bar = feed.bars[feed.cursor].clone();
        feed.cursor += 1;
        Some(DataFeedBar {
            feed_index,
            feed_name: feed.name.clone(),
            bar,
        })
    }

    pub fn len(&self) -> usize {
        self.feeds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.feeds.is_empty()
    }

    fn next_feed_index(&self) -> Option<usize> {
        self.feeds
            .iter()
            .enumerate()
            .filter_map(|(feed_index, feed)| {
                feed.bars
                    .get(feed.cursor)
                    .map(|bar| ((bar.ts, feed_index), feed_index))
            })
            .min_by_key(|(key, _)| *key)
            .map(|(_, feed_index)| feed_index)
    }
}

#[pyclass]
pub(crate) struct RustPrimaryClockPlan {
    cursors: Vec<Vec<isize>>,
}

#[pyclass]
pub(crate) struct RustBarRunner {
    cursors: Vec<Vec<isize>>,
}

#[pyclass]
pub(crate) struct RustClockedMultiDataRunner {
    cursors: Vec<Vec<isize>>,
    symbols: Vec<String>,
    timestamps: Vec<Vec<i64>>,
    opens: Vec<Vec<f64>>,
    highs: Vec<Vec<f64>>,
    lows: Vec<Vec<f64>>,
    closes: Vec<Vec<f64>>,
    volumes: Vec<Vec<f64>>,
}

pub(crate) fn register_pyclasses(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBarRunner>()?;
    m.add_class::<RustClockedMultiDataRunner>()?;
    m.add_class::<RustPrimaryClockPlan>()?;
    Ok(())
}

fn build_primary_clock_cursors(
    primary_timestamps: Vec<i64>,
    secondary_timestamps: Vec<Vec<i64>>,
) -> Vec<Vec<isize>> {
    let mut secondary_cursors = vec![0usize; secondary_timestamps.len()];
    let mut cursors = Vec::with_capacity(primary_timestamps.len());

    for (primary_cursor, primary_ts) in primary_timestamps.into_iter().enumerate() {
        let mut row = Vec::with_capacity(secondary_timestamps.len() + 1);
        row.push(primary_cursor as isize);
        for (feed_idx, timestamps) in secondary_timestamps.iter().enumerate() {
            while secondary_cursors[feed_idx] < timestamps.len()
                && timestamps[secondary_cursors[feed_idx]] <= primary_ts
            {
                secondary_cursors[feed_idx] += 1;
            }
            row.push(secondary_cursors[feed_idx] as isize - 1);
        }
        cursors.push(row);
    }

    cursors
}

fn extract_i64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<i64>>()
}

fn extract_i64_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i64>>> {
    let mut rows = Vec::new();
    for item in obj.try_iter()? {
        rows.push(extract_i64_vec(&item?)?);
    }
    Ok(rows)
}

fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<f64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<f64>>()
}

fn extract_f64_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let mut rows = Vec::new();
    for item in obj.try_iter()? {
        rows.push(extract_f64_vec(&item?)?);
    }
    Ok(rows)
}

#[pymethods]
impl RustPrimaryClockPlan {
    #[new]
    fn new(
        primary_timestamps: &Bound<'_, PyAny>,
        secondary_timestamps: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let primary = extract_i64_vec(primary_timestamps)?;
        let secondary = extract_i64_matrix(secondary_timestamps)?;
        Ok(Self {
            cursors: build_primary_clock_cursors(primary, secondary),
        })
    }

    fn len(&self) -> usize {
        self.cursors.len()
    }

    fn cursors_at(&self, primary_cursor: usize) -> Vec<isize> {
        self.cursors
            .get(primary_cursor)
            .cloned()
            .unwrap_or_default()
    }
}

#[pymethods]
impl RustBarRunner {
    #[new]
    fn new(
        primary_timestamps: &Bound<'_, PyAny>,
        secondary_timestamps: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let primary = extract_i64_vec(primary_timestamps)?;
        let secondary = extract_i64_matrix(secondary_timestamps)?;
        Ok(Self {
            cursors: build_primary_clock_cursors(primary, secondary),
        })
    }

    fn len(&self) -> usize {
        self.cursors.len()
    }

    fn cursors_at(&self, primary_cursor: usize) -> Vec<isize> {
        self.cursors
            .get(primary_cursor)
            .cloned()
            .unwrap_or_default()
    }

    fn run(
        &self,
        py: Python<'_>,
        mut engine: PyRefMut<'_, RustBacktestEngine>,
        broker: Py<PyAny>,
        on_bar: Py<PyAny>,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let stop = end.min(engine.inner.total_bars()).min(self.cursors.len());
        for cursor in start..stop {
            let fill_records = engine.inner.step_open(cursor);
            let fills = engine.map_fills_compact(fill_records);
            let cash = engine.inner.get_cash();
            let (size, price) = engine.inner.get_position();
            let data_cursors = self.cursors[cursor].clone();
            let drained = on_bar.call1(py, (cursor, data_cursors, fills, cash, size, price))?;
            if drained.is_none(py) {
                continue;
            }
            let orders: Vec<(u64, String, String, String, f64, Option<f64>, Option<f64>)> =
                drained.extract(py)?;
            let mut bindings: Vec<(u64, u64)> = Vec::with_capacity(orders.len());

            for (provisional_ref, symbol, side, order_type, order_size, limit_price, stop_price) in
                orders
            {
                let side = parse_order_side(&side)?;
                let order_type = parse_order_type(&order_type)?;
                let order_id = engine.inner.submit_order(
                    symbol,
                    side,
                    order_type,
                    order_size,
                    limit_price,
                    stop_price,
                );
                bindings.push((provisional_ref, order_id));
            }
            if !bindings.is_empty() {
                broker.call_method1(py, "bind_rust_order_refs", (bindings,))?;
            }
        }
        Ok(())
    }
}

#[pymethods]
impl RustClockedMultiDataRunner {
    #[new]
    fn new(
        symbols: Vec<String>,
        timestamps: &Bound<'_, PyAny>,
        opens: &Bound<'_, PyAny>,
        highs: &Bound<'_, PyAny>,
        lows: &Bound<'_, PyAny>,
        closes: &Bound<'_, PyAny>,
        volumes: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let timestamps = extract_i64_matrix(timestamps)?;
        let opens = extract_f64_matrix(opens)?;
        let highs = extract_f64_matrix(highs)?;
        let lows = extract_f64_matrix(lows)?;
        let closes = extract_f64_matrix(closes)?;
        let volumes = extract_f64_matrix(volumes)?;
        let feed_count = symbols.len();
        if feed_count == 0 {
            return Err(PyValueError::new_err("symbols must not be empty"));
        }
        if timestamps.len() != feed_count
            || opens.len() != feed_count
            || highs.len() != feed_count
            || lows.len() != feed_count
            || closes.len() != feed_count
            || volumes.len() != feed_count
        {
            return Err(PyValueError::new_err(
                "symbols and OHLCV matrices must have the same feed count",
            ));
        }
        for feed_idx in 0..feed_count {
            let len = timestamps[feed_idx].len();
            if opens[feed_idx].len() != len
                || highs[feed_idx].len() != len
                || lows[feed_idx].len() != len
                || closes[feed_idx].len() != len
                || volumes[feed_idx].len() != len
            {
                return Err(PyValueError::new_err(format!(
                    "OHLCV arrays must match timestamp length for feed {feed_idx}"
                )));
            }
        }
        let primary = timestamps[0].clone();
        let secondary = timestamps[1..].to_vec();
        Ok(Self {
            cursors: build_primary_clock_cursors(primary, secondary),
            symbols,
            timestamps,
            opens,
            highs,
            lows,
            closes,
            volumes,
        })
    }

    fn len(&self) -> usize {
        self.cursors.len()
    }

    fn cursors_at(&self, primary_cursor: usize) -> Vec<isize> {
        self.cursors
            .get(primary_cursor)
            .cloned()
            .unwrap_or_default()
    }

    fn active_count_at(&self, primary_cursor: usize) -> usize {
        self.cursors
            .get(primary_cursor)
            .map(|row| row.iter().filter(|cursor| **cursor >= 0).count())
            .unwrap_or(0)
    }

    fn run(
        &self,
        py: Python<'_>,
        engine: Py<RustBacktestEngine>,
        broker: Py<PyAny>,
        on_bar: Py<PyAny>,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let stop = end.min(self.cursors.len());
        for cursor in start..stop {
            let (fills, cash, size, price) = {
                let mut engine_ref = engine.borrow_mut(py);
                let bars = self.active_bars_at(cursor);
                let fill_records = engine_ref.inner.step_open_bars(bars);
                let fills = engine_ref.map_fills_compact(fill_records);
                let cash = engine_ref.inner.get_cash();
                let (size, price) = engine_ref.inner.get_position();
                (fills, cash, size, price)
            };
            let data_cursors = self.cursors[cursor].clone();
            let drained = on_bar.call1(py, (cursor, data_cursors, fills, cash, size, price))?;
            if drained.is_none(py) {
                continue;
            }
            let orders: Vec<(u64, String, String, String, f64, Option<f64>, Option<f64>)> =
                drained.extract(py)?;
            let mut bindings: Vec<(u64, u64)> = Vec::with_capacity(orders.len());
            let mut engine_ref = engine.borrow_mut(py);

            for (provisional_ref, symbol, side, order_type, order_size, limit_price, stop_price) in
                orders
            {
                let side = parse_order_side(&side)?;
                let order_type = parse_order_type(&order_type)?;
                let order_id = engine_ref.inner.submit_order(
                    symbol,
                    side,
                    order_type,
                    order_size,
                    limit_price,
                    stop_price,
                );
                bindings.push((provisional_ref, order_id));
            }
            if !bindings.is_empty() {
                broker.call_method1(py, "bind_rust_order_refs", (bindings,))?;
            }
        }
        Ok(())
    }
}

impl RustClockedMultiDataRunner {
    fn active_bars_at(&self, primary_cursor: usize) -> Vec<BarEvent> {
        let Some(cursors) = self.cursors.get(primary_cursor) else {
            return Vec::new();
        };
        let mut bars = Vec::with_capacity(cursors.len());
        for (feed_idx, cursor) in cursors.iter().enumerate() {
            if *cursor < 0 {
                continue;
            }
            let index = *cursor as usize;
            if index >= self.timestamps[feed_idx].len() {
                continue;
            }
            bars.push(BarEvent {
                ts: self.timestamps[feed_idx][index],
                symbol: self.symbols[feed_idx].clone(),
                open: self.opens[feed_idx][index],
                high: self.highs[feed_idx][index],
                low: self.lows[feed_idx][index],
                close: self.closes[feed_idx][index],
                volume: self.volumes[feed_idx][index],
            });
        }
        bars
    }
}
