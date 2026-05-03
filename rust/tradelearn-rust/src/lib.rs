use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod engine;
pub mod matching;
pub mod runner;
pub mod types;

use crate::engine::*;
use crate::matching::*;
use crate::types::*;

#[pyfunction]
fn tradelearn_rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (
    order_id,
    symbol,
    side,
    order_type,
    size,
    limit_price,
    stop_price,
    created_ts,
    ts,
    open,
    high,
    low,
    close,
    volume,
    trade_on_close,
    commission_ratio
))]
#[allow(clippy::too_many_arguments)]
fn match_order_fill(
    order_id: u64,
    symbol: String,
    side: &str,
    order_type: &str,
    size: f64,
    limit_price: Option<f64>,
    stop_price: Option<f64>,
    created_ts: i64,
    ts: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    trade_on_close: bool,
    commission_ratio: f64,
) -> PyResult<Option<(f64, f64, f64, f64)>> {
    let side = match side {
        "buy" => OrderSide::Buy,
        "sell" => OrderSide::Sell,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported order side: {other}"
            )));
        }
    };
    let order_type = match order_type {
        "market" => OrderType::Market,
        "limit" => OrderType::Limit,
        "stop" => OrderType::Stop,
        "stop_limit" => OrderType::StopLimit,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported order type: {other}"
            )));
        }
    };
    let order = OrderEvent {
        order_id,
        symbol: symbol.clone(),
        side,
        order_type,
        size,
        limit_price,
        stop_price,
        created_ts,
    };
    let bar = BarEvent {
        ts,
        symbol,
        open,
        high,
        low,
        close,
        volume,
    };
    let options = ExecutionOptions {
        trade_on_close,
        smart_matching: false,
        cheat_on_close: false,
        cheat_on_open: false,
        slip_perc: 0.0,
        slip_fixed: 0.0,
        slip_match: true,
        slip_limit: true,
        slip_out: false,
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
        commission: CommissionModel::Percent(PercentCommission {
            ratio: commission_ratio,
        }),
        mult: 1.0,
        margin: 1.0,
    };
    Ok(match_order(&order, &bar, &options)
        .map(|fill| (fill.size, fill.price, fill.commission, fill.slippage)))
}

#[pyclass]
struct RustBacktestEngine {
    inner: BacktestEngine,
}

#[pyclass]
struct RustPrimaryClockPlan {
    cursors: Vec<Vec<isize>>,
}

#[pyclass]
struct RustBarRunner {
    cursors: Vec<Vec<isize>>,
}

#[pyclass]
struct RustClockedMultiDataRunner {
    cursors: Vec<Vec<isize>>,
    symbols: Vec<String>,
    timestamps: Vec<Vec<i64>>,
    opens: Vec<Vec<f64>>,
    highs: Vec<Vec<f64>>,
    lows: Vec<Vec<f64>>,
    closes: Vec<Vec<f64>>,
    volumes: Vec<Vec<f64>>,
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

#[pymethods]
impl RustBacktestEngine {
    #[new]
    #[pyo3(signature = (timestamps, opens, highs, lows, closes, volumes, cash, commission_ratio, trade_on_close, cheat_on_close, cheat_on_open, slip_perc, slip_fixed, slip_match, slip_limit, slip_out, mult=1.0, margin=1.0, smart_matching=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        cash: f64,
        commission_ratio: f64,
        trade_on_close: bool,
        cheat_on_close: bool,
        cheat_on_open: bool,
        slip_perc: f64,
        slip_fixed: f64,
        slip_match: bool,
        slip_limit: bool,
        slip_out: bool,
        mult: f64,
        margin: f64,
        smart_matching: bool,
    ) -> Self {
        Self {
            inner: BacktestEngine::new(
                timestamps,
                opens,
                highs,
                lows,
                closes,
                volumes,
                cash,
                commission_ratio,
                trade_on_close,
                cheat_on_close,
                cheat_on_open,
                slip_perc,
                slip_fixed,
                slip_match,
                slip_limit,
                slip_out,
                mult,
                margin,
                smart_matching,
            ),
        }
    }

    fn total_bars(&self) -> usize {
        self.inner.total_bars()
    }

    fn step(&mut self, cursor: usize) -> Vec<(u64, String, f64, f64, f64, f64, f64)> {
        let fills = self.inner.step(cursor);
        self.map_fills(fills)
    }

    fn step_close(&mut self, cursor: usize) -> Vec<(u64, String, f64, f64, f64, f64, f64)> {
        let fills = self.inner.step_close(cursor);
        self.map_fills(fills)
    }

    fn step_close_compact(&mut self, cursor: usize) -> Option<CompactFills> {
        let fills = self.inner.step_close(cursor);
        self.map_fills_compact(fills)
    }

    fn step_open(&mut self, cursor: usize) -> Vec<(u64, String, f64, f64, f64, f64, f64)> {
        let fills = self.inner.step_open(cursor);
        self.map_fills(fills)
    }

    fn step_open_collect(
        &mut self,
        cursor: usize,
        _fill_start_idx: usize,
    ) -> (Vec<(u64, String, f64, f64, f64, f64, f64)>, f64, f64, f64) {
        let fill_records = self.inner.step_open(cursor);
        let fills = self.map_fills(fill_records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        (fills, cash, size, price)
    }

    fn step_open_collect_compact(
        &mut self,
        cursor: usize,
        _fill_start_idx: usize,
    ) -> (Option<CompactFills>, f64, f64, f64) {
        let fill_records = self.inner.step_open(cursor);
        let fills = self.map_fills_compact(fill_records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        (fills, cash, size, price)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_open_bars_compact(
        &mut self,
        symbols: Vec<String>,
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> (Option<CompactFills>, f64, f64, f64) {
        let bars = build_bars(symbols, timestamps, opens, highs, lows, closes, volumes);
        let fill_records = self.inner.step_open_bars(bars);
        let fills = self.map_fills_compact(fill_records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        (fills, cash, size, price)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_close_bars_compact(
        &mut self,
        symbols: Vec<String>,
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
    ) -> (Option<CompactFills>, f64, f64, f64) {
        let bars = build_bars(symbols, timestamps, opens, highs, lows, closes, volumes);
        let fill_records = self.inner.step_close_bars(bars);
        let fills = self.map_fills_compact(fill_records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        (fills, cash, size, price)
    }

    fn run_bar_loop(
        &mut self,
        py: Python<'_>,
        broker: Py<PyAny>,
        on_bar: Py<PyAny>,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let stop = end.min(self.inner.total_bars());
        for cursor in start..stop {
            let fill_records = self.inner.step_open(cursor);
            let fills = self.map_fills_compact(fill_records);
            let cash = self.inner.get_cash();
            let (size, price) = self.inner.get_position();
            let drained = on_bar.call1(py, (cursor, fills, cash, size, price))?;
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
                let order_id = self.inner.submit_order(
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

    #[pyo3(signature = (symbol, side, order_type, size, limit_price=None, stop_price=None))]
    fn submit_order_for_symbol(
        &mut self,
        symbol: String,
        side: &str,
        order_type: &str,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
    ) -> PyResult<u64> {
        let side = parse_order_side(side)?;
        let order_type = parse_order_type(order_type)?;
        Ok(self
            .inner
            .submit_order(symbol, side, order_type, size, limit_price, stop_price))
    }

    #[pyo3(signature = (side, order_type, size, limit_price=None, stop_price=None))]
    fn submit_order(
        &mut self,
        side: &str,
        order_type: &str,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
    ) -> PyResult<u64> {
        self.submit_order_for_symbol(
            "data0".to_string(),
            side,
            order_type,
            size,
            limit_price,
            stop_price,
        )
    }

    fn get_position(&self) -> (f64, f64) {
        self.inner.get_position()
    }

    fn get_position_for_symbol(&self, symbol: &str) -> (f64, f64) {
        self.inner.get_position_for_symbol(symbol)
    }

    fn get_cash(&self) -> f64 {
        self.inner.get_cash()
    }

    fn get_equity(&self) -> f64 {
        self.inner.get_equity()
    }

    fn get_equity_curve(&self) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        let results = self.inner.get_results();
        let mut ts = Vec::with_capacity(results.equity.len());
        let mut cash = Vec::with_capacity(results.equity.len());
        let mut value = Vec::with_capacity(results.equity.len());
        for rec in &results.equity {
            ts.push(rec.ts);
            cash.push(rec.cash);
            value.push(rec.value);
        }
        (ts, cash, value)
    }

    fn get_fills(&self) -> Vec<(u64, String, f64, f64, f64, f64, f64, i64)> {
        self.inner
            .get_results()
            .fills
            .iter()
            .map(|f| {
                let side_str = match f.side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                };
                (
                    f.order_id,
                    side_str,
                    f.size,
                    f.price,
                    f.commission,
                    f.slippage,
                    f.pnl,
                    f.ts,
                )
            })
            .collect()
    }

    fn get_new_fills(&self, start_idx: usize) -> Vec<(u64, String, f64, f64, f64, f64, f64, i64)> {
        let fills = &self.inner.get_results().fills;
        if start_idx >= fills.len() {
            return Vec::new();
        }
        fills[start_idx..]
            .iter()
            .map(|f| {
                let side_str = match f.side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                };
                (
                    f.order_id,
                    side_str,
                    f.size,
                    f.price,
                    f.commission,
                    f.slippage,
                    f.pnl,
                    f.ts,
                )
            })
            .collect()
    }

    fn get_new_fills_compact(&self, start_idx: usize) -> Option<CompactFills> {
        let fills = &self.inner.get_results().fills;
        if start_idx >= fills.len() {
            return None;
        }
        Some(compact_fills_from_iter(fills[start_idx..].iter()))
    }

    fn get_fills_compact(&self) -> Option<CompactFills> {
        let fills = &self.inner.get_results().fills;
        if fills.is_empty() {
            return None;
        }
        Some(compact_fills_from_iter(fills.iter()))
    }
}

fn parse_order_side(side: &str) -> PyResult<OrderSide> {
    match side {
        "buy" => Ok(OrderSide::Buy),
        "sell" => Ok(OrderSide::Sell),
        other => Err(PyValueError::new_err(format!(
            "unsupported order side: {other}"
        ))),
    }
}

fn parse_order_type(order_type: &str) -> PyResult<OrderType> {
    match order_type {
        "market" => Ok(OrderType::Market),
        "limit" => Ok(OrderType::Limit),
        "stop" => Ok(OrderType::Stop),
        "stop_limit" => Ok(OrderType::StopLimit),
        other => Err(PyValueError::new_err(format!(
            "unsupported order type: {other}"
        ))),
    }
}

fn build_bars(
    symbols: Vec<String>,
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
) -> Vec<BarEvent> {
    let len = symbols
        .len()
        .min(timestamps.len())
        .min(opens.len())
        .min(highs.len())
        .min(lows.len())
        .min(closes.len())
        .min(volumes.len());
    let mut bars = Vec::with_capacity(len);
    for index in 0..len {
        bars.push(BarEvent {
            ts: timestamps[index],
            symbol: symbols[index].clone(),
            open: opens[index],
            high: highs[index],
            low: lows[index],
            close: closes[index],
            volume: volumes[index],
        });
    }
    bars
}

type CompactFills = (Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn compact_fills_from_iter<'a>(fills: impl Iterator<Item = &'a FillRecord>) -> CompactFills {
    let mut order_ids = Vec::new();
    let mut sizes = Vec::new();
    let mut prices = Vec::new();
    let mut commissions = Vec::new();
    let mut slippages = Vec::new();
    let mut pnls = Vec::new();

    for fill in fills {
        order_ids.push(fill.order_id);
        sizes.push(fill.size);
        prices.push(fill.price);
        commissions.push(fill.commission);
        slippages.push(fill.slippage);
        pnls.push(fill.pnl);
    }

    (order_ids, sizes, prices, commissions, slippages, pnls)
}

impl RustBacktestEngine {
    fn map_fills(&self, fills: Vec<FillRecord>) -> Vec<(u64, String, f64, f64, f64, f64, f64)> {
        fills
            .into_iter()
            .map(|f| {
                let side_str = match f.side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                };
                (
                    f.order_id,
                    side_str,
                    f.size,
                    f.price,
                    f.commission,
                    f.slippage,
                    f.pnl,
                )
            })
            .collect()
    }

    fn map_fills_compact(&self, fills: Vec<FillRecord>) -> Option<CompactFills> {
        if fills.is_empty() {
            return None;
        }
        Some(compact_fills_from_iter(fills.iter()))
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(match_order_fill, m)?)?;
    m.add_class::<RustBacktestEngine>()?;
    m.add_class::<RustBarRunner>()?;
    m.add_class::<RustClockedMultiDataRunner>()?;
    m.add_class::<RustPrimaryClockPlan>()?;
    Ok(())
}
