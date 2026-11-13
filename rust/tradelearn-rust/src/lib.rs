use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod engine;
mod logging;
pub mod matching;
pub mod resampler;
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
        "stop_trail" => OrderType::StopTrail,
        "stop_trail_limit" => OrderType::StopTrailLimit,
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
        trail_amount: None,
        trail_percent: None,
        trail_watermark: None,
        trail_triggered: false,
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
pub(crate) struct RustBacktestEngine {
    pub(crate) inner: BacktestEngine,
}

#[pymethods]
impl RustBacktestEngine {
    #[new]
    #[pyo3(signature = (timestamps, opens, highs, lows, closes, volumes, cash, commission_ratio, trade_on_close, cheat_on_close, cheat_on_open, slip_perc, slip_fixed, slip_match, slip_limit, slip_out, mult=1.0, margin=1.0, smart_matching=false, cn_a_stock_commission=false, cn_commission_rate=0.00025, cn_min_commission=5.0, cn_stamp_tax_rate=0.001, cn_transfer_fee_rate=0.00002))]
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
        cn_a_stock_commission: bool,
        cn_commission_rate: f64,
        cn_min_commission: f64,
        cn_stamp_tax_rate: f64,
        cn_transfer_fee_rate: f64,
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
                cn_a_stock_commission,
                cn_commission_rate,
                cn_min_commission,
                cn_stamp_tax_rate,
                cn_transfer_fee_rate,
            ),
        }
    }

    fn total_bars(&self) -> usize {
        self.inner.total_bars()
    }

    /// Cancel a pending order in the Rust matching queue (synced from Python broker.cancel()).
    #[pyo3(name = "cancel_order")]
    fn cancel_order(&mut self, order_ref: u64) -> bool {
        self.inner.remove_order(order_ref)
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
    fn step_bars_with_quotes_compact(
        &mut self,
        symbols: Vec<String>,
        timestamps: Vec<i64>,
        opens: Vec<f64>,
        highs: Vec<f64>,
        lows: Vec<f64>,
        closes: Vec<f64>,
        volumes: Vec<f64>,
        quote_opens: Vec<f64>,
        quote_highs: Vec<f64>,
        quote_lows: Vec<f64>,
        quote_closes: Vec<f64>,
        trade_on_close: bool,
    ) -> PyResult<(Option<CompactFills>, f64, f64, f64)> {
        let (bars, quotes) = build_quote_bars(symbols, timestamps, opens, highs, lows, closes, volumes,
            quote_opens, quote_highs, quote_lows, quote_closes)?;
        let records = self
            .inner
            .step_bars_with_quotes(bars, &quotes, trade_on_close);
        let fills = self.map_fills_compact(records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        Ok((fills, cash, size, price))
    }

    #[allow(clippy::too_many_arguments)]
    fn step_bars_with_quote_views_compact(
        &mut self, symbols: Vec<String>, timestamps: Vec<i64>, opens: Vec<f64>, highs: Vec<f64>,
        lows: Vec<f64>, closes: Vec<f64>, volumes: Vec<f64>, quote_opens: Vec<f64>,
        quote_highs: Vec<f64>, quote_lows: Vec<f64>, quote_closes: Vec<f64>,
        trade_on_close: bool, delivery_views: Vec<(String, u64, i64)>,
    ) -> PyResult<(Option<CompactFills>, f64, f64, f64)> {
        let (bars, quotes) = build_quote_bars(symbols, timestamps, opens, highs, lows, closes, volumes,
            quote_opens, quote_highs, quote_lows, quote_closes)?;
        let count = delivery_views.len();
        let views: std::collections::HashMap<String, (u64, i64)> = delivery_views.into_iter()
            .map(|(symbol, turn, clock)| (symbol, (turn, clock))).collect();
        if views.len() != count { return Err(PyValueError::new_err("duplicate delivery view symbol")); }
        let records = self.inner.step_bars_with_quote_views(bars, &quotes, &views, trade_on_close)
            .map_err(PyValueError::new_err)?;
        let fills = self.map_fills_compact(records);
        let cash = self.inner.get_cash();
        let (size, price) = self.inner.get_position();
        Ok((fills, cash, size, price))
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
        slf: Py<Self>,
        py: Python<'_>,
        broker: Py<PyAny>,
        on_bar: Py<PyAny>,
        start: usize,
        end: usize,
    ) -> PyResult<()> {
        let stop = end.min(slf.borrow(py).inner.total_bars());
        for cursor in start..stop {
            let (fills,cash,size,price) = {
                let mut engine=slf.borrow_mut(py);
                engine.sync_broker_controls(py,&broker)?;
                let records=engine.inner.step_open(cursor);
                engine.deliver_order_events(py,&broker)?;
                let fills=engine.map_fills_compact(records);
                let cash=engine.inner.get_cash();
                let (size,price)=engine.inner.get_position();
                (fills,cash,size,price)
            };
            // No mutable owner loan spans arbitrary Python business. Public
            // BT reads/explicit valuation still address this exact account.
            let drained=on_bar.call1(py,(cursor,fills,cash,size,price))?;
            slf.borrow_mut(py).consume_broker_callback(py,&broker,drained)?;
        }
        Ok(())
    }

    #[pyo3(signature = (symbol, side, order_type, size, limit_price=None, stop_price=None, trail_amount=None, trail_percent=None))]
    fn submit_order_for_symbol(
        &mut self,
        symbol: String,
        side: &str,
        order_type: &str,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
        trail_amount: Option<f64>,
        trail_percent: Option<f64>,
    ) -> PyResult<u64> {
        let side = parse_order_side(side)?;
        let order_type = parse_order_type(order_type)?;
        Ok(self.inner.submit_order(
            symbol,
            side,
            order_type,
            size,
            limit_price,
            stop_price,
            trail_amount,
            trail_percent,
        ))
    }

    #[pyo3(signature = (side, order_type, size, limit_price=None, stop_price=None, trail_amount=None, trail_percent=None))]
    fn submit_order(
        &mut self,
        side: &str,
        order_type: &str,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
        trail_amount: Option<f64>,
        trail_percent: Option<f64>,
    ) -> PyResult<u64> {
        self.submit_order_for_symbol(
            "data0".to_string(),
            side,
            order_type,
            size,
            limit_price,
            stop_price,
            trail_amount,
            trail_percent,
        )
    }

    fn configure_order_expiry_clock(&mut self, order_id: u64, target_clock: bool) {
        self.inner
            .configure_order_expiry_clock(order_id, target_clock);
    }

    fn configure_market_close_reference(&mut self, order_id: u64, price: f64) -> PyResult<()> {
        self.inner.configure_market_close_reference(order_id,price).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn configure_market_eligibility(&mut self, order_id: u64, after: i64) {
        self.inner.configure_market_eligibility(order_id, after);
    }

    #[pyo3(signature = (order_ref, deadline=None, oco_peer=None))]
    fn configure_order(&mut self, order_ref: u64, deadline: Option<i64>, oco_peer: Option<u64>) {
        self.inner.configure_order(order_ref, deadline, oco_peer);
    }

    fn drain_order_events(&mut self) -> Vec<(u64, String)> {
        self.inner.drain_order_events()
    }

    fn get_position(&self) -> (f64, f64) {
        self.inner.get_position()
    }

    fn get_position_for_symbol(&self, symbol: &str) -> (f64, f64) {
        self.inner.get_position_for_symbol(symbol)
    }

    fn configure_submission_check(&mut self, order_ref: u64, price: f64) -> PyResult<()> {
        self.inner.configure_submission_check(order_ref, price).map_err(PyValueError::new_err)
    }

    fn apply_cash_controls(&mut self, commands: Vec<(u64, String, f64)>) -> PyResult<()> {
        self.inner.apply_cash_controls(commands).map_err(PyValueError::new_err)
    }

    fn drain_cash_events(&mut self) -> PyResult<Vec<(u64, String, f64, f64)>> {
        self.inner.drain_cash_events().map_err(PyValueError::new_err)
    }

    fn get_cash(&self) -> f64 {
        self.inner.get_cash()
    }

    #[pyo3(signature=(symbols,include_cash,settle_cash))]
    fn observe_valuation(&mut self, symbols: Option<Vec<String>>, include_cash: bool, settle_cash: bool) -> PyResult<(f64,f64,f64)> {
        self.inner.observe_valuation(symbols.as_deref(),include_cash,settle_cash).map_err(PyValueError::new_err)
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

pub(crate) fn parse_order_side(side: &str) -> PyResult<OrderSide> {
    match side {
        "buy" => Ok(OrderSide::Buy),
        "sell" => Ok(OrderSide::Sell),
        other => Err(PyValueError::new_err(format!(
            "unsupported order side: {other}"
        ))),
    }
}

pub(crate) fn parse_order_type(order_type: &str) -> PyResult<OrderType> {
    match order_type {
        "market" => Ok(OrderType::Market),
        "limit" => Ok(OrderType::Limit),
        "stop" => Ok(OrderType::Stop),
        "stop_limit" => Ok(OrderType::StopLimit),
        "stop_trail" => Ok(OrderType::StopTrail),
        "stop_trail_limit" => Ok(OrderType::StopTrailLimit),
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

    pub(crate) fn map_fills_compact(&self, fills: Vec<FillRecord>) -> Option<CompactFills> {
        if fills.is_empty() {
            return None;
        }
        Some(compact_fills_from_iter(fills.iter()))
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::register(m)?;
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(match_order_fill, m)?)?;
    resampler::register_pyfunctions(m)?;
    m.add_class::<RustBacktestEngine>()?;
    runner::register_pyclasses(m)?;
    Ok(())
}

// Every runner consumes the same broker command boundary, including empty callbacks.
impl RustBacktestEngine {
    pub(crate) fn sync_broker_controls(
        &mut self,
        py: Python<'_>,
        broker: &Py<PyAny>,
    ) -> PyResult<()> {
        if broker.bind(py).hasattr("drain_cash_controls")? {
            let commands: Vec<(u64, String, f64)> = broker.call_method0(py, "drain_cash_controls")?.extract(py)?;
            self.inner.apply_cash_controls(commands).map_err(PyValueError::new_err)?;
            self.deliver_cash_events(py, broker)?;
        }
        if broker.bind(py).hasattr("drain_submission_checks")? {
            let checks: Vec<(u64, f64)> = broker.call_method0(py, "drain_submission_checks")?.extract(py)?;
            for (id, price) in checks {
                self.inner.configure_submission_check(id, price).map_err(PyValueError::new_err)?;
            }
        }
        if broker.bind(py).hasattr("drain_order_controls")? {
            let (cancels, updates): (Vec<u64>, Vec<(u64, Option<i64>, Option<u64>)>) = broker
                .call_method0(py, "drain_order_controls")?
                .extract(py)?;
            for (order_id, deadline, peer) in updates {
                self.inner.configure_order(order_id, deadline, peer);
            }
            for order_id in cancels {
                self.inner.remove_order(order_id);
            }
        }
        if broker.bind(py).hasattr("drain_target_expiry_clocks")? {
            let updates: Vec<u64> = broker
                .call_method0(py, "drain_target_expiry_clocks")?
                .extract(py)?;
            for order_id in updates {
                self.inner.configure_order_expiry_clock(order_id, true);
            }
        }
        if broker.bind(py).hasattr("drain_market_eligibility")? {
            let updates: Vec<(u64, i64)> = broker
                .call_method0(py, "drain_market_eligibility")?
                .extract(py)?;
            for (order_id, after) in updates {
                self.inner.configure_market_eligibility(order_id, after);
            }
        }
        Ok(())
    }

    pub(crate) fn deliver_order_events(
        &mut self,
        py: Python<'_>,
        broker: &Py<PyAny>,
    ) -> PyResult<()> {
        self.deliver_cash_events(py, broker)?;
        let events = self.inner.drain_order_events();
        if !events.is_empty() && broker.bind(py).hasattr("receive_order_events")? {
            broker.call_method1(py, "receive_order_events", (events,))?;
        }
        Ok(())
    }

    fn deliver_cash_events(&mut self, py: Python<'_>, broker: &Py<PyAny>) -> PyResult<()> {
        let events = self.inner.drain_cash_events().map_err(PyValueError::new_err)?;
        if !events.is_empty() {
            broker.call_method1(py, "receive_cash_events", (events,))?;
        }
        Ok(())
    }

    pub(crate) fn consume_broker_callback(
        &mut self,
        py: Python<'_>,
        broker: &Py<PyAny>,
        drained: Py<PyAny>,
    ) -> PyResult<()> {
        if !drained.is_none(py) {
            let orders: Vec<(u64, String, String, String, f64, Option<f64>, Option<f64>)> =
                drained.extract(py)?;
            let mut bindings = Vec::with_capacity(orders.len());
            let has_trails = broker.bind(py).hasattr("pop_trail_params")?;
            for (provisional, symbol, side, kind, size, limit, stop) in orders {
                let (amount, percent): (Option<f64>, Option<f64>) = if has_trails {
                    broker
                        .call_method1(py, "pop_trail_params", (provisional,))?
                        .extract(py)?
                } else {
                    (None, None)
                };
                let id = self.inner.submit_order(
                    symbol,
                    parse_order_side(&side)?,
                    parse_order_type(&kind)?,
                    size,
                    limit,
                    stop,
                    amount,
                    percent,
                );
                bindings.push((provisional, id));
            }
            if !bindings.is_empty() {
                broker.call_method1(py, "bind_rust_order_refs", (bindings,))?;
            }
        }
        self.sync_broker_controls(py, broker)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_quote_bars(
    symbols: Vec<String>, timestamps: Vec<i64>, opens: Vec<f64>, highs: Vec<f64>,
    lows: Vec<f64>, closes: Vec<f64>, volumes: Vec<f64>, quote_opens: Vec<f64>,
    quote_highs: Vec<f64>, quote_lows: Vec<f64>, quote_closes: Vec<f64>,
) -> PyResult<(Vec<BarEvent>, Vec<BarEvent>)> {
        let count = symbols.len();
        if [
            timestamps.len(),
            opens.len(),
            highs.len(),
            lows.len(),
            closes.len(),
            volumes.len(),
            quote_opens.len(),
            quote_highs.len(),
            quote_lows.len(),
            quote_closes.len(),
        ]
        .iter()
        .any(|length| *length != count)
        {
            return Err(PyValueError::new_err(
                "execution quote and valuation bar lengths differ",
            ));
        }
        if quote_opens
            .iter()
            .chain(&quote_highs)
            .chain(&quote_lows)
            .chain(&quote_closes)
            .any(|price| !price.is_finite())
        {
            return Err(PyValueError::new_err("execution quotes must be finite"));
        }
        let quotes = build_bars(
            symbols.clone(),
            timestamps.clone(),
            quote_opens,
            quote_highs,
            quote_lows,
            quote_closes,
            volumes.clone(),
        );
        let bars = build_bars(symbols, timestamps, opens, highs, lows, closes, volumes);
    Ok((bars, quotes))
}
