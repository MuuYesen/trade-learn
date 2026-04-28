use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod core;
use crate::core::*;

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
            let fills = self.map_fills(fill_records);
            let cash = self.inner.get_cash();
            let (size, price) = self.inner.get_position();
            let drained = on_bar.call1(py, (cursor, fills, cash, size, price))?;
            let orders: Vec<(u64, String, String, f64, Option<f64>, Option<f64>)> =
                drained.extract(py)?;
            let mut bindings: Vec<(u64, u64)> = Vec::with_capacity(orders.len());

            for (provisional_ref, side, order_type, order_size, limit_price, stop_price) in orders {
                let side = parse_order_side(&side)?;
                let order_type = parse_order_type(&order_type)?;
                let order_id =
                    self.inner
                        .submit_order(side, order_type, order_size, limit_price, stop_price);
                bindings.push((provisional_ref, order_id));
            }
            if !bindings.is_empty() {
                broker.call_method1(py, "bind_rust_order_refs", (bindings,))?;
            }
        }
        Ok(())
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
        Ok(self
            .inner
            .submit_order(side, order_type, size, limit_price, stop_price))
    }

    fn get_position(&self) -> (f64, f64) {
        self.inner.get_position()
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
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(match_order_fill, m)?)?;
    m.add_class::<RustBacktestEngine>()?;
    Ok(())
}
