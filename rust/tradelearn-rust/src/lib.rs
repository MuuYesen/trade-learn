#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

pub mod core;

#[cfg(feature = "extension-module")]
use crate::core::{
    match_order, BacktestEngine, BarEvent, CommissionModel, ExecutionOptions, FixedSlippage,
    OrderEvent, OrderSide, OrderType, PercentCommission, SlippageModel,
};

#[cfg(feature = "extension-module")]
#[pyfunction]
fn tradelearn_rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "extension-module")]
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
        commission: CommissionModel::Percent(PercentCommission {
            ratio: commission_ratio,
        }),
    };
    Ok(match_order(&order, &bar, &options)
        .map(|fill| (fill.size, fill.price, fill.commission, fill.slippage)))
}

// ---------------------------------------------------------------------------
// RustBacktestEngine – PyO3 wrapper around core::BacktestEngine
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
#[pyclass]
struct RustBacktestEngine {
    inner: BacktestEngine,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl RustBacktestEngine {
    #[new]
    #[pyo3(signature = (timestamps, opens, highs, lows, closes, volumes, cash, commission_ratio, trade_on_close))]
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
            ),
        }
    }

    fn total_bars(&self) -> usize {
        self.inner.total_bars()
    }

    /// Process pending orders at given cursor. Returns list of fill tuples:
    /// (order_id, side_str, size, price, commission, slippage, pnl)
    fn step(&mut self, cursor: usize) -> Vec<(u64, String, f64, f64, f64, f64, f64)> {
        self.inner
            .step(cursor)
            .into_iter()
            .map(|f| {
                let side_str = match f.side {
                    OrderSide::Buy => "buy".to_string(),
                    OrderSide::Sell => "sell".to_string(),
                };
                (f.order_id, side_str, f.size, f.price, f.commission, f.slippage, f.pnl)
            })
            .collect()
    }

    /// Submit an order. Returns order_id.
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
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
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
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported order type: {other}"
                )));
            }
        };
        Ok(self.inner.submit_order(side, order_type, size, limit_price, stop_price))
    }

    /// Returns (size, avg_price) for the default data.
    fn get_position(&self) -> (f64, f64) {
        self.inner.get_position()
    }

    fn get_cash(&self) -> f64 {
        self.inner.get_cash()
    }

    fn get_equity(&self) -> f64 {
        self.inner.get_equity()
    }

    /// Returns (equity_ts, equity_cash, equity_value) as three lists.
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

    /// Returns fills as list of tuples: (order_id, side, size, price, commission, slippage, pnl, ts)
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
                (f.order_id, side_str, f.size, f.price, f.commission, f.slippage, f.pnl, f.ts)
            })
            .collect()
    }
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(match_order_fill, m)?)?;
    m.add_class::<RustBacktestEngine>()?;
    Ok(())
}
