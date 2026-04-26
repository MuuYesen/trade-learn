#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

pub mod core;

#[cfg(feature = "extension-module")]
use crate::core::{
    match_order, BarEvent, CommissionModel, ExecutionOptions, FixedSlippage, OrderEvent, OrderSide,
    OrderType, PercentCommission, PercentSlippage, SlippageModel,
};

#[cfg(feature = "extension-module")]
#[pyfunction]
fn tradelearn_rust_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(feature = "extension-module")]
#[pyfunction]
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

#[cfg(feature = "extension-module")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tradelearn_rust_version, m)?)?;
    m.add_function(wrap_pyfunction!(match_order_fill, m)?)?;
    Ok(())
}
