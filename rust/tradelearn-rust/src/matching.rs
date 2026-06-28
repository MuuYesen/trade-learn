use crate::types::*;

pub fn match_order_at_price(
    order: &OrderEvent,
    price: f64,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> Option<FillEvent> {
    let fill_price = match order.order_type {
        OrderType::Market => Some(price),
        OrderType::Limit => {
            if order.side == OrderSide::Buy {
                if price <= order.limit_price? {
                    Some(price)
                } else {
                    None
                }
            } else {
                if price >= order.limit_price? {
                    Some(price)
                } else {
                    None
                }
            }
        }
        OrderType::Stop => {
            if order.side == OrderSide::Buy {
                if price >= order.stop_price? {
                    Some(price)
                } else {
                    None
                }
            } else {
                if price <= order.stop_price? {
                    Some(price)
                } else {
                    None
                }
            }
        }
        OrderType::StopLimit => {
            // Complex. For simplicity, match against the current point price.
            // If stop is hit, it becomes a limit order.
            let stop = order.stop_price?;
            let limit = order.limit_price?;
            if order.side == OrderSide::Buy {
                if price >= stop && price <= limit {
                    Some(price)
                } else {
                    None
                }
            } else {
                if price <= stop && price >= limit {
                    Some(price)
                } else {
                    None
                }
            }
        }
    }?;

    let size = round_precision(signed_order_size(order));
    Some(FillEvent {
        order_id: order.order_id,
        symbol: order.symbol.clone(),
        size,
        price: round_precision(fill_price),
        commission: round_precision(calculate_commission(
            fill_price,
            size,
            order.side,
            options.commission,
            options.mult,
        )),
        slippage: 0.0,
        ts: bar.ts,
    })
}

fn apply_slippage(price: f64, side: OrderSide, model: SlippageModel) -> (f64, f64) {
    let adjusted = match model {
        SlippageModel::Fixed(model) => match side {
            OrderSide::Buy => price + model.amount,
            OrderSide::Sell => price - model.amount,
        },
        SlippageModel::Percent(model) => match side {
            OrderSide::Buy => price * (1.0 + model.ratio),
            OrderSide::Sell => price * (1.0 - model.ratio),
        },
    };
    let rounded = round_precision(adjusted);
    (rounded, round_precision(rounded - price))
}

pub(crate) fn fill_from_raw_price(
    order: &OrderEvent,
    raw_price: f64,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> FillEvent {
    let size = round_precision(signed_order_size(order));
    let (fill_price, slippage) = apply_slippage(raw_price, order.side, options.slippage);
    FillEvent {
        order_id: order.order_id,
        symbol: order.symbol.clone(),
        size,
        price: fill_price,
        commission: round_precision(calculate_commission(
            fill_price,
            size,
            order.side,
            options.commission,
            options.mult,
        )),
        slippage,
        ts: bar.ts,
    }
}

pub fn match_order(
    order: &OrderEvent,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> Option<FillEvent> {
    let execution_price = if options.trade_on_close {
        bar.close
    } else {
        bar.open
    };

    let raw_price = match order.order_type {
        OrderType::Market => Some(execution_price),
        OrderType::Limit => {
            let limit = order.limit_price?;
            match order.side {
                OrderSide::Buy if bar.low <= limit => Some(execution_price.min(limit)),
                OrderSide::Sell if bar.high >= limit => Some(execution_price.max(limit)),
                _ => None,
            }
        }
        OrderType::Stop => {
            let stop = order.stop_price?;
            match order.side {
                OrderSide::Buy if bar.high >= stop => Some(execution_price),
                OrderSide::Sell if bar.low <= stop => Some(execution_price),
                _ => None,
            }
        }
        OrderType::StopLimit => {
            let stop = order.stop_price?;
            let limit = order.limit_price?;
            match order.side {
                OrderSide::Buy if bar.high >= stop && bar.low <= limit => {
                    Some(execution_price.min(limit))
                }
                OrderSide::Sell if bar.low <= stop && bar.high >= limit => {
                    Some(execution_price.max(limit))
                }
                _ => None,
            }
        }
    }?;

    Some(fill_from_raw_price(order, raw_price, bar, options))
}

pub fn match_order_smart(
    order: &OrderEvent,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> Option<FillEvent> {
    let raw_price = smart_match_price(order, bar, options)?.1;
    Some(fill_from_raw_price(order, raw_price, bar, options))
}

pub(crate) fn smart_match_price(
    order: &OrderEvent,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> Option<(f64, f64)> {
    if options.trade_on_close {
        if order.order_type == OrderType::Market {
            return Some((3.0, bar.close));
        }
    }

    let path = if bar.close >= bar.open {
        [bar.open, bar.low, bar.high, bar.close]
    } else {
        [bar.open, bar.high, bar.low, bar.close]
    };

    if order.order_type == OrderType::Market {
        return Some((0.0, path[0]));
    }

    let mut stop_limit_armed = false;
    for (segment_idx, pair) in path.windows(2).enumerate() {
        let start = pair[0];
        let end = pair[1];
        match order.order_type {
            OrderType::Limit => {
                let limit = order.limit_price?;
                if let Some(price) = limit_cross_price(order.side, limit, start, end) {
                    return Some((smart_segment_rank(segment_idx, start, end, price), price));
                }
            }
            OrderType::Stop => {
                let stop = order.stop_price?;
                if let Some(price) = stop_cross_price(order.side, stop, start, end) {
                    return Some((smart_segment_rank(segment_idx, start, end, price), price));
                }
            }
            OrderType::StopLimit => {
                let stop = order.stop_price?;
                let limit = order.limit_price?;
                if !stop_limit_armed && stop_cross_price(order.side, stop, start, end).is_some() {
                    stop_limit_armed = true;
                }
                if stop_limit_armed {
                    if let Some(price) = limit_cross_price(order.side, limit, start, end) {
                        return Some((smart_segment_rank(segment_idx, start, end, price), price));
                    }
                }
            }
            OrderType::Market => unreachable!(),
        }
    }
    None
}

fn limit_cross_price(side: OrderSide, limit: f64, start: f64, end: f64) -> Option<f64> {
    match side {
        OrderSide::Buy if start <= limit => Some(start),
        OrderSide::Buy if segment_contains(start, end, limit) => Some(limit),
        OrderSide::Sell if start >= limit => Some(start),
        OrderSide::Sell if segment_contains(start, end, limit) => Some(limit),
        _ => None,
    }
}

fn stop_cross_price(side: OrderSide, stop: f64, start: f64, end: f64) -> Option<f64> {
    match side {
        OrderSide::Buy if start >= stop => Some(start),
        OrderSide::Buy if segment_contains(start, end, stop) => Some(stop),
        OrderSide::Sell if start <= stop => Some(start),
        OrderSide::Sell if segment_contains(start, end, stop) => Some(stop),
        _ => None,
    }
}

fn segment_contains(start: f64, end: f64, price: f64) -> bool {
    let low = start.min(end);
    let high = start.max(end);
    low <= price && price <= high
}

fn smart_segment_rank(segment_idx: usize, start: f64, end: f64, price: f64) -> f64 {
    let span = (end - start).abs();
    let ratio = if span < f64::EPSILON {
        0.0
    } else {
        (price - start).abs() / span
    };
    segment_idx as f64 + ratio
}

fn calculate_commission(
    price: f64,
    size: f64,
    side: OrderSide,
    commission: CommissionModel,
    mult: f64,
) -> f64 {
    match commission {
        CommissionModel::Fixed(model) => model.amount,
        CommissionModel::Percent(model) => price * size.abs() * model.ratio * mult,
        CommissionModel::CNAStock(model) => {
            let notional = price * size.abs() * mult;
            let commission = (notional * model.commission_rate).max(model.min_commission);
            let transfer_fee = notional * model.transfer_fee_rate;
            let stamp_tax = if side == OrderSide::Sell {
                notional * model.stamp_tax_rate
            } else {
                0.0
            };
            commission + transfer_fee + stamp_tax
        }
    }
}

fn round_precision(value: f64) -> f64 {
    const SCALE: f64 = 1_000_000.0;
    (value * SCALE).round() / SCALE
}

pub(crate) fn signed_order_size(order: &OrderEvent) -> f64 {
    match order.side {
        OrderSide::Buy => order.size.abs(),
        OrderSide::Sell => -order.size.abs(),
    }
}

pub(crate) fn smart_order_priority(order: &OrderEvent) -> u8 {
    match order.order_type {
        OrderType::Stop => 0,
        OrderType::StopLimit => 1,
        OrderType::Market => 2,
        OrderType::Limit => 3,
    }
}

pub(crate) fn is_exit_fill(old_size: f64, fill_size: f64) -> bool {
    old_size.abs() > 1e-9 && old_size.signum() != fill_size.signum()
}

pub(crate) fn is_exit_order_for_position(old_size: f64, order: &OrderEvent) -> bool {
    old_size.abs() > 1e-9 && old_size.signum() != signed_order_size(order).signum()
}
