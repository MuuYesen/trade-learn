use std::collections::BTreeMap;

pub type Timestamp = i64;
pub type OrderId = u64;

#[derive(Clone, Debug, PartialEq)]
pub struct BarEvent {
    pub ts: Timestamp,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OrderEvent {
    pub order_id: OrderId,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub size: f64,
    pub limit_price: Option<f64>,
    pub stop_price: Option<f64>,
    pub created_ts: Timestamp,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FillEvent {
    pub order_id: OrderId,
    pub symbol: String,
    pub size: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub ts: Timestamp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedSlippage {
    pub amount: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PercentSlippage {
    pub ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlippageModel {
    Fixed(FixedSlippage),
    Percent(PercentSlippage),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedCommission {
    pub amount: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PercentCommission {
    pub ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissionModel {
    Fixed(FixedCommission),
    Percent(PercentCommission),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutionOptions {
    pub trade_on_close: bool,
    pub slippage: SlippageModel,
    pub commission: CommissionModel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CancelEvent {
    pub order_id: OrderId,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectEvent {
    pub order_id: OrderId,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimerEvent {
    pub ts: Timestamp,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TickEvent {
    pub ts: Timestamp,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Bar(BarEvent),
    Order(OrderEvent),
    Fill(FillEvent),
    Cancel(CancelEvent),
    Reject(RejectEvent),
    Timer(TimerEvent),
    Tick(TickEvent),
}

pub trait Broker {
    fn submit_order(&mut self, order: OrderEvent) -> Event;
}

pub trait DataFeed {
    fn next_bar(&mut self) -> Option<BarEvent>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueuedEvent {
    pub timestamp: Timestamp,
    pub sequence: u64,
    pub event: Event,
}

#[derive(Default, Debug)]
pub struct EventQueue {
    next_sequence: u64,
    events: BTreeMap<(Timestamp, u64), Event>,
}

impl EventQueue {
    pub fn push(&mut self, timestamp: Timestamp, event: Event) -> u64 {
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        self.events.insert((timestamp, sequence), event);
        sequence
    }

    pub fn pop_next(&mut self) -> Option<QueuedEvent> {
        let key = *self.events.keys().next()?;
        let event = self.events.remove(&key)?;
        Some(QueuedEvent {
            timestamp: key.0,
            sequence: key.1,
            event,
        })
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

pub fn match_order(
    order: &OrderEvent,
    bar: &BarEvent,
    options: &ExecutionOptions,
) -> Option<FillEvent> {
    let raw_price = match order.order_type {
        OrderType::Market => Some(execution_price(bar, options)),
        OrderType::Limit => limit_fill_price(order, bar),
        OrderType::Stop => {
            if stop_triggered(order, bar) {
                Some(execution_price(bar, options))
            } else {
                None
            }
        }
        OrderType::StopLimit => {
            if stop_triggered(order, bar) {
                limit_fill_price(order, bar)
            } else {
                None
            }
        }
    }?;
    let price = apply_slippage(raw_price, order.side, options.slippage);
    Some(FillEvent {
        order_id: order.order_id,
        symbol: order.symbol.clone(),
        size: order.size,
        price,
        commission: calculate_commission(price, order.size, options.commission),
        slippage: price - raw_price,
        ts: bar.ts,
    })
}

fn execution_price(bar: &BarEvent, options: &ExecutionOptions) -> f64 {
    if options.trade_on_close {
        bar.close
    } else {
        bar.open
    }
}

fn limit_fill_price(order: &OrderEvent, bar: &BarEvent) -> Option<f64> {
    let limit = order.limit_price?;
    match order.side {
        OrderSide::Buy if bar.low <= limit => Some(limit.min(bar.open)),
        OrderSide::Sell if bar.high >= limit => Some(limit.max(bar.open)),
        _ => None,
    }
}

fn stop_triggered(order: &OrderEvent, bar: &BarEvent) -> bool {
    match (order.side, order.stop_price) {
        (OrderSide::Buy, Some(stop)) => bar.high >= stop,
        (OrderSide::Sell, Some(stop)) => bar.low <= stop,
        _ => false,
    }
}

fn apply_slippage(price: f64, side: OrderSide, slippage: SlippageModel) -> f64 {
    match (side, slippage) {
        (OrderSide::Buy, SlippageModel::Fixed(model)) => price + model.amount,
        (OrderSide::Sell, SlippageModel::Fixed(model)) => price - model.amount,
        (OrderSide::Buy, SlippageModel::Percent(model)) => price * (1.0 + model.ratio),
        (OrderSide::Sell, SlippageModel::Percent(model)) => price * (1.0 - model.ratio),
    }
}

fn calculate_commission(price: f64, size: f64, commission: CommissionModel) -> f64 {
    match commission {
        CommissionModel::Fixed(model) => model.amount,
        CommissionModel::Percent(model) => price * size.abs() * model.ratio,
    }
}
