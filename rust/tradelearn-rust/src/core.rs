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
