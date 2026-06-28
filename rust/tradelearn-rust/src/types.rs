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

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BarBatch {
    pub ts: Vec<Timestamp>,
    pub symbol: Vec<String>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CallbackBatch {
    pub sequence: usize,
    pub bars: BarBatch,
}

#[derive(Debug)]
pub struct CallbackBatcher {
    callback_batch: usize,
    next_sequence: usize,
    pending: Vec<BarEvent>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    pub symbol: String,
    pub size: f64,
    pub avg_price: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub mark_price: f64,
}

#[derive(Debug)]
pub struct Portfolio {
    cash: f64,
    positions: BTreeMap<String, Position>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DataFeedBar {
    pub feed_index: usize,
    pub feed_name: String,
    pub bar: BarEvent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BarDataFeed {
    pub(crate) name: String,
    pub(crate) bars: Vec<BarEvent>,
    pub(crate) cursor: usize,
}

#[derive(Default, Debug)]
pub struct MultiDataFeed {
    pub(crate) feeds: Vec<BarDataFeed>,
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
pub struct CNAStockCommission {
    pub commission_rate: f64,
    pub min_commission: f64,
    pub stamp_tax_rate: f64,
    pub transfer_fee_rate: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommissionModel {
    Fixed(FixedCommission),
    Percent(PercentCommission),
    CNAStock(CNAStockCommission),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutionOptions {
    pub trade_on_close: bool,
    pub smart_matching: bool,
    pub cheat_on_close: bool,
    pub cheat_on_open: bool,
    pub slip_perc: f64,
    pub slip_fixed: f64,
    pub slip_match: bool,
    pub slip_limit: bool,
    pub slip_out: bool,
    pub slippage: SlippageModel,
    pub commission: CommissionModel,
    pub mult: f64,
    pub margin: f64,
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

impl BarBatch {
    pub fn from_bars(bars: &[BarEvent]) -> Self {
        let mut batch = Self {
            ts: Vec::with_capacity(bars.len()),
            symbol: Vec::with_capacity(bars.len()),
            open: Vec::with_capacity(bars.len()),
            high: Vec::with_capacity(bars.len()),
            low: Vec::with_capacity(bars.len()),
            close: Vec::with_capacity(bars.len()),
            volume: Vec::with_capacity(bars.len()),
        };
        for bar in bars {
            batch.ts.push(bar.ts);
            batch.symbol.push(bar.symbol.clone());
            batch.open.push(bar.open);
            batch.high.push(bar.high);
            batch.low.push(bar.low);
            batch.close.push(bar.close);
            batch.volume.push(bar.volume);
        }
        batch
    }

    pub fn len(&self) -> usize {
        self.ts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ts.is_empty()
    }

    pub fn bar(&self, index: usize) -> Option<BarEvent> {
        Some(BarEvent {
            ts: *self.ts.get(index)?,
            symbol: self.symbol.get(index)?.clone(),
            open: *self.open.get(index)?,
            high: *self.high.get(index)?,
            low: *self.low.get(index)?,
            close: *self.close.get(index)?,
            volume: *self.volume.get(index)?,
        })
    }
}

impl CallbackBatcher {
    pub fn new(callback_batch: usize) -> Self {
        Self {
            callback_batch: callback_batch.max(1),
            next_sequence: 0,
            pending: Vec::new(),
        }
    }

    pub fn push_bar(&mut self, bar: BarEvent) -> Option<CallbackBatch> {
        self.pending.push(bar);
        if self.pending.len() >= self.callback_batch {
            self.drain_pending()
        } else {
            None
        }
    }

    pub fn flush(&mut self) -> Option<CallbackBatch> {
        if self.pending.is_empty() {
            None
        } else {
            self.drain_pending()
        }
    }

    pub fn callback_batch(&self) -> usize {
        self.callback_batch
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    fn drain_pending(&mut self) -> Option<CallbackBatch> {
        if self.pending.is_empty() {
            return None;
        }
        let sequence = self.next_sequence;
        self.next_sequence += 1;
        let bars = BarBatch::from_bars(&self.pending);
        self.pending.clear();
        Some(CallbackBatch { sequence, bars })
    }
}

impl Portfolio {
    pub fn new(cash: f64) -> Self {
        Self {
            cash,
            positions: BTreeMap::new(),
        }
    }

    pub fn apply_fill(&mut self, fill: &FillEvent, mult: f64) {
        // For non-stocklike (futures), we don't subtract price * size from cash.
        // Instead, we only subtract commission, and PnL is added later.
        // But for simplicity and to match the current engine's stock-like behavior by default,
        // we'll keep the current formula but allow mult to scale it.
        // Actually, Backtrader's mult affects the value.
        self.cash -= (fill.price * fill.size * mult) + fill.commission;
        let position = self
            .positions
            .entry(fill.symbol.clone())
            .or_insert_with(|| Position::new(fill.symbol.clone()));
        position.apply_fill(fill.size, fill.price, mult);
    }

    pub fn mark_to_market(&mut self, bars: &[BarEvent], mult: f64) {
        for bar in bars {
            if let Some(position) = self.positions.get_mut(&bar.symbol) {
                position.mark(bar.close, mult);
            }
        }
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    pub fn equity(&self, mult: f64) -> f64 {
        self.cash
            + self
                .positions
                .values()
                .map(|position| position.size * position.mark_price * mult)
                .sum::<f64>()
    }

    pub fn margin_used(&self, mult: f64, margin: f64) -> f64 {
        self.positions
            .values()
            .map(|position| (position.size * position.mark_price).abs() * mult * margin)
            .sum()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.positions
            .values()
            .map(|position| position.realized_pnl)
            .sum()
    }

    pub fn unrealized_pnl(&self) -> f64 {
        self.positions
            .values()
            .map(|position| position.unrealized_pnl)
            .sum()
    }

    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }
}

impl Position {
    fn new(symbol: String) -> Self {
        Self {
            symbol,
            size: 0.0,
            avg_price: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            mark_price: 0.0,
        }
    }

    fn apply_fill(&mut self, fill_size: f64, fill_price: f64, mult: f64) {
        if self.size == 0.0 || self.size.signum() == fill_size.signum() {
            let total_size = self.size + fill_size;
            self.avg_price = weighted_average(self.size, self.avg_price, fill_size, fill_price);
            self.size = total_size;
            self.mark_price = fill_price;
            self.mark(fill_price, mult);
            return;
        }

        let existing_sign = self.size.signum();
        let closing_size = self.size.abs().min(fill_size.abs());
        self.realized_pnl += (fill_price - self.avg_price) * closing_size * existing_sign * mult;
        let remaining_size = self.size + fill_size;
        self.size = remaining_size;
        if remaining_size == 0.0 {
            self.avg_price = 0.0;
        } else if remaining_size.signum() != existing_sign {
            self.avg_price = fill_price;
        }
        self.mark_price = fill_price;
        self.mark(fill_price, mult);
    }

    fn mark(&mut self, price: f64, mult: f64) {
        self.mark_price = price;
        self.unrealized_pnl = (price - self.avg_price) * self.size * mult;
    }
}

fn weighted_average(
    existing_size: f64,
    existing_price: f64,
    fill_size: f64,
    fill_price: f64,
) -> f64 {
    let total_abs_size = existing_size.abs() + fill_size.abs();
    if total_abs_size == 0.0 {
        0.0
    } else {
        (existing_size.abs() * existing_price + fill_size.abs() * fill_price) / total_abs_size
    }
}
