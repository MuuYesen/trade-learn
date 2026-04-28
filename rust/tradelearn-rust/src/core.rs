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
    name: String,
    bars: Vec<BarEvent>,
    cursor: usize,
}

#[derive(Default, Debug)]
pub struct MultiDataFeed {
    feeds: Vec<BarDataFeed>,
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

fn fill_from_raw_price(
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

fn calculate_commission(price: f64, size: f64, commission: CommissionModel, mult: f64) -> f64 {
    match commission {
        CommissionModel::Fixed(model) => model.amount,
        CommissionModel::Percent(model) => price * size.abs() * model.ratio * mult,
    }
}

fn round_precision(value: f64) -> f64 {
    const SCALE: f64 = 1_000_000.0;
    (value * SCALE).round() / SCALE
}

fn signed_order_size(order: &OrderEvent) -> f64 {
    match order.side {
        OrderSide::Buy => order.size.abs(),
        OrderSide::Sell => -order.size.abs(),
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

// ---------------------------------------------------------------------------
// BacktestEngine – drives the main loop from Rust
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EquityRecord {
    pub ts: Timestamp,
    pub cash: f64,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct FillRecord {
    pub order_id: OrderId,
    pub ts: Timestamp,
    pub side: OrderSide,
    pub size: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub pnl: f64,
}

#[derive(Clone, Debug, Default)]
pub struct BacktestResults {
    pub equity: Vec<EquityRecord>,
    pub fills: Vec<FillRecord>,
}

#[derive(Debug)]
pub struct BacktestEngine {
    // OHLCV data (columnar)
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    total_bars: usize,

    // Portfolio state
    portfolio: Portfolio,

    // Execution config
    options: ExecutionOptions,

    // Pending orders
    pending: Vec<OrderEvent>,
    next_order_id: OrderId,
    activation_bar: usize,

    // Records
    results: BacktestResults,
}

impl BacktestEngine {
    pub fn new(
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
    ) -> Self {
        let total_bars = timestamps.len();
        Self {
            timestamps,
            opens,
            highs,
            lows,
            closes,
            volumes,
            total_bars,
            portfolio: Portfolio::new(cash),
            options: ExecutionOptions {
                trade_on_close,
                cheat_on_close,
                cheat_on_open,
                slip_perc,
                slip_fixed,
                slip_match,
                slip_limit,
                slip_out,
                slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
                commission: CommissionModel::Percent(PercentCommission {
                    ratio: commission_ratio,
                }),
                mult,
                margin,
            },
            pending: Vec::new(),
            next_order_id: 1,
            activation_bar: 1,
            results: BacktestResults::default(),
        }
    }

    pub fn total_bars(&self) -> usize {
        self.total_bars
    }

    fn bar_at(&self, cursor: usize) -> BarEvent {
        BarEvent {
            ts: self.timestamps[cursor],
            symbol: "data0".to_string(),
            open: self.opens[cursor],
            high: self.highs[cursor],
            low: self.lows[cursor],
            close: self.closes[cursor],
            volume: self.volumes[cursor],
        }
    }

    /// Process pending orders against the bar at `cursor`.
    /// Returns list of fills that occurred.
    pub fn step(&mut self, cursor: usize) -> Vec<FillRecord> {
        let bar = self.bar_at(cursor);
        let fills = self.match_all_pending(&bar);

        // Snapshot portfolio
        self.portfolio.mark_to_market(&[bar], self.options.mult);
        self.results.equity.push(EquityRecord {
            ts: self.timestamps[cursor],
            cash: self.portfolio.cash(),
            value: self.portfolio.equity(self.options.mult),
        });

        fills
    }

    pub fn step_close(&mut self, cursor: usize) -> Vec<FillRecord> {
        let bar = self.bar_at(cursor);
        let mut options = self.options;
        options.trade_on_close = true;
        self.match_all_pending_internal(&bar, &options)
    }

    pub fn step_open(&mut self, cursor: usize) -> Vec<FillRecord> {
        let bar = self.bar_at(cursor);
        let mut options = self.options;
        options.trade_on_close = false;
        self.match_all_pending_internal(&bar, &options)
    }

    fn match_all_pending(&mut self, bar: &BarEvent) -> Vec<FillRecord> {
        let options = self.options;
        self.match_all_pending_internal(bar, &options)
    }

    fn match_all_pending_internal(
        &mut self,
        bar: &BarEvent,
        options: &ExecutionOptions,
    ) -> Vec<FillRecord> {
        let mut fills = Vec::new();
        let mut remaining = Vec::new();
        let current_pending = std::mem::take(&mut self.pending);

        for order in current_pending {
            if let Some(fill_event) = match_order(&order, bar, options) {
                let position = self.portfolio.position(&order.symbol);
                let old_size = position.map(|p| p.size).unwrap_or(0.0);
                let old_price = position.map(|p| p.avg_price).unwrap_or(0.0);
                let pnl = realized_pnl(
                    old_size,
                    old_price,
                    fill_event.size,
                    fill_event.price,
                    options.mult,
                );

                self.portfolio.apply_fill(&fill_event, options.mult);

                let record = FillRecord {
                    order_id: fill_event.order_id,
                    ts: fill_event.ts,
                    side: order.side,
                    size: fill_event.size,
                    price: fill_event.price,
                    commission: fill_event.commission,
                    slippage: fill_event.slippage,
                    pnl,
                };
                self.results.fills.push(record.clone());
                fills.push(record);
            } else {
                remaining.push(order);
            }
        }
        self.pending = remaining;
        fills
    }

    /// Submit a new order (called from Python's strategy.buy()/sell()).
    pub fn submit_order(
        &mut self,
        side: OrderSide,
        order_type: OrderType,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
    ) -> OrderId {
        let order_id = self.next_order_id;
        self.next_order_id += 1;
        self.pending.push(OrderEvent {
            order_id,
            symbol: "data0".to_string(),
            side,
            order_type,
            size: size.abs(),
            limit_price,
            stop_price,
            created_ts: 0,
        });
        order_id
    }

    pub fn get_position(&self) -> (f64, f64) {
        self.portfolio
            .position("data0")
            .map(|p| (p.size, p.avg_price))
            .unwrap_or((0.0, 0.0))
    }

    pub fn get_cash(&self) -> f64 {
        self.portfolio.cash()
    }

    pub fn get_equity(&self) -> f64 {
        self.portfolio.equity(self.options.mult)
    }

    pub fn get_results(&self) -> &BacktestResults {
        &self.results
    }
}

fn realized_pnl(old_size: f64, old_price: f64, fill_size: f64, fill_price: f64, mult: f64) -> f64 {
    if old_size == 0.0 || old_size.signum() == fill_size.signum() {
        return 0.0;
    }
    let closing_size = old_size.abs().min(fill_size.abs());
    (fill_price - old_price) * closing_size * old_size.signum() * mult
}
