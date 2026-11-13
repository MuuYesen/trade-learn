use std::collections::{HashMap, HashSet};

use crate::matching::{
    advance_trailing_order, calculate_commission, fill_from_raw_price, is_exit_fill, is_exit_order_for_position,
    match_order, match_order_smart, smart_match_price, smart_order_priority,
};
use crate::types::*;

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeliveryInput {
    turn: u64,
    clock: Timestamp,
    source_ts: Timestamp,
    values: [u64; 10],
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
    last_open_timestamps: HashMap<String, Timestamp>,
    active_delivery_inputs: HashMap<String, DeliveryInput>,
    last_delivery_inputs: HashMap<String, DeliveryInput>,
    open_clock: HashMap<String, Timestamp>,
    deadlines: HashMap<OrderId, (Timestamp, bool)>,
    oco_links: HashMap<OrderId, HashSet<OrderId>>,
    terminal_events: Vec<(OrderId, String)>,
    pending_cash: Vec<(u64, f64)>,
    applied_cash: Vec<(u64, String, f64, f64)>,
    last_cash_command: u64,
    cash_error: bool,
    submission_checks: HashMap<OrderId, f64>,
    market_close_references: HashMap<OrderId, f64>,
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
        smart_matching: bool,
        cn_a_stock_commission: bool,
        cn_commission_rate: f64,
        cn_min_commission: f64,
        cn_stamp_tax_rate: f64,
        cn_transfer_fee_rate: f64,
    ) -> Self {
        let total_bars = timestamps.len();
        let commission = if cn_a_stock_commission {
            CommissionModel::CNAStock(CNAStockCommission {
                commission_rate: cn_commission_rate,
                min_commission: cn_min_commission,
                stamp_tax_rate: cn_stamp_tax_rate,
                transfer_fee_rate: cn_transfer_fee_rate,
            })
        } else {
            CommissionModel::Percent(PercentCommission {
                ratio: commission_ratio,
            })
        };
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
                preserve_input_precision: false,
                trade_on_close,
                smart_matching,
                cheat_on_close,
                cheat_on_open,
                slip_perc,
                slip_fixed,
                slip_match,
                slip_limit,
                slip_out,
                slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
                commission,
                mult,
                margin,
            },
            pending: Vec::new(),
            next_order_id: 1,
            last_open_timestamps: HashMap::new(),
            active_delivery_inputs: HashMap::new(),
            last_delivery_inputs: HashMap::new(),
            open_clock: HashMap::new(),
            deadlines: HashMap::new(),
            oco_links: HashMap::new(),
            terminal_events: Vec::new(),
            pending_cash: Vec::new(),
            applied_cash: Vec::new(),
            last_cash_command: 0,
            cash_error: false,
            submission_checks: HashMap::new(),
            market_close_references: HashMap::new(),
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
        self.check_submissions();
        let fills = self.match_all_pending(&bar);
        self.apply_pending_cash();

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
        self.check_submissions();
        let fills = self.match_all_pending_internal(&bar, &options);
        self.apply_pending_cash();
        // The columnar fast path owns the same current-bar mark as the
        // multi-feed path, before any Python account observation.
        self.portfolio.mark_to_market(&[bar],self.options.mult);
        fills
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

        if options.smart_matching {
            return self.match_all_pending_smart(bar, options, current_pending);
        }

        let mut excluded = HashSet::new();
        for order in current_pending {
            if excluded.contains(&order.order_id) {
                continue;
            }
            if order.symbol != bar.symbol {
                remaining.push(order);
                continue;
            }
            if self.expire_order(&order, bar.ts) {
                continue;
            }
            if options.trade_on_close && order.order_type != OrderType::Market {
                remaining.push(order);
                continue;
            }
            if !Self::market_eligible(&order, bar, options) {
                remaining.push(order);
                continue;
            }
            // 跟踪止损单：先用「上一根 bar 的水位」参与撮合判定，
            // 避免同一根 bar 先拿 high 抬水位、再用自己的 low 触发（bar 内前视）。
            let matched = match_order(&order, bar, options);
            if let Some(fill_event) = matched {
                if !self.can_apply_fill(&fill_event, options.mult) {
                    self.terminal_events.push((order.order_id, "margin".into()));
                    continue;
                }
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
                excluded.extend(self.cancel_oco(order.order_id));

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
                // 未成交的跟踪单：用当前 bar 极值推进水位后回写，供下一根 bar 使用。
                let mut carried = order;
                if carried.order_type == OrderType::Market {
                    self.terminal_events
                        .push((carried.order_id, "rejected".into()));
                    continue;
                }
                advance_trailing_order(&mut carried, bar);
                remaining.push(carried);
            }
        }
        remaining.retain(|order| !excluded.contains(&order.order_id));
        self.pending = remaining;
        fills
    }

    fn match_all_pending_smart(
        &mut self,
        bar: &BarEvent,
        options: &ExecutionOptions,
        current_pending: Vec<OrderEvent>,
    ) -> Vec<FillRecord> {
        let mut fills = Vec::new();
        let mut candidates = Vec::new();

        let mut current_pending = current_pending;
        current_pending
            .retain(|order| order.symbol != bar.symbol || !self.expire_order(order, bar.ts));
        for (idx, order) in current_pending.iter().enumerate() {
            if order.symbol != bar.symbol
                || !Self::market_eligible(order, bar, options)
                || (options.trade_on_close && order.order_type != OrderType::Market)
            {
                continue;
            }
            if let Some((rank, raw_price)) = smart_match_price(order, bar, options) {
                let fill_event = fill_from_raw_price(order, raw_price, bar, options);
                candidates.push((rank, smart_order_priority(order), idx, fill_event));
            }
        }
        candidates.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });

        let mut filled = HashSet::new();
        let mut canceled = HashSet::new();
        for (_, _, idx, fill_event) in candidates {
            if filled.contains(&idx) || canceled.contains(&idx) {
                continue;
            }
            let order = &current_pending[idx];
            if !self.can_apply_fill(&fill_event, options.mult) {
                self.terminal_events.push((order.order_id, "margin".into()));
                filled.insert(idx);
                continue;
            }
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
            filled.insert(idx);

            let siblings = self.cancel_oco(order.order_id);
            // Preserve the existing smart-mode protective-exit behavior. Explicit
            // OCO additionally works in exact mode and across symbols.
            let new_size = self
                .portfolio
                .position(&order.symbol)
                .map(|p| p.size)
                .unwrap_or(0.0);
            if is_exit_fill(old_size, fill_event.size) && new_size.abs() < 1e-9 {
                for (other_idx, other) in current_pending.iter().enumerate() {
                    if other_idx != idx
                        && !filled.contains(&other_idx)
                        && !canceled.contains(&other_idx)
                        && other.symbol == order.symbol
                        && other.side == order.side
                        && is_exit_order_for_position(old_size, other)
                    {
                        canceled.insert(other_idx);
                        if !siblings.contains(&other.order_id) {
                            self.terminal_events
                                .push((other.order_id, "canceled".into()));
                        }
                    }
                }
            }
            for (other_idx, other) in current_pending.iter().enumerate() {
                if siblings.contains(&other.order_id) {
                    canceled.insert(other_idx);
                }
            }
        }

        self.pending = current_pending
            .into_iter()
            .enumerate()
            .filter_map(|(idx, mut order)| {
                if filled.contains(&idx) || canceled.contains(&idx) {
                    None
                } else {
                    if order.symbol == bar.symbol && !options.trade_on_close {
                        if !Self::market_eligible(&order, bar, options) {
                            return Some(order);
                        }
                        if order.order_type == OrderType::Market {
                            self.terminal_events
                                .push((order.order_id, "rejected".into()));
                            return None;
                        }
                        advance_trailing_order(&mut order, bar);
                    }
                    Some(order)
                }
            })
            .collect();
        fills
    }

    fn can_apply_fill(&self, fill: &FillEvent, mult: f64) -> bool {
        self.portfolio.can_apply_fill(fill, mult)
    }

    pub fn step_open_bars(&mut self, bars: Vec<BarEvent>) -> Vec<FillRecord> {
        self.step_bars_with_trade_on_close(bars, false)
    }

    pub fn step_close_bars(&mut self, bars: Vec<BarEvent>) -> Vec<FillRecord> {
        self.step_bars_with_trade_on_close(bars, true)
    }

    fn step_bars_with_trade_on_close(
        &mut self,
        bars: Vec<BarEvent>,
        trade_on_close: bool,
    ) -> Vec<FillRecord> {
        self.step_bars_with_quotes(bars, &[], trade_on_close)
    }

    // BT feeds expose execution ticks separately from the current valuation
    // bar, notably on EOF delivery. Both still use this one matching cycle.
    /// Private delivery metadata for BT replay. Quotes retain their source
    /// timestamp; date-gated admission/expiry uses the visible BT date instead.
    /// Each turn is scoped to one native symbol and must be immutable on replay.
    pub fn step_bars_with_quote_views(
        &mut self,
        bars: Vec<BarEvent>,
        quotes: &[BarEvent],
        views: &HashMap<String, (u64, Timestamp)>,
        trade_on_close: bool,
    ) -> Result<Vec<FillRecord>, String> {
        if views.is_empty() {
            return Ok(self.step_bars_with_quotes(bars, quotes, trade_on_close));
        }
        let symbols: HashSet<&str> = bars.iter().map(|bar| bar.symbol.as_str()).collect();
        if symbols.len() != bars.len() || views.keys().any(|key| !symbols.contains(key.as_str())) {
            return Err("delivery view has a foreign or duplicate symbol".into());
        }
        let mut inputs = HashMap::new();
        let mut fresh = false;
        for bar in &bars {
            let quote = quotes.iter().find(|quote| quote.symbol == bar.symbol && quote.ts == bar.ts)
                .ok_or_else(|| "delivery view requires its original execution quote".to_string())?;
            if ![quote.open, quote.high, quote.low, quote.close].iter().all(|value| value.is_finite()) {
                return Err("delivery execution quote must be finite".into());
            }
            if let Some(&(turn, clock)) = views.get(&bar.symbol) {
                if turn == 0 { return Err("delivery turn must be positive".into()); }
                let input = DeliveryInput { turn, clock, source_ts: bar.ts, values: [
                    bar.open, bar.high, bar.low, bar.close, bar.volume,
                    quote.open, quote.high, quote.low, quote.close, quote.volume,
                ].map(f64::to_bits) };
                let prior = self.last_delivery_inputs.get(&bar.symbol);
                if let Some(prior) = prior {
                    if turn < prior.turn || (turn == prior.turn && &input != prior) {
                        return Err("delivery turn is stale or conflicts with its frozen input".into());
                    }
                }
                if trade_on_close && prior != Some(&input) {
                    return Err("delivery close requires the same admitted open turn".into());
                }
                fresh |= prior != Some(&input);
                inputs.insert(bar.symbol.clone(), input);
            } else {
                fresh |= self.last_open_timestamps.get(&bar.symbol) != Some(&bar.ts);
            }
        }
        // An exact repeated delivery is not another account valuation/cash turn.
        if !trade_on_close && !fresh { return Ok(Vec::new()); }
        self.active_delivery_inputs = inputs;
        let result = self.step_bars_with_quotes(bars, quotes, trade_on_close);
        self.active_delivery_inputs.clear();
        Ok(result)
    }

    pub fn step_bars_with_quotes(
        &mut self,
        bars: Vec<BarEvent>,
        quotes: &[BarEvent],
        trade_on_close: bool,
    ) -> Vec<FillRecord> {
        let mut options = self.options;
        options.trade_on_close = trade_on_close;
        let clock = bars.first().map(|bar| bar.ts);
        // Validity follows the primary runtime clock even when a target feed
        // has no new bar. Expire all intents before matching or OCO resolution.
        if let Some(ts) = clock {
            let pending = std::mem::take(&mut self.pending);
            self.pending = pending
                .into_iter()
                .filter(|order| !self.expire_global_order(order, ts))
                .collect();
        }
        let matching_bars: Vec<BarEvent> = bars
            .iter()
            .filter(|bar| {
                if trade_on_close {
                    return self.open_clock.get(&bar.symbol).copied() == clock;
                }
                if let Some(input) = self.active_delivery_inputs.get(&bar.symbol).cloned() {
                    if self.last_delivery_inputs.get(&bar.symbol) == Some(&input) { return false; }
                    self.last_delivery_inputs.insert(bar.symbol.clone(), input);
                } else if self.last_open_timestamps.get(&bar.symbol) == Some(&bar.ts) {
                    return false;
                }
                self.last_open_timestamps.insert(bar.symbol.clone(), bar.ts);
                if let Some(clock) = clock {
                    self.open_clock.insert(bar.symbol.clone(), clock);
                }
                true
            })
            .map(|bar| {
                quotes
                    .iter()
                    .find(|quote| quote.symbol == bar.symbol && quote.ts == bar.ts)
                    .unwrap_or(bar)
                    .clone()
            })
            .collect();
        if !trade_on_close { self.check_submissions(); }
        let all_fills = self.match_all_pending_against_bars(&matching_bars, &options);
        if !trade_on_close { self.apply_pending_cash(); }
        self.portfolio.mark_to_market(&bars, self.options.mult);
        if let Some(primary) = bars.first() {
            self.results.equity.push(EquityRecord {
                ts: primary.ts,
                cash: self.portfolio.cash(),
                value: self.portfolio.equity(self.options.mult),
            });
        }
        all_fills
    }

    fn match_all_pending_against_bars(
        &mut self,
        bars: &[BarEvent],
        options: &ExecutionOptions,
    ) -> Vec<FillRecord> {
        let mut fills = Vec::new();
        let mut remaining = Vec::new();
        let bars_by_symbol: HashMap<&str, &BarEvent> =
            bars.iter().map(|bar| (bar.symbol.as_str(), bar)).collect();
        let current_pending = std::mem::take(&mut self.pending);

        let mut excluded = HashSet::new();
        let mut current_pending = current_pending;
        if options.smart_matching {
            current_pending.sort_by(|a, b| {
                let rank = |o: &OrderEvent| {
                    bars_by_symbol
                        .get(o.symbol.as_str())
                        .and_then(|bar| smart_match_price(o, bar, options))
                        .map(|(rank, _)| rank)
                        .unwrap_or(f64::INFINITY)
                };
                rank(a)
                    .partial_cmp(&rank(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(smart_order_priority(a).cmp(&smart_order_priority(b)))
            });
        }
        for order in current_pending {
            if excluded.contains(&order.order_id) {
                continue;
            }
            let Some(bar) = bars_by_symbol.get(order.symbol.as_str()) else {
                remaining.push(order);
                continue;
            };
            let admission_clock = self.active_delivery_inputs.get(&bar.symbol).map(|input| input.clock).unwrap_or(bar.ts);
            if self.expire_order(&order, admission_clock) {
                continue;
            }
            if options.trade_on_close && order.order_type != OrderType::Market {
                remaining.push(order);
                continue;
            }
            let captured_close=self.market_close_references.get(&order.order_id).copied();
            if captured_close.is_none() && !Self::market_eligible_at(&order, admission_clock, options) {
                remaining.push(order);
                continue;
            }
            // 跟踪止损单：先用「上一根 bar 的水位」参与撮合判定，未成交再推进水位，
            // 避免同一根 bar 用自身极值制造 bar 内前视。
            // BT COC captures the creating logical bar's close, which may
            // differ from the raw quote being delivered during replay.
            // This per-order price reference never replaces the source quote.
            let mut priced_bar;
            let mut pricing_options=*options;
            let pricing=if let Some(price)=captured_close {
                priced_bar=(*bar).clone();priced_bar.close=price;
                pricing_options.trade_on_close=true;&priced_bar
            } else { bar };
            let matched = if pricing_options.smart_matching {
                match_order_smart(&order, pricing, &pricing_options)
            } else {
                match_order(&order, pricing, &pricing_options)
            };
            if let Some(fill_event) = matched {
                if !self.can_apply_fill(&fill_event, options.mult) {
                    self.terminal_events.push((order.order_id, "margin".into()));
                    continue;
                }
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
                excluded.extend(self.cancel_oco(order.order_id));

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
                // 未成交的跟踪单：用当前 bar 极值推进水位后回写，供下一根 bar 使用。
                let mut carried = order;
                if carried.order_type == OrderType::Market {
                    self.terminal_events
                        .push((carried.order_id, "rejected".into()));
                    continue;
                }
                advance_trailing_order(&mut carried, bar);
                remaining.push(carried);
            }
        }
        remaining.retain(|order| !excluded.contains(&order.order_id));
        self.pending = remaining;
        self.market_close_references.retain(|id,_|self.pending.iter().any(|order|order.order_id==*id));
        fills
    }

    /// Submit a new order (called from Python's strategy.buy()/sell()).
    pub fn submit_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        size: f64,
        limit_price: Option<f64>,
        stop_price: Option<f64>,
        trail_amount: Option<f64>,
        trail_percent: Option<f64>,
    ) -> OrderId {
        let order_id = self.next_order_id;
        self.next_order_id += 1;
        self.pending.push(OrderEvent {
            order_id,
            symbol,
            side,
            order_type,
            size: size.abs(),
            limit_price,
            stop_price,
            // Unconfigured (Lite/native direct) orders have no BT date gate.
            created_ts: i64::MIN,
            trail_amount,
            trail_percent,
            trail_watermark: stop_price,
            trail_triggered: false,
        });
        order_id
    }

    pub fn configure_market_close_reference(&mut self, order_id: OrderId, price: f64) -> Result<(), String> {
        if !price.is_finite() { return Err("market close reference must be finite".into()); }
        if !self.pending.iter().any(|order| order.order_id==order_id && order.order_type==OrderType::Market) {
            return Err("market close reference requires a pending market order".into());
        }
        self.market_close_references.insert(order_id,price);
        Ok(())
    }

    pub fn configure_bt_precision(&mut self, preserve: bool) {
        // Selected by the existing private broker command boundary, not a new
        // Engine user option. Legacy Lite/native direct callers keep their policy.
        self.options.preserve_input_precision = preserve;
    }

    pub fn configure_market_eligibility(&mut self, order_id: OrderId, after: Timestamp) {
        if let Some(order) = self
            .pending
            .iter_mut()
            .find(|order| order.order_id == order_id)
        {
            if order.order_type == OrderType::Market {
                // Upper raw-tick boundary of BT's public creation date. Actual
                // submission clocks remain in the Python/raw event facts.
                order.created_ts = after;
            }
        }
    }

    fn market_eligible(order: &OrderEvent, bar: &BarEvent, options: &ExecutionOptions) -> bool {
        Self::market_eligible_at(order, bar.ts, options)
    }

    fn market_eligible_at(order: &OrderEvent, clock: Timestamp, options: &ExecutionOptions) -> bool {
        options.trade_on_close || order.order_type != OrderType::Market || clock > order.created_ts
    }

    pub fn configure_order(
        &mut self,
        order_id: OrderId,
        deadline: Option<Timestamp>,
        peer: Option<OrderId>,
    ) {
        if let Some(ts) = deadline {
            let target_clock = self
                .deadlines
                .get(&order_id)
                .is_some_and(|(_, target)| *target);
            self.deadlines.insert(order_id, (ts, target_clock));
        }
        if let Some(peer) = peer {
            let mut group = self.oco_links.get(&order_id).cloned().unwrap_or_default();
            group.extend(self.oco_links.get(&peer).cloned().unwrap_or_default());
            group.insert(order_id);
            group.insert(peer);
            for member in &group {
                self.oco_links.insert(*member, group.clone());
            }
        }
    }

    pub fn configure_order_expiry_clock(&mut self, order_id: OrderId, target_clock: bool) {
        if let Some((_, target)) = self.deadlines.get_mut(&order_id) {
            *target = target_clock;
        }
    }

    fn expire_global_order(&mut self, order: &OrderEvent, ts: Timestamp) -> bool {
        let target = self
            .deadlines
            .get(&order.order_id)
            .is_some_and(|(_, target)| *target);
        !target && self.expire_order(order, ts)
    }

    fn expire_order(&mut self, order: &OrderEvent, ts: Timestamp) -> bool {
        if self
            .deadlines
            .get(&order.order_id)
            .is_some_and(|(deadline, _)| ts > *deadline)
        {
            self.deadlines.remove(&order.order_id);
            self.terminal_events
                .push((order.order_id, "expired".into()));
            true
        } else {
            false
        }
    }

    fn cancel_oco(&mut self, order_id: OrderId) -> HashSet<OrderId> {
        let mut group = self.oco_links.remove(&order_id).unwrap_or_default();
        group.remove(&order_id);
        for peer in &group {
            self.oco_links.remove(peer);
            self.deadlines.remove(peer);
            self.submission_checks.remove(peer);
            self.terminal_events.push((*peer, "canceled".into()));
        }
        self.deadlines.remove(&order_id);
        group
    }

    pub fn drain_order_events(&mut self) -> Vec<(OrderId, String)> {
        std::mem::take(&mut self.terminal_events)
    }

    /// Remove a pending order from the matching queue (called by Python cancel()).
    /// Returns true if the order was found and removed.
    pub fn remove_order(&mut self, order_ref: u64) -> bool {
        self.submission_checks.remove(&order_ref);
        self.market_close_references.remove(&order_ref);
        let before = self.pending.len();
        self.pending.retain(|o| o.order_id != order_ref);
        self.pending.len() != before
    }

    pub fn get_position(&self) -> (f64, f64) {
        self.get_position_for_symbol("data0")
    }

    pub fn get_position_for_symbol(&self, symbol: &str) -> (f64, f64) {
        self.portfolio
            .position(symbol)
            .map(|p| (p.size, p.avg_price))
            .unwrap_or((0.0, 0.0))
    }

    pub fn configure_submission_check(&mut self, order_id: OrderId, price: f64) -> Result<(), String> {
        if !price.is_finite() { return Err("finite submission reference required".into()); }
        if self.pending.iter().any(|order| order.order_id == order_id) {
            self.submission_checks.insert(order_id, price);
        }
        Ok(())
    }

    fn check_submissions(&mut self) {
        // BT's stock-like admission is a temporary budget, not a second
        // portfolio or a fill. The real portfolio changes only on real matches.
        let mut budget = self.portfolio.cash();
        let mut canceled = HashSet::new();
        let mut admitted = Vec::with_capacity(self.pending.len());
        for order in std::mem::take(&mut self.pending) {
            if canceled.contains(&order.order_id) {
                self.submission_checks.remove(&order.order_id);
                continue;
            }
            if let Some(price) = self.submission_checks.remove(&order.order_id) {
                let signed = if order.side == OrderSide::Buy { order.size } else { -order.size };
                budget -= signed * price * self.options.mult
                    + calculate_commission(price, signed, order.side, self.options.commission, self.options.mult);
                if budget < 0.0 {
                    self.terminal_events.push((order.order_id, "submission_margin".into()));
                    canceled.extend(self.cancel_oco(order.order_id));
                    continue;
                }
            }
            admitted.push(order);
        }
        self.pending = admitted.into_iter().filter(|order| !canceled.contains(&order.order_id)).collect();
    }

    /// Commands affect this one portfolio; no strategy-side account is created.
    pub fn apply_cash_controls(&mut self, commands: Vec<(u64, String, f64)>) -> Result<(), String> {
        let mut previous = self.last_cash_command;
        let mut projected = self.portfolio.cash();
        let mut queued: f64 = self.pending_cash.iter().map(|(_, amount)| *amount).sum();
        projected += queued;
        let additions = commands.iter().filter(|(_, kind, _)| kind == "add").count();
        if self.pending_cash.len() + additions > 4096 || self.applied_cash.len() + commands.len() > 4096 || self.cash_error {
            return Err("native cash command capacity or prior failure".into());
        }
        for (id, kind, amount) in &commands {
            if *id <= previous || !amount.is_finite() || (kind != "set" && kind != "add") {
                return Err("invalid or repeated native cash command".into());
            }
            if kind == "set" { projected = *amount + queued; }
            else { queued += *amount; projected += *amount; }
            if !projected.is_finite() { return Err("native cash command overflows account".into()); }
            previous = *id;
        }
        for (id, kind, amount) in commands {
            if kind == "set" {
                self.portfolio.set_cash(amount);
                self.applied_cash.push((id, kind, amount, self.portfolio.cash()));
            } else {
                self.pending_cash.push((id, amount));
            }
            self.last_cash_command = id;
        }
        Ok(())
    }

    fn apply_pending_cash(&mut self) {
        let mut balance = self.portfolio.cash();
        for (_, amount) in &self.pending_cash {
            balance += amount;
            if !balance.is_finite() { self.cash_error = true; return; }
        }
        for (id, amount) in std::mem::take(&mut self.pending_cash) {
            self.portfolio.set_cash(self.portfolio.cash() + amount);
            self.applied_cash.push((id, "add".into(), amount, self.portfolio.cash()));
        }
    }

    pub fn drain_cash_events(&mut self) -> Result<Vec<(u64, String, f64, f64)>, String> {
        if self.cash_error { return Err("native cash application overflow; account unchanged".into()); }
        Ok(std::mem::take(&mut self.applied_cash))
    }

    pub fn get_cash(&self) -> f64 {
        self.portfolio.cash()
    }

    /// Observe the one account's selected positions. BT's explicit data
    /// valuation settles already-authorized cash additions at this boundary.
    pub fn observe_valuation(&mut self, symbols: Option<&[String]>, include_cash: bool, settle_cash: bool)
        -> Result<(f64,f64,f64),String> {
        let market=self.portfolio.position_value(symbols,self.options.mult);
        let projected=self.portfolio.cash()+if settle_cash { self.pending_cash.iter().map(|(_,amount)|*amount).sum::<f64>() } else {0.};
        let value=market+if include_cash {projected} else {0.};
        if !market.is_finite() || !projected.is_finite() || !value.is_finite() {
            return Err("native selected valuation is not finite; cash settlement unchanged".into());
        }
        if settle_cash {self.apply_pending_cash();}
        if self.cash_error {return Err("native cash settlement unavailable".into());}
        Ok((self.portfolio.cash(),value,market))
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
