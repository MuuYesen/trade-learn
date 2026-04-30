use std::collections::{HashMap, HashSet};

use crate::matching::{
    fill_from_raw_price, is_exit_fill, is_exit_order_for_position, match_order, match_order_smart,
    smart_match_price, smart_order_priority,
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
        smart_matching: bool,
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
                smart_matching,
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

        if options.smart_matching {
            return self.match_all_pending_smart(bar, options, current_pending);
        }

        for order in current_pending {
            if order.symbol != bar.symbol {
                remaining.push(order);
                continue;
            }
            if let Some(fill_event) = if options.smart_matching {
                match_order_smart(&order, bar, options)
            } else {
                match_order(&order, bar, options)
            } {
                if !self.can_apply_fill(&fill_event, options.mult) {
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
            } else {
                remaining.push(order);
            }
        }
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

        for (idx, order) in current_pending.iter().enumerate() {
            if order.symbol != bar.symbol {
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
                filled.insert(idx);
                continue;
            }
            let position = self.portfolio.position(&order.symbol);
            let old_size = position.map(|p| p.size).unwrap_or(0.0);
            let old_price = position.map(|p| p.avg_price).unwrap_or(0.0);
            let was_exit = is_exit_fill(old_size, fill_event.size);
            let pnl = realized_pnl(
                old_size,
                old_price,
                fill_event.size,
                fill_event.price,
                options.mult,
            );

            self.portfolio.apply_fill(&fill_event, options.mult);
            let new_size = self
                .portfolio
                .position(&order.symbol)
                .map(|p| p.size)
                .unwrap_or(0.0);

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

            if was_exit && new_size.abs() < 1e-9 {
                for (other_idx, other) in current_pending.iter().enumerate() {
                    if other_idx != idx
                        && other.symbol == order.symbol
                        && other.side == order.side
                        && is_exit_order_for_position(old_size, other)
                    {
                        canceled.insert(other_idx);
                    }
                }
            }
        }

        self.pending = current_pending
            .into_iter()
            .enumerate()
            .filter_map(|(idx, order)| {
                if filled.contains(&idx) || canceled.contains(&idx) {
                    None
                } else {
                    Some(order)
                }
            })
            .collect();
        fills
    }

    fn can_apply_fill(&self, fill: &FillEvent, mult: f64) -> bool {
        if fill.size <= 0.0 {
            return true;
        }
        let required_cash = fill.price * fill.size * mult + fill.commission;
        self.portfolio.cash() + 1e-9 >= required_cash
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
        let mut options = self.options;
        options.trade_on_close = trade_on_close;
        let all_fills = self.match_all_pending_against_bars(&bars, &options);
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

        for order in current_pending {
            let Some(bar) = bars_by_symbol.get(order.symbol.as_str()) else {
                remaining.push(order);
                continue;
            };
            let matched = if options.smart_matching {
                match_order_smart(&order, bar, options)
            } else {
                match_order(&order, bar, options)
            };
            if let Some(fill_event) = matched {
                if !self.can_apply_fill(&fill_event, options.mult) {
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
        symbol: String,
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
            symbol,
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
        self.get_position_for_symbol("data0")
    }

    pub fn get_position_for_symbol(&self, symbol: &str) -> (f64, f64) {
        self.portfolio
            .position(symbol)
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
