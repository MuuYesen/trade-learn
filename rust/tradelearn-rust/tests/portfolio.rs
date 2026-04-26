use _rust::core::{
    match_order, BarEvent, CommissionModel, ExecutionOptions, FillEvent, FixedCommission,
    FixedSlippage, OrderEvent, OrderSide, OrderType, Portfolio, SlippageModel,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn bar(symbol: &str, ts: i64, open: f64, close: f64) -> BarEvent {
    BarEvent {
        ts,
        symbol: symbol.to_string(),
        open,
        high: open.max(close),
        low: open.min(close),
        close,
        volume: 1000.0,
    }
}

fn fill(order_id: u64, symbol: &str, size: f64, price: f64, commission: f64) -> FillEvent {
    FillEvent {
        order_id,
        symbol: symbol.to_string(),
        size,
        price,
        commission,
        slippage: 0.0,
        ts: order_id as i64,
    }
}

fn market_order(order_id: u64, symbol: &str, side: OrderSide, size: f64) -> OrderEvent {
    OrderEvent {
        order_id,
        symbol: symbol.to_string(),
        side,
        order_type: OrderType::Market,
        size,
        limit_price: None,
        stop_price: None,
        created_ts: order_id as i64,
    }
}

fn close_options() -> ExecutionOptions {
    ExecutionOptions {
        trade_on_close: true,
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
        commission: CommissionModel::Fixed(FixedCommission { amount: 0.0 }),
    }
}

#[test]
fn portfolio_tracks_cash_positions_equity_margin_and_pnl() {
    let mut portfolio = Portfolio::new(10_000.0);

    portfolio.apply_fill(&fill(1, "AAPL", 10.0, 100.0, 1.0));
    portfolio.mark_to_market(&[bar("AAPL", 2, 105.0, 110.0)]);

    assert_close(portfolio.cash(), 8_999.0);
    assert_close(
        portfolio.position("AAPL").expect("AAPL position").size,
        10.0,
    );
    assert_close(
        portfolio.position("AAPL").expect("AAPL position").avg_price,
        100.0,
    );
    assert_close(portfolio.unrealized_pnl(), 100.0);
    assert_close(portfolio.realized_pnl(), 0.0);
    assert_close(portfolio.equity(), 10_099.0);
    assert_close(portfolio.margin_used(), 1_100.0);

    portfolio.apply_fill(&fill(2, "AAPL", -4.0, 120.0, 2.0));
    portfolio.mark_to_market(&[bar("AAPL", 3, 115.0, 90.0)]);

    assert_close(portfolio.cash(), 9_477.0);
    assert_close(portfolio.position("AAPL").expect("AAPL position").size, 6.0);
    assert_close(
        portfolio.position("AAPL").expect("AAPL position").avg_price,
        100.0,
    );
    assert_close(portfolio.realized_pnl(), 80.0);
    assert_close(portfolio.unrealized_pnl(), -60.0);
    assert_close(portfolio.equity(), 10_017.0);
    assert_close(portfolio.margin_used(), 540.0);
}

#[test]
fn portfolio_aggregates_multiple_assets_and_trade_on_close_fills() {
    let mut portfolio = Portfolio::new(5_000.0);
    let aapl_fill = match_order(
        &market_order(1, "AAPL", OrderSide::Buy, 10.0),
        &bar("AAPL", 2, 100.0, 103.0),
        &close_options(),
    )
    .expect("AAPL close fill");
    let msft_fill = match_order(
        &market_order(2, "MSFT", OrderSide::Buy, 5.0),
        &bar("MSFT", 2, 200.0, 198.0),
        &close_options(),
    )
    .expect("MSFT close fill");

    portfolio.apply_fill(&aapl_fill);
    portfolio.apply_fill(&msft_fill);
    portfolio.mark_to_market(&[bar("AAPL", 3, 104.0, 110.0), bar("MSFT", 3, 199.0, 190.0)]);

    assert_close(aapl_fill.price, 103.0);
    assert_close(msft_fill.price, 198.0);
    assert_close(portfolio.cash(), 2_980.0);
    assert_close(
        portfolio.position("AAPL").expect("AAPL position").avg_price,
        103.0,
    );
    assert_close(
        portfolio.position("MSFT").expect("MSFT position").avg_price,
        198.0,
    );
    assert_close(portfolio.unrealized_pnl(), 30.0);
    assert_close(portfolio.equity(), 5_030.0);
    assert_close(portfolio.margin_used(), 2_050.0);
}
