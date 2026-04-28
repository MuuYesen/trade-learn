use _rust::core::{
    match_order, match_order_smart, BarEvent, CommissionModel, ExecutionOptions, FixedCommission,
    FixedSlippage, OrderEvent, OrderSide, OrderType, PercentCommission, PercentSlippage,
    SlippageModel,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

fn bar() -> BarEvent {
    BarEvent {
        ts: 2,
        symbol: "AAPL".to_string(),
        open: 10.0,
        high: 12.0,
        low: 8.0,
        close: 11.0,
        volume: 1000.0,
    }
}

fn order(order_type: OrderType, side: OrderSide) -> OrderEvent {
    OrderEvent {
        order_id: 7,
        symbol: "AAPL".to_string(),
        side,
        order_type,
        size: 100.0,
        limit_price: None,
        stop_price: None,
        created_ts: 1,
    }
}

fn options() -> ExecutionOptions {
    ExecutionOptions {
        trade_on_close: false,
        smart_matching: false,
        cheat_on_close: false,
        cheat_on_open: false,
        slip_perc: 0.0,
        slip_fixed: 0.0,
        slip_match: true,
        slip_limit: true,
        slip_out: false,
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0 }),
        commission: CommissionModel::Fixed(FixedCommission { amount: 0.0 }),
        mult: 1.0,
        margin: 1.0,
    }
}

#[test]
fn market_orders_fill_at_open_or_close_with_side_aware_slippage() {
    let mut buy = order(OrderType::Market, OrderSide::Buy);
    let mut sell = order(OrderType::Market, OrderSide::Sell);
    let open_options = ExecutionOptions {
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.25 }),
        ..options()
    };
    let close_options = ExecutionOptions {
        trade_on_close: true,
        slippage: SlippageModel::Percent(PercentSlippage { ratio: 0.10 }),
        ..options()
    };

    let buy_fill = match_order(&buy, &bar(), &open_options).expect("buy market fills");
    let sell_fill = match_order(&sell, &bar(), &open_options).expect("sell market fills");
    buy.order_id = 8;
    sell.order_id = 9;
    let close_buy = match_order(&buy, &bar(), &close_options).expect("close buy fills");
    let close_sell = match_order(&sell, &bar(), &close_options).expect("close sell fills");

    assert_close(buy_fill.price, 10.25);
    assert_close(sell_fill.price, 9.75);
    assert_close(close_buy.price, 12.1);
    assert_close(close_sell.price, 9.9);
}

#[test]
fn limit_orders_fill_only_when_bar_crosses_limit_price() {
    let mut buy = order(OrderType::Limit, OrderSide::Buy);
    buy.limit_price = Some(9.0);
    let mut sell = order(OrderType::Limit, OrderSide::Sell);
    sell.limit_price = Some(11.0);
    let mut missed_buy = order(OrderType::Limit, OrderSide::Buy);
    missed_buy.limit_price = Some(7.5);
    let mut missed_sell = order(OrderType::Limit, OrderSide::Sell);
    missed_sell.limit_price = Some(12.5);

    let buy_fill = match_order(&buy, &bar(), &options()).expect("buy limit fills");
    let sell_fill = match_order(&sell, &bar(), &options()).expect("sell limit fills");

    assert_eq!(buy_fill.price, 9.0);
    assert_eq!(sell_fill.price, 11.0);
    assert!(match_order(&missed_buy, &bar(), &options()).is_none());
    assert!(match_order(&missed_sell, &bar(), &options()).is_none());
}

#[test]
fn stop_orders_trigger_to_market_and_stop_limit_requires_both_prices() {
    let mut stop_buy = order(OrderType::Stop, OrderSide::Buy);
    stop_buy.stop_price = Some(11.5);
    let mut stop_sell = order(OrderType::Stop, OrderSide::Sell);
    stop_sell.stop_price = Some(8.5);
    let mut missed_stop = order(OrderType::Stop, OrderSide::Buy);
    missed_stop.stop_price = Some(12.5);
    let mut stop_limit = order(OrderType::StopLimit, OrderSide::Buy);
    stop_limit.stop_price = Some(11.5);
    stop_limit.limit_price = Some(9.5);

    let stop_buy_fill = match_order(&stop_buy, &bar(), &options()).expect("buy stop fills");
    let stop_sell_fill = match_order(&stop_sell, &bar(), &options()).expect("sell stop fills");
    let stop_limit_fill = match_order(&stop_limit, &bar(), &options()).expect("stop limit fills");

    assert_eq!(stop_buy_fill.price, 10.0);
    assert_eq!(stop_sell_fill.price, 10.0);
    assert_eq!(stop_limit_fill.price, 9.5);
    assert!(match_order(&missed_stop, &bar(), &options()).is_none());
}

#[test]
fn commission_models_are_applied_to_fill_events() {
    let order = order(OrderType::Market, OrderSide::Buy);
    let fixed = ExecutionOptions {
        commission: CommissionModel::Fixed(FixedCommission { amount: 5.0 }),
        ..options()
    };
    let percent = ExecutionOptions {
        commission: CommissionModel::Percent(PercentCommission { ratio: 0.001 }),
        ..options()
    };

    let fixed_fill = match_order(&order, &bar(), &fixed).expect("fixed commission fills");
    let percent_fill = match_order(&order, &bar(), &percent).expect("percent commission fills");

    assert_eq!(fixed_fill.commission, 5.0);
    assert_eq!(percent_fill.commission, 1.0);
}

#[test]
fn fill_precision_is_frozen_after_slippage_before_commission() {
    let mut order = order(OrderType::Market, OrderSide::Buy);
    order.size = 3.333333333;
    let bar = BarEvent {
        open: 10.0000004,
        ..bar()
    };
    let options = ExecutionOptions {
        slippage: SlippageModel::Fixed(FixedSlippage { amount: 0.0000004 }),
        commission: CommissionModel::Percent(PercentCommission { ratio: 0.001 }),
        ..options()
    };

    let fill = match_order(&order, &bar, &options).expect("market fills");

    assert_eq!(fill.size, 3.333333);
    assert_eq!(fill.price, 10.000001);
    assert_eq!(fill.slippage, 0.000001);
    assert_eq!(fill.commission, 0.033333);
}

#[test]
fn smart_matching_follows_bullish_open_low_high_close_path() {
    let mut buy_limit = order(OrderType::Limit, OrderSide::Buy);
    buy_limit.limit_price = Some(9.0);
    let mut sell_limit = order(OrderType::Limit, OrderSide::Sell);
    sell_limit.limit_price = Some(11.0);
    let mut sell_stop = order(OrderType::Stop, OrderSide::Sell);
    sell_stop.stop_price = Some(8.5);
    let mut buy_stop = order(OrderType::Stop, OrderSide::Buy);
    buy_stop.stop_price = Some(11.5);

    let buy_limit_fill =
        match_order_smart(&buy_limit, &bar(), &options()).expect("buy limit fills on low leg");
    let sell_limit_fill =
        match_order_smart(&sell_limit, &bar(), &options()).expect("sell limit fills on high leg");
    let sell_stop_fill =
        match_order_smart(&sell_stop, &bar(), &options()).expect("sell stop fills on low leg");
    let buy_stop_fill =
        match_order_smart(&buy_stop, &bar(), &options()).expect("buy stop fills on high leg");

    assert_eq!(buy_limit_fill.price, 9.0);
    assert_eq!(sell_limit_fill.price, 11.0);
    assert_eq!(sell_stop_fill.price, 8.5);
    assert_eq!(buy_stop_fill.price, 11.5);
}
