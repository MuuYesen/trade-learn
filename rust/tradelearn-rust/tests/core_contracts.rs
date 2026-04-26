use _rust::core::{
    BarEvent, Broker, CancelEvent, DataFeed, Event, EventQueue, FillEvent, OrderEvent, OrderSide,
    OrderType, RejectEvent,
};

#[derive(Default)]
struct RecordingBroker {
    submitted: Vec<OrderEvent>,
}

impl Broker for RecordingBroker {
    fn submit_order(&mut self, order: OrderEvent) -> Event {
        self.submitted.push(order.clone());
        Event::Order(order)
    }
}

struct VecFeed {
    bars: Vec<BarEvent>,
}

impl DataFeed for VecFeed {
    fn next_bar(&mut self) -> Option<BarEvent> {
        if self.bars.is_empty() {
            None
        } else {
            Some(self.bars.remove(0))
        }
    }
}

fn bar(ts: i64, symbol: &str) -> BarEvent {
    BarEvent {
        ts,
        symbol: symbol.to_string(),
        open: 10.0,
        high: 12.0,
        low: 9.0,
        close: 11.0,
        volume: 1000.0,
    }
}

fn order(order_id: u64, ts: i64) -> OrderEvent {
    OrderEvent {
        order_id,
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Limit,
        size: 100.0,
        limit_price: Some(10.5),
        stop_price: None,
        created_ts: ts,
    }
}

#[test]
fn event_contracts_cover_stage3_week1_variants() {
    let bar_event = Event::Bar(bar(1, "AAPL"));
    let order_event = Event::Order(order(42, 1));
    let fill_event = Event::Fill(FillEvent {
        order_id: 42,
        symbol: "AAPL".to_string(),
        size: 100.0,
        price: 10.5,
        commission: 1.0,
        slippage: 0.0,
        ts: 2,
    });
    let cancel_event = Event::Cancel(CancelEvent {
        order_id: 42,
        reason: "expired".to_string(),
    });
    let reject_event = Event::Reject(RejectEvent {
        order_id: 43,
        reason: "insufficient cash".to_string(),
    });

    assert!(matches!(bar_event, Event::Bar(_)));
    assert!(matches!(order_event, Event::Order(_)));
    assert!(matches!(fill_event, Event::Fill(_)));
    assert!(matches!(cancel_event, Event::Cancel(_)));
    assert!(matches!(reject_event, Event::Reject(_)));
}

#[test]
fn broker_and_data_feed_traits_are_implementable() {
    let mut broker = RecordingBroker::default();
    let submitted = broker.submit_order(order(7, 1));
    assert!(matches!(submitted, Event::Order(_)));
    assert_eq!(broker.submitted.len(), 1);

    let mut feed = VecFeed {
        bars: vec![bar(1, "AAPL"), bar(2, "MSFT")],
    };
    assert_eq!(
        feed.next_bar().map(|event| event.symbol),
        Some("AAPL".to_string())
    );
    assert_eq!(
        feed.next_bar().map(|event| event.symbol),
        Some("MSFT".to_string())
    );
    assert!(feed.next_bar().is_none());
}

#[test]
fn event_queue_orders_by_timestamp_then_insertion_order() {
    let mut queue = EventQueue::default();
    queue.push(2, Event::Order(order(2, 2)));
    queue.push(1, Event::Bar(bar(1, "AAPL")));
    queue.push(2, Event::Order(order(3, 2)));

    assert_eq!(queue.len(), 3);
    assert_eq!(queue.pop_next().map(|item| item.timestamp), Some(1));
    assert_eq!(queue.pop_next().map(|item| item.sequence), Some(0));
    assert_eq!(queue.pop_next().map(|item| item.sequence), Some(2));
    assert!(queue.pop_next().is_none());
}
