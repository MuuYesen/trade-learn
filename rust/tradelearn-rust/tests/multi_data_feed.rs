use _rust::core::{BarEvent, MultiDataFeed};

fn bar(ts: i64, symbol: &str, close: f64) -> BarEvent {
    BarEvent {
        ts,
        symbol: symbol.to_string(),
        open: close - 1.0,
        high: close + 1.0,
        low: close - 2.0,
        close,
        volume: 1000.0,
    }
}

#[test]
fn multi_data_feed_merges_bars_by_timestamp_then_registration_order() {
    let mut feed = MultiDataFeed::new();
    let daily = feed.add_feed("daily", vec![bar(1, "AAPL", 10.0), bar(3, "AAPL", 12.0)]);
    let five_min = feed.add_feed("5min", vec![bar(1, "AAPL", 10.5), bar(2, "AAPL", 11.0)]);

    let first = feed.next_bar().expect("first bar");
    let second = feed.next_bar().expect("second bar");
    let third = feed.next_bar().expect("third bar");
    let fourth = feed.next_bar().expect("fourth bar");

    assert_eq!(daily, 0);
    assert_eq!(five_min, 1);
    assert_eq!(
        (first.feed_index, first.feed_name.as_str(), first.bar.ts),
        (0, "daily", 1)
    );
    assert_eq!(
        (second.feed_index, second.feed_name.as_str(), second.bar.ts),
        (1, "5min", 1)
    );
    assert_eq!(
        (third.feed_index, third.feed_name.as_str(), third.bar.ts),
        (1, "5min", 2)
    );
    assert_eq!(
        (fourth.feed_index, fourth.feed_name.as_str(), fourth.bar.ts),
        (0, "daily", 3)
    );
    assert!(feed.next_bar().is_none());
}

#[test]
fn multi_data_feed_skips_empty_feeds_and_preserves_symbol_payload() {
    let mut feed = MultiDataFeed::new();
    feed.add_feed("empty", vec![]);
    feed.add_feed("aapl_daily", vec![bar(5, "AAPL", 20.0)]);
    feed.add_feed("msft_daily", vec![bar(4, "MSFT", 30.0)]);

    let first = feed.next_bar().expect("MSFT comes first");
    let second = feed.next_bar().expect("AAPL comes second");

    assert_eq!(first.feed_name, "msft_daily");
    assert_eq!(first.bar.symbol, "MSFT");
    assert_eq!(first.bar.close, 30.0);
    assert_eq!(second.feed_name, "aapl_daily");
    assert_eq!(second.bar.symbol, "AAPL");
    assert_eq!(second.bar.close, 20.0);
    assert!(feed.next_bar().is_none());
}
