use _rust::core::{BarBatch, BarEvent, CallbackBatcher};

fn bar(ts: i64, symbol: &str, close: f64) -> BarEvent {
    BarEvent {
        ts,
        symbol: symbol.to_string(),
        open: close - 1.0,
        high: close + 1.0,
        low: close - 2.0,
        close,
        volume: 1000.0 + close,
    }
}

#[test]
fn bar_batch_uses_arrow_ready_columns_and_reconstructs_rows() {
    let bars = vec![bar(1, "AAPL", 10.0), bar(2, "MSFT", 20.0)];

    let batch = BarBatch::from_bars(&bars);

    assert_eq!(batch.len(), 2);
    assert_eq!(batch.ts, vec![1, 2]);
    assert_eq!(batch.symbol, vec!["AAPL".to_string(), "MSFT".to_string()]);
    assert_eq!(batch.open, vec![9.0, 19.0]);
    assert_eq!(batch.high, vec![11.0, 21.0]);
    assert_eq!(batch.low, vec![8.0, 18.0]);
    assert_eq!(batch.close, vec![10.0, 20.0]);
    assert_eq!(batch.volume, vec![1010.0, 1020.0]);
    assert_eq!(batch.bar(1), Some(bars[1].clone()));
    assert_eq!(batch.bar(2), None);
}

#[test]
fn callback_batcher_emits_full_batches_and_flushes_tail_in_order() {
    let mut batcher = CallbackBatcher::new(2);

    assert!(batcher.push_bar(bar(1, "AAPL", 10.0)).is_none());
    let first = batcher.push_bar(bar(2, "MSFT", 20.0)).expect("full batch");
    assert_eq!(first.sequence, 0);
    assert_eq!(first.bars.ts, vec![1, 2]);
    assert_eq!(
        first.bars.symbol,
        vec!["AAPL".to_string(), "MSFT".to_string()]
    );
    assert_eq!(batcher.pending_len(), 0);

    assert!(batcher.push_bar(bar(3, "NVDA", 30.0)).is_none());
    let tail = batcher.flush().expect("tail batch");
    assert_eq!(tail.sequence, 1);
    assert_eq!(tail.bars.ts, vec![3]);
    assert!(batcher.flush().is_none());
}

#[test]
fn callback_batcher_treats_zero_as_single_bar_mode() {
    let mut batcher = CallbackBatcher::new(0);

    assert_eq!(batcher.callback_batch(), 1);
    let batch = batcher
        .push_bar(bar(1, "AAPL", 10.0))
        .expect("single bar batch");
    assert_eq!(batch.sequence, 0);
    assert_eq!(batch.bars.len(), 1);
    assert_eq!(batch.bars.close, vec![10.0]);
}
