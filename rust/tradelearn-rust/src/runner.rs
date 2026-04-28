use crate::types::*;

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
