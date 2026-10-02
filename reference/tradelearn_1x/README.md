# Frozen local 1.x oracle

These files are the existing local TradeLearn 1.x reference snapshot, now included
so clean checkouts use the same oracle instead of an unrelated installed package.
They are test-only and excluded from the distributed Python package. Source copyright
and license headers are retained; see the repository NOTICE for attribution.

The exact bytes are pinned by `tests/golden/fixture_checksums.json`. This snapshot
has no verified upstream commit identifier; the manifest establishes reproducibility,
not upstream authenticity. Do not regenerate expected results to fit current code.

The same manifest pins the previously local AAPL TradingView daily fixture for
2020-01-01 through 2024-12-31. It is historical test data, never a live trading feed.
