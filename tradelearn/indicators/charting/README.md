# Charting study catalog

`catalog()` returns the 104 names captured from the local charting menu plus
Smart Money Concepts (105 entries), stable IDs,
parameter schemas, plot IDs/types, overlay flags, and semantic notes.
`calculate(frame, name_or_id, params=None, secondary=None, visible_range=None)`
returns JSON-safe series of the input length, profile bins, or SMC drawings. Unknown parameters,
invalid ranges, duplicate/unsorted timestamps, invalid OHLCV and nonfinite inputs
are rejected. UTC timestamps are retained externally. Secondary bars align by
exact timestamp with null output for missing pairs; no forward filling.

`catalog.json` is the portable frontend schema. Native PyneCore primitives run
through `tradelearn.indicators.tv`; compositions are in `formulas.py`, with causal
event/profile logic in `special.py`. Computation is serialized to protect native
PyneCore state. Native wrapper initialization rules apply to warmup.

## Formula evidence and scope

The primary source for otherwise ambiguous study names was the locally supplied
TradingView library `client/public/charting_library/bundles/library.6738f23786f3f834acd7.js`
in the Charting repository. Its per-study constructors establish, for example:
MA Double/Triple/Multiple mean independent MA curves; Hamming uses sine weights;
Adaptive MA uses a log-return volatility factor; Klinger uses signed-volume EMA
differences; Correlation - Log correlates log returns; Advance/Decline counts
up/down candle bodies; Volatility Index is Wilder's ATR trailing stop.
This package independently expresses those formulas in pandas and native TV
primitives. It does not claim bit-for-bit numerical parity with the proprietary
charting runtime or all defaults/visual controls of its UI.

Deliberate differences and limits:

* All calculations start at the first supplied bar. Cumulative values, EMA seeds,
  and recursive stops depend on supplied history. Warmup/undefined ratios return
  null; more history can change previously displayed values after a reload.
* Defaults and adjustable parameters are those in the catalog, not a promise of
  identical legacy defaults. Visual offsets and some fixed conventional periods
  are not editable. Schema only exposes parameters used by the calculation.
* VWAP and pivots use UTC day/week/month boundaries, not exchange sessions.
  VWAP bands use cumulative volume-weighted population variance.
* Historical Volatility uses explicit annualization. Zero-trend and OHLC
  volatility account for actual elapsed days; OHLC supports market-closed weight.
* Standard Error Bands smooth the regression curve and its residual-error bands.
* Chande Kroll uses the canonical lowest-low short stop; the bundled script
  unexpectedly uses lowest-high. Volatility Index initializes at length bars and
  advances even when consecutive closes match, unlike legacy suppression.
* Fractals and close-based percentage Zig Zag emit confirmed extremes at the
  confirmation timestamp; they never backdate or repaint earlier bars.
* Ichimoku plots conversion/base and forward-shifted available cloud spans. The
  lagging plot carries current closes and descriptor `offset: -25` for rendering;
  calculation does not read future bars.
* Profile volume is allocated uniformly across each OHLCV bar's high-low range;
  it is not tick-level/price-level trade volume. Fixed range uses the last `bars`
  bars; visible range honors supplied timestamp boundaries. Up/down split and
  value-area annotations are not fabricated.
* Chop Zone returns constant-height histogram values and a separate 0–8 color
  class for native chart rendering.

Verification: all original 104 entries have smoke/JSON-schema and prefix-causality coverage,
plus numerical checks for SMA, regression slope, advance/decline, PVT, VWAP,
previous-period pivots, secondary alignment and volume conservation. These checks
validate selected formulas and invariants, not exhaustive external parity.

## Smart Money Concepts

`calculate(frame, "smart-money-concepts", {"depth": 5})` adapts the existing
`tradelearn.indicators.smc.SMCAnalyzer` without changing its algorithm. It returns
`series: []` and `drawings` with `levels`, `bosLL`, `bosHH`, `zoneD`, `zoneS`,
`impLL`, and `impHH`. Times are UTC epoch seconds. Depth is an integer from 1 to
1000; this study accepts at most 5000 bars. The caller supplies completed bars
and truncates them before any historical replay cutoff. No final bar is removed
inside the package.

These are snapshot drawings, not per-bar strategy signals. Swing geometry is
anchored at the original pivot; `confirmedAt` is its later confirmation time,
and `provisional: true` marks a mutable terminal pivot. Zones can extend to the
last supplied bar, then stop on mitigation in a later snapshot. Recompute on a
closed-bar prefix for replay; filtering drawings calculated from a full future
window cannot reconstruct an earlier snapshot. Geometric BOS/imbalance origins
are likewise not a claim that the event was known at the origin.

Tests check all seven drawing groups against the existing analyzer, confirmed
pivot prefix stability, future-data isolation, output time bounds, empty/short
history and parameter/input limits.
