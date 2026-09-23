# PR #4 correction and release verification

The original PR commits are retained. Corrections preserve the shared Engine/Lite runtime and existing public order constants. Rust owns matching eligibility, explicit OCO exclusion, expiry and portfolio transitions; all runners use a common command bridge. Python normalizes metadata and delivers guarded terminal notifications.

Regression coverage includes cancellation before/after binding and inside Submitted notifications, parent cancellation, deferred market children, expiry callbacks that submit replacements, same-batch cancel/fill races, exact/smart OCO, chained OCO IDs, trailing limits and watermarks, asynchronous secondary bars, and Python runner fallback. `tag` and `info` survive order artifacts.

Existing local trade accounting, rejected target handling, indicator comparisons and benchmark fixture loading corrections are preserved in the release. Unrelated local uncommitted strategy, data, metrics and operational work is not included in this focused release and must be preserved when updating the local checkout.

Verified on macOS arm64 / Python 3.12:

- Backtest/core/indicator regressions and focused Backtrader oracle: 422 passed, 1 skipped (optional design document absent).
- Full golden release gate: 49 tests passed, 50 historical comparisons passed.
- Final historical comparison rerun after indicator integration: 50 passed.
- Rust formatting and Python F/E9 static checks passed.
- Optimized CPython 3.12 wheel and source distribution built as 0.2.6.
- Installed wheel outside source checkout passed native version, stop-price, cancellation, expiry and OCO smoke tests.
- Final wheel Python modules were byte-compared with the verified sources.
- cibuildwheel now runs the native lifecycle smoke on every release wheel.

Cross-platform builds and publishing are performed by the existing tag-triggered GitHub Release workflow.
