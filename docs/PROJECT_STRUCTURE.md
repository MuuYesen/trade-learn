# Project Structure

This repository is organized around a small shared event-driven core, facade-specific compatibility layers, and separate indicator families.

## Runtime Code

- `tradelearn/backtest/core/`: shared backtest runtime primitives only.
  - Owns the event loop, broker-neutral models, core strategy base, shared bar buffers, line primitives, indicator cache plumbing, and paper/live extension interfaces.
  - Must not import `tradelearn.compat.*`.
- `tradelearn/compat/backtrader/`: Backtrader-compatible facade.
  - Owns Backtrader context, metaclass behavior, strategy/sizer/analyzer shims, Backtrader-style data feeds, and Backtrader indicator aliases.
- `tradelearn/compat/backtesting/`: backtesting.py-compatible facade.
  - Owns `Backtest`, `Strategy.I(...)`, backtesting.py data proxies, position proxies, and facade-specific stats behavior.
- `tradelearn/core/`: event and streaming primitives shared by backtest, paper, and live style runners.
- `tradelearn/indicators/`: indicator integrations.
  - `core/`: pandas-ta-classic backed common indicators and cache adapters.
  - `tdx/`: TDX indicator family.
  - `tv/`: TradingView-style indicator family.
- `rust/tradelearn-rust/`: production Rust execution engine and PyO3 bindings.

## Tests And Benchmarks

- `tests/unit/`: focused unit tests by package area.
- `tests/consistency/`: cross-engine and cross-facade consistency checks.
- `tests/golden/`: golden datasets, strategies, expected values, and return fixtures.
- `tests/runners/`: executable benchmark/audit runners, including `benchmark_bt.py`.
- `benchmarks/`: saved benchmark baselines and migration blocker snapshots.

## Examples And Documentation

- `examples/backtrader/`: Backtrader facade strategy examples.
- `examples/backtesting/`: backtesting.py facade strategy examples.
- `tests/runners/`: executable benchmark/audit runners, including backtesting.py and Backtrader parity scripts.
- `benchmarks/data/`: datasets used by benchmark and parity runners.
- `docs/`: public architecture, compatibility, progress, and runbook documents.
- `docs/internal/`: internal design notes.
- `docs/release/`: release evaluation notes.

## Local-Only Areas

- `scratch/`: local debugging probes. Existing tracked checkpoints can remain, but new ad-hoc probes are ignored by default.
- `reference/`: local upstream reference checkouts, ignored by git.
- `.venv/`, `target/`, `.pytest_cache/`, `.ruff_cache/`: generated environment, build, and cache directories.

## Boundary Rules

- Core code may depend on `tradelearn.core`, `tradelearn.backtest.core`, and Rust bindings, but not on compatibility facades.
- Backtrader-only behavior belongs under `tradelearn/compat/backtrader`.
- backtesting.py-only behavior belongs under `tradelearn/compat/backtesting`.
- Generic indicator cache infrastructure can live in core; indicator formula ownership stays in pandas-ta-classic, TDX, or TradingView integration modules.
- Concrete broker adapters such as QMT should remain outside git until intentionally promoted; keep only generic extension interfaces in shared modules.
