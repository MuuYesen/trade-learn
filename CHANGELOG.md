# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-05-05

### Added
- **Intelligent Console Feedback System**: 
    - Implemented `SmartPbar` (tqdm subclass) and `ConsoleState` to manage terminal aesthetics.
    - Added "Smart Padding" logic that automatically ensures exactly one blank line between execution stages (Data -> Causal -> Backtest -> Report).
    - Integrated `smart_print` to synchronize manual output with the console state tracker, preventing redundant or missing newlines.
- **Enhanced Backtest Metrics**:
    - Introduced **SQN (System Quality Number)** for system robustness evaluation.
    - Added **Kelly Criterion** for optimal position sizing suggestions.
    - Integrated **Exposure Time (%)** to track market participation.
    - Expanded drawdown tracking with **Max/Avg Drawdown Duration**.
    - Standardized output to a high-density, single-column tabular format for professional reporting.

### Changed
- **Terminal UI Refinement**:
    - Replaced all legacy `print` and `logging` progress indicators with synchronized `smart_tqdm` bars.
    - Suppressed verbose third-party warnings from `tvDatafeed` for a cleaner research environment.
    - Refactored `Backtest.run` (Rust-core) to support zero-delay progress bar visibility.
- **Reporting Pipeline**:
    - Improved `Reporter` serialization logic to handle `pandas.Timestamp` and `pandas.Timedelta` objects gracefully in JSON outputs.
    - Standardized `CausalSelector` report embedding via `to_section()`.

### Fixed
- Resolved "split progress bar" issues where 0% and 100% states appeared on separate lines due to stream interleaving.
- Fixed JSON serialization errors in HTML report generation for strategies with complex time-based metrics.
- Corrected trade duration calculations by ensuring `dtopen` and `dtclose` are captured across all backtest engines.

## [2.0.0] - 2026-05-04

### Added
- **Trade-Learn 2.0 Official Release**.
- High-performance backtest engine written in Rust.
- Integrated Causal Inference suite for factor selection.
- Modern HTML interactive tear sheets using Bokeh.
- Support for multi-asset cross-sectional backtesting.
