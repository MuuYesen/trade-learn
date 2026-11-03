//! Shared durable recorder; framework adaptation stays in the thin bridges.
pub mod recorder;

// Historical performance tools are opt-in diagnostics, never a default SDK writer.
#[cfg(feature = "diagnostics")]
mod diagnostics;
#[cfg(feature = "diagnostics")]
pub use diagnostics::*;
