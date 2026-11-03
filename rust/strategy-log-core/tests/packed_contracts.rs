#![cfg(feature = "diagnostics")]
#[allow(dead_code)]
#[path = "../../../../../../监控连接器实验/商用外放/diagnostics/packed_core.rs"]
mod packed;
use packed as core_under_test;
include!("support/contracts.rs");
