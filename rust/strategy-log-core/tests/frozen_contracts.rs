#![cfg(feature = "diagnostics")]
#[allow(dead_code)]
#[path = "../../../../../../监控连接器实验/商用外放/docs/source-snapshot/core-v1.rs"]
mod frozen;
use frozen as core_under_test;
include!("support/contracts.rs");
