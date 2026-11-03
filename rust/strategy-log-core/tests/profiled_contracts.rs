#![cfg(feature = "diagnostics")]
use leafquant_external_log_core::{Failure, Level, LogOwner, Sink};
use std::{
    fmt,
    io::{self, Write},
    sync::{Arc, Condvar, Mutex},
};
#[derive(Default)]
struct Memory(Vec<u8>);
impl Write for Memory {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Sink for Memory {
    fn sync_all(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn profiled_success_stage_sum_and_closed_rejection() {
    let (owner, tx) = LogOwner::start(Memory::default(), 8).unwrap();
    let p = tx
        .try_log_profiled(Level::Info, "profile", 123, "exact")
        .unwrap();
    assert_eq!(
        p.internal_total_ns,
        p.admission_ns + p.channel_ns + p.release_ns
    );
    assert_eq!(owner.finish().unwrap().sink.0, b"123|INFO|profile|exact\n");
    assert_eq!(
        tx.try_log_profiled(Level::Info, "profile", 124, "late")
            .unwrap_err(),
        Failure::Closed
    );
}
struct Gated(Arc<(Mutex<(bool, bool)>, Condvar)>);
impl fmt::Display for Gated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lock, cv) = &*self.0;
        let mut state = lock.lock().unwrap();
        state.0 = true;
        cv.notify_all();
        let _guard = cv.wait_while(state, |s| !s.1).unwrap();
        write!(f, "gated")
    }
}
#[test]
fn profiled_full_is_sticky_and_does_not_lose_accepted_tail() {
    let gate = Arc::new((Mutex::new((false, false)), Condvar::new()));
    let (owner, tx) = LogOwner::start(Memory::default(), 1).unwrap();
    tx.try_log_profiled(Level::Info, "profile", 1, Gated(gate.clone()))
        .unwrap();
    let (lock, cv) = &*gate;
    let guard = lock.lock().unwrap();
    let (guard, timed) = cv
        .wait_timeout_while(guard, std::time::Duration::from_secs(2), |s| !s.0)
        .unwrap();
    assert!(!timed.timed_out());
    drop(guard);
    tx.try_log_profiled(Level::Info, "profile", 2, Gated(gate.clone()))
        .unwrap();
    assert_eq!(
        tx.try_log_profiled(Level::Info, "profile", 3, Gated(gate.clone()))
            .unwrap_err(),
        Failure::Full
    );
    lock.lock().unwrap().1 = true;
    cv.notify_all();
    assert_eq!(
        tx.try_log_profiled(Level::Info, "profile", 4, Gated(gate.clone()))
            .unwrap_err(),
        Failure::Full
    );
    let result = owner.finish().err().unwrap();
    assert_eq!(result.failure, Failure::Full);
    assert_eq!(result.records, 2);
    assert_eq!(
        result.sink.unwrap().0,
        b"1|INFO|profile|gated\n2|INFO|profile|gated\n"
    );
}
struct PanicDisplay;
impl fmt::Display for PanicDisplay {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("profile formatter panic")
    }
}
#[test]
fn profiled_formatter_panic_remains_visible_and_sticky() {
    let (owner, tx) = LogOwner::start(Memory::default(), 8).unwrap();
    tx.try_log_profiled(Level::Info, "profile", 1, PanicDisplay)
        .unwrap();
    assert_eq!(owner.finish().err().unwrap().failure, Failure::Format);
    assert_eq!(
        tx.try_log_profiled(Level::Info, "profile", 2, PanicDisplay)
            .unwrap_err(),
        Failure::Format
    );
}
struct Broken;
impl Write for Broken {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("profile write fault"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Sink for Broken {
    fn sync_all(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn profiled_writer_error_is_not_success() {
    let (owner, tx) = LogOwner::start(Broken, 8).unwrap();
    tx.try_log_profiled(Level::Info, "profile", 1, "row")
        .unwrap();
    assert_eq!(owner.finish().err().unwrap().failure, Failure::Write);
    assert_eq!(tx.check(), Err(Failure::Write));
}
