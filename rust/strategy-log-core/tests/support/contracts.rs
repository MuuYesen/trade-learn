use core_under_test::{Failure, Level, LogOwner, Sink};
use std::{
    fmt,
    io::{self, Write},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::Duration,
};
#[derive(Default)]
struct Memory {
    bytes: Vec<u8>,
    operations: Vec<&'static str>,
}
impl Write for Memory {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.operations.push("write");
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.operations.push("flush");
        Ok(())
    }
}
impl Sink for Memory {
    fn sync_all(&mut self) -> io::Result<()> {
        self.operations.push("sync");
        Ok(())
    }
}
#[test]
fn exact_content_source_time_fifo_and_shutdown_with_live_producer() {
    let (owner, tx) = LogOwner::start::<String>(Memory::default(), 16).unwrap();
    tx.try_log(Level::Info, "native", 123, "first".into())
        .unwrap();
    tx.try_log(Level::Warn, "native", 456, "last".into())
        .unwrap();
    let result = owner.finish().unwrap();
    assert_eq!(
        result.sink.bytes,
        b"123|INFO|native|first\n456|WARN|native|last\n"
    );
    assert_eq!(result.records, 2);
    assert_eq!(result.sink.operations.last(), Some(&"sync"));
    assert_eq!(
        tx.try_log(Level::Info, "native", 789, "closed".into()),
        Err(Failure::Closed)
    );
}
struct WorkerOnly(thread::ThreadId);
impl fmt::Display for WorkerOnly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert_ne!(thread::current().id(), self.0);
        write!(f, "worker text")
    }
}
#[test]
fn formatting_is_worker_only() {
    let (owner, tx) = LogOwner::start(Memory::default(), 8).unwrap();
    tx.try_log(Level::Info, "native", 1, WorkerOnly(thread::current().id()))
        .unwrap();
    assert_eq!(
        owner.finish().unwrap().sink.bytes,
        b"1|INFO|native|worker text\n"
    );
}
#[derive(Clone)]
struct Gate(Arc<(Mutex<(bool, bool)>, Condvar)>);
struct Gated(Gate);
impl fmt::Display for Gated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lock, cv) = &*self.0.0;
        let mut state = lock.lock().unwrap();
        state.0 = true;
        cv.notify_all();
        let _guard = cv.wait_while(state, |s| !s.1).unwrap();
        write!(f, "gated")
    }
}
#[test]
fn queue_full_is_sticky_nonblocking_and_accepted_tail_drains() {
    let gate = Gate(Arc::new((Mutex::new((false, false)), Condvar::new())));
    let (owner, tx) = LogOwner::start(Memory::default(), 1).unwrap();
    tx.try_log(Level::Info, "native", 1, Gated(gate.clone()))
        .unwrap();
    let (lock, cv) = &*gate.0;
    let state = lock.lock().unwrap();
    let (_guard, timeout) = cv
        .wait_timeout_while(state, Duration::from_secs(2), |s| !s.0)
        .unwrap();
    assert!(!timeout.timed_out());
    drop(_guard);
    tx.try_log(Level::Info, "native", 2, Gated(gate.clone()))
        .unwrap();
    let at = std::time::Instant::now();
    assert_eq!(
        tx.try_log(Level::Info, "native", 3, Gated(gate.clone())),
        Err(Failure::Full)
    );
    assert!(at.elapsed() < Duration::from_millis(100));
    lock.lock().unwrap().1 = true;
    cv.notify_all();
    assert_eq!(
        tx.try_log(Level::Info, "native", 4, Gated(gate)),
        Err(Failure::Full)
    );
    let failed = owner.finish().err().unwrap();
    assert_eq!(failed.failure, Failure::Full);
    assert_eq!(failed.records, 2);
    assert_eq!(
        failed.sink.unwrap().bytes,
        b"1|INFO|native|gated\n2|INFO|native|gated\n"
    );
}
#[derive(Clone, Copy)]
enum Stage {
    Write,
    Flush,
    Sync,
}
struct Broken(Stage);
impl Write for Broken {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if matches!(self.0, Stage::Write) {
            Err(io::Error::other("write"))
        } else {
            Ok(b.len())
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        if matches!(self.0, Stage::Flush) {
            Err(io::Error::other("flush"))
        } else {
            Ok(())
        }
    }
}
impl Sink for Broken {
    fn sync_all(&mut self) -> io::Result<()> {
        if matches!(self.0, Stage::Sync) {
            Err(io::Error::other("sync"))
        } else {
            Ok(())
        }
    }
}
#[test]
fn write_flush_sync_errors_are_not_success() {
    for (stage, expected) in [
        (Stage::Write, Failure::Write),
        (Stage::Flush, Failure::Flush),
        (Stage::Sync, Failure::Sync),
    ] {
        let (owner, tx) = LogOwner::start(Broken(stage), 8).unwrap();
        tx.try_log(Level::Info, "native", 1, "persist").unwrap();
        assert_eq!(owner.finish().err().unwrap().failure, expected);
        assert_eq!(tx.check(), Err(expected));
    }
}
struct PanicDisplay;
impl fmt::Display for PanicDisplay {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("formatter fault")
    }
}
#[test]
fn formatter_panic_is_sticky_failure() {
    let (owner, tx) = LogOwner::start(Memory::default(), 8).unwrap();
    tx.try_log(Level::Info, "native", 1, PanicDisplay).unwrap();
    assert_eq!(owner.finish().err().unwrap().failure, Failure::Format);
    assert_eq!(tx.check(), Err(Failure::Format));
}
#[test]
fn zero_capacity_rejected() {
    assert!(LogOwner::start::<String>(Memory::default(), 0).is_err());
}
#[test]
fn concurrent_close_accounts_for_every_success_and_rejects_late_calls() {
    for _ in 0..40 {
        let (owner, tx) = LogOwner::start(Memory::default(), 2000).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(5));
        let mut workers = Vec::new();
        for _ in 0..4 {
            let tx = tx.clone();
            let barrier = barrier.clone();
            workers.push(thread::spawn(move || {
                barrier.wait();
                let mut accepted = 0;
                for _ in 0..400 {
                    match tx.try_log(Level::Info, "race", 1, "race") {
                        Ok(()) => accepted += 1,
                        Err(Failure::Closed) => break,
                        other => panic!("unexpected {other:?}"),
                    }
                }
                accepted
            }));
        }
        barrier.wait();
        thread::yield_now();
        let result = owner.finish().unwrap();
        let accepted: u64 = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .sum();
        assert_eq!(result.records, accepted);
        assert_eq!(
            result.sink.bytes.split(|b| *b == b'\n').count() as u64,
            accepted + 1
        );
        assert_eq!(
            tx.try_log(Level::Info, "late", 2, "late"),
            Err(Failure::Closed)
        );
    }
}
struct PanicSink(Stage, Arc<std::sync::atomic::AtomicBool>);
impl Write for PanicSink {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if matches!(self.0, Stage::Write) {
            self.1.store(true, std::sync::atomic::Ordering::Release);
            panic!("sink write panic")
        }
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if matches!(self.0, Stage::Flush) {
            self.1.store(true, std::sync::atomic::Ordering::Release);
            panic!("sink flush panic")
        }
        Ok(())
    }
}
impl Sink for PanicSink {
    fn sync_all(&mut self) -> io::Result<()> {
        if matches!(self.0, Stage::Sync) {
            self.1.store(true, std::sync::atomic::Ordering::Release);
            panic!("sink sync panic")
        }
        Ok(())
    }
}
#[test]
fn sink_panics_become_visible_without_finish() {
    for stage in [Stage::Write, Stage::Flush, Stage::Sync] {
        let triggered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (owner, tx) = LogOwner::start(PanicSink(stage, triggered.clone()), 8).unwrap();
        tx.try_log(Level::Info, "panic", 1, "row").unwrap();
        // Flush/sync happen when the owner's explicit close starts; use another thread so
        // the producer can observe check() before finish() returns to this test.
        if matches!(stage, Stage::Write) {
            let until = std::time::Instant::now() + Duration::from_secs(2);
            while tx.check().is_ok() && std::time::Instant::now() < until {
                thread::yield_now();
            }
            assert_eq!(tx.check(), Err(Failure::WorkerPanic));
            assert_eq!(owner.finish().err().unwrap().failure, Failure::WorkerPanic);
        } else {
            let observer = tx.clone();
            drop(tx);
            let finished = thread::spawn(move || owner.finish().err().unwrap().failure);
            let until = std::time::Instant::now() + Duration::from_secs(2);
            while observer.check().is_ok() && std::time::Instant::now() < until {
                thread::yield_now();
            }
            assert_eq!(observer.check(), Err(Failure::WorkerPanic));
            assert_eq!(finished.join().unwrap(), Failure::WorkerPanic);
        }
        assert!(triggered.load(std::sync::atomic::Ordering::Acquire));
    }
}
struct FormatError;
impl fmt::Display for FormatError {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Err(fmt::Error)
    }
}
#[test]
fn formatter_error_is_failure_without_panic() {
    let (owner, tx) = LogOwner::start(Memory::default(), 8).unwrap();
    tx.try_log(Level::Info, "fmt", 1, FormatError).unwrap();
    assert_eq!(owner.finish().err().unwrap().failure, Failure::Format);
}
#[test]
fn real_file_finish_persists_exact_accepted_content() {
    let path = std::env::temp_dir().join(format!(
        "leafquant-external-core-{}-{}.log",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .unwrap();
    let (owner, tx) = LogOwner::start(file, 8).unwrap();
    tx.try_log(Level::Info, "file", 777, "persisted").unwrap();
    let output = owner.finish().unwrap();
    assert_eq!(output.records, 1);
    drop(output.sink);
    assert_eq!(std::fs::read(&path).unwrap(), b"777|INFO|file|persisted\n");
    std::fs::remove_file(path).unwrap();
}
