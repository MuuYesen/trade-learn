//! Owned-event logging core. No framework, network, global logger, or Python dependency.
//! Enqueue success means accepted into memory; only `LogOwner::finish` confirms persistence.
use std::{
    fmt,
    fs::File,
    io::{self, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        mpsc::{self, SyncSender},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Failure {
    Full = 1,
    Disconnected,
    Write,
    Flush,
    Sync,
    Format,
    WorkerPanic,
    Closed,
    InvalidCapacity,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Failure {}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Info,
    Warn,
    Error,
}
impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        })
    }
}
/// A single owned sink. `flush` and `sync_all` are distinct completion barriers.
pub trait Sink: Write + Send + 'static {
    fn sync_all(&mut self) -> io::Result<()>;
}
impl Sink for File {
    fn sync_all(&mut self) -> io::Result<()> {
        File::sync_all(self)
    }
}
// Highest bit is CLOSED; the remaining bits count admitted producer calls.
const CLOSED: usize = 1usize << (usize::BITS - 1);
const ACTIVE_MASK: usize = CLOSED - 1;
struct State {
    failure: AtomicU8,
    admission: AtomicUsize,
    stopping: AtomicBool,
}
impl State {
    fn fail(&self, failure: Failure) {
        let _ =
            self.failure
                .compare_exchange(0, failure as u8, Ordering::AcqRel, Ordering::Acquire);
    }
    fn check(&self) -> Result<(), Failure> {
        match self.failure.load(Ordering::Acquire) {
            0 => Ok(()),
            1 => Err(Failure::Full),
            2 => Err(Failure::Disconnected),
            3 => Err(Failure::Write),
            4 => Err(Failure::Flush),
            5 => Err(Failure::Sync),
            6 => Err(Failure::Format),
            7 => Err(Failure::WorkerPanic),
            _ => unreachable!("internal failure code"),
        }
    }
    fn enter(&self) -> Result<Active<'_>, Failure> {
        let mut observed = self.admission.load(Ordering::Acquire);
        loop {
            if observed & CLOSED != 0 {
                return Err(Failure::Closed);
            }
            // Saturation is not a normal queue-full condition, but must fail explicitly
            // rather than incrementing into CLOSED or wrapping the count.
            if observed & ACTIVE_MASK == ACTIVE_MASK {
                self.fail(Failure::Full);
                return Err(Failure::Full);
            }
            match self.admission.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(Active(&self.admission)),
                Err(current) => observed = current,
            }
        }
    }
    fn close(&self) {
        // fetch_or is the close linearization point in the same modification order
        // as admission CAS. Subtracting an active count can never clear CLOSED.
        self.admission.fetch_or(CLOSED, Ordering::AcqRel);
        while self.admission.load(Ordering::Acquire) & ACTIVE_MASK != 0 {
            thread::yield_now();
        }
        self.stopping.store(true, Ordering::Release);
    }
}
struct Envelope<T> {
    level: Level,
    component: &'static str,
    source_ns: u64,
    payload: T,
}
/// Cloneable producer. Capacity bounds queued records, excluding one worker in-flight record.
pub struct Producer<T> {
    tx: SyncSender<Envelope<T>>,
    state: Arc<State>,
}
impl<T> Clone for Producer<T> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            state: self.state.clone(),
        }
    }
}
impl<T> fmt::Debug for Producer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Producer").finish_non_exhaustive()
    }
}
struct Active<'a>(&'a AtomicUsize);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
impl<T: fmt::Display + Send + 'static> Producer<T> {
    /// Moves payload without rendering it. Caller supplies source/producer time in nanoseconds.
    /// Full/disconnected failures are sticky; there is no fallback or silent overwrite.
    pub fn try_log(
        &self,
        level: Level,
        component: &'static str,
        source_ns: u64,
        payload: T,
    ) -> Result<(), Failure> {
        self.check()?;
        let _active = self.state.enter()?;
        match self.tx.try_send(Envelope {
            level,
            component,
            source_ns,
            payload,
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.state.fail(Failure::Full);
                Err(Failure::Full)
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.state.fail(Failure::Disconnected);
                Err(Failure::Disconnected)
            }
        }
    }
    /// Instrumented diagnostic entry point. Extra Instant reads perturb latency;
    /// it is not a replacement for normal try_log performance measurement.
    #[cfg(feature = "diagnostics")]
    pub fn try_log_profiled(
        &self,
        level: Level,
        component: &'static str,
        source_ns: u64,
        payload: T,
    ) -> Result<EnqueueProfile, Failure> {
        let t0 = std::time::Instant::now();
        self.check()?;
        let active = self.state.enter()?;
        let t1 = std::time::Instant::now();
        let result = match self.tx.try_send(Envelope {
            level,
            component,
            source_ns,
            payload,
        }) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => {
                self.state.fail(Failure::Full);
                Err(Failure::Full)
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.state.fail(Failure::Disconnected);
                Err(Failure::Disconnected)
            }
        };
        let t2 = std::time::Instant::now();
        drop(active);
        let t3 = std::time::Instant::now();
        // Convert all markers to one t0-relative integer-nanosecond coordinate,
        // then subtract offsets so conversion rounding cannot break additivity.
        let admission_ns = t1.duration_since(t0).as_nanos();
        let through_channel_ns = t2.duration_since(t0).as_nanos();
        let internal_total_ns = t3.duration_since(t0).as_nanos();
        let profile = EnqueueProfile {
            admission_ns,
            channel_ns: through_channel_ns - admission_ns,
            release_ns: internal_total_ns - through_channel_ns,
            internal_total_ns,
        };
        debug_assert_eq!(
            profile.internal_total_ns,
            profile.admission_ns + profile.channel_ns + profile.release_ns
        );
        result.map(|()| profile)
    }
    pub fn check(&self) -> Result<(), Failure> {
        self.state.check()
    }
}
/// Nanosecond diagnostic intervals between four Instant markers. Each later
/// marker's sampling overhead is attributed to the preceding interval.
#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, Copy)]
pub struct EnqueueProfile {
    pub admission_ns: u128,
    pub channel_ns: u128,
    pub release_ns: u128,
    pub internal_total_ns: u128,
}
pub struct Finished<W> {
    pub sink: W,
    pub records: u64,
}
pub struct FinishFailure<W> {
    pub failure: Failure,
    pub detail: Option<String>,
    pub records: u64,
    pub sink: Option<W>,
}
impl<W> fmt::Debug for FinishFailure<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FinishFailure")
            .field("failure", &self.failure)
            .field("detail", &self.detail)
            .field("records", &self.records)
            .finish_non_exhaustive()
    }
}
struct WorkerReturn<W> {
    sink: W,
    records: u64,
    detail: Option<String>,
}
pub struct LogOwner<W: Sink> {
    state: Arc<State>,
    worker: Option<JoinHandle<WorkerReturn<W>>>,
}
impl<W: Sink> LogOwner<W> {
    pub fn start<T: fmt::Display + Send + 'static>(
        mut sink: W,
        capacity: usize,
    ) -> Result<(Self, Producer<T>), Failure> {
        if capacity == 0 {
            return Err(Failure::InvalidCapacity);
        }
        let (tx, rx) = mpsc::sync_channel::<Envelope<T>>(capacity);
        let state = Arc::new(State {
            failure: AtomicU8::new(0),
            admission: AtomicUsize::new(0),
            stopping: AtomicBool::new(false),
        });
        let worker_state = state.clone();
        let worker = thread::Builder::new()
            .name("external-business-log".into())
            .spawn(move || {
                let mut records = 0;
                let mut detail = None;
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    loop {
                        let row = if worker_state.stopping.load(Ordering::Acquire) {
                            match rx.try_recv() {
                                Ok(row) => row,
                                Err(_) => break,
                            }
                        } else {
                            match rx.recv_timeout(Duration::from_millis(2)) {
                                Ok(row) => row,
                                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        };
                        // Catch both a panic and a fmt::Error propagated by Display.
                        let line = match catch_unwind(AssertUnwindSafe(|| {
                            let mut text = String::new();
                            fmt::write(
                                &mut text,
                                format_args!(
                                    "{}|{}|{}|{}\n",
                                    row.source_ns, row.level, row.component, row.payload
                                ),
                            )
                            .map(|()| text)
                        })) {
                            Ok(Ok(line)) => line,
                            other => {
                                worker_state.fail(Failure::Format);
                                detail = Some(
                                    if other.is_err() {
                                        "formatter panicked"
                                    } else {
                                        "formatter returned fmt::Error"
                                    }
                                    .into(),
                                );
                                break;
                            }
                        };
                        if let Err(error) = sink.write_all(line.as_bytes()) {
                            worker_state.fail(Failure::Write);
                            detail = Some(error.to_string());
                            break;
                        }
                        records += 1;
                    }
                    if detail.is_none() {
                        if let Err(error) = sink.flush() {
                            worker_state.fail(Failure::Flush);
                            detail = Some(error.to_string());
                        }
                    }
                    if detail.is_none() {
                        if let Err(error) = sink.sync_all() {
                            worker_state.fail(Failure::Sync);
                            detail = Some(error.to_string());
                        }
                    }
                }));
                if outcome.is_err() {
                    worker_state.fail(Failure::WorkerPanic);
                    detail = Some("sink or worker panicked".into());
                }
                WorkerReturn {
                    sink,
                    records,
                    detail,
                }
            })
            .map_err(|_| Failure::WorkerPanic)?;
        Ok((
            Self {
                state: state.clone(),
                worker: Some(worker),
            },
            Producer { tx, state },
        ))
    }
    /// Stops admission even if cloned producers remain alive, drains accepted records, flushes,
    /// syncs and joins. Caller must stop legitimate producers before invoking this barrier.
    /// A rejected/full record makes finish fail even when all accepted records were persisted.
    pub fn finish(mut self) -> Result<Finished<W>, FinishFailure<W>> {
        self.state.close();
        let result = self.worker.take().expect("owner worker").join();
        match result {
            Ok(output) => match self.state.check() {
                Ok(()) => Ok(Finished {
                    sink: output.sink,
                    records: output.records,
                }),
                Err(failure) => Err(FinishFailure {
                    failure,
                    detail: output.detail,
                    records: output.records,
                    sink: Some(output.sink),
                }),
            },
            Err(_) => {
                self.state.fail(Failure::WorkerPanic);
                Err(FinishFailure {
                    failure: Failure::WorkerPanic,
                    detail: Some("worker panicked outside formatter".into()),
                    records: 0,
                    sink: None,
                })
            }
        }
    }
}
impl<W: Sink> Drop for LogOwner<W> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            self.state.close();
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    fn state(value: usize) -> State {
        State {
            failure: AtomicU8::new(0),
            admission: AtomicUsize::new(value),
            stopping: AtomicBool::new(false),
        }
    }
    #[test]
    fn saturated_count_fails_without_setting_closed_or_wrapping() {
        let state = state(ACTIVE_MASK);
        assert!(matches!(state.enter(), Err(Failure::Full)));
        assert_eq!(state.admission.load(Ordering::Acquire), ACTIVE_MASK);
        assert_eq!(state.check(), Err(Failure::Full));
    }
    #[test]
    fn release_preserves_closed_and_late_entry_is_rejected() {
        let state = state(0);
        let active = state.enter().unwrap();
        state.admission.fetch_or(CLOSED, Ordering::AcqRel);
        drop(active);
        assert_eq!(state.admission.load(Ordering::Acquire), CLOSED);
        assert!(matches!(state.enter(), Err(Failure::Closed)));
    }
}
