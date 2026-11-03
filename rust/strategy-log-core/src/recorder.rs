//! Shared durable JSON recorder. No Python or framework dependency.
#[path = "recorder_platform.rs"]
mod platform;
#[derive(Debug)]
pub enum RecorderError { Invalid(String), Io(std::io::Error), Runtime(String) }
impl std::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Invalid(s) | Self::Runtime(s) => f.write_str(s), Self::Io(e) => e.fmt(f) }
    }
}
impl std::error::Error for RecorderError {}
fn invalid(value: impl Into<String>) -> RecorderError { RecorderError::Invalid(value.into()) }
fn runtime_error(value: impl Into<String>) -> RecorderError { RecorderError::Runtime(value.into()) }

#[cfg(test)]
mod tests {
    use super::*;
    struct TestDir(std::path::PathBuf);
    impl TestDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn testdir() -> TestDir {
        let p = std::env::temp_dir().join(run_id());
        std::fs::create_dir(&p).unwrap();
        TestDir(p)
    }
    #[test]
    fn concurrent_close_drains_and_allows_immediate_reopen() {
        let dir = testdir();
        let r =
            Arc::new(Recorder::new(dir.path().to_str().unwrap().into(), 1024, 128, 16384).unwrap());
        for i in 0..100 {
            assert!(r.append(format!("{{\"n\":{i}}}")));
        }
        let a = r.clone();
        let b = r.clone();
        let t1 = thread::spawn(move || a.barrier(true));
        let t2 = thread::spawn(move || b.barrier(true));
        for result in [t1.join().unwrap(), t2.join().unwrap()] {
            let status: serde_json::Value = serde_json::from_str(&result).unwrap();
            assert_eq!(status["accepted"], 100);
            assert_eq!(status["durable"], 100);
            assert_eq!(status["closed"], true);
        }
        assert!(!r.append("{}".into()));
        let reopened = Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 16384).unwrap();
        assert_ne!(r.shared.run_id, reopened.shared.run_id);
        reopened.barrier(true);
    }
    #[test]
    fn invalid_record_and_capacity_failure_are_visible_and_sticky() {
        let dir = testdir();
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 1, 128, 1024).unwrap();
        assert!(!r.append("[1,2]".into()));
        assert!(!r.append("{}".into()));
        assert!(
            serde_json::from_str::<serde_json::Value>(&r.status()).unwrap()["error"].is_string()
        );
        r.barrier(true);
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 1, 128, 1024).unwrap();
        // Hold the outstanding reservation to deterministically exercise queue saturation.
        r.shared.state.lock().unwrap().outstanding = 1;
        assert!(!r.append("{}".into()));
        assert!(r.status().contains("capacity exhausted"));
        r.barrier(true);
    }
    #[test]
    fn retired_entry_disappearing_during_scan_is_not_storage_failure() {
        let dir = testdir();
        let file = dir.path().join("retired.log");
        std::fs::write(&file, b"sealed").unwrap();
        let entry = std::fs::read_dir(dir.path()).unwrap().next().unwrap().unwrap();
        std::fs::remove_file(file).unwrap();
        assert_eq!(entry_size_if_present(&entry).unwrap(), 0);
    }
    #[test]
    fn worker_retains_first_storage_error_after_later_queued_records() {
        let dir = testdir();
        let lock = std::fs::File::create(dir.path().join("lock")).unwrap();
        let shared = Arc::new(Shared { state: Mutex::new(State {
            error: Some("storage: original error".into()), outstanding: 1, accepted: 1,
            ..State::default()
        }), completion: Condvar::new(), run_id: run_id() });
        let (tx, rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        tx.send(Command::Record("{}".into(), 1, false)).unwrap();
        tx.send(Command::Close(done_tx)).unwrap();
        worker(dir.path().to_path_buf(), rx, shared.clone(), 128, 1024, lock);
        done_rx.recv().unwrap();
        assert_eq!(shared.state.lock().unwrap().error.as_deref(), Some("storage: original error"));
    }
    #[test]
    fn filesystem_failure_does_not_advance_durable() {
        let dir = testdir();
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 1024).unwrap();
        std::fs::remove_dir_all(dir.path()).unwrap();
        assert!(r.append("{}".into()));
        let status: serde_json::Value = serde_json::from_str(&r.barrier(false)).unwrap();
        assert_eq!(status["durable"], 0);
        assert!(status["error"].is_string());
        r.barrier(true);
    }
    #[test]
    fn owned_records_become_sealed_durable_files() {
        let dir = testdir();
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 1024).unwrap();
        assert!(r.append("{\"price\":\"1.00000001\"}".into()));
        let status: serde_json::Value = serde_json::from_str(&r.barrier(false)).unwrap();
        assert_eq!(status["durable"], 1);
        let logs: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|s| s == "log"))
            .collect();
        assert_eq!(logs.len(), 1);
        assert_eq!(
            std::fs::read_to_string(logs[0].path()).unwrap(),
            "{\"price\":\"1.00000001\"}\n"
        );
        r.barrier(true);
    }
    #[test]
    fn disk_limit_is_sticky_and_never_claims_durability() {
        let dir = testdir();
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 2).unwrap();
        assert!(r.append("{}".into()));
        let s: serde_json::Value = serde_json::from_str(&r.barrier(false)).unwrap();
        assert_eq!(s["durable"], 0);
        assert!(s["error"].is_string());
        assert!(!r.append("{}".into()));
        r.barrier(true);
    }
    #[test]
    fn recovery_preserves_sealed_and_unknown_unanchored_pending() {
        let dir = testdir();
        std::fs::write(dir.path().join("old-000.log"), "{}\n").unwrap();
        std::fs::write(dir.path().join("old-001.pending"), "{partial").unwrap();
        let r = Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 1024).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("old-000.log")).unwrap(),
            "{}\n"
        );
        assert_eq!(std::fs::read(dir.path().join("old-001.pending")).unwrap(), b"{partial");
        assert!(Recorder::new(dir.path().to_str().unwrap().into(), 4, 128, 1024).is_err());
        r.barrier(true);
    }
}

use std::{
    fs::{self, File, OpenOptions},
    io::{Read,Write},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

#[derive(Default, Clone)]
struct State {
    accepted: u64,
    durable: u64,
    outstanding: usize,
    error: Option<String>,
    closed: bool,
    segments: Vec<String>,
}
struct Shared {
    state: Mutex<State>,
    completion: Condvar,
    run_id: String,
}
enum Command {
    Record(String, u64, bool),
    Native(Box<dyn FnOnce() -> String + Send>, u64),
    InvalidNative,
    Barrier(mpsc::Sender<()>),
    Close(mpsc::Sender<()>),
}
struct Gate {
    tx: mpsc::Sender<Command>,
    closed: bool,
}
pub struct Recorder {
    shared: Arc<Shared>,
    gate: Mutex<Gate>,
    capacity: usize,
    segment_bytes: usize,
    native_text: bool,
}

impl Recorder {
    pub fn new(
        state_dir: String,
        capacity: usize,
        segment_bytes: usize,
        max_disk_bytes: u64,
    ) -> Result<Self, RecorderError> {
        Self::open(state_dir, capacity, segment_bytes, max_disk_bytes, false, None)
    }
    pub fn new_native(state_dir: String, capacity: usize, segment_bytes: usize, max_disk_bytes: u64) -> Result<Self, RecorderError> {
        Self::open(state_dir, capacity, segment_bytes, max_disk_bytes, true, None)
    }
    pub fn new_windowed(state_dir: String, capacity: usize, segment_bytes: usize, max_disk_bytes: u64, window: Duration) -> Result<Self, RecorderError> {
        if window.is_zero() { return Err(invalid("positive archive window required")); }
        Self::open(state_dir, capacity, segment_bytes, max_disk_bytes, false, Some(window))
    }
    fn open(state_dir: String, capacity: usize, segment_bytes: usize, max_disk_bytes: u64, native_text: bool, window: Option<Duration>) -> Result<Self, RecorderError> {
        if capacity == 0 || segment_bytes < 3 || max_disk_bytes == 0 {
            return Err(invalid(
                "positive capacity/disk and segment_bytes >= 3 required",
            ));
        }
        let segment_bytes = if window.is_some() { segment_bytes.min(4 * 1024 * 1024) } else { segment_bytes };
        let path = PathBuf::from(state_dir);
        if native_text && (!path.is_absolute() || fs::canonicalize(&path).map_err(RecorderError::Io)? != path) {
            return Err(invalid("explicit existing canonical native source directory required"));
        }
        fs::create_dir_all(&path).map_err(RecorderError::Io)?;
        #[cfg(windows)] let path = fs::canonicalize(&path).map_err(RecorderError::Io)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path.join("recorder.lock"))
            .map_err(RecorderError::Io)?;
        lock.try_lock()
            .map_err(|e| runtime_error(format!("state directory already locked: {e}")))?;
        #[cfg(unix)] if native_text {
            use std::os::unix::fs::PermissionsExt;
            let info = fs::symlink_metadata(&path).map_err(RecorderError::Io)?;
            if !info.is_dir() || info.permissions().mode() & 0o077 != 0 {
                return Err(invalid("private existing native source directory required"));
            }
        }
        let pin = path.join("recorder-format.json");
        let expected = if native_text { "native_text.v1" } else { "json_object.v1" };
        if native_text && !pin.exists() {
            for entry in fs::read_dir(&path).map_err(RecorderError::Io)? {
                let entry=entry.map_err(RecorderError::Io)?;
                if entry.path().file_stem().and_then(|s|s.to_str()).is_some_and(legal_segment_stem) {
                    return Err(invalid("existing recorder files without a format pin need explicit migration"));
                }
            }
        }
        match platform::create_new(&pin) {
            Ok(mut file) => {
                #[cfg(unix)] {use std::os::unix::fs::PermissionsExt;file.set_permissions(fs::Permissions::from_mode(0o600)).map_err(RecorderError::Io)?;}
                file.write_all(expected.as_bytes()).map_err(RecorderError::Io)?;
                file.sync_all().map_err(RecorderError::Io)?;
                #[cfg(unix)] platform::sync_directory(&path).map_err(RecorderError::Io)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let info=fs::symlink_metadata(&pin).map_err(RecorderError::Io)?;
                if !info.is_file() || info.len()>64 || fs::read(&pin).map_err(RecorderError::Io)? != expected.as_bytes() {
                    return Err(invalid("recorder format differs; explicit source migration required"));
                }
            }
            Err(error) => return Err(RecorderError::Io(error)),
        }
        if window.is_some() { recover_windowed(&path).map_err(RecorderError::Io)?; }
        let mut segments = Vec::new();
        for entry in fs::read_dir(&path).map_err(RecorderError::Io)? {
            let entry = entry.map_err(RecorderError::Io)?;
            if entry.path().extension().is_some_and(|e| e == "pending") {
                let anchor=entry.path().with_extension("native-time.json");
                let stem=entry.path().file_stem().unwrap().to_string_lossy().into_owned();
                let legal=stem.len()==53 && stem.as_bytes()[32]==b'-' && stem.as_bytes()[..32].iter().copied().all(|c|c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) && stem.as_bytes()[33..].iter().copied().all(|c|c.is_ascii_digit());
                if legal && anchor.is_file() {
                    let info=fs::symlink_metadata(&anchor).map_err(RecorderError::Io)?;
                    if !info.is_file() || info.len()>1024 {return Err(invalid("bounded regular native anchor required"));}
                    let mut raw=Vec::new();File::open(&anchor).map_err(RecorderError::Io)?.take(1025).read_to_end(&mut raw).map_err(RecorderError::Io)?;
                    if raw.len()>1024 {return Err(invalid("bounded native anchor required"));}
                    let value:serde_json::Value=serde_json::from_slice(&raw).map_err(|_|invalid("invalid native anchor"))?;
                    if value.as_object().is_none_or(|o|o.len()!=4 || ["schema_version","start_ts_ns","end_ts_ns","source_schema"].iter().any(|k|!o.contains_key(*k))) || value["source_schema"]!="native-log-file.v1" || value["schema_version"]!=1 || value["start_ts_ns"].as_u64().is_none_or(|n|n==0 || n>i64::MAX as u64) || !value["end_ts_ns"].is_null() {return Err(invalid("invalid native anchor"));}
                    platform::open_for_sync(&entry.path()).and_then(|f|f.sync_all()).map_err(RecorderError::Io)?;
                    let sealed=entry.path().with_extension("log");
                    if sealed.exists(){return Err(invalid("native recovery target exists"));}
                    platform::move_file(&entry.path(),&sealed,false).map_err(RecorderError::Io)?;#[cfg(unix)] platform::sync_directory(&path).map_err(RecorderError::Io)?;
                    if native_text {publish_native(&path,sealed.file_name().unwrap().to_str().unwrap(),&stem[..32],value["start_ts_ns"].as_u64().unwrap()).map_err(RecorderError::Io)?;}
                    segments.push(sealed.file_name().unwrap().to_string_lossy().into_owned());
                } // Unknown historical pending bytes are retained, never date-guessed or deleted.
            }
            if entry.path().extension().is_some_and(|e| e == "log") {
                if native_text {
                    let filename=entry.file_name().to_string_lossy().into_owned();
                    let stem=entry.path().file_stem().unwrap().to_string_lossy().into_owned();
                    if legal_segment_stem(&stem) {
                        let clock=native_publication_clock(&path,&filename).map_err(RecorderError::Io)?;
                        publish_native(&path,&filename,&stem[..32],clock).map_err(RecorderError::Io)?;
                    }
                }
                segments.push(entry.file_name().to_string_lossy().into_owned());
            }
        }
        #[cfg(unix)] platform::sync_directory(&path).map_err(RecorderError::Io)?;
        segments.sort();
        let shared = Arc::new(Shared {
            completion: Condvar::new(),
            state: Mutex::new(State {
                segments,
                ..State::default()
            }),
            run_id: run_id(),
        });
        let (tx, rx) = mpsc::channel();
        let worker_shared = shared.clone();
        thread::Builder::new()
            .name("monitor-recorder".into())
            .spawn(move || {
                if let Some(window) = window { window_worker(path, rx, worker_shared, segment_bytes, max_disk_bytes, lock, window); }
                else { worker(path, rx, worker_shared, segment_bytes, max_disk_bytes, lock); }
            })
            .map_err(RecorderError::Io)?;
        Ok(Self {
            shared,
            gate: Mutex::new(Gate { tx, closed: false }),
            capacity,
            segment_bytes,
            native_text,
        })
    }
    pub fn append(&self, record_json: String) -> bool {
        let gate = self.gate.lock().unwrap();
        let mut s = self.shared.state.lock().unwrap();
        if gate.closed || s.error.is_some() {
            return false;
        }
        if self.native_text { s.error=Some("native recorder requires native payload".into()); return false; }
        let valid = !record_json.contains('\n')
            && !record_json.contains('\r')
            && serde_json::from_str::<serde_json::Value>(&record_json).is_ok_and(|v| v.is_object());
        if !valid || record_json.len() + 1 > self.segment_bytes {
            s.error = Some("invalid object JSON or record exceeds segment_bytes".into());
            return false;
        }
        if s.outstanding >= self.capacity {
            s.error = Some("recorder queue capacity exhausted".into());
            return false;
        }
        s.outstanding += 1;
        s.accepted += 1;
        if gate.tx.send(Command::Record(record_json, native_clock_ns(), false)).is_err() {
            s.error = Some("recorder worker disconnected".into());
            return false;
        }
        true
    }
    /// Enqueue an owned native formatter. It runs on this recorder's sole worker.
    /// `native_time` is captured by the producer before enqueue, never inferred.
    pub fn append_native(&self, render: impl FnOnce() -> String + Send + 'static, native_time: u64) -> bool {
        let gate=self.gate.lock().unwrap();let mut s=self.shared.state.lock().unwrap();
        if gate.closed || s.error.is_some() {return false;}
        if !self.native_text || native_time==0 || native_time>i64::MAX as u64 {
            s.error=Some("explicit native recorder and valid producer clock required".into());return false;
        }
        if s.outstanding>=self.capacity {s.error=Some("recorder queue capacity exhausted".into());return false;}
        s.outstanding+=1;s.accepted+=1;
        if gate.tx.send(Command::Native(Box::new(render),native_time)).is_err() {
            s.error=Some("recorder worker disconnected".into());return false;
        }
        true
    }
    pub fn check(&self) -> Result<(), RecorderError> {
        let state=self.shared.state.lock().unwrap();
        if let Some(error)=&state.error {return Err(runtime_error(error.clone()));}
        if state.closed {return Err(runtime_error("recorder closed"));}
        Ok(())
    }
    pub fn is_native_text(&self) -> bool { self.native_text }
    pub fn status(&self) -> String {
        self.snapshot()
    }
    pub fn flush(&self) -> String { self.barrier(false) }
    pub fn close(&self) -> String { self.barrier(true) }
}
impl Recorder {
    fn snapshot(&self) -> String {
        let s = self.shared.state.lock().unwrap().clone();
        serde_json::json!({"accepted":s.accepted,"durable":s.durable,"error":s.error,"run_id":self.shared.run_id,"segments":s.segments,"closed":s.closed}).to_string()
    }
    fn barrier(&self, close: bool) -> String {
        let (tx, rx) = mpsc::channel();
        {
            let mut gate = self.gate.lock().unwrap();
            // A second close still waits behind the first close via a completion flag.
            if gate.closed {
                drop(gate);
                let mut state = self.shared.state.lock().unwrap();
                while !state.closed {
                    state = self.shared.completion.wait(state).unwrap();
                }
                drop(state);
                return self.snapshot();
            }
            if close {
                gate.closed = true;
            }
            let cmd = if close {
                Command::Close(tx)
            } else {
                Command::Barrier(tx)
            };
            if gate.tx.send(cmd).is_err() {
                let mut s = self.shared.state.lock().unwrap();
                s.error.get_or_insert("recorder worker disconnected".into());
                s.closed = true;
                self.shared.completion.notify_all();
                return drop_and_snapshot(s, self);
            }
        }
        if rx.recv().is_err() {
            let mut s = self.shared.state.lock().unwrap();
            s.error.get_or_insert("recorder worker disconnected".into());
            s.closed = true;
            self.shared.completion.notify_all();
        }
        self.snapshot()
    }
}
fn drop_and_snapshot(s: std::sync::MutexGuard<'_, State>, r: &Recorder) -> String {
    drop(s);
    r.snapshot()
}
impl Drop for Recorder {
    fn drop(&mut self) {
        self.barrier(true);
    }
}
fn legal_segment_stem(stem: &str) -> bool {
    let bytes=stem.as_bytes();bytes.len()==53 && bytes[32]==b'-'
        && bytes[..32].iter().copied().all(|c|c.is_ascii_hexdigit()&&!c.is_ascii_uppercase())
        && bytes[33..].iter().copied().all(|c|c.is_ascii_digit())
}
fn read_native_anchor(path: &PathBuf) -> std::io::Result<u64> {
    let info=fs::symlink_metadata(path)?;
    if !info.is_file() || info.len()>1024 {return Err(std::io::Error::other("bounded regular native anchor required"));}
    let mut raw=Vec::new();File::open(path)?.take(1025).read_to_end(&mut raw)?;
    let value:serde_json::Value=serde_json::from_slice(&raw).map_err(std::io::Error::other)?;
    if raw.len()>1024 || value.as_object().is_none_or(|o|o.len()!=4)
        || value["schema_version"]!=1 || value["source_schema"]!="native-log-file.v1" || !value["end_ts_ns"].is_null()
        || value["start_ts_ns"].as_u64().is_none_or(|n|n==0 || n>i64::MAX as u64) {
        return Err(std::io::Error::other("invalid native anchor"));
    }
    Ok(value["start_ts_ns"].as_u64().unwrap())
}
fn regular_private(path: &std::path::Path) -> std::io::Result<bool> {
    let info=fs::symlink_metadata(path)?;
    if !info.is_file() || info.file_type().is_symlink() { return Ok(false); }
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; if info.permissions().mode() & 0o777 != 0o600 { return Ok(false); } }
    Ok(true)
}

fn entry_size_if_present(entry: &fs::DirEntry) -> std::io::Result<u64> {
    let size = || -> std::io::Result<u64> {
        if entry.file_type()?.is_file() { Ok(entry.metadata()?.len()) } else { Ok(0) }
    };
    match size() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        result => result,
    }
}
fn disk_usage(path: &PathBuf) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let e = entry?;
        total = total
            .checked_add(entry_size_if_present(&e)?)
            .ok_or_else(|| std::io::Error::other("disk size overflow"))?;
    }
    Ok(total)
}
fn persist(
    path: &PathBuf,
    shared: &Shared,
    sequence: u64,
    records: &[String],
    max_disk: u64,
    native_start: u64,
    native_text: bool,
) -> std::io::Result<String> {
    let size: u64 = records.iter().map(|r| r.len() as u64 + 1).sum();
    let stem = format!("{}-{sequence:020}", shared.run_id);
    if native_start == 0 || native_start > i64::MAX as u64 {return Err(std::io::Error::other("invalid native producer clock"));}
    // Native enqueue-time is persisted before creating or writing raw bytes.
    let anchor = serde_json::json!({"schema_version":1,"start_ts_ns":native_start,"end_ts_ns":null,"source_schema":"native-log-file.v1"}).to_string();
    if disk_usage(path)?.saturating_add(size).saturating_add(anchor.len() as u64).saturating_add(if native_text {4096} else {0})>max_disk {return Err(std::io::Error::other("recorder disk limit exceeded"));}
    let mut metadata=platform::create_new(&path.join(format!("{stem}.native-time.json")))?;
    #[cfg(unix)] {use std::os::unix::fs::PermissionsExt;metadata.set_permissions(fs::Permissions::from_mode(0o600))?;}
    metadata.write_all(anchor.as_bytes())?;metadata.sync_all()?;#[cfg(unix)] platform::sync_directory(path)?;
    let pending = path.join(format!("{stem}.pending"));
    let name = format!("{stem}.log");
    let mut file = platform::create_new(&pending)?;
    #[cfg(unix)] if native_text {use std::os::unix::fs::PermissionsExt;file.set_permissions(fs::Permissions::from_mode(0o600))?;}
    for r in records {
        file.write_all(r.as_bytes())?;
        file.write_all(b"\n")?;
    }
    file.sync_all()?;
    platform::move_file(&pending, &path.join(&name), false)?;
    #[cfg(unix)] platform::sync_directory(path)?;
    if native_text { publish_native(path, &name, &shared.run_id, native_start)?; }
    Ok(name)
}

fn native_publication_clock(path: &PathBuf, name: &str) -> std::io::Result<u64> {
    let sidecar=path.join(format!("{name}.source.json"));
    match fs::symlink_metadata(&sidecar) {
        Ok(info)=>{
            if !regular_private(&sidecar)? || info.len()>4096 {return Err(std::io::Error::other("private bounded native publication required"));}
            let mut raw=Vec::new();File::open(sidecar)?.take(4097).read_to_end(&mut raw)?;
            let value:serde_json::Value=serde_json::from_slice(&raw).map_err(|_|std::io::Error::other("invalid native publication"))?;
            value["native_time_anchor"]["start_ts_ns"].as_u64().filter(|n|*n>0 && *n<=i64::MAX as u64).ok_or_else(||std::io::Error::other("invalid native publication clock"))
        }
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>read_native_anchor(&path.join(name).with_extension("native-time.json")),
        Err(error)=>Err(error),
    }
}

fn publish_native(path: &PathBuf, name: &str, generation: &str, native_start: u64) -> std::io::Result<()> {
    {
        let info=fs::symlink_metadata(path.join(name))?;
        if !regular_private(&path.join(name))? {return Err(std::io::Error::other("private native file required"));}
        let file=platform::open_for_sync(&path.join(name))?;
        let (device,inode,links)=platform::identity(&file)?;
        if links!=1 {return Err(std::io::Error::other("native source must have one hard link"));}
        let expected=serde_json::json!({"schema_version":1,"path":name,"owner_generation":generation,
            "device":device,"inode":inode,"published_length":info.len(),
            "publication_sequence":1,"sealed":true,"native_time_anchor":{
                "schema_version":1,"start_ts_ns":native_start,"end_ts_ns":null,"source_schema":"native-log-file.v1"}});
        let sidecar=path.join(format!("{name}.source.json"));
        match fs::symlink_metadata(&sidecar) {
            Ok(meta)=>{
                if !regular_private(&sidecar)? || meta.len()>4096 {return Err(std::io::Error::other("private bounded native publication required"));}
                let mut raw=Vec::new();File::open(&sidecar)?.take(4097).read_to_end(&mut raw)?;
                let actual:serde_json::Value=serde_json::from_slice(&raw).map_err(|_|std::io::Error::other("invalid native publication"))?;
                if actual!=expected {return Err(std::io::Error::other("native publication contract changed"));}
            }
            Err(error) if error.kind()==std::io::ErrorKind::NotFound=>{
                let temporary=path.join(format!("{name}.source.pending"));
                let mut file=platform::create_new(&temporary)?;
                file.write_all(expected.to_string().as_bytes())?;file.sync_all()?;
                // Atomic no-overwrite publication; a crash exposes either no sidecar
                // or the complete fsynced inode, never a partial public document.
                #[cfg(unix)] fs::hard_link(&temporary,&sidecar)?;
                #[cfg(windows)] platform::move_file(&temporary,&sidecar,false)?;#[cfg(unix)] platform::sync_directory(path)?;
                #[cfg(unix)] fs::remove_file(temporary)?;#[cfg(unix)] platform::sync_directory(path)?;
            }
            Err(error)=>return Err(error),
        }
        let anchor=path.join(name).with_extension("native-time.json");
        match fs::symlink_metadata(&anchor) {
            Ok(_)=>{if read_native_anchor(&anchor)?!=native_start {return Err(std::io::Error::other("native anchor changed during publication"));}fs::remove_file(anchor)?;#[cfg(unix)] platform::sync_directory(path)?;}
            Err(error) if error.kind()==std::io::ErrorKind::NotFound=>{},
            Err(error)=>return Err(error),
        }
        Ok(())
    }
}

fn materialize(command: Command, segment_bytes: usize) -> Command {
    match command {
        Command::Native(render, clock) => {
            let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(render));
            match result {
                Ok(mut line) if line.ends_with('\n') && line.len()<=segment_bytes && line.len()<=96*1024 => {
                    line.pop();
                    if line.is_empty() || line.contains(['\n','\r']) {Command::InvalidNative}
                    else {Command::Record(line, clock, true)}
                }
                _ => Command::InvalidNative,
            }
        }
        other => other,
    }
}
fn worker(
    path: PathBuf,
    rx: mpsc::Receiver<Command>,
    shared: Arc<Shared>,
    segment_bytes: usize,
    max_disk: u64,
    lock: File,
) {
    let mut lock = Some(lock);
    let mut sequence = 0;
    let mut pending = None;
    loop {
        let command = pending.take().map(Ok).unwrap_or_else(|| rx.recv()).map(|c|materialize(c,segment_bytes));
        match command {
            Ok(Command::Record(first, native_start, native_text)) => {
                let mut size = first.len() + 1;
                let mut records = vec![first];
                // Records frequently arrive just after the previous record
                // was written. Bound the batching window from the first
                // record so those bursts share one durable source file. A
                // barrier or close remains an immediate batch boundary.
                let batch_deadline = Instant::now() + Duration::from_secs(1);
                loop {
                    let next = match rx.recv_timeout(batch_deadline.saturating_duration_since(Instant::now())) {
                        Ok(next) => next,
                        Err(_) => break,
                    };
                    match materialize(next,segment_bytes) {
                        Command::Record(r, _, native) if native==native_text && size + r.len() < segment_bytes => {
                            size += r.len() + 1;
                            records.push(r);
                        }
                        other => {
                            pending = Some(other);
                            break;
                        }
                    }
                }
                // Capacity errors do not discard records already accepted.
                let failed = shared
                    .state
                    .lock()
                    .unwrap()
                    .error
                    .as_ref()
                    .is_some_and(|e| e.starts_with("storage:"));
                let result = if failed {
                    Err(std::io::Error::other("storage already failed"))
                } else {
                    persist(&path, &shared, sequence, &records, max_disk, native_start, native_text)
                };
                sequence += 1;
                let mut s = shared.state.lock().unwrap();
                s.outstanding -= records.len();
                match result {
                    Ok(name) => {
                        s.durable += records.len() as u64;
                        s.segments.push(name);
                    }
                    Err(e) => {
                        if !failed { s.error = Some(format!("storage: {e}")); }
                    }
                }
            }
            Ok(Command::InvalidNative) => {
                let mut state=shared.state.lock().unwrap();state.outstanding-=1;
                state.error=Some("storage: native formatter failed, panicked or exceeded the exact line bound".into());
            }
            Ok(Command::Native(_, _)) => unreachable!("native payload is materialized once by this worker"),
            Ok(Command::Barrier(done)) => {
                let _ = done.send(());
            }
            Ok(Command::Close(done)) => {
                drop(lock.take());
                shared.state.lock().unwrap().closed = true;
                shared.completion.notify_all();
                let _ = done.send(());
                break;
            }
            Err(_) => {
                shared.state.lock().unwrap().closed = true;
                shared.completion.notify_all();
                break;
            }
        }
    }
}
fn run_id() -> String {
    let mut bytes = [0u8; 16];
    platform::random(&mut bytes).expect("OS entropy unavailable");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn native_clock_ns()->u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().and_then(|d|u64::try_from(d.as_nanos()).ok()).filter(|n|*n<=i64::MAX as u64).unwrap_or(0)
}

fn window_publication(path: &PathBuf, name: &str, file: &File, generation: &str, anchor: u64, seq: u64, sealed: bool) -> std::io::Result<()> {
    let info=file.metadata()?;
    let (device,inode,_)=platform::identity(file)?;
    let value=serde_json::json!({"schema_version":1,"path":name,"owner_generation":generation,"device":device,"inode":inode,"published_length":info.len(),"publication_sequence":seq,"sealed":sealed,"native_time_anchor":{"schema_version":1,"start_ts_ns":anchor,"end_ts_ns":null,"source_schema":"native-log-file.v1"}});
    let tmp=path.join(format!(".{name}.{}.source.tmp",run_id()));
    let mut output=platform::create_new(&tmp)?;
    output.write_all(value.to_string().as_bytes())?;output.sync_all()?;
    platform::move_file(&tmp,&path.join(format!("{name}.source.json")),true)?;
    #[cfg(unix)] platform::sync_directory(path)?;
    Ok(())
}

// Recovery runs under the recorder lock, before a new generation can write.
fn recover_windowed(path: &PathBuf) -> std::io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry=entry?;let side=entry.path();let filename=entry.file_name().to_string_lossy().into_owned();
        let Some(name)=filename.strip_suffix(".source.json") else {continue};
        let Some(stem)=name.strip_suffix(".log") else {continue};
        if !legal_segment_stem(stem) {continue}
        let meta=fs::symlink_metadata(&side)?;
        if !regular_private(&side)? || meta.len()>4096 {return Err(std::io::Error::other("invalid window publication file"));}
        let doc:serde_json::Value=serde_json::from_slice(&fs::read(&side)?).map_err(std::io::Error::other)?;
        let sealed=doc["sealed"].as_bool().ok_or_else(||std::io::Error::other("invalid window seal"))?;
        let anchor=doc["native_time_anchor"]["start_ts_ns"].as_u64().filter(|n|*n>0 && *n<=i64::MAX as u64).ok_or_else(||std::io::Error::other("invalid window anchor"))?;
        let expected_anchor=serde_json::json!({"schema_version":1,"start_ts_ns":anchor,"end_ts_ns":null,"source_schema":"native-log-file.v1"});
        let seq=doc["publication_sequence"].as_u64().filter(|n|*n>0).ok_or_else(||std::io::Error::other("invalid window sequence"))?;
        let length=doc["published_length"].as_u64().ok_or_else(||std::io::Error::other("invalid window length"))?;
        if doc.as_object().is_none_or(|o|o.len()!=9) || doc["schema_version"]!=1 || doc["path"]!=name || doc["owner_generation"]!=&stem[..32] || doc["native_time_anchor"]!=expected_anchor {return Err(std::io::Error::other("invalid window identity"));}
        let final_path=path.join(name);let pending=path.join(format!("{stem}.pending"));
        if sealed && !final_path.exists() && !pending.exists() {continue} // Source may already have been proof-retired by the sender.
        let actual=if final_path.exists() {if pending.exists(){return Err(std::io::Error::other("duplicate window source"));} final_path.clone()} else {pending.clone()};
        let info=fs::symlink_metadata(&actual)?;
        let file=platform::open_for_sync(&actual)?;
        let (device,inode,links)=platform::identity(&file)?;
        if !regular_private(&actual)? || links!=1 || doc["device"].as_u64()!=Some(device) || doc["inode"].as_u64()!=Some(inode) || length>info.len() || (sealed && length!=info.len()) {return Err(std::io::Error::other("window source identity or length changed"));}
        file.sync_all()?;
        if !sealed {window_publication(path,name,&file,&stem[..32],anchor,seq.checked_add(1).ok_or_else(||std::io::Error::other("publication sequence overflow"))?,true)?;}
        if actual==pending {platform::move_file(&pending,&final_path,false)?;#[cfg(unix)] platform::sync_directory(path)?;}
        match fs::remove_file(path.join(format!("{stem}.window"))) {Ok(())=>{#[cfg(unix)] platform::sync_directory(path)?;},Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{},Err(e)=>return Err(e)}
    }
    // A crash before the first publication leaves the explicit window marker,
    // anchor and possibly a partial pending file. Publish before making it visible.
    for entry in fs::read_dir(path)? {
        let entry=entry?;let filename=entry.file_name().to_string_lossy().into_owned();
        let Some(stem)=filename.strip_suffix(".pending") else {continue};
        if !legal_segment_stem(stem) || !path.join(format!("{stem}.window")).exists() {continue}
        let name=format!("{stem}.log");
        if path.join(format!("{name}.source.json")).exists() {return Err(std::io::Error::other("unresolved window publication"));}
        if !regular_private(&entry.path())? || platform::identity(&platform::open_for_sync(&entry.path())?)?.2!=1 {return Err(std::io::Error::other("invalid pending window"));}
        let anchor=read_native_anchor(&path.join(format!("{stem}.native-time.json")))?;
        let file=platform::open_for_sync(&entry.path())?;file.sync_all()?;
        window_publication(path,&name,&file,&stem[..32],anchor,1,true)?;
        platform::move_file(&entry.path(),&path.join(name),false)?;#[cfg(unix)] platform::sync_directory(path)?;
        fs::remove_file(path.join(format!("{stem}.window")))?;#[cfg(unix)] platform::sync_directory(path)?;
    }
    Ok(())
}


struct WindowFile { file: File, name: String, anchor: u64, seq: u64, started: Instant }
fn window_seal(path:&PathBuf, shared:&Shared, active:&mut Option<WindowFile>) -> std::io::Result<()> {
    if let Some(mut a)=active.take() {
        a.file.sync_all()?;a.seq=a.seq.checked_add(1).ok_or_else(||std::io::Error::other("publication sequence overflow"))?;
        window_publication(path,&a.name,&a.file,&shared.run_id,a.anchor,a.seq,true)?;
        let stem=a.name.strip_suffix(".log").ok_or_else(||std::io::Error::other("invalid window name"))?;
        fs::remove_file(path.join(format!("{stem}.window")))?;#[cfg(unix)] platform::sync_directory(path)?;
    }
    Ok(())
}
fn window_worker(path:PathBuf,rx:mpsc::Receiver<Command>,shared:Arc<Shared>,segment_bytes:usize,max_disk:u64,lock:File,window:Duration) {
    let mut active:Option<WindowFile>=None;let mut sequence=0u64;let mut clean=Instant::now();let mut pending=None;let mut held_lock=Some(lock);
    loop {
        let timeout=active.as_ref().map(|a|window.saturating_sub(a.started.elapsed()).min(Duration::from_secs(1))).unwrap_or(Duration::from_secs(1));
        let cmd=match pending.take() {Some(c)=>Ok(c),None=>rx.recv_timeout(timeout)};
        if clean.elapsed()>=Duration::from_secs(60) {shared.state.lock().unwrap().segments.retain(|n|path.join(n).is_file());clean=Instant::now();}
        if active.as_ref().is_some_and(|a|a.started.elapsed()>=window || native_clock_ns()/86_400_000_000_000>a.anchor/86_400_000_000_000) {
            if let Err(e)=window_seal(&path,&shared,&mut active) {shared.state.lock().unwrap().error.get_or_insert(format!("storage: {e}"));}
        }
        match cmd {
            Ok(Command::Record(first,clock,false)) => {
                let mut records=vec![first];let mut size=records[0].len()+1;
                let batch_start=Instant::now();
                let batch_deadline=batch_start+Duration::from_secs(1).min(window);
                while let Ok(next)=rx.recv_timeout(batch_deadline.saturating_duration_since(Instant::now())) {
                    match next {Command::Record(line,t,false) if t/86_400_000_000_000==clock/86_400_000_000_000 && size+line.len()+1<=segment_bytes=>{size+=line.len()+1;records.push(line);}, other=>{pending=Some(other);break}}
                }
                let result=(|| -> std::io::Result<()> {
                    if clock==0 || clock>i64::MAX as u64 {return Err(std::io::Error::other("invalid producer clock"));}
                    if shared.state.lock().unwrap().error.as_ref().is_some_and(|e|e.starts_with("storage:")) {return Err(std::io::Error::other("storage already failed"));}
                    if active.as_ref().is_some_and(|a|a.anchor/86_400_000_000_000!=clock/86_400_000_000_000 || a.file.metadata().map_or(true,|m|m.len()+size as u64>segment_bytes as u64)) {window_seal(&path,&shared,&mut active)?;}
                    if disk_usage(&path)?.saturating_add(size as u64).saturating_add(8192)>max_disk {return Err(std::io::Error::other("recorder disk limit exceeded"));}
                    if active.is_none() {
                        let stem=format!("{}-{sequence:020}",shared.run_id);sequence+=1;let name=format!("{stem}.log");
                        let marker=platform::create_new(&path.join(format!("{stem}.window")))?;
                        marker.sync_all()?;#[cfg(unix)] platform::sync_directory(&path)?;
                        let anchor_path=path.join(format!("{stem}.native-time.json"));
                        let mut anchor_file=platform::create_new(&anchor_path)?;
                        anchor_file.write_all(serde_json::json!({"schema_version":1,"start_ts_ns":clock,"end_ts_ns":null,"source_schema":"native-log-file.v1"}).to_string().as_bytes())?;
                        anchor_file.sync_all()?;#[cfg(unix)] platform::sync_directory(&path)?;
                        let temp=path.join(format!("{stem}.pending"));let mut file=platform::create_new(&temp)?;
                        for r in &records {file.write_all(r.as_bytes())?;file.write_all(b"\n")?;}file.sync_all()?;
                        window_publication(&path,&name,&file,&shared.run_id,clock,1,false)?;
                        platform::move_file(&temp,&path.join(&name),false)?;#[cfg(unix)] platform::sync_directory(&path)?;
                        shared.state.lock().unwrap().segments.push(name.clone());
                        active=Some(WindowFile{file,name,anchor:clock,seq:1,started:batch_start});
                    } else {
                        let a=active.as_mut().unwrap();for r in &records {a.file.write_all(r.as_bytes())?;a.file.write_all(b"\n")?;}a.file.sync_all()?;
                        a.seq=a.seq.checked_add(1).ok_or_else(||std::io::Error::other("publication sequence overflow"))?;
                        window_publication(&path,&a.name,&a.file,&shared.run_id,a.anchor,a.seq,false)?;
                    }
                    if active.as_ref().is_some_and(|a|a.seq>=192 || a.file.metadata().map_or(true,|m|m.len()>=segment_bytes as u64)) {window_seal(&path,&shared,&mut active)?;}Ok(())
                })();
                let mut s=shared.state.lock().unwrap();s.outstanding-=records.len();match result {Ok(())=>s.durable+=records.len() as u64,Err(e)=>{s.error.get_or_insert(format!("storage: {e}"));}}
            }
            Ok(Command::Barrier(done)) => {let _=done.send(());}
            Ok(Command::Close(done)) => {if let Err(e)=window_seal(&path,&shared,&mut active){shared.state.lock().unwrap().error.get_or_insert(format!("storage: {e}"));}drop(active.take());drop(held_lock.take());shared.state.lock().unwrap().closed=true;shared.completion.notify_all();let _=done.send(());break;}
            Ok(_) => {shared.state.lock().unwrap().error.get_or_insert("storage: invalid window command".into());}
            Err(mpsc::RecvTimeoutError::Timeout)=>{},
            Err(mpsc::RecvTimeoutError::Disconnected)=>{let _=window_seal(&path,&shared,&mut active);shared.state.lock().unwrap().closed=true;shared.completion.notify_all();break;}
        }
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    #[test]
    fn batch_crossing_utc_day_rotates_at_record_boundary() {
        let path=std::env::temp_dir().join(run_id());fs::create_dir(&path).unwrap();
        let lock=File::create(path.join("lock")).unwrap();let shared=Arc::new(Shared{state:Mutex::new(State{accepted:2,outstanding:2,..State::default()}),completion:Condvar::new(),run_id:run_id()});
        let (tx,rx)=mpsc::channel();let (done,finished)=mpsc::channel();
        let day=native_clock_ns()/86_400_000_000_000;
        tx.send(Command::Record("{\"a\":1}".into(),day*86_400_000_000_000+1,false)).unwrap();
        tx.send(Command::Record("{\"b\":2}".into(),(day+1)*86_400_000_000_000+1,false)).unwrap();tx.send(Command::Close(done)).unwrap();
        window_worker(path.clone(),rx,shared,1024,65536,lock,Duration::from_secs(300));finished.recv().unwrap();
        let mut logs=fs::read_dir(&path).unwrap().flatten().filter(|e|e.path().extension().is_some_and(|v|v=="log")).map(|e|fs::read_to_string(e.path()).unwrap()).collect::<Vec<_>>();logs.sort();assert_eq!(logs,vec!["{\"a\":1}\n","{\"b\":2}\n"]);fs::remove_dir_all(path).unwrap();
    }
}
