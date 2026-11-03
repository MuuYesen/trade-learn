//! Real filesystem contracts, run unchanged on Windows, Linux and macOS.
use leafquant_external_log_core::recorder::Recorder;
use std::{fs, path::PathBuf, sync::Arc, thread, time::Duration};

fn directory(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("lq-platform-{label}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir(&path).unwrap();
    path
}

#[test]
fn sealed_published_tail_truncation_is_rejected_without_rewriting_evidence() {
    let path = directory("corrupt");
    let r = Recorder::new_windowed(path.to_str().unwrap().into(), 16, 4096, 65536, Duration::from_secs(300)).unwrap();
    assert!(r.append("{\"n\":1}".into()));
    let state: serde_json::Value = serde_json::from_str(&r.close()).unwrap();
    assert_eq!(state["durable"], 1);
    drop(r);
    let raw = fs::read_dir(&path).unwrap().flatten().find(|e| e.path().extension().is_some_and(|s| s == "log")).unwrap().path();
    let mut bytes = fs::read(&raw).unwrap();
    bytes.pop();
    fs::write(&raw, &bytes).unwrap();
    assert!(Recorder::new_windowed(path.to_str().unwrap().into(), 16, 4096, 65536, Duration::from_secs(300)).is_err());
    assert_eq!(fs::read(&raw).unwrap(), bytes);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn concurrent_close_preserves_records_and_releases_process_lock() {
    let path = directory("close");
    let r = Arc::new(Recorder::new_windowed(path.to_str().unwrap().into(), 256, 65536, 1048576, Duration::from_secs(300)).unwrap());
    for n in 0..100 { assert!(r.append(format!("{{\"n\":{n}}}"))); }
    let handles: Vec<_> = (0..4).map(|_| { let r = r.clone(); thread::spawn(move || r.close()) }).collect();
    for h in handles {
        let s: serde_json::Value = serde_json::from_str(&h.join().unwrap()).unwrap();
        assert_eq!(s["durable"], 100);
        assert_eq!(s["closed"], true);
        assert!(s["error"].is_null());
    }
    assert!(!r.append("{}".into()));
    let next = Recorder::new_windowed(path.to_str().unwrap().into(), 4, 4096, 1048576, Duration::from_secs(300)).unwrap();
    next.close(); drop(next); drop(r);
    fs::remove_dir_all(path).unwrap();
}

#[test]
#[ignore = "subprocess fixture: invoked explicitly by process_crash_recovers"]
fn crash_child() {
    let path=std::env::var("LEAFQUANT_RECORDER_CRASH_DIRECTORY").unwrap();
    let r=Recorder::new_windowed(path,16,4096,65536,Duration::from_secs(300)).unwrap();
    assert!(r.append("{\"crash\":\"exact\"}".into()));
    let state:serde_json::Value=serde_json::from_str(&r.flush()).unwrap();
    assert_eq!(state["durable"],1);
    assert!(state["error"].is_null());
    println!("RECORDER_DURABLE");
    use std::io::Write;
    std::io::stdout().flush().unwrap();
    loop { thread::park(); }
}

#[test]
fn process_crash_recovers_acknowledged_prefix_and_original_identity() {
    use std::io::{BufRead,BufReader};
    use std::process::{Command,Stdio};
    let path=directory("crash");
    let mut child=Command::new(std::env::current_exe().unwrap())
        .args(["--ignored","--exact","crash_child","--nocapture"])
        .env("LEAFQUANT_RECORDER_CRASH_DIRECTORY",&path)
        .stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();
    let mut reader=BufReader::new(child.stdout.take().unwrap());
    let mut line=String::new();
    loop { line.clear(); assert_ne!(reader.read_line(&mut line).unwrap(),0,"child failed before durable acknowledgement"); if line.contains("RECORDER_DURABLE") {break;} }
    let side=fs::read_dir(&path).unwrap().flatten().find(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).unwrap().path();
    let before:serde_json::Value=serde_json::from_slice(&fs::read(&side).unwrap()).unwrap();
    assert_eq!(before["sealed"],false);
    // A different process owns the lock, rather than just another Rust object.
    assert!(Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).is_err());
    child.kill().unwrap(); assert!(!child.wait().unwrap().success());
    let r=Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).unwrap();
    let after:serde_json::Value=serde_json::from_slice(&fs::read(&side).unwrap()).unwrap();
    assert_eq!(after["sealed"],true);
    for key in ["device","inode","owner_generation","native_time_anchor","path","published_length"] { assert_eq!(before[key],after[key],"{key}"); }
    assert_eq!(fs::read(path.join(after["path"].as_str().unwrap())).unwrap(),b"{\"crash\":\"exact\"}\n");
    r.close();drop(r);
    // The same bytes at a different file identity must not be silently adopted.
    let raw=path.join(after["path"].as_str().unwrap());
    let renamed=raw.with_extension("original");
    fs::rename(&raw,&renamed).unwrap();fs::copy(&renamed,&raw).unwrap();
    #[cfg(unix)] {use std::os::unix::fs::PermissionsExt;fs::set_permissions(&raw,fs::Permissions::from_mode(0o600)).unwrap();}
    assert!(Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).is_err());
    assert_eq!(fs::read(&raw).unwrap(),fs::read(&renamed).unwrap());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn corrupt_publication_metadata_is_rejected_without_replacing_it() {
    for case in ["json", "generation", "length"] {
        let path=directory(case);
        let r=Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).unwrap();
        assert!(r.append("{}".into()));r.close();drop(r);
        let side=fs::read_dir(&path).unwrap().flatten().find(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).unwrap().path();
        let mut document:serde_json::Value=serde_json::from_slice(&fs::read(&side).unwrap()).unwrap();
        let bytes=match case {
            "json" => b"{truncated".to_vec(),
            "generation" => {document["owner_generation"]=serde_json::json!("0".repeat(32));serde_json::to_vec(&document).unwrap()},
            _ => {document["published_length"]=serde_json::json!(999);serde_json::to_vec(&document).unwrap()},
        };
        fs::write(&side,&bytes).unwrap();
        assert!(Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).is_err(),"{case}");
        assert_eq!(fs::read(&side).unwrap(),bytes);
        fs::remove_dir_all(path).unwrap();
    }
}

#[test]
fn uncommitted_partial_non_utf8_tail_is_preserved_as_original_bytes() {
    use std::io::Write;
    let path=directory("uncommitted-tail");
    let r=Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).unwrap();
    assert!(r.append("{}".into()));r.close();drop(r);
    let side=fs::read_dir(&path).unwrap().flatten().find(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).unwrap().path();
    let mut doc:serde_json::Value=serde_json::from_slice(&fs::read(&side).unwrap()).unwrap();
    doc["sealed"]=serde_json::json!(false);
    fs::write(&side,serde_json::to_vec(&doc).unwrap()).unwrap();
    let raw=path.join(doc["path"].as_str().unwrap());
    let mut file=fs::OpenOptions::new().append(true).open(&raw).unwrap();
    file.write_all(b"\xff{partial").unwrap();file.sync_all().unwrap();drop(file);
    let original=fs::read(&raw).unwrap();
    let recovered=Recorder::new_windowed(path.to_str().unwrap().into(),16,4096,65536,Duration::from_secs(300)).unwrap();
    assert_eq!(fs::read(&raw).unwrap(),original);
    assert!(recovered.append("{\"new\":true}".into()));
    let state:serde_json::Value=serde_json::from_str(&recovered.close()).unwrap();
    assert_eq!(state["durable"],1);assert!(state["error"].is_null());drop(recovered);
    fs::remove_dir_all(path).unwrap();
}
