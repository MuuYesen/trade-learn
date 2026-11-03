//! The same durable recorder is usable without Python or a framework.
use leafquant_external_log_core::recorder::Recorder;

#[test]
fn native_records_arriving_close_together_share_one_durable_segment() {
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;
    let root=std::env::temp_dir().join(format!("lq-native-batch-{}",std::process::id()));
    std::fs::create_dir(&root).unwrap();
    std::fs::set_permissions(&root,std::fs::Permissions::from_mode(0o700)).unwrap();
    let root=std::fs::canonicalize(root).unwrap();
    let recorder=Recorder::new_native(root.to_str().unwrap().into(),16,4096,65536).unwrap();
    for n in 0..3 {
        assert!(recorder.append_native(move || format!("native record {n}\n"),1793059200000000001+n));
        std::thread::sleep(Duration::from_millis(30));
    }
    assert!(recorder.flush().contains("\"durable\":3"));
    let logs:Vec<_>=std::fs::read_dir(&root).unwrap().flatten().filter(|entry|entry.path().extension().is_some_and(|s|s=="log")).collect();
    assert_eq!(logs.len(),1);
    assert_eq!(std::fs::read_to_string(logs[0].path()).unwrap(),"native record 0\nnative record 1\nnative record 2\n");
    recorder.close();drop(recorder);std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn json_records_arriving_close_together_share_one_durable_segment() {
    use std::time::Duration;
    let root=std::env::temp_dir().join(format!("lq-json-batch-{}",std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let recorder=Recorder::new(root.to_str().unwrap().into(),16,4096,65536).unwrap();
    for n in 0..3 {
        assert!(recorder.append(format!("{{\"n\":{n}}}")));
        std::thread::sleep(Duration::from_millis(30));
    }
    assert!(recorder.flush().contains("\"durable\":3"));
    let logs:Vec<_>=std::fs::read_dir(&root).unwrap().flatten().filter(|entry|entry.path().extension().is_some_and(|s|s=="log")).collect();
    assert_eq!(logs.len(),1);
    assert_eq!(std::fs::read_to_string(logs[0].path()).unwrap(),"{\"n\":0}\n{\"n\":1}\n{\"n\":2}\n");
    recorder.close();drop(recorder);std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_recorder_preserves_exact_json_and_rejects_after_close() {
    let directory = std::env::temp_dir().join(format!("lq-shared-recorder-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let recorder = Recorder::new(directory.to_str().unwrap().into(), 4, 1024, 4096).unwrap();
    let original = r#"{"schema_version":1,"amount":"1.0000000001"}"#;
    assert!(recorder.append(original.into()));
    assert!(recorder.flush().contains("\"durable\":1"));
    assert!(recorder.close().contains("\"closed\":true"));
    assert!(!recorder.append("{}".into()));
    let file = std::fs::read_dir(&directory).unwrap().flatten()
        .find(|entry| entry.path().extension().is_some_and(|s| s == "log")).unwrap();
    assert_eq!(std::fs::read_to_string(file.path()).unwrap(), format!("{original}\n"));
    drop(recorder);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn native_line_formats_on_worker_and_publishes_existing_source_contract() {
    use std::os::unix::fs::PermissionsExt;
    let directory = std::env::temp_dir().join(format!("lq-native-recorder-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).unwrap();
    let directory = std::fs::canonicalize(directory).unwrap();
    let recorder = Recorder::new_native(directory.to_str().unwrap().into(), 4, 1024, 16384).unwrap();
    let caller = std::thread::current().id();
    assert!(recorder.append_native(move || {
        assert_ne!(std::thread::current().id(), caller);
        "2026-10-27T00:00:00.000000001Z [INFO] TRIRUST-001.nautilus_trading::strategy: exact event\n".into()
    }, 1793059200000000001));
    assert!(recorder.flush().contains("\"durable\":1"));
    recorder.close();
    let source = std::fs::read_dir(&directory).unwrap().flatten()
        .find(|entry| entry.file_name().to_str().unwrap().ends_with(".source.json")).unwrap();
    let publication: serde_json::Value = serde_json::from_slice(&std::fs::read(source.path()).unwrap()).unwrap();
    assert_eq!(publication["sealed"], true);
    assert_eq!(publication["native_time_anchor"]["start_ts_ns"], 1793059200000000001u64);
    let raw = std::fs::read(directory.join(publication["path"].as_str().unwrap())).unwrap();
    assert!(raw.starts_with(b"2026-10-27T"));
    assert_eq!(publication["published_length"].as_u64().unwrap(), raw.len() as u64);
    assert!(!std::fs::read_dir(&directory).unwrap().flatten().any(|entry|entry.file_name().to_str().unwrap().ends_with(".native-time.json")));
    drop(recorder);
    assert!(Recorder::new(directory.to_str().unwrap().into(), 4, 1024, 16384).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn native_formatter_failure_is_sticky_and_never_durable() {
    use std::os::unix::fs::PermissionsExt;
    for case in 0..4 {
        let root=std::env::temp_dir().join(format!("lq-native-invalid-{}-{case}",std::process::id()));
        std::fs::create_dir(&root).unwrap();std::fs::set_permissions(&root,std::fs::Permissions::from_mode(0o700)).unwrap();
        let root=std::fs::canonicalize(root).unwrap();
        let recorder=Recorder::new_native(root.to_str().unwrap().into(),4,128,65536).unwrap();
        assert!(recorder.append_native(move || match case {
            0=>"embedded\nnewline\n".into(), 1=>"no terminal newline".into(),
            2=>format!("{}\n","x".repeat(128)), _=>panic!("synthetic formatter panic"),
        },1793059200000000001));
        let status:serde_json::Value=serde_json::from_str(&recorder.flush()).unwrap();
        assert_eq!(status["durable"],0);assert!(status["error"].is_string());
        assert!(!recorder.append_native(||"later\n".into(),1793059200000000002));
        assert!(recorder.close().contains("\"closed\":true"));
        drop(recorder);std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn native_crash_recovery_preserves_file_and_original_time_generation() {
    use std::os::unix::fs::PermissionsExt;
    let root=std::env::temp_dir().join(format!("lq-native-recovery-{}",std::process::id()));
    std::fs::create_dir(&root).unwrap();std::fs::set_permissions(&root,std::fs::Permissions::from_mode(0o700)).unwrap();
    let root=std::fs::canonicalize(root).unwrap();
    let recorder=Recorder::new_native(root.to_str().unwrap().into(),4,1024,65536).unwrap();recorder.close();drop(recorder);
    let generation="a".repeat(32);let stem=format!("{generation}-00000000000000000001");
    let pending=root.join(format!("{stem}.pending"));std::fs::write(&pending,b"original native line\n").unwrap();
    std::fs::set_permissions(&pending,std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(root.join(format!("{stem}.native-time.json")),r#"{"schema_version":1,"start_ts_ns":1793059200000000001,"end_ts_ns":null,"source_schema":"native-log-file.v1"}"#).unwrap();
    let recorder=Recorder::new_native(root.to_str().unwrap().into(),4,1024,65536).unwrap();
    let source:serde_json::Value=serde_json::from_slice(&std::fs::read(root.join(format!("{stem}.log.source.json"))).unwrap()).unwrap();
    assert_eq!(source["owner_generation"],generation);assert_eq!(source["native_time_anchor"]["start_ts_ns"],1793059200000000001u64);
    assert_eq!(std::fs::read(root.join(format!("{stem}.log"))).unwrap(),b"original native line\n");
    recorder.close();drop(recorder);std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_reopen_validates_publication_before_accepting_writer() {
    use std::os::unix::fs::PermissionsExt;
    for corrupt in [false, true] {
        let root=std::env::temp_dir().join(format!("lq-native-publication-{}-{corrupt}",std::process::id()));
        std::fs::create_dir(&root).unwrap();std::fs::set_permissions(&root,std::fs::Permissions::from_mode(0o700)).unwrap();
        let root=std::fs::canonicalize(root).unwrap();
        let recorder=Recorder::new_native(root.to_str().unwrap().into(),4,1024,65536).unwrap();
        assert!(recorder.append_native(||"original native line\n".into(),1793059200000000001));
        recorder.close();drop(recorder);
        let sidecar=std::fs::read_dir(&root).unwrap().flatten().find(|e|e.file_name().to_str().unwrap().ends_with(".source.json")).unwrap().path();
        let value:serde_json::Value=serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        let anchor=root.join(value["path"].as_str().unwrap()).with_extension("native-time.json");
        std::fs::write(&anchor,serde_json::to_vec(&value["native_time_anchor"]).unwrap()).unwrap();
        std::fs::set_permissions(&anchor,std::fs::Permissions::from_mode(0o600)).unwrap();
        if corrupt {std::fs::write(&sidecar,b"{truncated").unwrap();}
        let reopened=Recorder::new_native(root.to_str().unwrap().into(),4,1024,65536);
        if corrupt {assert!(reopened.is_err());} else {let recorder=reopened.unwrap();recorder.close();drop(recorder);assert!(!anchor.exists());}
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn windowed_flush_publishes_live_prefix_and_close_seals_same_file() {
    let dir=std::env::temp_dir().join(format!("lq-window-basic-{}",std::process::id()));
    let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),16,4096,65536,std::time::Duration::from_secs(300)).unwrap();
    assert!(r.append("{\"a\":1}".into()));r.flush();
    let side=std::fs::read_dir(&dir).unwrap().flatten().find(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).unwrap().path();
    let live:serde_json::Value=serde_json::from_slice(&std::fs::read(&side).unwrap()).unwrap();
    assert_eq!(live["sealed"],false);assert_eq!(live["published_length"],8);
    let raw=dir.join(live["path"].as_str().unwrap());assert_eq!(std::fs::read(&raw).unwrap(),b"{\"a\":1}\n");
    r.close();let sealed:serde_json::Value=serde_json::from_slice(&std::fs::read(&side).unwrap()).unwrap();
    assert_eq!(sealed["sealed"],true);assert!(sealed["publication_sequence"].as_u64().unwrap()>live["publication_sequence"].as_u64().unwrap());
    assert_eq!(std::fs::read(raw).unwrap(),b"{\"a\":1}\n");drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn window_recovery_seals_unpublished_tail_without_changing_inode_or_anchor() {
    use std::os::unix::fs::{MetadataExt,PermissionsExt};
    let dir=std::env::temp_dir().join(format!("lq-window-recover-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let stem=format!("{}-{:020}","a".repeat(32),0);let name=format!("{stem}.log");let pending=dir.join(format!("{stem}.pending"));std::fs::write(&pending,b"{\"n\":1}\npartial").unwrap();std::fs::set_permissions(&pending,std::fs::Permissions::from_mode(0o600)).unwrap();let info=std::fs::metadata(&pending).unwrap();
    let side=dir.join(format!("{name}.source.json"));let anchor=serde_json::json!({"schema_version":1,"start_ts_ns":1793059200000000001u64,"end_ts_ns":null,"source_schema":"native-log-file.v1"});
    std::fs::write(&side,serde_json::json!({"schema_version":1,"path":name,"owner_generation":"a".repeat(32),"device":info.dev(),"inode":info.ino(),"published_length":8,"publication_sequence":3,"sealed":false,"native_time_anchor":anchor}).to_string()).unwrap();std::fs::set_permissions(&side,std::fs::Permissions::from_mode(0o600)).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),4,1024,65536,std::time::Duration::from_secs(300)).unwrap();
    let v:serde_json::Value=serde_json::from_slice(&std::fs::read(&side).unwrap()).unwrap();assert_eq!(v["sealed"],true);assert_eq!(v["publication_sequence"],4);assert_eq!(v["native_time_anchor"],anchor);assert_eq!(v["inode"],info.ino());assert_eq!(std::fs::read(dir.join(name)).unwrap(),b"{\"n\":1}\npartial");r.close();drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn window_timeout_seals_without_more_records_and_empty_creates_no_source() {
    let dir=std::env::temp_dir().join(format!("lq-window-time-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),8,1024,65536,std::time::Duration::from_millis(30)).unwrap();r.flush();assert!(!std::fs::read_dir(&dir).unwrap().flatten().any(|e|e.path().extension().is_some_and(|x|x=="log")));
    assert!(r.append("{}".into()));r.flush();std::thread::sleep(std::time::Duration::from_millis(100));
    let side=std::fs::read_dir(&dir).unwrap().flatten().find(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).unwrap();let v:serde_json::Value=serde_json::from_slice(&std::fs::read(side.path()).unwrap()).unwrap();assert_eq!(v["sealed"],true);r.close();drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn window_size_rotates_without_losing_records() {
    let dir=std::env::temp_dir().join(format!("lq-window-size-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),8,8,65536,std::time::Duration::from_secs(300)).unwrap();assert!(r.append("{\"a\":1}".into()));r.flush();assert!(r.append("{}".into()));r.close();
    let mut all=std::fs::read_dir(&dir).unwrap().flatten().filter(|e|e.path().extension().is_some_and(|x|x=="log")).map(|e|std::fs::read_to_string(e.path()).unwrap()).collect::<Vec<_>>();all.sort();assert_eq!(all,vec!["{\"a\":1}\n","{}\n"]);drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn window_recovery_before_first_publication_publishes_before_rename() {
    use std::os::unix::fs::PermissionsExt;
    let dir=std::env::temp_dir().join(format!("lq-window-prepub-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();let stem=format!("{}-{:020}","b".repeat(32),0);
    std::fs::write(dir.join(format!("{stem}.window")),b"").unwrap();std::fs::write(dir.join(format!("{stem}.pending")),b"{partial").unwrap();std::fs::set_permissions(dir.join(format!("{stem}.pending")),std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(dir.join(format!("{stem}.native-time.json")),r#"{"schema_version":1,"start_ts_ns":1793059200000000001,"end_ts_ns":null,"source_schema":"native-log-file.v1"}"#).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),4,1024,65536,std::time::Duration::from_secs(300)).unwrap();let v:serde_json::Value=serde_json::from_slice(&std::fs::read(dir.join(format!("{stem}.log.source.json"))).unwrap()).unwrap();assert_eq!(v["sealed"],true);assert_eq!(std::fs::read(dir.join(format!("{stem}.log"))).unwrap(),b"{partial");r.close();drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn window_rotates_before_193rd_live_publication() {
    let dir=std::env::temp_dir().join(format!("lq-window-budget-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),8,8*1024*1024,20*1024*1024,std::time::Duration::from_secs(300)).unwrap();
    for _ in 0..193 {assert!(r.append("{}".into()));r.flush();}
    r.close();
    let mut pubs=std::fs::read_dir(&dir).unwrap().flatten().filter(|e|e.file_name().to_string_lossy().ends_with(".log.source.json")).map(|e|serde_json::from_slice::<serde_json::Value>(&std::fs::read(e.path()).unwrap()).unwrap()).collect::<Vec<_>>();pubs.sort_by_key(|p|p["path"].as_str().unwrap().to_owned());assert_eq!(pubs.len(),2);assert_eq!(pubs[0]["publication_sequence"],193);assert_eq!(pubs[0]["published_length"],576);assert_eq!(pubs[1]["published_length"],3);drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn window_real_sender_streams_300_mixed_prefixes_within_chunk_limit() {
    use std::io::{BufRead,BufReader,Write};use std::process::{Command,Stdio};
    let core=std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let script=core.join("tests/support/window_capture.py");
    let protocol_root=std::env::var_os("LEAFQUANT_PROTOCOL_ROOT").map(std::path::PathBuf::from).unwrap_or_else(||core.parent().unwrap().parent().unwrap().to_path_buf());
    let dir=std::env::temp_dir().join(format!("lq-window-capture-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),8,4*1024*1024,64*1024*1024,std::time::Duration::from_secs(300)).unwrap();
    let mut child=Command::new("python3").arg(script).arg(&dir).arg(protocol_root).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().unwrap();let mut input=child.stdin.take().unwrap();let mut output=BufReader::new(child.stdout.take().unwrap());
    for i in 0..300 {let x="x".repeat(if i%10==0 {40000} else {3});assert!(r.append(format!("{{\"n\":{i},\"x\":\"{x}\"}}")));r.flush();writeln!(input,"{i}").unwrap();input.flush().unwrap();let mut answer=String::new();output.read_line(&mut answer).unwrap();assert_eq!(answer.trim(),"OK","iteration {i}");}
    r.close();writeln!(input,"seal").unwrap();input.flush().unwrap();let mut answer=String::new();output.read_line(&mut answer).unwrap();assert!(answer.starts_with("OK chunks="),"{answer}");println!("{}",answer.trim());drop(input);assert!(child.wait().unwrap().success());drop(r);std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn window_caps_configured_segment_bytes_at_four_mib() {
    let dir=std::env::temp_dir().join(format!("lq-window-cap-{}",std::process::id()));let _=std::fs::remove_dir_all(&dir);std::fs::create_dir(&dir).unwrap();
    let r=Recorder::new_windowed(dir.to_str().unwrap().into(),4,8*1024*1024,20*1024*1024,std::time::Duration::from_secs(300)).unwrap();
    assert!(!r.append(format!("{{\"x\":\"{}\"}}","x".repeat(4*1024*1024))));r.close();drop(r);std::fs::remove_dir_all(dir).unwrap();
}
