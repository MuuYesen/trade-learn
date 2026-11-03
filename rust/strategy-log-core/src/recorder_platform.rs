//! Recorder filesystem primitives. Windows is not a Unix directory-fsync emulation.
//!
//! Windows publishes same-directory names with MoveFileExW(WRITE_THROUGH), never
//! COPY_ALLOWED, after FlushFileBuffers (File::sync_all) on WRITE_THROUGH files.
//! This supports the tested process-crash contract; storage controllers and
//! filesystems can impose additional limits under sudden power loss.
//! https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-movefileexw
//! https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers
//! https://learn.microsoft.com/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
//! https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle
use std::{fs::{self, File, OpenOptions}, io, path::Path};

pub(super) fn create_new(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        // READ | WRITE | DELETE sharing keeps an owned handle valid across an
        // atomic rename. Writer exclusion comes from recorder.lock.try_lock().
        options.custom_flags(0x80000000).share_mode(1 | 2 | 4);
    }
    #[cfg(unix)] { options.open(path) }
    #[cfg(windows)] {
        // Persist the new name with the same write-through namespace operation
        // used for completed publications. No cross-volume copy fallback.
        let temporary = path.with_file_name(format!(".{}.{}.creating", path.file_name().unwrap().to_string_lossy(), super::run_id()));
        let file = options.open(&temporary)?;
        file.sync_all()?;
        if let Err(error) = move_file(&temporary, path, false) { drop(file); let _ = fs::remove_file(temporary); return Err(error); }
        Ok(file)
    }
}

pub(super) fn open_for_sync(path: &Path) -> io::Result<File> {
    #[cfg(unix)] { File::open(path) }
    #[cfg(windows)] {
        use std::os::windows::fs::OpenOptionsExt;
        OpenOptions::new().read(true).write(true).custom_flags(0x80000000).share_mode(1 | 2 | 4).open(path)
    }
}

#[cfg(unix)]
pub(super) fn sync_directory(path: &Path) -> io::Result<()> { File::open(path)?.sync_all() }

pub(super) fn move_file(source: &Path, destination: &Path, replace: bool) -> io::Result<()> {
    if source.parent() != destination.parent() { return Err(io::Error::other("recorder publication must remain in one directory")); }
    #[cfg(unix)] {
        if !replace && destination.try_exists()? { return Err(io::ErrorKind::AlreadyExists.into()); }
        fs::rename(source, destination)
    }
    #[cfg(windows)] {
        use std::os::windows::ffi::OsStrExt;
        let src: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let dst: Vec<u16> = destination.as_os_str().encode_wide().chain(Some(0)).collect();
        if src[..src.len()-1].contains(&0) || dst[..dst.len()-1].contains(&0) { return Err(io::Error::other("embedded NUL in recorder path")); }
        // WRITE_THROUGH | optional REPLACE_EXISTING. COPY_ALLOWED is absent.
        if unsafe { MoveFileExW(src.as_ptr(), dst.as_ptr(), 8 | u32::from(replace)) } == 0 { return Err(io::Error::last_os_error()); }
        Ok(())
    }
}

pub(super) fn identity(file: &File) -> io::Result<(u64, u64, u64)> {
    #[cfg(unix)] { use std::os::unix::fs::MetadataExt; let m=file.metadata()?; Ok((m.dev(), m.ino(), m.nlink())) }
    #[cfg(windows)] {
        use std::os::windows::io::AsRawHandle;
        let mut info = std::mem::MaybeUninit::<FileInformation>::uninit();
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), info.as_mut_ptr()) } == 0 { return Err(io::Error::last_os_error()); }
        let info=unsafe { info.assume_init() };
        Ok((u64::from(info.volume_serial), (u64::from(info.index_high)<<32)|u64::from(info.index_low), u64::from(info.links)))
    }
}

pub(super) fn random(bytes: &mut [u8; 16]) -> io::Result<()> {
    #[cfg(unix)] { use std::io::Read; File::open("/dev/urandom")?.read_exact(bytes) }
    #[cfg(windows)] {
        let status=unsafe { BCryptGenRandom(std::ptr::null_mut(), bytes.as_mut_ptr(), 16, 2) };
        if status < 0 { return Err(io::Error::other(format!("Windows system entropy failed: {status:#x}"))); }
        Ok(())
    }
}

#[cfg(windows)]
#[repr(C)]
struct FileInformation {
    attributes: u32, creation: [u32;2], access: [u32;2], write: [u32;2],
    volume_serial: u32, size_high: u32, size_low: u32, links: u32, index_high: u32, index_low: u32,
}
#[cfg(windows)]
#[link(name="kernel32")]
unsafe extern "system" {
    fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    fn GetFileInformationByHandle(handle: *mut std::ffi::c_void, information: *mut FileInformation) -> i32;
}
#[cfg(windows)]
#[link(name="bcrypt")]
unsafe extern "system" {
    fn BCryptGenRandom(algorithm: *mut std::ffi::c_void, buffer: *mut u8, length: u32, flags: u32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn entropy_and_atomic_publication_never_overwrite_without_permission() {
        let mut first=[0;16]; let mut second=[0;16];
        random(&mut first).unwrap(); random(&mut second).unwrap();
        assert_ne!(first,second);
        let root=std::env::temp_dir().join(format!("lq-platform-publication-{}",super::super::run_id()));
        fs::create_dir(&root).unwrap();
        let source=root.join("source"); let target=root.join("target");
        let mut file=create_new(&source).unwrap();file.write_all(b"complete-new-value").unwrap();file.sync_all().unwrap();
        let mut old=create_new(&target).unwrap();old.write_all(b"old").unwrap();old.sync_all().unwrap();
        assert!(move_file(&source,&target,false).is_err());
        assert_eq!(fs::read(&target).unwrap(),b"old");
        move_file(&source,&target,true).unwrap();
        assert_eq!(fs::read(&target).unwrap(),b"complete-new-value");
        assert!(!source.exists());
        assert!(move_file(&target,&root.parent().unwrap().join("outside"),true).is_err());
        let identity_before=identity(&file).unwrap();
        assert_eq!(identity_before,identity(&open_for_sync(&target).unwrap()).unwrap());
        drop(old);drop(file);fs::remove_dir_all(root).unwrap();
    }
}
