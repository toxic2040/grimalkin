//! Localized-unsafe fanotify wrapper. CAP_SYS_ADMIN required (the Plan B spike
//! showed fanotify_init returns EPERM otherwise, even inside unshare -Ur). Every
//! `unsafe` block is annotated with why it is sound.
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

/// Initialize a fanotify group reporting the accessing PID (FAN_REPORT_PIDFD) and
/// classed for notification only (FAN_CLASS_NOTIF).
pub fn init() -> io::Result<OwnedFd> {
    // SAFETY: fanotify_init takes two scalar flag words and returns a new fd or
    // -1. No memory is shared; we wrap the returned fd in OwnedFd for RAII.
    let fd = unsafe {
        libc::fanotify_init(
            libc::FAN_CLASS_NOTIF | libc::FAN_REPORT_PIDFD | libc::FAN_CLOEXEC,
            (libc::O_RDONLY | libc::O_CLOEXEC) as u32,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is a fresh, valid, owned fd returned by fanotify_init above.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Mark the directory `path` so opens of the files *inside* it generate events
/// (`FAN_EVENT_ON_CHILD`). A bare directory mark reports only events on the
/// directory inode itself; `FAN_EVENT_ON_CHILD` extends it to the directory's
/// children — which is exactly "watch this sensitive directory" without the
/// firehose (and privacy leak, and queue-overflow) of marking the whole mount.
///
/// v0.1 watches the *direct* children of each configured directory; nested
/// subtrees need their subdirectories marked too (a recursive walk, or eBPF, is
/// the v0.2 upgrade). The caller still filters by path prefix as defense in depth.
pub fn mark(group: &OwnedFd, path: &str) -> io::Result<()> {
    let c = std::ffi::CString::new(path).map_err(|_| io::Error::other("nul in path"))?;
    // SAFETY: group.as_raw_fd() is a valid fd for the call; `c` is a valid
    // NUL-terminated C string that outlives the call; the mask/flags are scalars.
    let rc = unsafe {
        libc::fanotify_mark(
            group.as_raw_fd(),
            libc::FAN_MARK_ADD,
            libc::FAN_OPEN | libc::FAN_ACCESS | libc::FAN_EVENT_ON_CHILD,
            libc::AT_FDCWD,
            c.as_ptr(),
        )
    };
    if rc < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// One decoded event: the accessing PID and the fd referring to the accessed
/// file. The caller resolves the path via /proc/self/fd/<fd> and must close it.
pub struct RawEvent {
    pub pid: i32,
    pub fd: RawFd,
}

/// Read and decode the next batch of events. Blocks until at least one arrives.
pub fn read_events(group: &OwnedFd) -> io::Result<Vec<RawEvent>> {
    let mut buf = [0u8; 4096];
    // SAFETY: read into a valid, sufficiently sized local buffer; on success the
    // kernel has initialized exactly `rc` bytes.
    let rc = unsafe {
        libc::read(
            group.as_raw_fd(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut out = Vec::new();
    let mut off = 0usize;
    let meta_len = std::mem::size_of::<libc::fanotify_event_metadata>();
    while off + meta_len <= rc as usize {
        // SAFETY: off..off+meta_len lies within the `rc` bytes the kernel wrote;
        // fanotify guarantees event_len-aligned, fully-populated metadata records.
        let meta = unsafe {
            std::ptr::read_unaligned(buf.as_ptr().add(off) as *const libc::fanotify_event_metadata)
        };
        if meta.event_len as usize == 0 {
            break;
        }
        out.push(RawEvent {
            pid: meta.pid,
            fd: meta.fd,
        });
        off += meta.event_len as usize;
    }
    Ok(out)
}

/// Whether an event's fd is real and forwardable. A fanotify queue overflow
/// yields `FAN_NOFD` (-1); such an event must never be wrapped in `OwnedFd`.
pub fn should_forward(fd: RawFd) -> bool {
    fd >= 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_or_nofd_is_not_forwarded() {
        assert!(
            !should_forward(-1),
            "FAN_NOFD must be dropped, never wrapped"
        );
        assert!(should_forward(3), "a real fd is forwarded");
    }
}
