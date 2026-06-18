//! familiar-fanotify-helper — the privileged file-read sensor. Holds
//! CAP_SYS_ADMIN; does nothing but watch the configured prefixes and stream
//! FileRead events to the daemon. Minimal by design: the broad capability is
//! isolated here, away from the network daemon. This is the only crate in the
//! workspace that uses `unsafe`.
mod fanotify;

use serde::Serialize;
use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::{SystemTime, UNIX_EPOCH};

/// Mirrors familiar_linux::wire::FileReadEvent. Duplicated (not a dep) to keep
/// the privileged binary's dependency surface minimal.
#[derive(Serialize)]
struct FileReadEvent {
    at: u64,
    pid: u32,
    exe: String,
    path: String,
}

fn is_watched(path: &str, prefixes: &[String]) -> bool {
    prefixes.iter().any(|p| path.starts_with(p.as_str()))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn exe_of(pid: i32) -> String {
    std::fs::read_link(format!("/proc/{pid}/exe"))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let socket = args
        .next()
        .expect("usage: familiar-fanotify-helper <socket> <prefix>...");
    let prefixes: Vec<String> = args.collect();
    assert!(!prefixes.is_empty(), "at least one watched prefix required");

    let group = fanotify::init().expect("fanotify_init (needs CAP_SYS_ADMIN)");
    for p in &prefixes {
        if let Err(e) = fanotify::mark(&group, p) {
            eprintln!("[helper] mark {p} failed: {e}");
        }
    }
    eprintln!(
        "[helper] marks placed for {} prefix(es); connecting to {socket}",
        prefixes.len()
    );

    let mut out = UnixStream::connect(&socket).expect("connect to daemon socket");
    eprintln!("[helper] connected; watching for reads");
    loop {
        let events = match fanotify::read_events(&group) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[helper] read error: {e}");
                continue;
            }
        };
        for ev in events {
            if !fanotify::should_forward(ev.fd) {
                // FAN_Q_OVERFLOW / FAN_NOFD: events were dropped by the kernel
                // queue. Do NOT wrap fd<0 in OwnedFd. Log loudly — a lost read is
                // a sensing gap, not a silent success.
                eprintln!(
                    "[helper] fanotify overflow/no-fd (fd={}); file-read events were dropped",
                    ev.fd
                );
                continue;
            }
            let path = std::fs::read_link(format!("/proc/self/fd/{}", ev.fd))
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            // SAFETY: should_forward guaranteed ev.fd >= 0; it is the valid fd the
            // kernel handed us in this event. Wrap it so it closes once on drop.
            let _owned = unsafe { OwnedFd::from_raw_fd(ev.fd) };
            if ev.pid > 0 && is_watched(&path, &prefixes) {
                let fr = FileReadEvent {
                    at: now_ms(),
                    pid: ev.pid as u32,
                    exe: exe_of(ev.pid),
                    path,
                };
                let line = serde_json::to_string(&fr).expect("serialize");
                eprintln!("[helper] emit FileRead pid={} -> daemon", fr.pid);
                if writeln!(out, "{line}").is_err() {
                    eprintln!("[helper] daemon socket closed; exiting");
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_watched_prefix() {
        let prefixes = vec!["/home/u/.ssh".to_string(), "/etc/shadow".to_string()];
        assert!(is_watched("/home/u/.ssh/id_ed25519", &prefixes));
        assert!(!is_watched("/home/u/Documents/notes.txt", &prefixes));
    }
}
