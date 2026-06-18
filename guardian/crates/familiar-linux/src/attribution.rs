//! Best-effort `/proc` socket → PID attribution: `/proc/net/tcp{,6}` maps a
//! local port to a socket inode; `/proc/<pid>/fd/*` maps the inode to a PID.
//! Racy by nature (a process that exits before the scan is unattributable);
//! eBPF socket attribution is the v0.2 upgrade. Confirmed by the Plan B spike.
use familiar_core::Pid;
use familiar_core::events::ProcessRef;
use std::fs;

fn inode_for_local_port(path: &str, port: u16) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 {
            continue;
        }
        let Some((_, port_hex)) = f[1].split_once(':') else {
            continue;
        };
        if u16::from_str_radix(port_hex, 16).ok()? == port
            && let Ok(inode) = f[9].parse::<u64>()
        {
            return Some(inode);
        }
    }
    None
}

fn pid_for_inode(inode: u64) -> Option<Pid> {
    let needle = format!("socket:[{inode}]");
    for entry in fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<Pid>().ok())
        else {
            continue;
        };
        let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else {
            continue;
        };
        for fd in fds.flatten() {
            if fs::read_link(fd.path())
                .map(|t| t.to_string_lossy() == needle)
                .unwrap_or(false)
            {
                return Some(pid);
            }
        }
    }
    None
}

/// Best-effort attribution of a local source port to the owning process.
/// Returns None when the owner cannot be found (the documented exit race) — the
/// caller must treat that as "unknown process", never guess.
pub fn attribute(src_port: u16) -> Option<ProcessRef> {
    let inode = inode_for_local_port("/proc/net/tcp", src_port)
        .or_else(|| inode_for_local_port("/proc/net/tcp6", src_port))?;
    let pid = pid_for_inode(inode)?;
    let exe = fs::read_link(format!("/proc/{pid}/exe"))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(ProcessRef { pid, exe })
}
