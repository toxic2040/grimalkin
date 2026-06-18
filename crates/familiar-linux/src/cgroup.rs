//! cgroup-v2 freezer. Freezes a process by moving it into a per-target child
//! cgroup under a daemon-owned root and writing `cgroup.freeze`. Reversible via
//! `thaw`. Confirmed against a delegated user cgroup by the Plan B spike.
use familiar_core::Pid;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CgroupError {
    #[error("cgroup io: {0}")]
    Io(String),
    #[error("process did not enter the frozen state")]
    NotFrozen,
}

fn io<E: std::fmt::Display>(e: E) -> CgroupError {
    CgroupError::Io(e.to_string())
}

/// Freezes processes by moving each into a per-target child cgroup under a
/// daemon-owned root and writing `cgroup.freeze`. Reversible via `thaw`.
pub struct Freezer {
    root: PathBuf,
}

impl Freezer {
    pub fn new(cgroup_root: impl Into<PathBuf>) -> Self {
        Self {
            root: cgroup_root.into(),
        }
    }

    fn cg_for(&self, pid: Pid) -> PathBuf {
        self.root.join(format!("familiar-freeze-{pid}"))
    }

    /// Create a child cgroup, move `pid` into it, freeze it. Returns the cgroup
    /// path as the reversal handle.
    pub fn freeze(&self, pid: Pid) -> Result<String, CgroupError> {
        let cg = self.cg_for(pid);
        fs::create_dir_all(&cg).map_err(io)?;
        // Move the pid into the child cgroup.
        fs::write(cg.join("cgroup.procs"), pid.to_string()).map_err(io)?;
        fs::write(cg.join("cgroup.freeze"), "1").map_err(io)?;
        // Confirm via cgroup.events (the authoritative signal; /proc status does
        // not flip to a distinct frozen value).
        let events = fs::read_to_string(cg.join("cgroup.events")).map_err(io)?;
        if !events.lines().any(|l| l == "frozen 1") {
            return Err(CgroupError::NotFrozen);
        }
        Ok(cg.to_string_lossy().into_owned())
    }

    /// Thaw the per-target cgroup. Best-effort rmdir afterward (a cgroup that
    /// still holds the live process cannot be removed yet — that is fine).
    pub fn thaw(&self, pid: Pid) -> Result<(), CgroupError> {
        let cg = self.cg_for(pid);
        fs::write(cg.join("cgroup.freeze"), "0").map_err(io)?;
        let _ = fs::remove_dir(&cg);
        Ok(())
    }
}
