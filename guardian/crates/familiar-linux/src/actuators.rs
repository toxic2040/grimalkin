//! The Linux actuator: a reversible nft drop rule for BlockOutbound and the
//! cgroup-v2 freezer for FreezeProcess. Tracks active blocks so it can reverse
//! them. Satisfies the same `Actuators` trait the testkit fake does.
use crate::{cgroup, nft};
use familiar_core::Pid;
use familiar_core::policy::ProposedAction;
use familiar_platform::{ActuationError, ActuationOutcome, Actuators};
use std::net::Ipv4Addr;
use std::path::PathBuf;

/// The Linux actuator. `apply` installs a block or freezes a process;
/// `reverse_all` removes every block and thaws every frozen process.
pub struct LinuxActuators {
    freezer: cgroup::Freezer,
    active_blocks: Vec<(Ipv4Addr, u16)>,
    active_freezes: Vec<Pid>,
}

impl LinuxActuators {
    pub fn new(cgroup_root: impl Into<PathBuf>) -> Result<Self, ActuationError> {
        Ok(Self {
            freezer: cgroup::Freezer::new(cgroup_root),
            active_blocks: Vec::new(),
            active_freezes: Vec::new(),
        })
    }

    /// Reverse every containment familiar installed: flush the block chain and
    /// thaw every frozen process. The table and the NFQUEUE sense chain are
    /// preserved, so sensing keeps running. Best-effort across both kinds — a
    /// failed flush must not leave a process frozen, and vice versa — so it tries
    /// each one and returns the first error. Idempotent.
    pub fn reverse_all(&mut self) -> Result<(), ActuationError> {
        let mut first_err = None;
        match nft::flush_block_chain() {
            Ok(()) => self.active_blocks.clear(),
            Err(e) => first_err = Some(ActuationError::Failed(e.to_string())),
        }
        for pid in std::mem::take(&mut self.active_freezes) {
            if let Err(e) = self.freezer.thaw(pid) {
                first_err.get_or_insert_with(|| ActuationError::Failed(e.to_string()));
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The destinations currently blocked (for status/Plan C).
    pub fn active_blocks(&self) -> &[(Ipv4Addr, u16)] {
        &self.active_blocks
    }

    /// The pids currently frozen (mirrors `active_blocks` for teardown/status).
    pub fn active_freezes(&self) -> &[Pid] {
        &self.active_freezes
    }
}

impl Actuators for LinuxActuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        match action {
            ProposedAction::BlockOutbound {
                dst_ip, dst_port, ..
            } => {
                let ip: Ipv4Addr = dst_ip.parse().map_err(|_| {
                    ActuationError::Failed(format!("non-IPv4 dst {dst_ip} (v0.1 is IPv4-only)"))
                })?;
                nft::ensure_table().map_err(|e| ActuationError::Failed(e.to_string()))?;
                let note = nft::block_outbound(ip, *dst_port)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_blocks.push((ip, *dst_port));
                Ok(ActuationOutcome { note })
            }
            ProposedAction::FreezeProcess { pid } => {
                // Track the freeze so disarm/reverse_all can thaw it. (No v0.1
                // detector proposes FreezeProcess yet; before one does, the
                // freeze path still needs process-identity revalidation to close
                // pid-reuse races — see SECURITY.md v0.1 limits.)
                let handle = self
                    .freezer
                    .freeze(*pid)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_freezes.push(*pid);
                Ok(ActuationOutcome {
                    note: format!("froze pid {pid} ({handle})"),
                })
            }
        }
    }

    fn reverse(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        match action {
            ProposedAction::BlockOutbound {
                dst_ip, dst_port, ..
            } => {
                let ip: Ipv4Addr = dst_ip
                    .parse()
                    .map_err(|_| ActuationError::Failed(format!("non-IPv4 dst {dst_ip}")))?;
                let note = nft::unblock_outbound(ip, *dst_port)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_blocks
                    .retain(|(i, p)| !(*i == ip && *p == *dst_port));
                Ok(ActuationOutcome { note })
            }
            ProposedAction::FreezeProcess { pid } => {
                self.freezer
                    .thaw(*pid)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_freezes.retain(|p| p != pid);
                Ok(ActuationOutcome {
                    note: format!("thawed pid {pid}"),
                })
            }
        }
    }

    fn reverse_all(&mut self) -> Result<ActuationOutcome, ActuationError> {
        LinuxActuators::reverse_all(self)?;
        Ok(ActuationOutcome {
            note: "flushed all outbound blocks and thawed all freezes".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A throwaway cgroup-shaped tree the real `Freezer` can drive without root:
    /// `freeze` reads `cgroup.events`, so seed the child cgroup as already frozen.
    fn seeded_root(pid: Pid) -> PathBuf {
        let root = std::env::temp_dir().join(format!("fam-act-{}-{}", std::process::id(), pid));
        let cg = root.join(format!("familiar-freeze-{pid}"));
        fs::create_dir_all(&cg).unwrap();
        fs::write(cg.join("cgroup.events"), "frozen 1\n").unwrap();
        root
    }

    #[test]
    fn reverse_all_thaws_tracked_freezes() {
        let pid: Pid = 4242;
        let root = seeded_root(pid);
        let cg = root.join(format!("familiar-freeze-{pid}"));
        let mut act = LinuxActuators::new(root.clone()).unwrap();

        act.apply(&ProposedAction::FreezeProcess { pid }).unwrap();
        assert_eq!(act.active_freezes(), [pid]);
        assert_eq!(fs::read_to_string(cg.join("cgroup.freeze")).unwrap(), "1");

        // reverse_all also flushes the nft block chain, which shells out to `nft`
        // and may fail in this environment; that must not stop the thaw, so we
        // ignore its Result and assert the freeze side-effects directly.
        let _ = LinuxActuators::reverse_all(&mut act);
        assert!(
            act.active_freezes().is_empty(),
            "disarm must stop tracking the freeze"
        );
        assert_eq!(
            fs::read_to_string(cg.join("cgroup.freeze")).unwrap(),
            "0",
            "disarm must thaw the frozen process"
        );

        let _ = fs::remove_dir_all(&root);
    }
}
