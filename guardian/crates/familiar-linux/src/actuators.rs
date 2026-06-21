//! The Linux actuator: a reversible nft drop rule for BlockOutbound and the
//! cgroup-v2 freezer for FreezeProcess. Tracks active blocks so it can reverse
//! them. Satisfies the same `Actuators` trait the testkit fake does.
use crate::{cgroup, nft};
use familiar_core::policy::ProposedAction;
use familiar_platform::{ActuationError, ActuationOutcome, Actuators};
use std::net::Ipv4Addr;
use std::path::PathBuf;

/// The Linux actuator. `apply` installs a block or freezes a process;
/// `reverse_all` removes every block.
pub struct LinuxActuators {
    freezer: cgroup::Freezer,
    active_blocks: Vec<(Ipv4Addr, u16)>,
}

impl LinuxActuators {
    pub fn new(cgroup_root: impl Into<PathBuf>) -> Result<Self, ActuationError> {
        Ok(Self {
            freezer: cgroup::Freezer::new(cgroup_root),
            active_blocks: Vec::new(),
        })
    }

    /// Reverse every block familiar installed by flushing the block chain. The
    /// table and the NFQUEUE sense chain are preserved, so sensing keeps running.
    /// Idempotent.
    pub fn reverse_all(&mut self) -> Result<(), ActuationError> {
        nft::flush_block_chain().map_err(|e| ActuationError::Failed(e.to_string()))?;
        self.active_blocks.clear();
        Ok(())
    }

    /// The destinations currently blocked (for status/Plan C).
    pub fn active_blocks(&self) -> &[(Ipv4Addr, u16)] {
        &self.active_blocks
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
                let handle = self
                    .freezer
                    .freeze(*pid)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
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
                Ok(ActuationOutcome {
                    note: format!("thawed pid {pid}"),
                })
            }
        }
    }

    fn reverse_all(&mut self) -> Result<ActuationOutcome, ActuationError> {
        LinuxActuators::reverse_all(self)?;
        Ok(ActuationOutcome {
            note: "flushed all outbound blocks".into(),
        })
    }
}
