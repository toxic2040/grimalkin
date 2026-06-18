use crate::Timestamp;
use crate::audit::{AuditKind, AuditLog};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Every sensor, detector, and actuator is a named capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CapabilityId {
    SensorOutboundConn,
    SensorSensitiveRead,
    DetectorExfil,
    ActuatorBlockConn,
    ActuatorFreezeProcess,
}

impl CapabilityId {
    /// Every capability the v0.1 spine knows about.
    pub const ALL: [CapabilityId; 5] = [
        CapabilityId::SensorOutboundConn,
        CapabilityId::SensorSensitiveRead,
        CapabilityId::DetectorExfil,
        CapabilityId::ActuatorBlockConn,
        CapabilityId::ActuatorFreezeProcess,
    ];
}

/// A serializable snapshot of capability states, for the daemon to persist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    pub states: BTreeMap<CapabilityId, bool>,
}

/// The Capability Registry: every capability default-off, fail-closed, with
/// every toggle written to the audit log.
#[derive(Clone, Debug)]
pub struct CapabilityRegistry {
    states: BTreeMap<CapabilityId, bool>,
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityRegistry {
    /// A fresh registry with every known capability registered and OFF.
    pub fn new() -> Self {
        let mut states = BTreeMap::new();
        for id in CapabilityId::ALL {
            states.insert(id, false);
        }
        Self { states }
    }

    /// Fail-closed: an unknown/missing capability reads as disabled.
    pub fn is_enabled(&self, id: CapabilityId) -> bool {
        self.states.get(&id).copied().unwrap_or(false)
    }

    /// Toggle a capability and record the change. Returns the new state. The
    /// toggle is physical — it is not a preference a model can override.
    pub fn set(
        &mut self,
        id: CapabilityId,
        enabled: bool,
        at: Timestamp,
        audit: &mut AuditLog,
    ) -> bool {
        self.states.insert(id, enabled);
        audit.append(
            at,
            AuditKind::CapabilityToggled,
            format!("{id:?} -> {}", if enabled { "on" } else { "off" }),
        );
        enabled
    }

    pub fn snapshot(&self) -> CapabilitySnapshot {
        CapabilitySnapshot {
            states: self.states.clone(),
        }
    }

    /// Restore from a snapshot, re-registering any capability the snapshot omits
    /// as OFF (fail-closed across version skew).
    pub fn restore(snapshot: CapabilitySnapshot) -> Self {
        let mut reg = Self::new();
        for (id, on) in snapshot.states {
            reg.states.insert(id, on);
        }
        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_capability_defaults_off() {
        let reg = CapabilityRegistry::new();
        for id in CapabilityId::ALL {
            assert!(!reg.is_enabled(id), "{id:?} should default off");
        }
    }

    #[test]
    fn set_flips_state_and_audits_the_toggle() {
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 100, &mut audit);
        assert!(reg.is_enabled(CapabilityId::DetectorExfil));
        assert_eq!(audit.records().len(), 1);
        assert_eq!(audit.records()[0].kind, AuditKind::CapabilityToggled);
        assert!(audit.records()[0].detail.contains("DetectorExfil"));
    }

    #[test]
    fn restore_treats_a_missing_capability_as_off() {
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 1, &mut audit);
        let mut snap = reg.snapshot();
        snap.states.remove(&CapabilityId::DetectorExfil); // simulate version skew
        let restored = CapabilityRegistry::restore(snap);
        assert!(!restored.is_enabled(CapabilityId::DetectorExfil));
    }

    #[test]
    fn snapshot_round_trips_through_json() {
        // Proves the persistence seam the daemon (Plan B) relies on.
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::ActuatorFreezeProcess, true, 1, &mut audit);
        let json = serde_json::to_string(&reg.snapshot()).unwrap();
        let restored = CapabilityRegistry::restore(serde_json::from_str(&json).unwrap());
        assert!(restored.is_enabled(CapabilityId::ActuatorFreezeProcess));
        assert!(!restored.is_enabled(CapabilityId::DetectorExfil));
    }
}
