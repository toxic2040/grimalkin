#![forbid(unsafe_code)]
//! familiar-runtime — the OS-agnostic supervisor that drives the guardian loop.
//!
//! It owns the engine, the permission ledger, and the audit log, and drives them
//! over the `Sensors`/`Actuators`/`Notifier` seam. The daemon instantiates it
//! with the real Linux adapter; tests use the testkit fakes.

use familiar_core::Timestamp;
use familiar_core::advisor::Advisor;
use familiar_core::audit::{AuditKind, AuditLog};
use familiar_core::permission::{PermissionLedger, RequestId};
use familiar_core::policy::{Decision, Disposition, Engine, ProposedAction};
use familiar_platform::{Actuators, Notifier, Sensors};

/// Wires the deterministic core to a platform adapter and an advisor, and drives
/// the detect -> decide -> act/ask -> audit -> notify loop.
pub struct Supervisor<S, A, N, V> {
    pub engine: Engine,
    pub ledger: PermissionLedger,
    pub audit: AuditLog,
    sensors: S,
    actuators: A,
    notifier: N,
    advisor: V,
    default_timeout_ms: u64,
}

impl<S, A, N, V> Supervisor<S, A, N, V>
where
    S: Sensors,
    A: Actuators,
    N: Notifier,
    V: Advisor,
{
    pub fn new(
        engine: Engine,
        sensors: S,
        actuators: A,
        notifier: N,
        advisor: V,
        default_timeout_ms: u64,
    ) -> Self {
        Self {
            engine,
            ledger: PermissionLedger::new(),
            audit: AuditLog::new(),
            sensors,
            actuators,
            notifier,
            advisor,
            default_timeout_ms,
        }
    }

    /// Borrow the notifier (tests inspect captured messages/requests).
    pub fn notifier(&self) -> &N {
        &self.notifier
    }
    /// Borrow the actuators (tests inspect applied actions).
    pub fn actuators(&self) -> &A {
        &self.actuators
    }

    /// Expire every overdue request — a timeout is a denial — and record each.
    /// Called at the start of every tick and before resolving any human
    /// decision, so a grant that arrives past the deadline resolves to deny.
    fn sweep_expired(&mut self, now: Timestamp) {
        for (_outcome, req) in self.ledger.expire_due(now) {
            self.audit.append(
                now,
                AuditKind::PermissionResolved,
                format!("request {} timed out -> deny", req.id),
            );
        }
    }

    /// One tick: expire overdue requests (timeout => deny), then poll sensors and
    /// handle each event's decision.
    pub fn drive_once(&mut self, now: Timestamp) {
        self.sweep_expired(now);
        let events = self.sensors.poll();
        for ev in events {
            if let Some(decision) = self.engine.intake(&ev, &self.advisor, &mut self.audit) {
                self.dispatch(decision, now);
            }
        }
    }

    /// An idle control tick: expire pending requests without polling sensors.
    /// The daemon uses this while disarmed.
    pub fn drive_idle(&mut self, now: Timestamp) {
        self.sweep_expired(now);
    }

    /// Resolve a pending request by explicit human decision. A grant acts; a
    /// denial records no-action.
    pub fn resolve_permission(&mut self, id: RequestId, granted: bool, now: Timestamp) {
        // A decision arriving at/after the deadline is a timeout, not a grant
        // (spec §4.2): sweep first so an overdue request resolves to deny rather
        // than acting on a late click or a delayed UI event.
        self.sweep_expired(now);
        if let Some((outcome, req)) = self.ledger.resolve(id, granted) {
            self.audit.append(
                now,
                AuditKind::PermissionResolved,
                format!("request {id} -> {outcome:?}"),
            );
            if outcome.permits_action() {
                let action = req.detection.proposed.clone();
                // Defensive re-check (spec §7, fail-closed): the actuator
                // capability may have been switched OFF between raising this
                // prompt and the grant. A disabled actuator cannot fire even on
                // an explicit grant — the toggle is physical.
                if self
                    .engine
                    .registry()
                    .is_enabled(action.actuator_capability())
                {
                    self.act(&action, now);
                } else {
                    self.audit.append(
                        now,
                        AuditKind::NoAction,
                        format!(
                            "grant ignored: {:?} actuator disabled",
                            action.actuator_capability()
                        ),
                    );
                    self.notifier
                        .notify("Grant ignored: the required capability is now disabled");
                }
            }
        }
    }

    fn dispatch(&mut self, decision: Decision, now: Timestamp) {
        match decision.disposition {
            Disposition::ActAutonomously => {
                let action = decision.detection.proposed.clone();
                // Timestamp the actuation with when it actually happens (now),
                // not the event time, for an honest forensic timeline.
                self.act(&action, now);
            }
            Disposition::RequirePermission => {
                let id = self
                    .ledger
                    .open(now, self.default_timeout_ms, decision.detection.clone());
                self.audit.append(
                    now,
                    AuditKind::PermissionRequested,
                    format!("request {id}: {}", decision.detection.rationale),
                );
                if let Some(req) = self.ledger.get(id).cloned() {
                    self.notifier.request_permission(&req);
                }
            }
            // Emitted by the actuator-capability gate (fail-closed) when the
            // required actuator is disabled. Records no-action and returns.
            Disposition::Deny => {
                self.audit
                    .append(now, AuditKind::NoAction, "denied by gate");
            }
        }
    }

    /// Lift a previously-applied containment (remove a block / thaw a process).
    /// This is the only public actuation path besides the gated `act` — and it
    /// can only *reduce* containment. Audited and surfaced like an actuation.
    // Result<(), ()> is intentional: the daemon maps Ok→IPC-ack, Err→IPC-nack.
    #[allow(clippy::result_unit_err)]
    pub fn reverse_action(&mut self, action: &ProposedAction, now: Timestamp) -> Result<(), ()> {
        match self.actuators.reverse(action) {
            Ok(outcome) => {
                self.audit.append(
                    now,
                    AuditKind::Actuation,
                    format!("reversed {action:?}: {}", outcome.note),
                );
                self.notifier
                    .notify(&format!("Lifted containment {action:?} ({})", outcome.note));
                Ok(())
            }
            Err(e) => {
                self.audit.append(
                    now,
                    AuditKind::NoAction,
                    format!("reversal failed for {action:?}: {e}"),
                );
                self.notifier
                    .notify(&format!("Could not lift containment: {e}"));
                Err(())
            }
        }
    }

    /// Lift every containment the platform adapter can track. This is used by
    /// the master disarm path and can only reduce containment.
    #[allow(clippy::result_unit_err)]
    pub fn reverse_all_containment(&mut self, now: Timestamp) -> Result<(), ()> {
        match self.actuators.reverse_all() {
            Ok(outcome) => {
                self.audit.append(
                    now,
                    AuditKind::Actuation,
                    format!("reversed all containment: {}", outcome.note),
                );
                self.notifier
                    .notify(&format!("Lifted all containment ({})", outcome.note));
                Ok(())
            }
            Err(e) => {
                self.audit.append(
                    now,
                    AuditKind::NoAction,
                    format!("reverse all containment failed: {e}"),
                );
                self.notifier
                    .notify(&format!("Could not lift all containment: {e}"));
                Err(())
            }
        }
    }

    /// Carry out a reversible action. Fail-closed: an actuation error degrades to
    /// a recorded no-action, never to a silent pass.
    fn act(&mut self, action: &ProposedAction, at: Timestamp) {
        match self.actuators.apply(action) {
            Ok(outcome) => {
                self.audit.append(
                    at,
                    AuditKind::Actuation,
                    format!("{action:?}: {}", outcome.note),
                );
                self.notifier
                    .notify(&format!("Contained {action:?} ({})", outcome.note));
            }
            Err(e) => {
                self.audit.append(
                    at,
                    AuditKind::NoAction,
                    format!("actuation failed for {action:?}: {e}"),
                );
                self.notifier
                    .notify(&format!("Containment failed, no action taken: {e}"));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_advisor::NullAdvisor;
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
    use familiar_core::events::{Event, ProcessRef};
    use familiar_core::policy::{ExfilConfig, ExfilDetector};
    use familiar_platform::testkit::{CapturingNotifier, FakeSensors, RecordingActuators};

    fn proc(pid: u32) -> ProcessRef {
        ProcessRef {
            pid,
            exe: "/usr/bin/curl".into(),
        }
    }

    fn armed_engine() -> Engine {
        let mut reg = CapabilityRegistry::new();
        let mut throwaway = AuditLog::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
            CapabilityId::ActuatorBlockConn,
        ] {
            reg.set(cap, true, 0, &mut throwaway);
        }
        let det = ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            ..ExfilConfig::default()
        });
        Engine::new(reg, det)
    }

    #[test]
    fn reverse_action_reverses_audits_and_notifies_but_never_installs() {
        use familiar_core::events::ProcessRef;
        let mut sup = Supervisor::new(
            armed_engine(),
            FakeSensors::new(vec![]),
            RecordingActuators::default(),
            CapturingNotifier::default(),
            NullAdvisor,
            30_000,
        );
        let block = ProposedAction::BlockOutbound {
            process: ProcessRef {
                pid: 7,
                exe: "/usr/bin/curl".into(),
            },
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        };
        sup.reverse_action(&block, 1000).expect("reverse ok");
        // It went through reverse(), NOT apply(): nothing was installed.
        assert!(
            sup.actuators().applied.is_empty(),
            "reverse must never install"
        );
        assert_eq!(sup.actuators().reversed.len(), 1);
        assert!(
            sup.audit
                .records()
                .iter()
                .any(|r| r.kind == AuditKind::Actuation && r.detail.contains("reversed")),
            "the reversal is audited"
        );
        assert!(
            !sup.notifier().messages.is_empty(),
            "the reversal is surfaced"
        );
        assert!(sup.audit.verify().is_ok());
    }

    #[test]
    fn a_grant_after_the_actuator_is_disabled_does_not_act() {
        use familiar_core::capabilities::CapabilityId;
        // Arm everything, raise an ambiguous prompt (confidence 50 => ask).
        let sensors = FakeSensors::new(vec![vec![Event::OutboundConn {
            at: 1000,
            process: proc(7),
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }]]);
        let mut sup = Supervisor::new(
            armed_engine(),
            sensors,
            RecordingActuators::default(),
            CapturingNotifier::default(),
            NullAdvisor,
            30_000,
        );
        sup.drive_once(1000);
        let id = sup.notifier().requests[0].id;
        // The operator now disables the block actuator, THEN grants the stale prompt.
        sup.engine
            .set_capability(CapabilityId::ActuatorBlockConn, false, 1100, &mut sup.audit);
        sup.resolve_permission(id, true, 1200);
        assert!(
            sup.actuators().applied.is_empty(),
            "a grant cannot resurrect a disabled actuator"
        );
        assert!(
            sup.audit
                .records()
                .iter()
                .any(|r| r.kind == AuditKind::NoAction && r.detail.contains("disabled")),
            "the ignored grant is recorded"
        );
    }

    #[test]
    fn high_confidence_exfil_is_contained_audited_and_notified() {
        let sensors = FakeSensors::new(vec![vec![
            Event::FileRead {
                at: 1000,
                process: proc(7),
                path: "/home/u/.ssh/id_ed25519".into(),
            },
            Event::OutboundConn {
                at: 1500,
                process: proc(7),
                dst_ip: "203.0.113.9".into(),
                dst_port: 443,
            },
        ]]);
        let mut sup = Supervisor::new(
            armed_engine(),
            sensors,
            RecordingActuators::default(),
            CapturingNotifier::default(),
            NullAdvisor,
            30_000,
        );
        sup.drive_once(2000);
        assert_eq!(sup.actuators().applied.len(), 1, "blocked reversibly");
        assert!(matches!(
            sup.actuators().applied[0],
            ProposedAction::BlockOutbound { .. }
        ));
        assert!(!sup.notifier().messages.is_empty(), "notified");
        assert!(sup.audit.verify().is_ok(), "audit chain intact");
    }
}
