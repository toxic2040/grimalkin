use crate::advisor::{Advice, Advisor, apply_caution};
use crate::audit::{AuditKind, AuditLog};
use crate::capabilities::{CapabilityId, CapabilityRegistry};
use crate::events::{Event, ProcessRef};
use crate::{Pid, Timestamp};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Part A — action vocabulary and the authority envelope
// ---------------------------------------------------------------------------

/// Rule-derived confidence, 0..=100. Never model-derived in v0.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Confidence(pub u8);

/// At or above this, a reversible action may run autonomously.
pub const HIGH_CONFIDENCE: u8 = 80;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reversibility {
    Reversible,
    Irreversible,
}

/// A platform-neutral action the core may propose. The platform layer maps each
/// to a concrete OS operation. v0.1 ships two, both reversible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposedAction {
    /// Block/hold an outbound connection (e.g. a removable firewall rule).
    BlockOutbound {
        process: ProcessRef,
        dst_ip: String,
        dst_port: u16,
    },
    /// Freeze the offending process pending a decision.
    FreezeProcess { pid: Pid },
}

impl ProposedAction {
    pub fn reversibility(&self) -> Reversibility {
        match self {
            ProposedAction::BlockOutbound { .. } | ProposedAction::FreezeProcess { .. } => {
                Reversibility::Reversible
            }
        }
    }

    /// The actuator capability that must be enabled to carry this out.
    pub fn actuator_capability(&self) -> CapabilityId {
        match self {
            ProposedAction::BlockOutbound { .. } => CapabilityId::ActuatorBlockConn,
            ProposedAction::FreezeProcess { .. } => CapabilityId::ActuatorFreezeProcess,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionKind {
    ExfilSuspected,
}

/// A detector's output: what it saw, how sure it is, and what it proposes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detection {
    pub at: Timestamp,
    pub kind: DetectionKind,
    pub confidence: Confidence,
    pub proposed: ProposedAction,
    pub rationale: String,
}

/// The authority envelope's verdict for a proposed action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// Reversible, high-confidence, permitted: act then notify.
    ActAutonomously,
    /// Irreversible or ambiguous: ask the human; deny on timeout.
    RequirePermission,
    /// Do nothing (a gate refused).
    Deny,
}

/// The authority envelope as a total function of the two facts that matter.
/// Pure and exhaustively testable — including the irreversible branch that no
/// v0.1 action exercises yet but which guards every future actuator.
pub fn classify_parts(reversibility: Reversibility, confidence: Confidence) -> Disposition {
    match reversibility {
        Reversibility::Irreversible => Disposition::RequirePermission,
        Reversibility::Reversible => {
            if confidence.0 >= HIGH_CONFIDENCE {
                Disposition::ActAutonomously
            } else {
                Disposition::RequirePermission
            }
        }
    }
}

/// The authority envelope for a concrete detection.
pub fn classify(detection: &Detection) -> Disposition {
    classify_parts(detection.proposed.reversibility(), detection.confidence)
}

// ---------------------------------------------------------------------------
// Part B — the exfiltration detector
// ---------------------------------------------------------------------------

/// Configuration for the v0.1 exfiltration detector.
#[derive(Clone, Debug)]
pub struct ExfilConfig {
    /// Path prefixes treated as sensitive (e.g. `/home/u/.ssh`).
    pub sensitive_prefixes: Vec<String>,
    /// Destination IPs with an established basis (never flagged).
    pub established_dsts: Vec<String>,
    /// How long after a sensitive read an outbound connection counts as linked.
    pub link_window_ms: u64,
    /// Confidence when a recent sensitive read is linked to the connection.
    pub linked_confidence: u8,
    /// Confidence for an unestablished outbound with no recent sensitive read.
    pub unlinked_confidence: u8,
}

impl Default for ExfilConfig {
    fn default() -> Self {
        Self {
            sensitive_prefixes: Vec::new(),
            established_dsts: Vec::new(),
            link_window_ms: 5_000,
            linked_confidence: 90,
            unlinked_confidence: 50,
        }
    }
}

/// Rule-based exfiltration detector. Stateful only in the small: it remembers
/// the most recent sensitive read per process so it can link a later outbound
/// connection to it. No model.
#[derive(Clone, Debug)]
pub struct ExfilDetector {
    cfg: ExfilConfig,
    recent_sensitive_read: BTreeMap<Pid, Timestamp>,
}

impl ExfilDetector {
    pub fn new(cfg: ExfilConfig) -> Self {
        Self {
            cfg,
            recent_sensitive_read: BTreeMap::new(),
        }
    }

    fn is_sensitive(&self, path: &str) -> bool {
        // An empty prefix would match every path, silently marking all reads
        // sensitive; skip empties so a config slip cannot blanket-flag the disk.
        self.cfg
            .sensitive_prefixes
            .iter()
            .any(|p| !p.is_empty() && path.starts_with(p.as_str()))
    }

    /// Forget all per-process read linkage. The engine calls this when the
    /// sensitive-read sensor or the detector itself is disabled, so stale state
    /// can never drive a live decision after the user cuts a capability.
    pub fn clear(&mut self) {
        self.recent_sensitive_read.clear();
    }

    /// Feed one event. Returns a detection when the rules fire.
    pub fn on_event(&mut self, ev: &Event) -> Option<Detection> {
        match ev {
            Event::FileRead { at, process, path } => {
                if self.is_sensitive(path) {
                    self.recent_sensitive_read.insert(process.pid, *at);
                }
                None // a sensitive read alone is not a threat
            }
            Event::OutboundConn {
                at,
                process,
                dst_ip,
                dst_port,
            } => {
                if self.cfg.established_dsts.iter().any(|d| d == dst_ip) {
                    return None; // established basis
                }
                // Causality: the read must precede the connection (`*at >=
                // *read_at`). A backdated/clock-skewed outbound that appears to
                // happen *before* the read must not link — otherwise a stale or
                // reordered timestamp manufactures high confidence.
                let linked = self
                    .recent_sensitive_read
                    .get(&process.pid)
                    .is_some_and(|read_at| {
                        *at >= *read_at && *at - *read_at <= self.cfg.link_window_ms
                    });
                let (confidence, rationale) = if linked {
                    (
                        self.cfg.linked_confidence,
                        format!(
                            "pid {} read a sensitive path then opened an outbound connection to {dst_ip}:{dst_port} with no established basis",
                            process.pid
                        ),
                    )
                } else {
                    (
                        self.cfg.unlinked_confidence,
                        format!(
                            "pid {} opened an outbound connection to {dst_ip}:{dst_port} with no established basis",
                            process.pid
                        ),
                    )
                };
                Some(Detection {
                    at: *at,
                    kind: DetectionKind::ExfilSuspected,
                    confidence: Confidence(confidence),
                    proposed: ProposedAction::BlockOutbound {
                        process: process.clone(),
                        dst_ip: dst_ip.clone(),
                        dst_port: *dst_port,
                    },
                    rationale,
                })
            }
            Event::ProcessExit { pid, .. } => {
                self.recent_sensitive_read.remove(pid);
                None
            }
            Event::ProcessStart { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Part C — the decision engine
// ---------------------------------------------------------------------------

/// The sensor capability that gates an event, if it is sensor-gated. Process
/// lifecycle events carry no sensitive data and are needed for bookkeeping, so
/// they are ungated.
fn sensor_capability(ev: &Event) -> Option<CapabilityId> {
    match ev {
        Event::FileRead { .. } => Some(CapabilityId::SensorSensitiveRead),
        Event::OutboundConn { .. } => Some(CapabilityId::SensorOutboundConn),
        Event::ProcessStart { .. } | Event::ProcessExit { .. } => None,
    }
}

/// A fully-formed decision: the detection, the final disposition after gates and
/// advisor, and the advice that informed it.
#[derive(Clone, Debug)]
pub struct Decision {
    pub detection: Detection,
    pub disposition: Disposition,
    pub advice: Advice,
}

/// The decision engine: a capability registry plus the exfil detector, wired
/// through the two gates, the authority envelope, and the heighten-only advisor.
#[derive(Clone, Debug)]
pub struct Engine {
    registry: CapabilityRegistry,
    detector: ExfilDetector,
}

impl Engine {
    pub fn new(registry: CapabilityRegistry, detector: ExfilDetector) -> Self {
        Self { registry, detector }
    }

    pub fn registry(&self) -> &CapabilityRegistry {
        &self.registry
    }

    /// Toggle a capability — the single mutation point. Disabling the
    /// sensitive-read sensor or the detector purges the detector's per-process
    /// linkage, so a capability the user has switched OFF can never keep
    /// influencing a live decision (spec §7, user sovereignty), and PID reuse
    /// across a detector off/on cycle cannot mis-link an old read to a new
    /// process. Returns the new state.
    pub fn set_capability(
        &mut self,
        id: CapabilityId,
        enabled: bool,
        at: Timestamp,
        audit: &mut AuditLog,
    ) -> bool {
        let state = self.registry.set(id, enabled, at, audit);
        if !enabled
            && matches!(
                id,
                CapabilityId::SensorSensitiveRead | CapabilityId::DetectorExfil
            )
        {
            self.detector.clear();
        }
        state
    }

    /// Intake one event. Returns a decision only when the detector fires under
    /// the gates. A disabled sensor or detector yields None and records nothing
    /// beyond toggle history (fail-closed).
    pub fn intake(
        &mut self,
        ev: &Event,
        advisor: &dyn Advisor,
        audit: &mut AuditLog,
    ) -> Option<Decision> {
        // Gate 1a — sensor capability.
        if let Some(cap) = sensor_capability(ev)
            && !self.registry.is_enabled(cap)
        {
            return None;
        }
        // Gate 1b — detector capability.
        if !self.registry.is_enabled(CapabilityId::DetectorExfil) {
            return None;
        }
        let detection = self.detector.on_event(ev)?;
        audit.append(
            detection.at,
            AuditKind::Detection,
            detection.rationale.clone(),
        );

        // Authority envelope (rule-only).
        let mut disposition = classify(&detection);
        // Actuator capability gate, fail-closed: if the actuator this action
        // needs is disabled, the action can never be carried out — not
        // autonomously and not via a later human grant. Deny outright (the
        // toggle physically cuts the ability; it is not a "downgrade to ask").
        if !self
            .registry
            .is_enabled(detection.proposed.actuator_capability())
        {
            disposition = Disposition::Deny;
        }
        // Advisor (heighten-only; can never open a gate — and cannot relax Deny).
        let advice = advisor.assess(&detection);
        disposition = apply_caution(disposition, advice.caution);

        audit.append(
            detection.at,
            AuditKind::Decision,
            format!("disposition={disposition:?}"),
        );
        Some(Decision {
            detection,
            disposition,
            advice,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod envelope_tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn reversible_and_high_confidence_acts_autonomously() {
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(HIGH_CONFIDENCE)),
            Disposition::ActAutonomously
        );
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(100)),
            Disposition::ActAutonomously
        );
    }

    #[test]
    fn reversible_but_ambiguous_requires_permission() {
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(HIGH_CONFIDENCE - 1)),
            Disposition::RequirePermission
        );
    }

    proptest! {
        /// Headline §8 invariant: no irreversible action is ever dispositioned to
        /// fire autonomously, at any confidence.
        #[test]
        fn irreversible_never_acts_autonomously(c in 0u8..=100) {
            prop_assert_ne!(
                classify_parts(Reversibility::Irreversible, Confidence(c)),
                Disposition::ActAutonomously
            );
        }
    }
}

#[cfg(test)]
mod detector_tests {
    use super::*;

    fn cfg() -> ExfilConfig {
        ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            established_dsts: vec!["10.0.0.1".into()],
            ..ExfilConfig::default()
        }
    }
    fn proc(pid: Pid) -> ProcessRef {
        ProcessRef {
            pid,
            exe: "/usr/bin/curl".into(),
        }
    }
    fn read(at: Timestamp, pid: Pid, path: &str) -> Event {
        Event::FileRead {
            at,
            process: proc(pid),
            path: path.into(),
        }
    }
    fn out(at: Timestamp, pid: Pid, ip: &str) -> Event {
        Event::OutboundConn {
            at,
            process: proc(pid),
            dst_ip: ip.into(),
            dst_port: 443,
        }
    }

    #[test]
    fn sensitive_read_then_outbound_is_high_confidence() {
        let mut d = ExfilDetector::new(cfg());
        assert!(
            d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"))
                .is_none()
        );
        let det = d
            .on_event(&out(1500, 7, "203.0.113.9"))
            .expect("should fire");
        assert_eq!(det.kind, DetectionKind::ExfilSuspected);
        assert_eq!(det.confidence, Confidence(90));
        assert!(matches!(det.proposed, ProposedAction::BlockOutbound { .. }));
    }

    #[test]
    fn outbound_with_no_recent_read_is_ambiguous() {
        let mut d = ExfilDetector::new(cfg());
        let det = d
            .on_event(&out(1000, 7, "203.0.113.9"))
            .expect("should fire");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn established_destination_is_not_flagged() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        assert!(d.on_event(&out(1100, 7, "10.0.0.1")).is_none());
    }

    #[test]
    fn read_outside_the_window_does_not_link() {
        let mut d = ExfilDetector::new(cfg()); // default link_window_ms = 5000
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        let det = d
            .on_event(&out(7000, 7, "203.0.113.9"))
            .expect("still fires, unlinked");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn process_exit_clears_the_linkage() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        d.on_event(&Event::ProcessExit { at: 1100, pid: 7 });
        let det = d
            .on_event(&out(1200, 7, "203.0.113.9"))
            .expect("fires, unlinked");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn outbound_exactly_at_link_window_links() {
        let mut d = ExfilDetector::new(cfg()); // window 5000
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        // delta == link_window_ms exactly: the window is inclusive, so it links.
        let det = d.on_event(&out(6000, 7, "203.0.113.9")).expect("fires");
        assert_eq!(det.confidence, Confidence(90));
    }

    #[test]
    fn outbound_one_ms_past_link_window_does_not_link() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        let det = d.on_event(&out(6001, 7, "203.0.113.9")).expect("fires");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn one_process_does_not_inherit_another_processes_read() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519")); // pid 7 reads a secret
        // pid 9 connects out within the window — it must NOT link to pid 7's read.
        let det = d.on_event(&out(1500, 9, "203.0.113.9")).expect("fires");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn outbound_before_the_read_does_not_link() {
        let mut d = ExfilDetector::new(cfg());
        // Read stamped later than the connection (clock skew / reordering).
        d.on_event(&read(2000, 7, "/home/u/.ssh/id_ed25519"));
        let det = d.on_event(&out(1500, 7, "203.0.113.9")).expect("fires");
        assert_eq!(
            det.confidence,
            Confidence(50),
            "a connection before the read cannot be linked"
        );
    }

    #[test]
    fn empty_sensitive_prefix_does_not_match_every_file() {
        let mut d = ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec![String::new()], // misconfiguration
            ..ExfilConfig::default()
        });
        d.on_event(&read(1000, 7, "/etc/hostname"));
        let det = d.on_event(&out(1100, 7, "203.0.113.9")).expect("fires");
        assert_eq!(
            det.confidence,
            Confidence(50),
            "an empty prefix must not blanket-flag every read"
        );
    }
}

#[cfg(test)]
mod engine_tests {
    use super::*;
    use crate::advisor::Caution;

    struct AbstainAdvisor;
    impl Advisor for AbstainAdvisor {
        fn assess(&self, _d: &Detection) -> Advice {
            Advice::none()
        }
    }

    fn proc(pid: Pid) -> ProcessRef {
        ProcessRef {
            pid,
            exe: "/usr/bin/curl".into(),
        }
    }
    fn read(at: Timestamp, pid: Pid) -> Event {
        Event::FileRead {
            at,
            process: proc(pid),
            path: "/home/u/.ssh/id_ed25519".into(),
        }
    }
    fn out(at: Timestamp, pid: Pid) -> Event {
        Event::OutboundConn {
            at,
            process: proc(pid),
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }
    }
    fn detector() -> ExfilDetector {
        ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            ..ExfilConfig::default()
        })
    }
    fn armed_engine(audit: &mut AuditLog) -> Engine {
        let mut reg = CapabilityRegistry::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
            CapabilityId::ActuatorBlockConn,
        ] {
            reg.set(cap, true, 0, audit);
        }
        Engine::new(reg, detector())
    }

    #[test]
    fn disabled_detector_yields_no_decision_and_no_detection_record() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        reg.set(CapabilityId::SensorOutboundConn, true, 0, &mut audit);
        // DetectorExfil left OFF.
        let mut engine = Engine::new(reg, detector());
        let before = audit.records().len();
        assert!(
            engine
                .intake(&out(1000, 7), &AbstainAdvisor, &mut audit)
                .is_none()
        );
        assert_eq!(
            audit.records().len(),
            before,
            "nothing recorded beyond toggles"
        );
    }

    #[test]
    fn disabled_sensor_drops_the_event() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        reg.set(CapabilityId::DetectorExfil, true, 0, &mut audit);
        // SensorOutboundConn left OFF.
        let mut engine = Engine::new(reg, detector());
        assert!(
            engine
                .intake(&out(1000, 7), &AbstainAdvisor, &mut audit)
                .is_none()
        );
    }

    #[test]
    fn high_confidence_exfil_acts_autonomously_and_audits() {
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        assert!(
            engine
                .intake(&read(1000, 7), &AbstainAdvisor, &mut audit)
                .is_none()
        );
        let decision = engine
            .intake(&out(1500, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.disposition, Disposition::ActAutonomously);
        assert!(audit.verify().is_ok());
        assert!(
            audit
                .records()
                .iter()
                .any(|r| r.kind == AuditKind::Detection)
        );
        assert!(
            audit
                .records()
                .iter()
                .any(|r| r.kind == AuditKind::Decision)
        );
    }

    #[test]
    fn disabled_actuator_denies_the_action() {
        // Fail-closed: ActuatorBlockConn is OFF, so the block cannot fire — not
        // autonomously and not via a later human grant. The engine denies.
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
        ] {
            reg.set(cap, true, 0, &mut audit); // ActuatorBlockConn deliberately OFF
        }
        let mut engine = Engine::new(reg, detector());
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit);
        let decision = engine
            .intake(&out(1500, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.disposition, Disposition::Deny);
    }

    #[test]
    fn ambiguous_exfil_requires_permission() {
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        let decision = engine
            .intake(&out(1000, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(50));
        assert_eq!(decision.disposition, Disposition::RequirePermission);
    }

    struct HeightenAdvisor;
    impl Advisor for HeightenAdvisor {
        fn assess(&self, _d: &Detection) -> Advice {
            Advice {
                explanation: "review this".into(),
                caution: Caution::Heighten,
            }
        }
    }

    #[test]
    fn advisor_heighten_escalates_autonomous_to_an_ask() {
        // Exercises the apply_caution wiring through the engine: rules say 90 ->
        // ActAutonomously, the advisor heightens it to RequirePermission.
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        engine.intake(&read(1000, 7), &HeightenAdvisor, &mut audit);
        let decision = engine
            .intake(&out(1500, 7), &HeightenAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(90));
        assert_eq!(decision.disposition, Disposition::RequirePermission);
        assert_eq!(decision.advice.caution, Caution::Heighten);
    }

    #[test]
    fn engine_acts_autonomously_exactly_at_the_confidence_threshold() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
            CapabilityId::ActuatorBlockConn,
        ] {
            reg.set(cap, true, 0, &mut audit);
        }
        let det = ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            linked_confidence: HIGH_CONFIDENCE, // exactly the boundary
            ..ExfilConfig::default()
        });
        let mut engine = Engine::new(reg, det);
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit);
        let decision = engine
            .intake(&out(1500, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(HIGH_CONFIDENCE));
        assert_eq!(decision.disposition, Disposition::ActAutonomously);
    }

    #[test]
    fn engine_audits_detection_then_decision_in_order() {
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        let before = audit.records().len();
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit); // no detection
        engine.intake(&out(1500, 7), &AbstainAdvisor, &mut audit);
        let new_kinds: Vec<_> = audit.records()[before..].iter().map(|r| r.kind).collect();
        assert_eq!(new_kinds, vec![AuditKind::Detection, AuditKind::Decision]);
    }

    #[test]
    fn disabling_the_read_sensor_purges_stale_linkage() {
        // Bug A regression: a read captured while the sensor was on must not keep
        // driving an autonomous block after the user cuts the sensor.
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit);
        engine.set_capability(CapabilityId::SensorSensitiveRead, false, 1100, &mut audit);
        engine.set_capability(CapabilityId::SensorSensitiveRead, true, 1200, &mut audit);
        let decision = engine
            .intake(&out(1300, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(50));
        assert_eq!(decision.disposition, Disposition::RequirePermission);
    }

    #[test]
    fn detector_off_on_cycle_does_not_mislink_reused_pid() {
        // Bug B regression: toggling the detector off clears its memory, so a
        // reused PID cannot inherit an old process's sensitive read.
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit); // old pid 7 read a secret
        engine.set_capability(CapabilityId::DetectorExfil, false, 1100, &mut audit);
        engine.set_capability(CapabilityId::DetectorExfil, true, 1200, &mut audit);
        // The OS reuses pid 7 for an unrelated process that connects out.
        let decision = engine
            .intake(&out(1300, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(50));
    }
}
