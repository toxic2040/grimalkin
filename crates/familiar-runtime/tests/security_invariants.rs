mod common;
use common::*;
use familiar_core::audit::AuditKind;
use familiar_core::policy::ProposedAction;
use familiar_platform::testkit::{FakeSensors, RecordingActuators};

/// §8: a disabled capability runs no detector and no actuator.
#[test]
fn disabled_detector_takes_no_action() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps {
            detector: false,
            ..Default::default()
        }),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert!(sup.actuators().applied.is_empty());
    assert!(
        !sup.audit
            .records()
            .iter()
            .any(|r| r.kind == AuditKind::Detection)
    );
}

/// §8: no action without an explicit grant — the ask path acts only on a grant.
#[test]
fn ambiguous_exfil_acts_only_after_a_grant() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]); // confidence 50 => ask
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000);
    assert!(
        sup.actuators().applied.is_empty(),
        "must not act before a grant"
    );
    assert_eq!(sup.notifier().requests.len(), 1, "a prompt was raised");
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, true, 1100);
    assert_eq!(
        sup.actuators().applied.len(),
        1,
        "acts only after the grant"
    );
}

/// §8: a denied request never acts.
#[test]
fn denied_request_never_acts() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000);
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, false, 1100);
    assert!(sup.actuators().applied.is_empty());
}

/// §8: a permission timeout resolves to deny — no action.
#[test]
fn timed_out_request_denies_and_takes_no_action() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000); // raises an ask; default_timeout_ms = 30_000
    let id = sup.notifier().requests[0].id;
    sup.drive_once(1000 + 30_000); // next tick is past the deadline (sensors drained)
    assert!(!sup.ledger.is_open(id), "request expired");
    assert!(sup.actuators().applied.is_empty(), "timeout took no action");
    assert!(
        sup.audit
            .records()
            .iter()
            .any(|r| r.kind == AuditKind::PermissionResolved && r.detail.contains("timed out"))
    );
}

/// §8: an actuator error degrades to a recorded no-action, never a silent pass.
#[test]
fn actuation_failure_degrades_to_recorded_no_action() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]); // autonomous
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::failing(),
    );
    sup.drive_once(2000);
    assert!(
        sup.actuators().applied.is_empty(),
        "the failing actuator recorded nothing"
    );
    assert!(
        sup.audit
            .records()
            .iter()
            .any(|r| r.kind == AuditKind::NoAction && r.detail.contains("actuation failed"))
    );
    assert!(sup.audit.verify().is_ok());
}

/// §8: a grant arriving past the deadline is a timeout-deny, never an action —
/// even if no tick swept the request first (a late click or delayed UI event).
#[test]
fn a_grant_after_the_deadline_is_denied() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000); // ask raised, deadline = 1000 + 30_000
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, true, 999_999); // granted far past the deadline
    assert!(
        sup.actuators().applied.is_empty(),
        "a grant past the deadline must not act"
    );
    assert!(!sup.ledger.is_open(id));
    assert!(
        !sup.audit
            .records()
            .iter()
            .any(|r| r.kind == AuditKind::Actuation)
    );
    assert!(
        sup.audit
            .records()
            .iter()
            .any(|r| r.kind == AuditKind::PermissionResolved && r.detail.contains("timed out"))
    );
}

/// §7 fail-closed: a high-confidence detection with the actuator capability
/// disabled neither acts nor asks — a disabled actuator cannot fire at all.
#[test]
fn high_confidence_with_disabled_actuator_denies_with_no_prompt() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps {
            actuator_block: false,
            ..Default::default()
        }),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert!(
        sup.actuators().applied.is_empty(),
        "a disabled actuator must not act"
    );
    assert!(
        sup.notifier().requests.is_empty(),
        "and must not raise a prompt"
    );
}

/// Two ambiguous connections raise two asks; resolving them out of order acts on
/// exactly the granted one.
#[test]
fn multiple_requests_resolved_out_of_order() {
    let sensors = FakeSensors::new(vec![vec![
        out(1000, 7, "203.0.113.9"),
        out(1000, 8, "198.51.100.4"),
    ]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000);
    assert_eq!(sup.notifier().requests.len(), 2);
    let id1 = sup.notifier().requests[0].id;
    let id2 = sup.notifier().requests[1].id;
    sup.resolve_permission(id2, true, 1100); // grant the second
    sup.resolve_permission(id1, false, 1100); // deny the first
    assert_eq!(
        sup.actuators().applied.len(),
        1,
        "only the granted request acts"
    );
    match &sup.actuators().applied[0] {
        ProposedAction::BlockOutbound { dst_ip, .. } => assert_eq!(dst_ip, "198.51.100.4"),
        other => panic!("unexpected action: {other:?}"),
    }
}

/// One batch where a high-confidence event is contained autonomously while an
/// ambiguous one only asks — each event is handled on its own merits.
#[test]
fn mixed_batch_one_acts_one_asks() {
    let sensors = FakeSensors::new(vec![vec![
        read(1000, 7),
        out(1100, 7, "203.0.113.9"),
        out(1100, 8, "198.51.100.4"),
    ]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert_eq!(
        sup.actuators().applied.len(),
        1,
        "the high-confidence one is contained"
    );
    assert_eq!(sup.notifier().requests.len(), 1, "the ambiguous one asks");
}
