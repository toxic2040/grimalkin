mod common;
use common::{TestSupervisor, *};
use familiar_core::audit::AuditKind;
use familiar_core::policy::ProposedAction;
use familiar_platform::testkit::{FakeSensors, RecordingActuators};

fn kinds(sup: &TestSupervisor) -> Vec<AuditKind> {
    sup.audit.records().iter().map(|r| r.kind).collect()
}

/// Scenario A — read a secret then connect out: high confidence, contained
/// autonomously, fully recorded.
#[test]
fn scenario_autonomous_containment() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert!(matches!(
        sup.actuators().applied[..],
        [ProposedAction::BlockOutbound { .. }]
    ));
    assert_eq!(
        kinds(&sup),
        vec![
            AuditKind::Detection,
            AuditKind::Decision,
            AuditKind::Actuation
        ]
    );
    assert!(sup.audit.verify().is_ok());
}

/// Scenario B — an ambiguous connection asks; the human grants; then it is
/// contained.
#[test]
fn scenario_ask_then_grant() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000);
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, true, 1200);
    assert_eq!(sup.actuators().applied.len(), 1);
    assert_eq!(
        kinds(&sup),
        vec![
            AuditKind::Detection,
            AuditKind::Decision,
            AuditKind::PermissionRequested,
            AuditKind::PermissionResolved,
            AuditKind::Actuation,
        ]
    );
    assert!(sup.audit.verify().is_ok());
}

/// Scenario C — an ambiguous connection asks; no one answers; the timeout denies
/// it and nothing is contained.
#[test]
fn scenario_ask_then_timeout_denies() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps::default()),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(1000);
    sup.drive_once(1000 + 30_000); // past the deadline; sensors drained
    assert!(sup.actuators().applied.is_empty());
    assert_eq!(
        kinds(&sup),
        vec![
            AuditKind::Detection,
            AuditKind::Decision,
            AuditKind::PermissionRequested,
            AuditKind::PermissionResolved, // timed out -> deny
        ]
    );
    assert!(sup.audit.verify().is_ok());
}
