//! §8 acceptance, network-only path (no file sensor): an unlinked outbound to an
//! unestablished destination (confidence 50) must ask the human; an explicit
//! grant installs the reversible block; a denial or timeout installs nothing.
//! Each case runs in its own private netns.
use familiar_core::events::{Event, ProcessRef};
use familiar_daemon::config::DaemonConfig;
use familiar_daemon::run::build_supervisor_with_sensors;
use std::process::Command;

fn reexec_in_netns(name: &str) -> bool {
    if std::env::var("FAMILIAR_IN_NETNS").is_ok() {
        return false;
    }
    let exe = std::env::current_exe().unwrap();
    let out = Command::new("unshare")
        .args(["-Urn"])
        .arg(&exe)
        .arg("--exact")
        .arg(name)
        .arg("--test-threads=1")
        .arg("--nocapture")
        .env("FAMILIAR_IN_NETNS", "1")
        .output()
        .expect("unshare");
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("Operation not permitted") || err.contains("namespace") {
            eprintln!("SKIP {name}: unprivileged userns unavailable");
            return true;
        }
        panic!("netns child failed for {name}:\n{err}");
    }
    true
}

fn ruleset() -> String {
    String::from_utf8(
        Command::new("nft")
            .args(["list", "ruleset"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
}

fn arm(state_dir: &std::path::Path) {
    use familiar_core::audit::AuditLog;
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
    let mut reg = CapabilityRegistry::new();
    let mut audit = AuditLog::new();
    for cap in [
        CapabilityId::SensorOutboundConn,
        CapabilityId::DetectorExfil,
        CapabilityId::ActuatorBlockConn,
    ] {
        reg.set(cap, true, 0, &mut audit);
    }
    familiar_daemon::persistence::save_capabilities(state_dir, &reg.snapshot()).unwrap();
}

/// A sensor that yields one unlinked outbound on its first poll, then nothing.
struct OneOutbound(std::cell::Cell<bool>);
impl familiar_platform::Sensors for OneOutbound {
    fn poll(&mut self) -> Vec<Event> {
        if self.0.replace(true) {
            return vec![];
        }
        vec![Event::OutboundConn {
            at: 1000,
            process: ProcessRef {
                pid: 7,
                exe: "/usr/bin/curl".into(),
            },
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }]
    }
}

fn armed_sup(
    tag: &str,
) -> familiar_runtime::Supervisor<
    OneOutbound,
    familiar_linux::LinuxActuators,
    familiar_linux::LinuxNotifier,
    familiar_advisor::NullAdvisor,
> {
    let cfg = DaemonConfig {
        cgroup_root: "/sys/fs/cgroup".into(),
        state_dir: std::env::temp_dir().join(format!("fam-rt-{}-{}", std::process::id(), tag)),
        ..DaemonConfig::default()
    };
    let _ = std::fs::remove_dir_all(&cfg.state_dir);
    arm(&cfg.state_dir);
    build_supervisor_with_sensors(&cfg, OneOutbound(std::cell::Cell::new(false))).expect("build")
}

#[test]
fn unlinked_outbound_asks_then_grant_blocks() {
    if reexec_in_netns("unlinked_outbound_asks_then_grant_blocks") {
        return;
    }
    let mut sup = armed_sup("grant");
    sup.drive_once(2000);
    assert!(!ruleset().contains("drop"), "must ask before blocking");
    assert!(sup.ledger.is_open(1), "a request should be open");

    sup.resolve_permission(1, true, 2500);
    assert!(ruleset().contains("drop"), "grant should install the block");
    assert!(sup.audit.verify().is_ok());
}

#[test]
fn denied_outbound_installs_nothing() {
    if reexec_in_netns("denied_outbound_installs_nothing") {
        return;
    }
    let mut sup = armed_sup("deny");
    sup.drive_once(2000);
    assert!(sup.ledger.is_open(1));
    sup.resolve_permission(1, false, 2500);
    assert!(!ruleset().contains("drop"), "denial must not block");
    assert!(sup.audit.verify().is_ok());
}

#[test]
fn timeout_denies_and_records() {
    if reexec_in_netns("timeout_denies_and_records") {
        return;
    }
    let mut sup = armed_sup("timeout");
    sup.drive_once(2000); // opens request at created_at=2000, timeout 30000
    assert!(sup.ledger.is_open(1));
    sup.drive_once(40_000); // past the deadline -> swept to deny
    assert!(!sup.ledger.is_open(1), "expired request should be closed");
    assert!(!ruleset().contains("drop"), "timeout must not block");
    let timed_out = sup
        .audit
        .records()
        .iter()
        .any(|r| r.detail.contains("timed out") && r.detail.contains("deny"));
    assert!(timed_out, "a timeout->deny must be recorded");
    assert!(sup.audit.verify().is_ok());
}
