//! Drives the real Supervisor wiring (LinuxActuators) with a scripted sensor, in
//! a private netns. Proves build_supervisor arms persisted capabilities and the
//! loop contains a high-confidence (linked) exfil autonomously.
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

/// Write a capability snapshot enabling sensors + detector + block actuator.
fn arm(state_dir: &std::path::Path) {
    use familiar_core::audit::AuditLog;
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
    let mut reg = CapabilityRegistry::new();
    let mut audit = AuditLog::new();
    for cap in [
        CapabilityId::SensorSensitiveRead,
        CapabilityId::SensorOutboundConn,
        CapabilityId::DetectorExfil,
        CapabilityId::ActuatorBlockConn,
    ] {
        reg.set(cap, true, 0, &mut audit);
    }
    familiar_daemon::persistence::save_capabilities(state_dir, &reg.snapshot()).unwrap();
}

#[test]
fn high_confidence_exfil_is_contained_end_to_end() {
    if reexec_in_netns("high_confidence_exfil_is_contained_end_to_end") {
        return;
    }
    use familiar_core::events::{Event, ProcessRef};
    use familiar_daemon::config::DaemonConfig;
    use familiar_daemon::run::build_supervisor_with_sensors;

    // A scripted sensor: one linked sensitive-read + outbound on the first poll.
    struct Script(std::cell::Cell<bool>);
    impl familiar_platform::Sensors for Script {
        fn poll(&mut self) -> Vec<Event> {
            if self.0.replace(true) {
                return vec![];
            }
            let p = ProcessRef {
                pid: 7,
                exe: "/usr/bin/curl".into(),
            };
            vec![
                Event::FileRead {
                    at: 1000,
                    process: p.clone(),
                    path: "/home/u/.ssh/id_ed25519".into(),
                },
                Event::OutboundConn {
                    at: 1500,
                    process: p,
                    dst_ip: "203.0.113.9".into(),
                    dst_port: 443,
                },
            ]
        }
    }

    let cfg = DaemonConfig {
        sensitive_prefixes: vec!["/home/u/.ssh".into()],
        cgroup_root: "/sys/fs/cgroup".into(),
        state_dir: std::env::temp_dir().join(format!("fam-loop-{}", std::process::id())),
        ..DaemonConfig::default()
    };
    let _ = std::fs::remove_dir_all(&cfg.state_dir);
    arm(&cfg.state_dir);

    let mut sup =
        build_supervisor_with_sensors(&cfg, Script(std::cell::Cell::new(false))).expect("build");
    sup.drive_once(2000);

    let rs = ruleset();
    assert!(
        rs.contains("drop"),
        "linked exfil should be contained autonomously:\n{rs}"
    );
    assert!(sup.audit.verify().is_ok());
}
