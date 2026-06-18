use familiar_core::capabilities::CapabilityId;
use familiar_core::events::{Event, ProcessRef};
use familiar_daemon::config::DaemonConfig;
use familiar_daemon::control::apply_command;
use familiar_daemon::run::build_supervisor_with_sensors;
use familiar_ipc::{ControlRequest, ControlResponse};
use std::process::Command;

#[test]
fn only_operator_or_root_is_authorized() {
    use familiar_daemon::control::authorized;
    assert!(authorized(1000, 1000), "operator allowed");
    assert!(authorized(0, 1000), "root always allowed");
    assert!(!authorized(1001, 1000), "another user rejected");
    assert!(!authorized(33, 1000), "www-data rejected");
}

#[test]
fn serve_control_round_trips_a_request() {
    use familiar_daemon::control::serve_control;
    use familiar_ipc::client::ControlClient;

    let dir = std::env::temp_dir().join(format!("fam-srv-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let sock = dir.join("control.sock");
    let me = rustix::process::getuid().as_raw();

    let (rx, _h) = serve_control(&sock, me).expect("serve");
    // Stub "tick loop": answer one command then stop.
    let loop_h = std::thread::spawn(move || {
        if let Ok((req, reply)) = rx.recv() {
            assert!(matches!(req, ControlRequest::ListCapabilities));
            let _ = reply.send(ControlResponse::Ok);
        }
    });

    let mut client = ControlClient::connect(&sock).expect("connect");
    let resp = client
        .request(&ControlRequest::ListCapabilities)
        .expect("request");
    assert_eq!(resp, ControlResponse::Ok);
    loop_h.join().unwrap();
}

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

/// One unlinked outbound (confidence 50 => ask) on the first poll, then nothing.
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
) -> (
    DaemonConfig,
    familiar_runtime::Supervisor<
        OneOutbound,
        familiar_linux::LinuxActuators,
        familiar_linux::LinuxNotifier,
        familiar_advisor::NullAdvisor,
    >,
) {
    let cfg = DaemonConfig {
        cgroup_root: "/sys/fs/cgroup".into(),
        state_dir: std::env::temp_dir().join(format!("fam-ctl-{}-{}", std::process::id(), tag)),
        ..DaemonConfig::default()
    };
    let _ = std::fs::remove_dir_all(&cfg.state_dir);
    arm(&cfg.state_dir);
    let sup = build_supervisor_with_sensors(&cfg, OneOutbound(std::cell::Cell::new(false)))
        .expect("build");
    (cfg, sup)
}

#[test]
fn answer_prompt_grant_installs_the_block_via_ipc() {
    if reexec_in_netns("answer_prompt_grant_installs_the_block_via_ipc") {
        return;
    }
    let (cfg, mut sup) = armed_sup("grant");
    sup.drive_once(2000);
    // A prompt is now open; status shows it.
    let status = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::GetStatus,
        2000,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    let id = match status {
        ControlResponse::Status(s) => {
            assert_eq!(s.prompts.len(), 1);
            s.prompts[0].id
        }
        o => panic!("{o:?}"),
    };
    assert!(!ruleset().contains("drop"), "no block before the grant");
    let r = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::AnswerPrompt { id, granted: true },
        2500,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    assert_eq!(r, ControlResponse::Ok);
    assert!(ruleset().contains("drop"), "grant installs the block");
}

#[test]
fn unblock_via_ipc_lifts_the_block() {
    if reexec_in_netns("unblock_via_ipc_lifts_the_block") {
        return;
    }
    let (cfg, mut sup) = armed_sup("unblock");
    sup.drive_once(2000);
    let status = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::GetStatus,
        2000,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    let id = match status {
        ControlResponse::Status(s) => s.prompts[0].id,
        o => panic!("{o:?}"),
    };
    apply_command(
        &mut sup,
        &cfg,
        ControlRequest::AnswerPrompt { id, granted: true },
        2500,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    assert!(ruleset().contains("drop"));
    let r = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::Unblock {
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        },
        2600,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    assert_eq!(r, ControlResponse::Ok);
    assert!(!ruleset().contains("drop"), "unblock lifts containment");
}

#[test]
fn set_capability_toggles_and_persists() {
    if reexec_in_netns("set_capability_toggles_and_persists") {
        return;
    }
    let (cfg, mut sup) = armed_sup("toggle");
    let r = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::SetCapability {
            id: CapabilityId::ActuatorFreezeProcess,
            enabled: true,
        },
        100,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    assert_eq!(r, ControlResponse::Ok);
    // Persisted to capabilities.json: a fresh load sees it on.
    let reloaded = familiar_daemon::persistence::load_capabilities(&cfg.state_dir);
    assert!(reloaded.is_enabled(CapabilityId::ActuatorFreezeProcess));
}

/// The headline invariant: NO control command installs a block absent a real
/// detection + grant. Drive a benign tick (sensor drained), then fire every
/// non-grant command and confirm nothing got contained.
#[test]
fn no_command_can_install_containment() {
    if reexec_in_netns("no_command_can_install_containment") {
        return;
    }
    let (cfg, mut sup) = armed_sup("invariant");
    // Drain the one scripted outbound by answering its prompt with a DENY.
    sup.drive_once(2000);
    let status = apply_command(
        &mut sup,
        &cfg,
        ControlRequest::GetStatus,
        2000,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    let id = match status {
        ControlResponse::Status(s) => s.prompts[0].id,
        o => panic!("{o:?}"),
    };
    apply_command(
        &mut sup,
        &cfg,
        ControlRequest::AnswerPrompt { id, granted: false },
        2100,
        familiar_daemon::run::SensorHealth {
            network_ok: true,
            file_ok: true,
        },
    );
    assert!(!ruleset().contains("drop"), "deny installs nothing");
    // Now hammer every other command; none may contain.
    for cmd in [
        ControlRequest::ListCapabilities,
        ControlRequest::GetStatus,
        ControlRequest::GetAudit { since_seq: 0 },
        ControlRequest::SetCapability {
            id: CapabilityId::ActuatorBlockConn,
            enabled: true,
        },
        ControlRequest::Unblock {
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }, // nothing to remove
    ] {
        let _ = apply_command(
            &mut sup,
            &cfg,
            cmd,
            3000,
            familiar_daemon::run::SensorHealth {
                network_ok: true,
                file_ok: true,
            },
        );
        assert!(
            !ruleset().contains("drop"),
            "no non-grant command may install a block"
        );
    }
}
