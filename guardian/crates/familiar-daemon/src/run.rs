//! Build the Supervisor from config + persisted state, and drive the tick loop.
use crate::config::DaemonConfig;
use crate::persistence;
use familiar_advisor::NullAdvisor;
use familiar_core::policy::{Engine, ExfilConfig, ExfilDetector};
use familiar_linux::nfqueue::{self, OutboundSyn};
use familiar_linux::wire::FileReadEvent;
use familiar_linux::{LinuxActuators, LinuxNotifier, LinuxSensors};
use familiar_runtime::Supervisor;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A point-in-time snapshot of whether each sensor's backing source is alive.
#[derive(Clone, Copy)]
pub struct SensorHealth {
    pub network_ok: bool,
    pub file_ok: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("actuators: {0}")]
    Actuators(String),
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn detector_from(cfg: &DaemonConfig) -> ExfilDetector {
    ExfilDetector::new(ExfilConfig {
        sensitive_prefixes: cfg.sensitive_prefixes.clone(),
        established_dsts: cfg.established_dsts.clone(),
        link_window_ms: cfg.link_window_ms,
        ..ExfilConfig::default()
    })
}

/// Build the Supervisor with the real Linux sensors over the given channels.
pub fn build_supervisor(
    cfg: &DaemonConfig,
    syn_rx: Receiver<OutboundSyn>,
    file_rx: Receiver<FileReadEvent>,
) -> Result<Supervisor<LinuxSensors, LinuxActuators, LinuxNotifier, NullAdvisor>, BuildError> {
    build_supervisor_with_sensors(cfg, LinuxSensors::new(syn_rx, file_rx))
}

/// Like `build_supervisor` but with a caller-supplied Sensors impl. Used by the
/// red-team fixtures to script exact event sequences without the OS.
pub fn build_supervisor_with_sensors<S: familiar_platform::Sensors>(
    cfg: &DaemonConfig,
    sensors: S,
) -> Result<Supervisor<S, LinuxActuators, LinuxNotifier, NullAdvisor>, BuildError> {
    let registry = persistence::load_capabilities(&cfg.state_dir);
    let engine = Engine::new(registry, detector_from(cfg));
    let actuators = LinuxActuators::new(cfg.cgroup_root.clone())
        .map_err(|e| BuildError::Actuators(e.to_string()))?;
    let notifier = LinuxNotifier::new(cfg.desktop_notify);
    Ok(Supervisor::new(
        engine,
        sensors,
        actuators,
        notifier,
        NullAdvisor,
        cfg.permission_timeout_ms,
    ))
}

/// The daemon's run loop. Spawns the NFQUEUE reader and the helper socket
/// source, then ticks the Supervisor and persists any new audit records.
pub fn main_loop(cfg: DaemonConfig) -> ! {
    // Sensing requires both the userspace reader and the nft rule that diverts
    // outbound TCP into the queue. ensure_table (via LinuxActuators::new in
    // build_supervisor) creates the table; install the sense chain here.
    familiar_linux::nft::ensure_table().expect("create familiar table");
    familiar_linux::nft::install_queue_rule(cfg.queue_num).expect("install nfqueue divert rule");

    let (syn_tx, syn_rx) = channel::<OutboundSyn>();
    let queue_num = cfg.queue_num;
    let network_ok = Arc::new(AtomicBool::new(true));
    {
        let flag = network_ok.clone();
        std::thread::spawn(move || {
            if let Err(e) = nfqueue::run_reader(queue_num, syn_tx) {
                eprintln!("[familiar] nfqueue reader stopped: {e}");
            }
            flag.store(false, Ordering::Relaxed); // reader is gone => network-blind
        });
    }
    let file_ok = Arc::new(AtomicBool::new(true));
    let (file_rx, _file_handle) = match crate::filereads::spawn_socket_source(&cfg.helper_socket) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[familiar] file-read source unavailable ({e}); running network-only");
            file_ok.store(false, Ordering::Relaxed);
            let (_tx, rx) = channel();
            (rx, std::thread::spawn(|| {}))
        }
    };

    // Control deck IPC. If the socket cannot be bound, log and run headless —
    // the guardian must not depend on the UI being present.
    let (ctl_rx, _ctl_handle) =
        match crate::control::serve_control(&cfg.control_socket, cfg.operator_uid) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[familiar] control socket unavailable ({e}); running without the deck");
                let (_tx, rx) = std::sync::mpsc::channel();
                (rx, std::thread::spawn(|| {}))
            }
        };

    let mut sup = build_supervisor(&cfg, syn_rx, file_rx).expect("build supervisor");
    // F2: reload + verify the persisted audit chain before driving. A tampered
    // or corrupt log is rotated aside and flagged, never silently trusted and
    // never appended-to (which would start a second genesis chain).
    let (audit_log, mut persisted) = persistence::restore_audit(&cfg.state_dir, now_ms());
    sup.audit = audit_log;
    // F4: one-shot latches — we fire an IntegrityAlert once per sensor going down,
    // not every tick, to avoid flooding the audit chain.
    let mut net_alerted = false;
    let mut file_alerted = false;
    loop {
        let now = now_ms();
        sup.drive_once(now);
        // F4: build sensor health snapshot and alert once on blindness under an
        // enabled capability.
        let health = SensorHealth {
            network_ok: network_ok.load(Ordering::Relaxed),
            file_ok: file_ok.load(Ordering::Relaxed),
        };
        if sup
            .engine
            .registry()
            .is_enabled(familiar_core::capabilities::CapabilityId::SensorOutboundConn)
            && !health.network_ok
            && !net_alerted
        {
            sup.audit.append(
                now,
                familiar_core::audit::AuditKind::IntegrityAlert,
                "outbound sensor (NFQUEUE reader) is down while SensorOutboundConn is enabled",
            );
            net_alerted = true;
        }
        if sup
            .engine
            .registry()
            .is_enabled(familiar_core::capabilities::CapabilityId::SensorSensitiveRead)
            && !health.file_ok
            && !file_alerted
        {
            sup.audit.append(
                now,
                familiar_core::audit::AuditKind::IntegrityAlert,
                "file-read sensor (fanotify helper) is unavailable while SensorSensitiveRead is enabled",
            );
            file_alerted = true;
        }
        // Apply any queued control commands (single-owner: only this loop mutates sup).
        while let Ok((req, reply)) = ctl_rx.try_recv() {
            let resp = crate::control::apply_command(&mut sup, &cfg, req, now, health);
            let _ = reply.send(resp);
        }
        // F3: persist record-by-record; on failure, stop and retry from the same
        // index next tick — never advance past a record that did not reach disk.
        let recs = sup.audit.records();
        let mut i = persisted;
        while i < recs.len() {
            match persistence::append_audit(&cfg.state_dir, &recs[i]) {
                Ok(()) => i += 1,
                Err(e) => {
                    eprintln!("[familiar] audit persist failed at seq {i} ({e}); will retry");
                    break;
                }
            }
        }
        persisted = i;
        std::thread::sleep(Duration::from_millis(cfg.tick_ms));
    }
}
