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
use std::sync::mpsc::Sender;
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;
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

struct SensingHandles {
    _network: JoinHandle<()>,
    _file: JoinHandle<()>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn activate_sensing(
    cfg: &DaemonConfig,
    syn_tx: Sender<OutboundSyn>,
    file_tx: Sender<FileReadEvent>,
    network_ok: Arc<AtomicBool>,
    file_ok: Arc<AtomicBool>,
) -> Result<SensingHandles, String> {
    familiar_linux::nft::ensure_table().map_err(|e| format!("create familiar table: {e}"))?;
    if let Err(e) = familiar_linux::nft::install_queue_rule(cfg.queue_num) {
        let _ = familiar_linux::nft::delete_table();
        return Err(format!("install nfqueue divert rule: {e}"));
    }

    network_ok.store(true, Ordering::Relaxed);
    let queue_num = cfg.queue_num;
    let net_flag = network_ok.clone();
    let network = std::thread::spawn(move || {
        if let Err(e) = nfqueue::run_reader(queue_num, syn_tx) {
            eprintln!("[familiar] nfqueue reader stopped: {e}");
        }
        net_flag.store(false, Ordering::Relaxed);
    });

    file_ok.store(true, Ordering::Relaxed);
    let file = match crate::filereads::spawn_socket_source_to(&cfg.helper_socket, file_tx) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[familiar] file-read source unavailable ({e}); running network-only");
            file_ok.store(false, Ordering::Relaxed);
            std::thread::spawn(|| {})
        }
    };

    Ok(SensingHandles {
        _network: network,
        _file: file,
    })
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
    let (syn_tx, syn_rx) = channel::<OutboundSyn>();
    let (file_tx, file_rx) = channel::<FileReadEvent>();
    let network_ok = Arc::new(AtomicBool::new(false));
    let file_ok = Arc::new(AtomicBool::new(false));

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
    let mut guardian_state = persistence::load_guardian_state(&cfg.state_dir);
    let mut sensing: Option<SensingHandles> = None;
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
        if guardian_state.armed && sensing.is_none() {
            match activate_sensing(
                &cfg,
                syn_tx.clone(),
                file_tx.clone(),
                network_ok.clone(),
                file_ok.clone(),
            ) {
                Ok(handles) => {
                    sensing = Some(handles);
                    sup.audit.append(
                        now,
                        familiar_core::audit::AuditKind::GuardianState,
                        "sensing activated",
                    );
                }
                Err(e) => {
                    eprintln!("[familiar] arming failed: {e}");
                    guardian_state.armed = false;
                    if let Err(e2) =
                        persistence::save_guardian_state(&cfg.state_dir, &guardian_state)
                    {
                        eprintln!("[familiar] could not persist failed arm state: {e2}");
                    }
                    sup.audit.append(
                        now,
                        familiar_core::audit::AuditKind::IntegrityAlert,
                        format!("arming failed; returned to disarmed: {e}"),
                    );
                }
            }
        }
        if !guardian_state.armed && sensing.is_some() {
            let _ = sup.reverse_all_containment(now);
            if let Err(e) = familiar_linux::nft::delete_table() {
                sup.audit.append(
                    now,
                    familiar_core::audit::AuditKind::NoAction,
                    format!("disarm teardown could not delete nft table: {e}"),
                );
            }
            sensing = None;
            network_ok.store(false, Ordering::Relaxed);
            file_ok.store(false, Ordering::Relaxed);
        }
        if guardian_state.armed && sensing.is_some() {
            sup.drive_once(now);
        } else {
            sup.drive_idle(now);
        }
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
        while let Ok((peer_uid, req, reply)) = ctl_rx.try_recv() {
            let resp = crate::control::apply_command_from(
                &mut sup,
                &cfg,
                &mut guardian_state,
                peer_uid,
                req,
                now,
                health,
            );
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
