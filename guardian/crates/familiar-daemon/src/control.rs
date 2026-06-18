//! The control surface: the trusted mapping from a `ControlRequest` to an
//! operation on the owned `Supervisor`. Every arm goes through a gated
//! entrypoint or a read accessor — there is no path here to `Actuators::apply`,
//! so the IPC can never install containment. The tick loop is the only caller
//! that mutates the Supervisor; this function runs inside it.
use crate::config::DaemonConfig;
use crate::persistence;
use familiar_advisor::NullAdvisor;
use familiar_core::permission::PermissionRequest;
use familiar_core::policy::ProposedAction;
use familiar_ipc::{BlockDto, ControlRequest, ControlResponse, PromptDto, StatusSnapshot};
use familiar_linux::{LinuxActuators, LinuxNotifier};
use familiar_platform::Sensors;
use familiar_runtime::Supervisor;

type Sup<S> = Supervisor<S, LinuxActuators, LinuxNotifier, NullAdvisor>;

fn prompt_dto(r: &PermissionRequest) -> PromptDto {
    PromptDto {
        id: r.id,
        created_at: r.created_at,
        timeout_ms: r.timeout_ms,
        confidence: r.detection.confidence.0,
        kind: format!("{:?}", r.detection.kind),
        proposed: format!("{:?}", r.detection.proposed),
        rationale: r.detection.rationale.clone(),
    }
}

fn status<S: Sensors>(sup: &Sup<S>, health: crate::run::SensorHealth) -> StatusSnapshot {
    StatusSnapshot {
        capabilities: sup.engine.registry().snapshot(),
        prompts: sup.ledger.open_requests().map(prompt_dto).collect(),
        active_blocks: sup
            .actuators()
            .active_blocks()
            .iter()
            .map(|(ip, p)| BlockDto {
                dst_ip: ip.to_string(),
                dst_port: *p,
            })
            .collect(),
        audit_ok: sup.audit.verify().is_ok(),
        audit_head: sup.audit.head_hash().to_string(),
        audit_len: sup.audit.records().len() as u64,
        network_sensor_ok: health.network_ok,
        file_sensor_ok: health.file_ok,
    }
}

/// Apply one control request. Pure side-effects on `sup` + persistence; returns
/// the response to send back over the socket.
pub fn apply_command<S: Sensors>(
    sup: &mut Sup<S>,
    cfg: &DaemonConfig,
    req: ControlRequest,
    now: u64,
    health: crate::run::SensorHealth,
) -> ControlResponse {
    match req {
        ControlRequest::ListCapabilities => {
            ControlResponse::Capabilities(sup.engine.registry().snapshot())
        }
        ControlRequest::SetCapability { id, enabled } => {
            // Disjoint borrows of two distinct public fields.
            sup.engine.set_capability(id, enabled, now, &mut sup.audit);
            match persistence::save_capabilities(&cfg.state_dir, &sup.engine.registry().snapshot())
            {
                Ok(()) => ControlResponse::Ok,
                Err(e) => ControlResponse::Error(format!("persist failed: {e}")),
            }
        }
        ControlRequest::AnswerPrompt { id, granted } => {
            sup.resolve_permission(id, granted, now);
            ControlResponse::Ok
        }
        ControlRequest::Unblock { dst_ip, dst_port } => {
            // Reversal keys on (dst_ip, dst_port); the process field is irrelevant.
            let action = ProposedAction::BlockOutbound {
                process: familiar_core::events::ProcessRef {
                    pid: 0,
                    exe: String::new(),
                },
                dst_ip,
                dst_port,
            };
            match sup.reverse_action(&action, now) {
                Ok(()) => ControlResponse::Ok,
                Err(()) => ControlResponse::Error("unblock failed (see audit log)".into()),
            }
        }
        ControlRequest::GetStatus => ControlResponse::Status(status(sup, health)),
        ControlRequest::GetAudit { since_seq } => {
            let recs = sup
                .audit
                .records()
                .iter()
                .filter(|r| r.seq >= since_seq)
                .cloned()
                .collect();
            ControlResponse::Audit(recs)
        }
    }
}

use std::io;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::{self, JoinHandle};

/// A control request paired with the one-shot channel to answer it.
pub type ControlEnvelope = (ControlRequest, Sender<ControlResponse>);

/// Only the configured operator uid, or root, may drive the deck.
pub fn authorized(peer_uid: u32, operator_uid: u32) -> bool {
    peer_uid == operator_uid || peer_uid == 0
}

fn peer_uid(stream: &UnixStream) -> io::Result<u32> {
    // rustix reads SO_PEERCRED through a safe wrapper (no `unsafe` here).
    let cred = rustix::net::sockopt::get_socket_peercred(stream)?;
    Ok(cred.uid.as_raw())
}

/// Bind the control socket and accept operator connections. Each request line is
/// parsed and forwarded — with a one-shot reply channel — to the tick loop over
/// the returned receiver. The loop applies it and sends the response back, which
/// this thread writes to the socket. One client at a time (the deck).
pub fn serve_control(
    socket: &Path,
    operator_uid: u32,
) -> io::Result<(Receiver<ControlEnvelope>, JoinHandle<()>)> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(socket)?;
    // Defense in depth: operator group + owner only. The uid check is authoritative.
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o660))?;

    let (tx, rx) = channel::<ControlEnvelope>();
    let handle = thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            match peer_uid(&conn) {
                Ok(uid) if authorized(uid, operator_uid) => {}
                Ok(uid) => {
                    eprintln!("[familiar] control: rejecting uid {uid} (operator {operator_uid})");
                    continue;
                }
                Err(e) => {
                    eprintln!("[familiar] control: cannot read peer cred: {e}");
                    continue;
                }
            }
            if handle_conn(conn, &tx).is_err() {
                // client gone; accept the next one
            }
        }
    });
    Ok((rx, handle))
}

fn handle_conn(conn: UnixStream, tx: &Sender<ControlEnvelope>) -> io::Result<()> {
    let mut reader = std::io::BufReader::new(conn.try_clone()?);
    let mut writer = conn;
    loop {
        let req: ControlRequest = match familiar_ipc::recv(&mut reader) {
            Ok(r) => r,
            Err(_) => return Ok(()), // EOF / parse error closes the connection
        };
        let (reply_tx, reply_rx) = channel::<ControlResponse>();
        if tx.send((req, reply_tx)).is_err() {
            return Ok(()); // daemon gone
        }
        let resp = reply_rx
            .recv()
            .unwrap_or(ControlResponse::Error("daemon busy".into()));
        familiar_ipc::send(&mut writer, &resp)?;
    }
}
