#![forbid(unsafe_code)]
//! familiar-ipc — the control protocol shared by the daemon and the UI.
//!
//! Newline-delimited JSON over a Unix socket. The protocol is deliberately
//! narrow: it can toggle capabilities, answer or read prompts, lift a block,
//! and read status/audit. It has NO variant that installs a block or freeze —
//! the authority envelope lives in the daemon and is never reachable here.

pub mod client;

use familiar_core::audit::AuditRecord;
use familiar_core::capabilities::{CapabilityId, CapabilitySnapshot};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};

/// Write one message as a single JSON line and flush.
pub fn send<T: Serialize, W: Write>(w: &mut W, msg: &T) -> io::Result<()> {
    let line = serde_json::to_string(msg).map_err(io::Error::other)?;
    w.write_all(line.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()
}

/// Read exactly one JSON line and parse it. EOF before a line => `UnexpectedEof`.
pub fn recv<T: DeserializeOwned, R: BufRead>(r: &mut R) -> io::Result<T> {
    let mut line = String::new();
    let n = r.read_line(&mut line)?;
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "peer closed"));
    }
    serde_json::from_str(line.trim_end()).map_err(io::Error::other)
}

/// A request from the control deck to the daemon.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlRequest {
    /// The current capability snapshot.
    ListCapabilities,
    /// Toggle a capability at runtime (persisted by the daemon).
    SetCapability { id: CapabilityId, enabled: bool },
    /// Answer an open permission prompt. `granted == false` denies.
    AnswerPrompt { id: u64, granted: bool },
    /// Lift containment for a destination (remove the nft DROP rule).
    Unblock { dst_ip: String, dst_port: u16 },
    /// A compact status snapshot for the live view.
    GetStatus,
    /// Audit records with `seq >= since_seq`.
    GetAudit { since_seq: u64 },
}

/// The daemon's reply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlResponse {
    Capabilities(CapabilitySnapshot),
    Status(StatusSnapshot),
    Audit(Vec<AuditRecord>),
    Ok,
    Error(String),
}

/// A pending permission prompt, flattened for the wire (core's `PermissionRequest`
/// is intentionally not `Serialize`; the daemon converts at the boundary).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptDto {
    pub id: u64,
    pub created_at: u64,
    pub timeout_ms: u64,
    pub confidence: u8,
    pub kind: String,
    pub proposed: String,
    pub rationale: String,
}

/// A currently-installed block (for the "active containment" panel).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDto {
    pub dst_ip: String,
    pub dst_port: u16,
}

/// The compact live status the deck polls.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub capabilities: CapabilitySnapshot,
    pub prompts: Vec<PromptDto>,
    pub active_blocks: Vec<BlockDto>,
    /// Result of re-verifying the in-memory hash chain (the deck's verify indicator).
    pub audit_ok: bool,
    /// The audit chain head hash (hex).
    pub audit_head: String,
    /// Number of audit records so far (so the deck can fetch only the tail).
    pub audit_len: u64,
    /// The NFQUEUE outbound sensor reader is alive.
    pub network_sensor_ok: bool,
    /// The fanotify helper file-read source is connected.
    pub file_sensor_ok: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_then_recv_round_trips_over_a_buffer() {
        use std::io::Cursor;
        let mut buf: Vec<u8> = Vec::new();
        let req = ControlRequest::AnswerPrompt {
            id: 9,
            granted: true,
        };
        send(&mut buf, &req).unwrap();
        // Two messages back to back must frame independently.
        send(&mut buf, &ControlRequest::GetStatus).unwrap();
        let mut cur = Cursor::new(buf);
        let a: ControlRequest = recv(&mut cur).unwrap();
        let b: ControlRequest = recv(&mut cur).unwrap();
        assert_eq!(a, req);
        assert_eq!(b, ControlRequest::GetStatus);
        // Third read hits EOF.
        assert!(recv::<ControlRequest, _>(&mut cur).is_err());
    }

    #[test]
    fn request_round_trips_through_json() {
        let reqs = [
            ControlRequest::ListCapabilities,
            ControlRequest::SetCapability {
                id: CapabilityId::DetectorExfil,
                enabled: true,
            },
            ControlRequest::AnswerPrompt {
                id: 3,
                granted: false,
            },
            ControlRequest::Unblock {
                dst_ip: "203.0.113.9".into(),
                dst_port: 443,
            },
            ControlRequest::GetStatus,
            ControlRequest::GetAudit { since_seq: 7 },
        ];
        for r in reqs {
            let json = serde_json::to_string(&r).unwrap();
            let back: ControlRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }

    #[test]
    fn all_response_variants_round_trip_through_json() {
        use familiar_core::audit::{AuditKind, AuditRecord};
        use familiar_core::capabilities::CapabilitySnapshot;
        use std::collections::BTreeMap;

        let audit_record = AuditRecord {
            seq: 1,
            at: 1_000_000,
            kind: AuditKind::Detection,
            detail: "test".into(),
            prev_hash: "0".repeat(64),
            hash: "a".repeat(64),
        };
        let responses: Vec<ControlResponse> = vec![
            ControlResponse::Ok,
            ControlResponse::Error("denied".into()),
            ControlResponse::Capabilities(CapabilitySnapshot {
                states: BTreeMap::new(),
            }),
            ControlResponse::Audit(vec![audit_record]),
        ];
        for r in responses {
            let json = serde_json::to_string(&r).unwrap();
            let back: ControlResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(r, back);
        }
    }

    #[test]
    fn status_response_round_trips_through_json() {
        let snap = StatusSnapshot {
            capabilities: CapabilitySnapshot {
                states: Default::default(),
            },
            prompts: vec![PromptDto {
                id: 1,
                created_at: 1000,
                timeout_ms: 30_000,
                confidence: 50,
                kind: "ExfilSuspected".into(),
                proposed: "BlockOutbound".into(),
                rationale: "unestablished outbound".into(),
            }],
            active_blocks: vec![BlockDto {
                dst_ip: "203.0.113.9".into(),
                dst_port: 443,
            }],
            audit_ok: true,
            audit_head: "0".repeat(64),
            audit_len: 5,
            network_sensor_ok: true,
            file_sensor_ok: true,
        };
        let r = ControlResponse::Status(snap);
        let json = serde_json::to_string(&r).unwrap();
        let back: ControlResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
