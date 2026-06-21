use crate::Timestamp;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// The kind of transition recorded in the audit log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditKind {
    GuardianState,
    CapabilityToggled,
    Detection,
    Decision,
    Actuation,
    NoAction,
    PermissionRequested,
    PermissionResolved,
    IntegrityAlert,
}

/// One append-only, hash-chained record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditRecord {
    pub seq: u64,
    pub at: Timestamp,
    pub kind: AuditKind,
    pub detail: String,
    pub prev_hash: String,
    pub hash: String,
}

/// Genesis hash: 64 hex zeros, the `prev_hash` of the first record.
pub const GENESIS_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AuditError {
    #[error("record {seq} hash does not match its contents")]
    BadHash { seq: u64 },
    #[error("record {seq} does not link to the previous record")]
    BrokenChain { seq: u64 },
}

fn compute_hash(seq: u64, at: Timestamp, kind: AuditKind, detail: &str, prev_hash: &str) -> String {
    let mut h = Sha256::new();
    h.update(seq.to_be_bytes());
    h.update(at.to_be_bytes());
    h.update((kind as u8).to_be_bytes());
    h.update((detail.len() as u64).to_be_bytes()); // length-prefix => unambiguous preimage
    h.update(detail.as_bytes());
    h.update(prev_hash.as_bytes());
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        let _ = write!(s, "{b:02x}"); // writing to a String is infallible
    }
    s
}

/// An append-only, hash-chained log held in memory.
///
/// `verify()` recomputes the chain and detects any record edit that does not
/// also recompute every later hash — i.e. naive tampering. It does NOT, on its
/// own, defeat an adversary who can rewrite the whole stored log and recompute
/// the chain forward: that requires comparing `head_hash()` against an
/// independently-sealed anchor, or a keyed MAC. Holding that anchor/key is the
/// daemon's job (Plan B); the core supplies the chain and exposes the head.
#[derive(Clone, Debug, Default)]
pub struct AuditLog {
    records: Vec<AuditRecord>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Reload a log from persisted records, preserving their stored hashes so
    /// `verify()` can still detect tampering. The daemon uses this on startup
    /// and pairs it with an external check of `head_hash()` against a sealed
    /// anchor.
    pub fn from_records(records: Vec<AuditRecord>) -> Self {
        Self { records }
    }

    /// Append a record linked to the current head. Returns the new record.
    pub fn append(
        &mut self,
        at: Timestamp,
        kind: AuditKind,
        detail: impl Into<String>,
    ) -> &AuditRecord {
        let detail = detail.into();
        let seq = self.records.len() as u64;
        let prev_hash = self.head_hash().to_string();
        let hash = compute_hash(seq, at, kind, &detail, &prev_hash);
        self.records.push(AuditRecord {
            seq,
            at,
            kind,
            detail,
            prev_hash,
            hash,
        });
        self.records.last().expect("just pushed")
    }

    /// The most recent record's hash, or the genesis hash if empty.
    pub fn head_hash(&self) -> &str {
        self.records
            .last()
            .map(|r| r.hash.as_str())
            .unwrap_or(GENESIS_HASH)
    }

    pub fn records(&self) -> &[AuditRecord] {
        &self.records
    }

    /// Recompute the whole chain and confirm nothing has been altered.
    pub fn verify(&self) -> Result<(), AuditError> {
        let mut prev = GENESIS_HASH.to_string();
        for (i, r) in self.records.iter().enumerate() {
            if r.seq != i as u64 || r.prev_hash != prev {
                return Err(AuditError::BrokenChain { seq: r.seq });
            }
            if compute_hash(r.seq, r.at, r.kind, &r.detail, &r.prev_hash) != r.hash {
                return Err(AuditError::BadHash { seq: r.seq });
            }
            prev = r.hash.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_head_is_genesis_and_verifies() {
        let log = AuditLog::new();
        assert_eq!(log.head_hash(), GENESIS_HASH);
        assert!(log.verify().is_ok());
    }

    #[test]
    fn appends_link_into_a_verifiable_chain() {
        let mut log = AuditLog::new();
        log.append(1, AuditKind::Detection, "exfil suspected");
        log.append(2, AuditKind::Decision, "ActAutonomously");
        log.append(3, AuditKind::Actuation, "blocked 1.1.1.1:443");
        assert_eq!(log.records().len(), 3);
        assert_eq!(log.records()[0].seq, 0);
        assert_eq!(log.records()[0].prev_hash, GENESIS_HASH);
        assert_eq!(log.records()[1].prev_hash, log.records()[0].hash);
        assert_eq!(log.head_hash(), log.records()[2].hash);
        assert!(log.verify().is_ok());
    }

    #[test]
    fn tampering_with_a_record_breaks_verification() {
        let mut log = AuditLog::new();
        log.append(1, AuditKind::Detection, "a");
        log.append(2, AuditKind::Decision, "b");
        // Reach into the private field (child module) to simulate tampering.
        log.records[0].detail = "forged".into();
        assert_eq!(log.verify(), Err(AuditError::BadHash { seq: 0 }));
    }

    #[test]
    fn verify_detects_a_reordered_seq() {
        let mut log = AuditLog::new();
        log.append(1, AuditKind::Detection, "a");
        log.append(2, AuditKind::Decision, "b");
        log.records[1].seq = 5; // should be 1
        assert_eq!(log.verify(), Err(AuditError::BrokenChain { seq: 5 }));
    }

    #[test]
    fn verify_detects_a_broken_prev_hash_link() {
        let mut log = AuditLog::new();
        log.append(1, AuditKind::Detection, "a");
        log.append(2, AuditKind::Decision, "b");
        log.records[1].prev_hash = GENESIS_HASH.to_string(); // no longer links to record 0
        assert_eq!(log.verify(), Err(AuditError::BrokenChain { seq: 1 }));
    }

    #[test]
    fn from_records_preserves_hashes_and_verifies() {
        let mut log = AuditLog::new();
        log.append(1, AuditKind::Detection, "a");
        log.append(2, AuditKind::Decision, "b");
        let reloaded = AuditLog::from_records(log.records().to_vec());
        assert!(reloaded.verify().is_ok());
        assert_eq!(reloaded.head_hash(), log.head_hash());
    }

    #[test]
    fn a_recomputed_forgery_verifies_but_moves_the_head() {
        // The honest log; the daemon seals its head hash out of band.
        let mut honest = AuditLog::new();
        honest.append(1, AuditKind::Decision, "RequirePermission");
        let sealed_head = honest.head_hash().to_string();

        // An adversary who can rewrite storage forges a different history and
        // recomputes the chain. verify() only checks internal consistency, so it
        // cannot tell the forgery apart...
        let mut forged = AuditLog::new();
        forged.append(1, AuditKind::Decision, "ActAutonomously");
        assert!(forged.verify().is_ok());

        // ...but the forged head differs from the sealed anchor, which is how a
        // daemon holding the anchor detects the tamper. This is precisely why
        // verify() alone is not tamper-evidence against a recomputing adversary.
        assert_ne!(forged.head_hash(), sealed_head);
    }
}
