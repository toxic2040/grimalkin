//! On-disk state: the capability snapshot (atomic JSON) and the audit log
//! (append-and-flush JSONL). The audit chain is re-verified on load, so a
//! tampered file is detected, not silently trusted. The core already derives
//! Serialize + Deserialize on AuditRecord/AuditKind and CapabilitySnapshot, so
//! no core change is needed.
use familiar_core::audit::{AuditKind, AuditLog, AuditRecord};
use familiar_core::capabilities::{CapabilityRegistry, CapabilitySnapshot};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Persisted daemon posture. Missing or corrupt state is disarmed: fail closed
/// toward no sensing, no containment, and no background kernel footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardianState {
    pub armed: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io: {0}")]
    Io(String),
    #[error("corrupt audit record: {0}")]
    Corrupt(String),
    #[error("audit chain failed verification: {0}")]
    Tampered(String),
}

pub fn save_capabilities(dir: &Path, snap: &CapabilitySnapshot) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let tmp = dir.join("capabilities.json.tmp");
    fs::write(
        &tmp,
        serde_json::to_vec_pretty(snap).expect("snapshot serializes"),
    )?;
    fs::rename(tmp, dir.join("capabilities.json")) // atomic replace
}

/// Fail-closed: any problem reading/parsing yields a fresh all-off registry.
pub fn load_capabilities(dir: &Path) -> CapabilityRegistry {
    let path = dir.join("capabilities.json");
    match fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<CapabilitySnapshot>(&t).ok())
    {
        Some(snap) => CapabilityRegistry::restore(snap),
        None => CapabilityRegistry::new(),
    }
}

pub fn save_guardian_state(dir: &Path, state: &GuardianState) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let tmp = dir.join("guardian-state.json.tmp");
    fs::write(
        &tmp,
        serde_json::to_vec_pretty(state).expect("state serializes"),
    )?;
    fs::rename(tmp, dir.join("guardian-state.json"))
}

/// Fail-closed: any problem reading/parsing yields disarmed.
pub fn load_guardian_state(dir: &Path) -> GuardianState {
    let path = dir.join("guardian-state.json");
    fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<GuardianState>(&t).ok())
        .unwrap_or_default()
}

pub fn append_audit(dir: &Path, rec: &AuditRecord) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("audit.jsonl"))?;
    let line = serde_json::to_string(rec).expect("record serializes");
    f.write_all(line.as_bytes())?;
    f.write_all(b"\n")?;
    f.flush()
}

pub fn load_audit(dir: &Path) -> Result<AuditLog, PersistError> {
    let path = dir.join("audit.jsonl");
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(AuditLog::new()),
        Err(e) => return Err(PersistError::Io(e.to_string())),
    };
    let mut records = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let rec: AuditRecord =
            serde_json::from_str(line).map_err(|e| PersistError::Corrupt(e.to_string()))?;
        records.push(rec);
    }
    let log = AuditLog::from_records(records);
    log.verify()
        .map_err(|e| PersistError::Tampered(e.to_string()))?;
    Ok(log)
}

/// Move a failed-verification `audit.jsonl` aside to the first free
/// `audit.jsonl.corrupt-N`, preserving the tampered evidence (never clobbered).
pub fn rotate_corrupt_audit(dir: &Path) -> io::Result<()> {
    let src = dir.join("audit.jsonl");
    if !src.exists() {
        return Ok(());
    }
    let mut n = 0u32;
    let dst = loop {
        let cand = dir.join(format!("audit.jsonl.corrupt-{n}"));
        if !cand.exists() {
            break cand;
        }
        n += 1;
    };
    fs::rename(src, dst)
}

/// Startup audit restore. If the on-disk chain verifies, return it and the count
/// already persisted. If it is tampered/corrupt, rotate it aside and return a
/// fresh log carrying a single `IntegrityAlert` (persisted count 0, so the alert
/// and everything after it get written to the clean file). Never silently trusts
/// a bad chain and never appends a second genesis to a bad file.
pub fn restore_audit(dir: &Path, now: u64) -> (AuditLog, usize) {
    match load_audit(dir) {
        Ok(log) => {
            let n = log.records().len();
            (log, n)
        }
        Err(e) => {
            if let Err(e) = rotate_corrupt_audit(dir) {
                eprintln!("[familiar] could not rotate the corrupt audit file aside: {e}");
            }
            let mut log = AuditLog::new();
            log.append(
                now,
                AuditKind::IntegrityAlert,
                format!("prior audit.jsonl failed verification and was rotated: {e}"),
            );
            (log, 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};

    fn tempdir(tag: u32) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fam-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn capabilities_persist_and_reload_fail_closed() {
        let dir = tempdir(1);
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 1, &mut audit);
        save_capabilities(&dir, &reg.snapshot()).unwrap();
        let reloaded = load_capabilities(&dir);
        assert!(reloaded.is_enabled(CapabilityId::DetectorExfil));
        assert!(!reloaded.is_enabled(CapabilityId::ActuatorBlockConn));
        let fresh = load_capabilities(std::path::Path::new("/nonexistent/familiar"));
        for id in CapabilityId::ALL {
            assert!(!fresh.is_enabled(id));
        }
    }

    #[test]
    fn guardian_state_persists_and_fails_disarmed() {
        let dir = tempdir(20);
        assert!(!load_guardian_state(&dir).armed);

        let state = GuardianState { armed: true };
        save_guardian_state(&dir, &state).unwrap();
        assert!(load_guardian_state(&dir).armed);

        std::fs::write(dir.join("guardian-state.json"), "{not json").unwrap();
        assert!(!load_guardian_state(&dir).armed);
    }

    #[test]
    fn audit_appends_reload_and_verify() {
        let dir = tempdir(2);
        let mut log = AuditLog::new();
        let recs = [
            log.append(1, AuditKind::Detection, "a").clone(),
            log.append(2, AuditKind::Actuation, "b").clone(),
        ];
        for r in &recs {
            append_audit(&dir, r).unwrap();
        }
        let reloaded = load_audit(&dir).expect("reload");
        assert_eq!(reloaded.records().len(), 2);
        assert!(reloaded.verify().is_ok());
    }

    #[test]
    fn tampered_audit_line_is_detected_on_load() {
        let dir = tempdir(3);
        let mut log = AuditLog::new();
        let r = log.append(1, AuditKind::Detection, "real").clone();
        append_audit(&dir, &r).unwrap();
        let p = dir.join("audit.jsonl");
        let mut v: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&p).unwrap().trim()).unwrap();
        v["detail"] = serde_json::Value::String("forged".into());
        std::fs::write(&p, format!("{v}\n")).unwrap();
        assert!(matches!(load_audit(&dir), Err(PersistError::Tampered(_))));
    }

    #[test]
    fn restore_audit_returns_the_verified_chain_intact() {
        let dir = tempdir(10);
        let mut log = AuditLog::new();
        for r in [
            log.append(1, AuditKind::Detection, "a").clone(),
            log.append(2, AuditKind::Decision, "b").clone(),
        ] {
            append_audit(&dir, &r).unwrap();
        }
        let (restored, persisted) = restore_audit(&dir, 5);
        assert_eq!(persisted, 2, "both records already on disk");
        assert_eq!(restored.records().len(), 2);
        assert!(restored.verify().is_ok());
    }

    #[test]
    fn restore_audit_rotates_a_tampered_file_and_alerts() {
        let dir = tempdir(11);
        let mut log = AuditLog::new();
        let r = log.append(1, AuditKind::Detection, "real").clone();
        append_audit(&dir, &r).unwrap();
        // Tamper: rewrite the detail without recomputing the hash.
        let p = dir.join("audit.jsonl");
        let mut v: serde_json::Value =
            serde_json::from_str(std::fs::read_to_string(&p).unwrap().trim()).unwrap();
        v["detail"] = serde_json::Value::String("forged".into());
        std::fs::write(&p, format!("{v}\n")).unwrap();

        let (restored, persisted) = restore_audit(&dir, 99);
        assert_eq!(persisted, 0, "fresh chain: nothing persisted yet");
        assert_eq!(restored.records().len(), 1, "one IntegrityAlert");
        assert_eq!(restored.records()[0].kind, AuditKind::IntegrityAlert);
        assert!(
            dir.join("audit.jsonl.corrupt-0").exists(),
            "bad file rotated aside"
        );
        assert!(restored.verify().is_ok());
    }
}
