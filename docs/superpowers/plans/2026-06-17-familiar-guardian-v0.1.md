# Familiar Guardian v0.1 — Implementation Plan (deterministic spine)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the portable, deterministic security spine of the Familiar guardian — events, capabilities, hash-chained audit, policy/decision engine, authority envelope, permission protocol, the null advisor, the platform trait seam, and an OS-agnostic supervisor loop — fully unit- and property-tested against synthetic exfil event streams, with no OS calls and no UI.

**Architecture:** A Rust workspace of single-purpose crates. The invariant everything respects: **`familiar-core` makes no OS calls and runs no model** — it is pure policy and bookkeeping, takes all timestamps/ids from its caller, and is identical on every platform. `familiar-platform` defines the `Sensors`/`Actuators`/`Notifier` trait seam (no OS impl in this plan; a fake adapter drives the tests). `familiar-runtime` is the OS-agnostic supervisor that drives the loop over those traits. `familiar-advisor` ships a null advisor behind the core `Advisor` trait. The real Linux adapter, the daemon, and the Tauri UI are deliberately **out of this plan** (see "Follow-on plans").

**Tech Stack:** Rust 1.95.0, edition 2024. `sha2` (audit hash chain), `serde`/`serde_json` (snapshots + audit serialization), `thiserror` (typed errors), `proptest` (security-invariant property tests). No async, no tokio, no network, no model in any crate this plan builds.

## Global Constraints

These apply to **every** task. Exact values, copied from the spec and the workspace rules:

- **`familiar-core` makes no OS calls and runs no model.** No `std::fs`, no `std::net`, no `std::process`, no `std::time::SystemTime::now()`, no threads, no RNG. All timestamps (`Timestamp = u64`, unix epoch millis) and all ids are supplied by the caller. This keeps the core pure, deterministic, and identical on every platform (spec §4).
- **`#![forbid(unsafe_code)]`** at the top of every crate this plan builds. (The future linux adapter will localize `unsafe` for syscalls — out of scope here.)
- **Every capability is default-OFF and fail-closed.** Unknown capability ⇒ treated as disabled. A disabled capability runs no sensor, no detector, no actuator, and records nothing beyond toggle history (spec §4.2, §5, §10).
- **Two gates in series, then the envelope** (spec §4.2): capability gate → permission gate → authority envelope (reversible + high-confidence ⇒ act then notify; irreversible or ambiguous ⇒ require explicit confirmation; deny on timeout). Every gate is deterministic.
- **The advisor can inform but never holds a gate** (spec §4.2, §7). Structurally: advisor influence is monotonic toward caution — it may route an autonomous action to the human (`ActAutonomously → RequirePermission`) and may do nothing else. It can never authorize an action and never unilaterally deny one.
- **Audit is append-only and hash-chained (SHA-256), tamper-evident** (spec §4.1, §7). Every transition is recorded, including no-action outcomes, capability toggles, and permission decisions (spec §6).
- **No automation fingerprints** in commit messages, docs, or any committed artifact (global rule). Write like a human engineer. No `Co-Authored-By`/AI-provenance trailers (a commit hook blocks them anyway).
- **Repo is local-only.** `repos/familiar` is `private_local`: local commits OK, **no remote, no push, until the user explicitly authorizes** (global "repos private until authorized" rule).

---

## File structure (whole v0.1 workspace; this plan builds the first four crates)

```
repos/familiar/
├── Cargo.toml                  # virtual workspace, resolver "3", members
├── rust-toolchain.toml         # pin 1.95.0 + rustfmt, clippy
├── .gitignore                  # /target, *.db, etc.
├── docs/
│   └── superpowers/
│       ├── specs/2026-06-17-familiar-guardian-v0.1-design.md   # the approved spec (copied in)
│       └── plans/2026-06-17-familiar-guardian-v0.1.md          # this plan
├── crates/
│   ├── familiar-core/          # THIS PLAN — pure spine, no OS, no model
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # module decls + re-exports; #![forbid(unsafe_code)]
│   │       ├── events.rs       # Event vocabulary (FileRead, OutboundConn, Process*)
│   │       ├── audit.rs        # AuditRecord, AuditLog, SHA-256 chain, verify()
│   │       ├── capabilities.rs # CapabilityId, CapabilityRegistry (default-off, audited)
│   │       ├── advisor.rs      # Advisor trait, Advice/Caution, apply_caution (heighten-only)
│   │       ├── policy.rs       # Confidence, ProposedAction, Detection, classify, ExfilDetector, Engine
│   │       └── permission.rs   # PermissionRequest/Outcome, PermissionLedger (timeout⇒deny)
│   ├── familiar-advisor/       # THIS PLAN — NullAdvisor behind core::Advisor
│   │   ├── Cargo.toml
│   │   └── src/lib.rs
│   ├── familiar-platform/      # THIS PLAN — trait seam + fake adapter (no OS impl)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Sensors, Actuators, Notifier traits + error types
│   │       └── testkit.rs      # FakeSensors, RecordingActuators, CapturingNotifier (feature = "testkit")
│   ├── familiar-runtime/       # THIS PLAN — OS-agnostic supervisor loop over the traits
│   │   ├── Cargo.toml
│   │   └── src/lib.rs          # Supervisor::drive_once / resolve_permission / act
│   ├── familiar-daemon/        # FOLLOW-ON PLAN B — not built here
│   └── familiar-ui/            # FOLLOW-ON PLAN C — not built here
```

**Dependency direction (acyclic):** `platform → core`, `advisor → core`, `runtime → core + platform`. Nothing depends on `runtime`. `core` depends on nothing internal. This is what keeps the core portable: it never names a platform.

---

## Phase 0 — Workspace scaffold

### Task 0.1: Initialize the repo, workspace, and toolchain pin

**Files:**
- Create: `repos/familiar/Cargo.toml`
- Create: `repos/familiar/rust-toolchain.toml`
- Create: `repos/familiar/.gitignore`
- Modify: `/home/toxic2040/work/workspace.yaml` (add the `familiar` path entry)
- (Already present: `docs/superpowers/plans/2026-06-17-familiar-guardian-v0.1.md` — this file)

**Interfaces:**
- Produces: a buildable virtual workspace whose members are added crate-by-crate in later tasks.

- [ ] **Step 1: Copy the approved spec into the repo**

```bash
mkdir -p /home/toxic2040/work/repos/familiar/docs/superpowers/specs
cp /home/toxic2040/work/repos/grimalkin/docs/design/2026-06-17-familiar-guardian-v0.1-design.md \
   /home/toxic2040/work/repos/familiar/docs/superpowers/specs/2026-06-17-familiar-guardian-v0.1-design.md
```

- [ ] **Step 2: Write the workspace manifest**

`repos/familiar/Cargo.toml`:

```toml
[workspace]
resolver = "3"
members = [
    "crates/familiar-core",
    # familiar-advisor, familiar-platform, and familiar-runtime are appended to
    # this list by the tasks that create them (1.6, 2.1, 2.3), so the workspace
    # manifest is valid — and `cargo build` passes — at the end of every task.
]

[workspace.package]
edition = "2024"
rust-version = "1.95"
license = "AGPL-3.0-or-later"
publish = false

[workspace.dependencies]
familiar-core = { path = "crates/familiar-core" }
familiar-platform = { path = "crates/familiar-platform" }
familiar-advisor = { path = "crates/familiar-advisor" }
sha2 = "0.10"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
proptest = "1"
```

- [ ] **Step 3: Pin the toolchain**

`repos/familiar/rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.95.0"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 4: Write `.gitignore`**

`repos/familiar/.gitignore`:

```gitignore
/target
**/*.rs.bk
*.db
*.db-shm
*.db-wal
```

- [ ] **Step 5: Register the repo in the workspace contract**

Add this entry to the `paths` array in `/home/toxic2040/work/workspace.yaml`, mirroring the `perc-oracle-rs` local-only Rust precedent:

```yaml
    {
      "name": "familiar",
      "root": "/home/toxic2040/work/repos/familiar",
      "kind": "git_repo",
      "visibility": "private_local",
      "closure": "local_commit_ok_no_remote",
      "commit": "local_commit_ok",
      "push": "forbidden_until_remote_decided",
      "notes": "Ground-up Rust rewrite of the grimalkin familiar (guardian daemon). Local-only Rust workspace; no remote until the user authorizes release. target/ stays untracked. grimalkin stays intact as the hardened reference."
    }
```

- [ ] **Step 6: Verify the empty workspace is coherent and classified**

Run: `cd /home/toxic2040/work/repos/familiar && cargo metadata --no-deps --format-version 1 >/dev/null && echo OK`
Expected: `OK` (no members yet resolve to crates, but the manifest parses). If cargo complains that members do not exist, that is expected until Task 0.2 — proceed; the members are created next.

Run: `cd /home/toxic2040/work && ./bin/ws guard repos/familiar/Cargo.toml`
Expected: classification resolves to the `familiar` entry with `commit: local_commit_ok`, `push: forbidden_until_remote_decided`.

- [ ] **Step 7: git init and first commit**

```bash
cd /home/toxic2040/work/repos/familiar
git init
git add -A
git commit -m "Scaffold familiar workspace: spec, manifest, toolchain pin"
```

(No push — the repo is local-only by policy. Do not add a remote.)

### Task 0.2: Create the `familiar-core` crate skeleton

**Files:**
- Create: `repos/familiar/crates/familiar-core/Cargo.toml`
- Create: `repos/familiar/crates/familiar-core/src/lib.rs`

**Interfaces:**
- Produces: an empty-but-compiling `familiar-core` crate with `#![forbid(unsafe_code)]` and the module skeleton later tasks fill in.

- [ ] **Step 1: Write the crate manifest**

`crates/familiar-core/Cargo.toml`:

```toml
[package]
name = "familiar-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
sha2.workspace = true
serde.workspace = true
thiserror.workspace = true

[dev-dependencies]
proptest.workspace = true
serde_json.workspace = true
```

- [ ] **Step 2: Write the lib root with the module skeleton**

`crates/familiar-core/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-core — the portable, deterministic guardian spine.
//!
//! Invariant: this crate makes no OS calls and runs no model. All timestamps
//! and ids are supplied by the caller. It is identical on every platform.

pub mod audit;
pub mod capabilities;
pub mod events;
pub mod permission;
pub mod policy;
pub mod advisor;

/// Unix epoch milliseconds, supplied by the caller. The core never reads a clock.
pub type Timestamp = u64;
/// OS process id, supplied by an adapter. The core never enumerates processes.
pub type Pid = u32;
```

- [ ] **Step 3: Create empty module files so the crate compiles**

Create each of `audit.rs`, `capabilities.rs`, `events.rs`, `permission.rs`, `policy.rs`, `advisor.rs` in `crates/familiar-core/src/` containing only:

```rust
// Filled in by a later task.
```

- [ ] **Step 4: Verify it builds**

Run: `cd /home/toxic2040/work/repos/familiar && cargo build -p familiar-core`
Expected: compiles clean (warnings about empty modules are acceptable).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core
git commit -m "Add familiar-core crate skeleton with no-unsafe, no-OS module layout"
```

---

## Phase 1 — `familiar-core`: the deterministic spine

### Task 1.1: Event vocabulary (`events.rs`)

**Files:**
- Modify: `crates/familiar-core/src/events.rs`

**Interfaces:**
- Consumes: `crate::{Pid, Timestamp}`.
- Produces: `ProcessRef { pid, exe }`; `Event` enum with variants `FileRead { at, process, path }`, `OutboundConn { at, process, dst_ip, dst_port }`, `ProcessStart { at, process, parent }`, `ProcessExit { at, pid }`; methods `Event::at() -> Timestamp`, `Event::pid() -> Pid`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-core/src/events.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn p(pid: Pid) -> ProcessRef {
        ProcessRef { pid, exe: "/usr/bin/curl".into() }
    }

    #[test]
    fn at_returns_the_stamped_time() {
        let ev = Event::OutboundConn { at: 42, process: p(7), dst_ip: "1.1.1.1".into(), dst_port: 443 };
        assert_eq!(ev.at(), 42);
    }

    #[test]
    fn pid_extracts_the_subject_process() {
        assert_eq!(Event::FileRead { at: 1, process: p(7), path: "/x".into() }.pid(), 7);
        assert_eq!(Event::ProcessExit { at: 1, pid: 9 }.pid(), 9);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core events`
Expected: FAIL — `ProcessRef`, `Event`, `at`, `pid` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment at the top of `crates/familiar-core/src/events.rs` with:

```rust
use crate::{Pid, Timestamp};

/// A process as an adapter sees it: its id and the resolved executable path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessRef {
    pub pid: Pid,
    pub exe: String,
}

/// The normalized, platform-neutral event vocabulary. Adapters translate OS
/// specifics into this; the core only ever sees this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A process read a file.
    FileRead { at: Timestamp, process: ProcessRef, path: String },
    /// A process opened (or attempted) an outbound connection.
    OutboundConn { at: Timestamp, process: ProcessRef, dst_ip: String, dst_port: u16 },
    /// A process started.
    ProcessStart { at: Timestamp, process: ProcessRef, parent: Pid },
    /// A process exited.
    ProcessExit { at: Timestamp, pid: Pid },
}

impl Event {
    /// The timestamp the adapter stamped on this event.
    pub fn at(&self) -> Timestamp {
        match self {
            Event::FileRead { at, .. }
            | Event::OutboundConn { at, .. }
            | Event::ProcessStart { at, .. }
            | Event::ProcessExit { at, .. } => *at,
        }
    }

    /// The subject process id.
    pub fn pid(&self) -> Pid {
        match self {
            Event::FileRead { process, .. }
            | Event::OutboundConn { process, .. }
            | Event::ProcessStart { process, .. } => process.pid,
            Event::ProcessExit { pid, .. } => *pid,
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core events`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/events.rs
git commit -m "Add normalized event vocabulary to familiar-core"
```

### Task 1.2: Hash-chained audit log (`audit.rs`)

**Files:**
- Modify: `crates/familiar-core/src/audit.rs`

**Interfaces:**
- Consumes: `crate::Timestamp`, `sha2`, `serde::Serialize`, `thiserror`.
- Produces: `AuditKind` (enum: `CapabilityToggled, Detection, Decision, Actuation, NoAction, PermissionRequested, PermissionResolved, IntegrityAlert`); `AuditRecord { seq, at, kind, detail, prev_hash, hash }`; `GENESIS_HASH`; `AuditLog` with `new()`, `append(at, kind, detail) -> &AuditRecord`, `head_hash() -> &str`, `records() -> &[AuditRecord]`, `verify() -> Result<(), AuditError>`; `AuditError::{BadHash, BrokenChain}`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-core/src/audit.rs`:

```rust
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
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core audit`
Expected: FAIL — types not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment in `crates/familiar-core/src/audit.rs` with:

```rust
use crate::Timestamp;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// The kind of transition recorded in the audit log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum AuditKind {
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AuditRecord {
    pub seq: u64,
    pub at: Timestamp,
    pub kind: AuditKind,
    pub detail: String,
    pub prev_hash: String,
    pub hash: String,
}

/// Genesis hash: 64 hex zeros, the `prev_hash` of the first record.
pub const GENESIS_HASH: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

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
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// An append-only, hash-chained, tamper-evident log held in memory. Persistence
/// is the daemon's job; the core only computes and verifies the chain.
#[derive(Clone, Debug, Default)]
pub struct AuditLog {
    records: Vec<AuditRecord>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self { records: Vec::new() }
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
        self.records.push(AuditRecord { seq, at, kind, detail, prev_hash, hash });
        self.records.last().expect("just pushed")
    }

    /// The most recent record's hash, or the genesis hash if empty.
    pub fn head_hash(&self) -> &str {
        self.records.last().map(|r| r.hash.as_str()).unwrap_or(GENESIS_HASH)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core audit`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/audit.rs
git commit -m "Add SHA-256 hash-chained, tamper-evident audit log"
```

### Task 1.3: Capability registry (`capabilities.rs`)

**Files:**
- Modify: `crates/familiar-core/src/capabilities.rs`

**Interfaces:**
- Consumes: `crate::audit::{AuditKind, AuditLog}`, `crate::Timestamp`, `serde`.
- Produces: `CapabilityId` (enum: `SensorOutboundConn, SensorSensitiveRead, DetectorExfil, ActuatorBlockConn, ActuatorFreezeProcess`) with `CapabilityId::ALL`; `CapabilitySnapshot`; `CapabilityRegistry` with `new()`, `is_enabled(id) -> bool`, `set(id, enabled, at, &mut AuditLog) -> bool`, `snapshot() -> CapabilitySnapshot`, `restore(CapabilitySnapshot) -> Self`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-core/src/capabilities.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{AuditKind, AuditLog};

    #[test]
    fn every_capability_defaults_off() {
        let reg = CapabilityRegistry::new();
        for id in CapabilityId::ALL {
            assert!(!reg.is_enabled(id), "{id:?} should default off");
        }
    }

    #[test]
    fn set_flips_state_and_audits_the_toggle() {
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 100, &mut audit);
        assert!(reg.is_enabled(CapabilityId::DetectorExfil));
        assert_eq!(audit.records().len(), 1);
        assert_eq!(audit.records()[0].kind, AuditKind::CapabilityToggled);
        assert!(audit.records()[0].detail.contains("DetectorExfil"));
    }

    #[test]
    fn restore_treats_a_missing_capability_as_off() {
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 1, &mut audit);
        let mut snap = reg.snapshot();
        snap.states.remove(&CapabilityId::DetectorExfil); // simulate version skew
        let restored = CapabilityRegistry::restore(snap);
        assert!(!restored.is_enabled(CapabilityId::DetectorExfil));
    }

    #[test]
    fn snapshot_round_trips_through_json() {
        // Proves the persistence seam the daemon (Plan B) relies on.
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::ActuatorFreezeProcess, true, 1, &mut audit);
        let json = serde_json::to_string(&reg.snapshot()).unwrap();
        let restored = CapabilityRegistry::restore(serde_json::from_str(&json).unwrap());
        assert!(restored.is_enabled(CapabilityId::ActuatorFreezeProcess));
        assert!(!restored.is_enabled(CapabilityId::DetectorExfil));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core capabilities`
Expected: FAIL — types not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment in `crates/familiar-core/src/capabilities.rs` with:

```rust
use crate::audit::{AuditKind, AuditLog};
use crate::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Every sensor, detector, and actuator is a named capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CapabilityId {
    SensorOutboundConn,
    SensorSensitiveRead,
    DetectorExfil,
    ActuatorBlockConn,
    ActuatorFreezeProcess,
}

impl CapabilityId {
    /// Every capability the v0.1 spine knows about.
    pub const ALL: [CapabilityId; 5] = [
        CapabilityId::SensorOutboundConn,
        CapabilityId::SensorSensitiveRead,
        CapabilityId::DetectorExfil,
        CapabilityId::ActuatorBlockConn,
        CapabilityId::ActuatorFreezeProcess,
    ];
}

/// A serializable snapshot of capability states, for the daemon to persist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySnapshot {
    pub states: BTreeMap<CapabilityId, bool>,
}

/// The Capability Registry: every capability default-off, fail-closed, with
/// every toggle written to the audit log.
#[derive(Clone, Debug)]
pub struct CapabilityRegistry {
    states: BTreeMap<CapabilityId, bool>,
}

impl Default for CapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CapabilityRegistry {
    /// A fresh registry with every known capability registered and OFF.
    pub fn new() -> Self {
        let mut states = BTreeMap::new();
        for id in CapabilityId::ALL {
            states.insert(id, false);
        }
        Self { states }
    }

    /// Fail-closed: an unknown/missing capability reads as disabled.
    pub fn is_enabled(&self, id: CapabilityId) -> bool {
        self.states.get(&id).copied().unwrap_or(false)
    }

    /// Toggle a capability and record the change. Returns the new state. The
    /// toggle is physical — it is not a preference a model can override.
    pub fn set(
        &mut self,
        id: CapabilityId,
        enabled: bool,
        at: Timestamp,
        audit: &mut AuditLog,
    ) -> bool {
        self.states.insert(id, enabled);
        audit.append(
            at,
            AuditKind::CapabilityToggled,
            format!("{id:?} -> {}", if enabled { "on" } else { "off" }),
        );
        enabled
    }

    pub fn snapshot(&self) -> CapabilitySnapshot {
        CapabilitySnapshot { states: self.states.clone() }
    }

    /// Restore from a snapshot, re-registering any capability the snapshot omits
    /// as OFF (fail-closed across version skew).
    pub fn restore(snapshot: CapabilitySnapshot) -> Self {
        let mut reg = Self::new();
        for (id, on) in snapshot.states {
            reg.states.insert(id, on);
        }
        reg
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core capabilities`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/capabilities.rs
git commit -m "Add default-off, fail-closed, audited capability registry"
```

### Task 1.4: Action vocabulary and the authority envelope (`policy.rs`, part A)

**Files:**
- Modify: `crates/familiar-core/src/policy.rs`

**Interfaces:**
- Consumes: `crate::capabilities::CapabilityId`, `crate::events::ProcessRef`, `crate::{Pid, Timestamp}`.
- Produces: `Confidence(u8)`; `HIGH_CONFIDENCE = 80`; `Reversibility`; `ProposedAction::{BlockOutbound, FreezeProcess}` with `reversibility()` and `actuator_capability()`; `DetectionKind::ExfilSuspected`; `Detection { at, kind, confidence, proposed, rationale }`; `Disposition::{ActAutonomously, RequirePermission, Deny}`; `classify_parts(Reversibility, Confidence) -> Disposition`; `classify(&Detection) -> Disposition`.

- [ ] **Step 1: Write the failing tests (incl. the headline invariant as a property test)**

Append to `crates/familiar-core/src/policy.rs`:

```rust
#[cfg(test)]
mod envelope_tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn reversible_and_high_confidence_acts_autonomously() {
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(HIGH_CONFIDENCE)),
            Disposition::ActAutonomously
        );
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(100)),
            Disposition::ActAutonomously
        );
    }

    #[test]
    fn reversible_but_ambiguous_requires_permission() {
        assert_eq!(
            classify_parts(Reversibility::Reversible, Confidence(HIGH_CONFIDENCE - 1)),
            Disposition::RequirePermission
        );
    }

    proptest! {
        /// Headline §8 invariant: no irreversible action is ever dispositioned to
        /// fire autonomously, at any confidence.
        #[test]
        fn irreversible_never_acts_autonomously(c in 0u8..=100) {
            prop_assert_ne!(
                classify_parts(Reversibility::Irreversible, Confidence(c)),
                Disposition::ActAutonomously
            );
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core envelope`
Expected: FAIL — `classify_parts`, `Confidence`, etc. not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment in `crates/familiar-core/src/policy.rs` with:

```rust
use crate::capabilities::CapabilityId;
use crate::events::ProcessRef;
use crate::{Pid, Timestamp};

/// Rule-derived confidence, 0..=100. Never model-derived in v0.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Confidence(pub u8);

/// At or above this, a reversible action may run autonomously.
pub const HIGH_CONFIDENCE: u8 = 80;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reversibility {
    Reversible,
    Irreversible,
}

/// A platform-neutral action the core may propose. The platform layer maps each
/// to a concrete OS operation. v0.1 ships two, both reversible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposedAction {
    /// Block/hold an outbound connection (e.g. a removable firewall rule).
    BlockOutbound { process: ProcessRef, dst_ip: String, dst_port: u16 },
    /// Freeze the offending process pending a decision.
    FreezeProcess { pid: Pid },
}

impl ProposedAction {
    pub fn reversibility(&self) -> Reversibility {
        match self {
            ProposedAction::BlockOutbound { .. } | ProposedAction::FreezeProcess { .. } => {
                Reversibility::Reversible
            }
        }
    }

    /// The actuator capability that must be enabled to carry this out.
    pub fn actuator_capability(&self) -> CapabilityId {
        match self {
            ProposedAction::BlockOutbound { .. } => CapabilityId::ActuatorBlockConn,
            ProposedAction::FreezeProcess { .. } => CapabilityId::ActuatorFreezeProcess,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionKind {
    ExfilSuspected,
}

/// A detector's output: what it saw, how sure it is, and what it proposes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detection {
    pub at: Timestamp,
    pub kind: DetectionKind,
    pub confidence: Confidence,
    pub proposed: ProposedAction,
    pub rationale: String,
}

/// The authority envelope's verdict for a proposed action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Disposition {
    /// Reversible, high-confidence, permitted: act then notify.
    ActAutonomously,
    /// Irreversible or ambiguous: ask the human; deny on timeout.
    RequirePermission,
    /// Do nothing (a gate refused).
    Deny,
}

/// The authority envelope as a total function of the two facts that matter.
/// Pure and exhaustively testable — including the irreversible branch that no
/// v0.1 action exercises yet but which guards every future actuator.
pub fn classify_parts(reversibility: Reversibility, confidence: Confidence) -> Disposition {
    match reversibility {
        Reversibility::Irreversible => Disposition::RequirePermission,
        Reversibility::Reversible => {
            if confidence.0 >= HIGH_CONFIDENCE {
                Disposition::ActAutonomously
            } else {
                Disposition::RequirePermission
            }
        }
    }
}

/// The authority envelope for a concrete detection.
pub fn classify(detection: &Detection) -> Disposition {
    classify_parts(detection.proposed.reversibility(), detection.confidence)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core envelope`
Expected: PASS (3 tests, including the property test over all confidences).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/policy.rs
git commit -m "Add proposed-action vocabulary and the deterministic authority envelope"
```

### Task 1.5: Advisor interface and heighten-only caution (`advisor.rs`)

**Files:**
- Modify: `crates/familiar-core/src/advisor.rs`

**Interfaces:**
- Consumes: `crate::policy::{Detection, Disposition}`.
- Produces: `Caution::{NoOpinion, Heighten}`; `Advice { explanation, caution }` with `Advice::none()`; `trait Advisor { fn assess(&self, &Detection) -> Advice }`; `apply_caution(Disposition, Caution) -> Disposition`.

- [ ] **Step 1: Write the failing tests (incl. the "never holds a gate" property)**

Append to `crates/familiar-core/src/advisor.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::Disposition;
    use proptest::prelude::*;

    #[test]
    fn heighten_routes_an_autonomous_action_to_the_human() {
        assert_eq!(
            apply_caution(Disposition::ActAutonomously, Caution::Heighten),
            Disposition::RequirePermission
        );
    }

    #[test]
    fn no_opinion_changes_nothing() {
        assert_eq!(
            apply_caution(Disposition::ActAutonomously, Caution::NoOpinion),
            Disposition::ActAutonomously
        );
    }

    fn dispositions() -> impl Strategy<Value = Disposition> {
        prop_oneof![
            Just(Disposition::ActAutonomously),
            Just(Disposition::RequirePermission),
            Just(Disposition::Deny),
        ]
    }
    fn cautions() -> impl Strategy<Value = Caution> {
        prop_oneof![Just(Caution::NoOpinion), Just(Caution::Heighten)]
    }

    proptest! {
        /// The advisor never holds a gate: it can neither manufacture an
        /// autonomous action nor convert a human-ask into a unilateral deny, nor
        /// relax a deny. Its only move is Act -> RequirePermission.
        #[test]
        fn advisor_never_moves_a_gate(d in dispositions(), c in cautions()) {
            let out = apply_caution(d, c);
            if d != Disposition::ActAutonomously {
                prop_assert_ne!(out, Disposition::ActAutonomously);
            }
            if d == Disposition::RequirePermission {
                prop_assert_eq!(out, Disposition::RequirePermission);
            }
            if d == Disposition::Deny {
                prop_assert_eq!(out, Disposition::Deny);
            }
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core advisor`
Expected: FAIL — `Advisor`, `Advice`, `apply_caution` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment in `crates/familiar-core/src/advisor.rs` with:

```rust
use crate::policy::{Detection, Disposition};

/// How much more cautious the advisor wants the harness to be. The advisor can
/// only ever move toward caution — never away. This is the structural form of
/// "the advisor never holds a gate" (spec §4.2, §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Caution {
    /// No escalation.
    NoOpinion,
    /// Route an otherwise-autonomous action to the human instead.
    Heighten,
}

/// The advisor's contribution to a decision: a human-readable explanation and,
/// at most, a request to be more cautious.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Advice {
    pub explanation: String,
    pub caution: Caution,
}

impl Advice {
    /// The advice a null/absent advisor gives.
    pub fn none() -> Self {
        Advice { explanation: String::new(), caution: Caution::NoOpinion }
    }
}

/// The interface the core may call for fuzzy judgment or explanation. An advisor
/// can inform a decision; it can never hold a gate.
pub trait Advisor {
    fn assess(&self, detection: &Detection) -> Advice;
}

/// Apply the advisor's caution to a rule-derived disposition. Monotonic toward
/// caution: the only move is `ActAutonomously -> RequirePermission`. The advisor
/// can never authorize an action and never unilaterally deny one.
pub fn apply_caution(disposition: Disposition, caution: Caution) -> Disposition {
    match (disposition, caution) {
        (Disposition::ActAutonomously, Caution::Heighten) => Disposition::RequirePermission,
        (d, _) => d,
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core advisor`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/advisor.rs
git commit -m "Add advisor interface with heighten-only, gate-safe caution"
```

### Task 1.6: The null advisor crate (`familiar-advisor`)

**Files:**
- Create: `crates/familiar-advisor/Cargo.toml`
- Create: `crates/familiar-advisor/src/lib.rs`
- Modify: `Cargo.toml` (add `crates/familiar-advisor` to `members`)

**Interfaces:**
- Consumes: `familiar_core::advisor::{Advice, Advisor}`, `familiar_core::policy::Detection`.
- Produces: `NullAdvisor` implementing `Advisor`, always returning `Advice::none()`.

- [ ] **Step 1: Add the crate to the workspace members**

In `repos/familiar/Cargo.toml`, extend `members`:

```toml
members = [
    "crates/familiar-core",
    "crates/familiar-advisor",
]
```

- [ ] **Step 2: Write the crate manifest**

`crates/familiar-advisor/Cargo.toml`:

```toml
[package]
name = "familiar-advisor"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
familiar-core.workspace = true
```

- [ ] **Step 3: Write the failing test + implementation**

`crates/familiar-advisor/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-advisor — implementations of the core `Advisor` trait.
//!
//! v0.1 ships only the null advisor: it explains nothing and never escalates.
//! The spine runs rule-only; this proves the seam without a model.

use familiar_core::advisor::{Advice, Advisor};
use familiar_core::policy::Detection;

/// An advisor that always abstains. The v0.1 default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullAdvisor;

impl Advisor for NullAdvisor {
    fn assess(&self, _detection: &Detection) -> Advice {
        Advice::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::advisor::Caution;
    use familiar_core::events::ProcessRef;
    use familiar_core::policy::{Confidence, DetectionKind, ProposedAction};

    #[test]
    fn null_advisor_always_abstains() {
        let d = Detection {
            at: 1,
            kind: DetectionKind::ExfilSuspected,
            confidence: Confidence(90),
            proposed: ProposedAction::FreezeProcess { pid: 7 },
            rationale: "x".into(),
        };
        let advice = NullAdvisor.assess(&d);
        assert_eq!(advice.caution, Caution::NoOpinion);
        assert!(advice.explanation.is_empty());
        let _ = ProcessRef { pid: 1, exe: "x".into() }; // ProcessRef stays in the public surface
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-advisor`
Expected: PASS (1 test). (Write-then-pass in one step is acceptable here: the impl is a single trait method with no branching to drive out.)

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/familiar-advisor
git commit -m "Add NullAdvisor: the v0.1 rule-only advisor"
```

### Task 1.7: The exfiltration detector (`policy.rs`, part B)

**Files:**
- Modify: `crates/familiar-core/src/policy.rs`

**Interfaces:**
- Consumes: `crate::events::Event`, `crate::{Pid, Timestamp}`, and the part-A types.
- Produces: `ExfilConfig { sensitive_prefixes, established_dsts, link_window_ms, linked_confidence, unlinked_confidence }` with `Default`; `ExfilDetector` with `new(ExfilConfig)` and `on_event(&Event) -> Option<Detection>`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-core/src/policy.rs`:

```rust
#[cfg(test)]
mod detector_tests {
    use super::*;
    use crate::events::{Event, ProcessRef};

    fn cfg() -> ExfilConfig {
        ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            established_dsts: vec!["10.0.0.1".into()],
            ..ExfilConfig::default()
        }
    }
    fn proc(pid: Pid) -> ProcessRef {
        ProcessRef { pid, exe: "/usr/bin/curl".into() }
    }
    fn read(at: Timestamp, pid: Pid, path: &str) -> Event {
        Event::FileRead { at, process: proc(pid), path: path.into() }
    }
    fn out(at: Timestamp, pid: Pid, ip: &str) -> Event {
        Event::OutboundConn { at, process: proc(pid), dst_ip: ip.into(), dst_port: 443 }
    }

    #[test]
    fn sensitive_read_then_outbound_is_high_confidence() {
        let mut d = ExfilDetector::new(cfg());
        assert!(d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519")).is_none());
        let det = d.on_event(&out(1500, 7, "203.0.113.9")).expect("should fire");
        assert_eq!(det.kind, DetectionKind::ExfilSuspected);
        assert_eq!(det.confidence, Confidence(90));
        assert!(matches!(det.proposed, ProposedAction::BlockOutbound { .. }));
    }

    #[test]
    fn outbound_with_no_recent_read_is_ambiguous() {
        let mut d = ExfilDetector::new(cfg());
        let det = d.on_event(&out(1000, 7, "203.0.113.9")).expect("should fire");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn established_destination_is_not_flagged() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        assert!(d.on_event(&out(1100, 7, "10.0.0.1")).is_none());
    }

    #[test]
    fn read_outside_the_window_does_not_link() {
        let mut d = ExfilDetector::new(cfg()); // default link_window_ms = 5000
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        let det = d.on_event(&out(7000, 7, "203.0.113.9")).expect("still fires, unlinked");
        assert_eq!(det.confidence, Confidence(50));
    }

    #[test]
    fn process_exit_clears_the_linkage() {
        let mut d = ExfilDetector::new(cfg());
        d.on_event(&read(1000, 7, "/home/u/.ssh/id_ed25519"));
        d.on_event(&Event::ProcessExit { at: 1100, pid: 7 });
        let det = d.on_event(&out(1200, 7, "203.0.113.9")).expect("fires, unlinked");
        assert_eq!(det.confidence, Confidence(50));
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core detector`
Expected: FAIL — `ExfilConfig`, `ExfilDetector` not found.

- [ ] **Step 3: Extend the imports and write the implementation**

At the top of `crates/familiar-core/src/policy.rs`, add to the existing `use` block:

```rust
use crate::events::Event;
use std::collections::BTreeMap;
```

Then append the detector to `crates/familiar-core/src/policy.rs`:

```rust
/// Configuration for the v0.1 exfiltration detector.
#[derive(Clone, Debug)]
pub struct ExfilConfig {
    /// Path prefixes treated as sensitive (e.g. `/home/u/.ssh`).
    pub sensitive_prefixes: Vec<String>,
    /// Destination IPs with an established basis (never flagged).
    pub established_dsts: Vec<String>,
    /// How long after a sensitive read an outbound connection counts as linked.
    pub link_window_ms: u64,
    /// Confidence when a recent sensitive read is linked to the connection.
    pub linked_confidence: u8,
    /// Confidence for an unestablished outbound with no recent sensitive read.
    pub unlinked_confidence: u8,
}

impl Default for ExfilConfig {
    fn default() -> Self {
        Self {
            sensitive_prefixes: Vec::new(),
            established_dsts: Vec::new(),
            link_window_ms: 5_000,
            linked_confidence: 90,
            unlinked_confidence: 50,
        }
    }
}

/// Rule-based exfiltration detector. Stateful only in the small: it remembers
/// the most recent sensitive read per process so it can link a later outbound
/// connection to it. No model.
#[derive(Clone, Debug)]
pub struct ExfilDetector {
    cfg: ExfilConfig,
    recent_sensitive_read: BTreeMap<Pid, Timestamp>,
}

impl ExfilDetector {
    pub fn new(cfg: ExfilConfig) -> Self {
        Self { cfg, recent_sensitive_read: BTreeMap::new() }
    }

    fn is_sensitive(&self, path: &str) -> bool {
        self.cfg.sensitive_prefixes.iter().any(|p| path.starts_with(p.as_str()))
    }

    /// Feed one event. Returns a detection when the rules fire.
    pub fn on_event(&mut self, ev: &Event) -> Option<Detection> {
        match ev {
            Event::FileRead { at, process, path } => {
                if self.is_sensitive(path) {
                    self.recent_sensitive_read.insert(process.pid, *at);
                }
                None // a sensitive read alone is not a threat
            }
            Event::OutboundConn { at, process, dst_ip, dst_port } => {
                if self.cfg.established_dsts.iter().any(|d| d == dst_ip) {
                    return None; // established basis
                }
                let linked = self
                    .recent_sensitive_read
                    .get(&process.pid)
                    .map(|read_at| at.saturating_sub(*read_at) <= self.cfg.link_window_ms)
                    .unwrap_or(false);
                let (confidence, rationale) = if linked {
                    (
                        self.cfg.linked_confidence,
                        format!(
                            "pid {} read a sensitive path then opened an outbound connection to {dst_ip}:{dst_port} with no established basis",
                            process.pid
                        ),
                    )
                } else {
                    (
                        self.cfg.unlinked_confidence,
                        format!(
                            "pid {} opened an outbound connection to {dst_ip}:{dst_port} with no established basis",
                            process.pid
                        ),
                    )
                };
                Some(Detection {
                    at: *at,
                    kind: DetectionKind::ExfilSuspected,
                    confidence: Confidence(confidence),
                    proposed: ProposedAction::BlockOutbound {
                        process: process.clone(),
                        dst_ip: dst_ip.clone(),
                        dst_port: *dst_port,
                    },
                    rationale,
                })
            }
            Event::ProcessExit { pid, .. } => {
                self.recent_sensitive_read.remove(pid);
                None
            }
            Event::ProcessStart { .. } => None,
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core detector`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/policy.rs
git commit -m "Add rule-based exfiltration detector with graded confidence"
```

### Task 1.8: Permission protocol (`permission.rs`)

**Files:**
- Modify: `crates/familiar-core/src/permission.rs`

**Interfaces:**
- Consumes: `crate::policy::Detection`, `crate::Timestamp`.
- Produces: `RequestId = u64`; `PermissionRequest { id, created_at, timeout_ms, detection }`; `PermissionOutcome::{Granted, Denied, TimedOut}` with `permits_action() -> bool`; `PermissionLedger` with `new()`, `open(created_at, timeout_ms, detection) -> RequestId`, `is_open(id) -> bool`, `get(id) -> Option<&PermissionRequest>`, `resolve(id, granted) -> Option<(PermissionOutcome, PermissionRequest)>`, `expire_due(now) -> Vec<(PermissionOutcome, PermissionRequest)>`.

- [ ] **Step 1: Write the failing tests (incl. the timeout-denies property)**

Append to `crates/familiar-core/src/permission.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ProcessRef;
    use crate::policy::{Confidence, Detection, DetectionKind, ProposedAction};
    use proptest::prelude::*;

    fn sample(at: Timestamp) -> Detection {
        Detection {
            at,
            kind: DetectionKind::ExfilSuspected,
            confidence: Confidence(90),
            proposed: ProposedAction::BlockOutbound {
                process: ProcessRef { pid: 7, exe: "/usr/bin/curl".into() },
                dst_ip: "203.0.113.9".into(),
                dst_port: 443,
            },
            rationale: "x".into(),
        }
    }

    #[test]
    fn only_a_grant_permits_action() {
        assert!(PermissionOutcome::Granted.permits_action());
        assert!(!PermissionOutcome::Denied.permits_action());
        assert!(!PermissionOutcome::TimedOut.permits_action());
    }

    #[test]
    fn open_then_resolve_returns_the_request() {
        let mut led = PermissionLedger::new();
        let id = led.open(100, 5_000, sample(100));
        assert!(led.is_open(id));
        let (outcome, req) = led.resolve(id, true).expect("open");
        assert_eq!(outcome, PermissionOutcome::Granted);
        assert_eq!(req.id, id);
        assert!(!led.is_open(id));
        assert!(led.resolve(id, true).is_none()); // already resolved
    }

    proptest! {
        /// §8 invariant: a request never survives its deadline, and a timeout is
        /// a denial. Before the deadline it stays open; at/after, it is TimedOut.
        #[test]
        fn timeout_resolves_to_deny(
            created in 0u64..1_000_000,
            timeout in 1u64..100_000,
            delta in 0u64..200_000,
        ) {
            let mut led = PermissionLedger::new();
            let id = led.open(created, timeout, sample(created));
            let expired = led.expire_due(created + delta);
            if delta >= timeout {
                prop_assert_eq!(expired.len(), 1);
                prop_assert_eq!(expired[0].0, PermissionOutcome::TimedOut);
                prop_assert!(!expired[0].0.permits_action());
                prop_assert!(!led.is_open(id));
            } else {
                prop_assert!(expired.is_empty());
                prop_assert!(led.is_open(id));
            }
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core permission`
Expected: FAIL — types not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder comment in `crates/familiar-core/src/permission.rs` with:

```rust
use crate::policy::Detection;
use crate::Timestamp;
use std::collections::BTreeMap;

pub type RequestId = u64;

/// A pending request for the human to authorize an action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermissionRequest {
    pub id: RequestId,
    pub created_at: Timestamp,
    pub timeout_ms: u64,
    pub detection: Detection,
}

/// The resolution of a permission request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionOutcome {
    Granted,
    Denied,
    TimedOut,
}

impl PermissionOutcome {
    /// Only an explicit grant permits the action. A timeout is a denial.
    pub fn permits_action(self) -> bool {
        matches!(self, PermissionOutcome::Granted)
    }
}

/// Tracks open permission requests. Deterministic: ids are a monotonic counter,
/// expiry is computed against a caller-supplied `now` (the core reads no clock).
#[derive(Clone, Debug)]
pub struct PermissionLedger {
    next_id: RequestId,
    open: BTreeMap<RequestId, PermissionRequest>,
}

impl Default for PermissionLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionLedger {
    pub fn new() -> Self {
        Self { next_id: 1, open: BTreeMap::new() }
    }

    /// Open a request and return its id.
    pub fn open(&mut self, created_at: Timestamp, timeout_ms: u64, detection: Detection) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.open.insert(id, PermissionRequest { id, created_at, timeout_ms, detection });
        id
    }

    pub fn is_open(&self, id: RequestId) -> bool {
        self.open.contains_key(&id)
    }

    /// Borrow an open request by id (the supervisor surfaces it to the user).
    pub fn get(&self, id: RequestId) -> Option<&PermissionRequest> {
        self.open.get(&id)
    }

    /// Resolve an open request by explicit human decision. Returns the outcome
    /// and the request (so the caller can act on a grant), or None if the id is
    /// unknown or already resolved.
    pub fn resolve(
        &mut self,
        id: RequestId,
        granted: bool,
    ) -> Option<(PermissionOutcome, PermissionRequest)> {
        let req = self.open.remove(&id)?;
        let outcome = if granted { PermissionOutcome::Granted } else { PermissionOutcome::Denied };
        Some((outcome, req))
    }

    /// Expire every open request whose deadline has passed. Each expiry resolves
    /// to TimedOut (a denial). Returns the expired requests.
    pub fn expire_due(&mut self, now: Timestamp) -> Vec<(PermissionOutcome, PermissionRequest)> {
        let due: Vec<RequestId> = self
            .open
            .iter()
            .filter(|(_, r)| now.saturating_sub(r.created_at) >= r.timeout_ms)
            .map(|(id, _)| *id)
            .collect();
        due.into_iter()
            .map(|id| (PermissionOutcome::TimedOut, self.open.remove(&id).expect("listed as due")))
            .collect()
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-core permission`
Expected: PASS (3 tests, including the timeout property).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/permission.rs
git commit -m "Add permission ledger where timeouts resolve to deny"
```

### Task 1.9: The decision engine (`policy.rs`, part C)

**Files:**
- Modify: `crates/familiar-core/src/policy.rs`

**Interfaces:**
- Consumes: `crate::advisor::{apply_caution, Advice, Advisor}`, `crate::audit::{AuditKind, AuditLog}`, `crate::capabilities::{CapabilityId, CapabilityRegistry}`, and the part-A/B types.
- Produces: `Decision { detection, disposition, advice }`; `Engine` with `new(CapabilityRegistry, ExfilDetector)`, `registry()`, `registry_mut()`, `intake(&Event, &dyn Advisor, &mut AuditLog) -> Option<Decision>`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-core/src/policy.rs`:

```rust
#[cfg(test)]
mod engine_tests {
    use super::*;
    use crate::advisor::Advisor;
    use crate::audit::AuditLog;
    use crate::capabilities::{CapabilityId, CapabilityRegistry};
    use crate::events::{Event, ProcessRef};

    struct AbstainAdvisor;
    impl Advisor for AbstainAdvisor {
        fn assess(&self, _d: &Detection) -> Advice {
            Advice::none()
        }
    }

    fn proc(pid: Pid) -> ProcessRef {
        ProcessRef { pid, exe: "/usr/bin/curl".into() }
    }
    fn read(at: Timestamp, pid: Pid) -> Event {
        Event::FileRead { at, process: proc(pid), path: "/home/u/.ssh/id_ed25519".into() }
    }
    fn out(at: Timestamp, pid: Pid) -> Event {
        Event::OutboundConn { at, process: proc(pid), dst_ip: "203.0.113.9".into(), dst_port: 443 }
    }
    fn detector() -> ExfilDetector {
        ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            ..ExfilConfig::default()
        })
    }
    /// An engine with the sensor + detector + block-actuator capabilities all on.
    fn armed_engine(audit: &mut AuditLog) -> Engine {
        let mut reg = CapabilityRegistry::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
            CapabilityId::ActuatorBlockConn,
        ] {
            reg.set(cap, true, 0, audit);
        }
        Engine::new(reg, detector())
    }

    #[test]
    fn disabled_detector_yields_no_decision_and_no_detection_record() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        reg.set(CapabilityId::SensorOutboundConn, true, 0, &mut audit);
        // DetectorExfil left OFF.
        let mut engine = Engine::new(reg, detector());
        let before = audit.records().len();
        assert!(engine.intake(&out(1000, 7), &AbstainAdvisor, &mut audit).is_none());
        assert_eq!(audit.records().len(), before, "nothing recorded beyond toggles");
    }

    #[test]
    fn disabled_sensor_drops_the_event() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        reg.set(CapabilityId::DetectorExfil, true, 0, &mut audit);
        // SensorOutboundConn left OFF.
        let mut engine = Engine::new(reg, detector());
        assert!(engine.intake(&out(1000, 7), &AbstainAdvisor, &mut audit).is_none());
    }

    #[test]
    fn high_confidence_exfil_acts_autonomously_and_audits() {
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        assert!(engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit).is_none());
        let decision = engine.intake(&out(1500, 7), &AbstainAdvisor, &mut audit).expect("fires");
        assert_eq!(decision.disposition, Disposition::ActAutonomously);
        assert!(audit.verify().is_ok());
        assert!(audit.records().iter().any(|r| r.kind == crate::audit::AuditKind::Detection));
        assert!(audit.records().iter().any(|r| r.kind == crate::audit::AuditKind::Decision));
    }

    #[test]
    fn disabled_actuator_downgrades_autonomy_to_an_ask() {
        let mut audit = AuditLog::new();
        let mut reg = CapabilityRegistry::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
        ] {
            reg.set(cap, true, 0, &mut audit); // ActuatorBlockConn deliberately OFF
        }
        let mut engine = Engine::new(reg, detector());
        engine.intake(&read(1000, 7), &AbstainAdvisor, &mut audit);
        let decision = engine.intake(&out(1500, 7), &AbstainAdvisor, &mut audit).expect("fires");
        assert_eq!(decision.disposition, Disposition::RequirePermission);
    }

    #[test]
    fn ambiguous_exfil_requires_permission() {
        let mut audit = AuditLog::new();
        let mut engine = armed_engine(&mut audit);
        let decision = engine.intake(&out(1000, 7), &AbstainAdvisor, &mut audit).expect("fires");
        assert_eq!(decision.detection.confidence, Confidence(50));
        assert_eq!(decision.disposition, Disposition::RequirePermission);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p familiar-core engine`
Expected: FAIL — `Engine`, `Decision` not found.

- [ ] **Step 3: Extend the imports and write the implementation**

At the top of `crates/familiar-core/src/policy.rs`, add to the existing `use` block:

```rust
use crate::advisor::{apply_caution, Advice, Advisor};
use crate::audit::{AuditKind, AuditLog};
use crate::capabilities::{CapabilityId, CapabilityRegistry};
```

Then append to `crates/familiar-core/src/policy.rs`:

```rust
/// The sensor capability that gates an event, if it is sensor-gated. Process
/// lifecycle events carry no sensitive data and are needed for bookkeeping, so
/// they are ungated.
fn sensor_capability(ev: &Event) -> Option<CapabilityId> {
    match ev {
        Event::FileRead { .. } => Some(CapabilityId::SensorSensitiveRead),
        Event::OutboundConn { .. } => Some(CapabilityId::SensorOutboundConn),
        Event::ProcessStart { .. } | Event::ProcessExit { .. } => None,
    }
}

/// A fully-formed decision: the detection, the final disposition after gates and
/// advisor, and the advice that informed it.
#[derive(Clone, Debug)]
pub struct Decision {
    pub detection: Detection,
    pub disposition: Disposition,
    pub advice: Advice,
}

/// The decision engine: a capability registry plus the exfil detector, wired
/// through the two gates, the authority envelope, and the heighten-only advisor.
#[derive(Clone, Debug)]
pub struct Engine {
    registry: CapabilityRegistry,
    detector: ExfilDetector,
}

impl Engine {
    pub fn new(registry: CapabilityRegistry, detector: ExfilDetector) -> Self {
        Self { registry, detector }
    }

    pub fn registry(&self) -> &CapabilityRegistry {
        &self.registry
    }
    pub fn registry_mut(&mut self) -> &mut CapabilityRegistry {
        &mut self.registry
    }

    /// Intake one event. Returns a decision only when the detector fires under
    /// the gates. A disabled sensor or detector yields None and records nothing
    /// beyond toggle history (fail-closed).
    pub fn intake(
        &mut self,
        ev: &Event,
        advisor: &dyn Advisor,
        audit: &mut AuditLog,
    ) -> Option<Decision> {
        // Gate 1a — sensor capability.
        if let Some(cap) = sensor_capability(ev) {
            if !self.registry.is_enabled(cap) {
                return None;
            }
        }
        // Gate 1b — detector capability.
        if !self.registry.is_enabled(CapabilityId::DetectorExfil) {
            return None;
        }
        let detection = self.detector.on_event(ev)?;
        audit.append(detection.at, AuditKind::Detection, detection.rationale.clone());

        // Authority envelope (rule-only).
        let mut disposition = classify(&detection);
        // Actuator capability gate: if the chosen actuator is disabled, we may
        // not act autonomously — route to the human (fail-closed heighten).
        if disposition == Disposition::ActAutonomously
            && !self.registry.is_enabled(detection.proposed.actuator_capability())
        {
            disposition = Disposition::RequirePermission;
        }
        // Advisor (heighten-only; can never open a gate).
        let advice = advisor.assess(&detection);
        disposition = apply_caution(disposition, advice.caution);

        audit.append(detection.at, AuditKind::Decision, format!("disposition={disposition:?}"));
        Some(Decision { detection, disposition, advice })
    }
}
```

- [ ] **Step 4: Run the full core test suite**

Run: `cargo test -p familiar-core`
Expected: PASS — every module's tests (events, audit, capabilities, envelope, advisor, detector, permission, engine).

- [ ] **Step 5: Lint and format gate**

Run: `cargo clippy -p familiar-core -- -D warnings && cargo fmt --check`
Expected: clean (no warnings, no diff). Fix any clippy findings before committing.

- [ ] **Step 6: Commit**

```bash
git add crates/familiar-core/src/policy.rs
git commit -m "Wire the decision engine: two gates, envelope, heighten-only advisor"
```

---

## Phase 2 — The platform seam, the supervisor, and the security invariants

This phase proves §10's portability seam ("the core compiles and its tests run with no platform adapter present") and turns §8's security invariants into runnable property tests, all without any OS. The real Linux adapter is Plan B.

### Task 2.1: Platform trait seam (`familiar-platform`)

**Files:**
- Create: `crates/familiar-platform/Cargo.toml`
- Create: `crates/familiar-platform/src/lib.rs`
- Modify: `Cargo.toml` (add `crates/familiar-platform` to `members`)

**Interfaces:**
- Consumes: `familiar_core::events::Event`, `familiar_core::permission::PermissionRequest`, `familiar_core::policy::ProposedAction`.
- Produces: `trait Sensors { fn poll(&mut self) -> Vec<Event> }`; `ActuationOutcome { note }`; `ActuationError::{Unsupported, Failed(String)}`; `trait Actuators { fn apply(&mut self, &ProposedAction) -> Result<ActuationOutcome, ActuationError> }`; `trait Notifier { fn notify(&mut self, &str); fn request_permission(&mut self, &PermissionRequest) }`; a `testkit` feature gate (filled in Task 2.2).

- [ ] **Step 1: Add the crate to the workspace members**

In `repos/familiar/Cargo.toml`, extend `members`:

```toml
members = [
    "crates/familiar-core",
    "crates/familiar-advisor",
    "crates/familiar-platform",
]
```

- [ ] **Step 2: Write the crate manifest**

`crates/familiar-platform/Cargo.toml`:

```toml
[package]
name = "familiar-platform"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[features]
testkit = []

[dependencies]
familiar-core.workspace = true
thiserror.workspace = true
```

- [ ] **Step 3: Write the failing test**

`crates/familiar-platform/src/lib.rs` — start with the test so the seam is exercised as trait objects:

```rust
#![forbid(unsafe_code)]
//! familiar-platform — the adapter seam.
//!
//! Traits the daemon implements per OS. v0.1 defines the seam and ships a
//! `testkit` fake adapter; the real Linux adapter is a follow-on plan. The core
//! never names this crate — the dependency flows platform -> core only.

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::policy::ProposedAction;

    struct OkActuators;
    impl Actuators for OkActuators {
        fn apply(&mut self, _a: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
            Ok(ActuationOutcome { note: "ok".into() })
        }
    }

    #[test]
    fn actuators_are_usable_as_trait_objects() {
        let mut a: Box<dyn Actuators> = Box::new(OkActuators);
        let outcome = a.apply(&ProposedAction::FreezeProcess { pid: 1 }).unwrap();
        assert_eq!(outcome.note, "ok");
    }
}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test -p familiar-platform`
Expected: FAIL — `Actuators`, `ActuationOutcome`, `ActuationError` not found.

- [ ] **Step 5: Write the implementation**

Above the `#[cfg(test)]` block in `crates/familiar-platform/src/lib.rs`, insert:

```rust
use familiar_core::events::Event;
use familiar_core::permission::PermissionRequest;
use familiar_core::policy::ProposedAction;

#[cfg(feature = "testkit")]
pub mod testkit;

/// A source of normalized events. An adapter polls the OS and returns core
/// events; the core only ever sees `Event`.
pub trait Sensors {
    /// Return any events observed since the last poll (possibly empty).
    fn poll(&mut self) -> Vec<Event>;
}

/// The result of carrying out a reversible action.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActuationOutcome {
    /// A human-readable note for the audit/notify trail (e.g. the firewall rule
    /// handle that can later reverse this).
    pub note: String,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ActuationError {
    #[error("actuator does not support this action")]
    Unsupported,
    #[error("actuation failed: {0}")]
    Failed(String),
}

/// Carries out reversible containment actions. An error degrades to no-action
/// upstream — never to a silent unguarded pass.
pub trait Actuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError>;
}

/// Surfaces notifications and permission prompts to the user (the UI in the
/// daemon; a capture buffer in tests).
pub trait Notifier {
    fn notify(&mut self, message: &str);
    fn request_permission(&mut self, request: &PermissionRequest);
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p familiar-platform`
Expected: PASS (1 test).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/familiar-platform
git commit -m "Add platform adapter seam: Sensors, Actuators, Notifier traits"
```

### Task 2.2: The fake adapter (`familiar-platform::testkit`)

**Files:**
- Create: `crates/familiar-platform/src/testkit.rs`

**Interfaces:**
- Consumes: the Task 2.1 traits + `familiar_core` event/action/permission types.
- Produces (under `feature = "testkit"`): `FakeSensors::new(Vec<Vec<Event>>)` with `is_drained()`; `RecordingActuators { applied, fail }` with `failing()`; `CapturingNotifier { messages, requests }`.

- [ ] **Step 1: Write the failing test**

Append to `crates/familiar-platform/src/lib.rs` test module a check that the testkit is wired (only compiled with the feature):

Add to the bottom of the existing `#[cfg(test)] mod tests` block in `lib.rs`:

```rust
    #[cfg(feature = "testkit")]
    #[test]
    fn fake_sensors_replay_one_batch_per_poll() {
        use crate::testkit::FakeSensors;
        use familiar_core::events::{Event, ProcessRef};
        let mut s = FakeSensors::new(vec![vec![Event::ProcessExit { at: 1, pid: 9 }]]);
        assert_eq!(s.poll().len(), 1);
        assert!(s.is_drained());
        let _ = ProcessRef { pid: 1, exe: "x".into() };
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p familiar-platform --features testkit`
Expected: FAIL — `crate::testkit` module not found.

- [ ] **Step 3: Write the implementation**

`crates/familiar-platform/src/testkit.rs`:

```rust
//! Fake adapters for driving the runtime in tests without an OS.

use std::collections::VecDeque;

use familiar_core::events::Event;
use familiar_core::permission::PermissionRequest;
use familiar_core::policy::ProposedAction;

use crate::{ActuationError, ActuationOutcome, Actuators, Notifier, Sensors};

/// Sensors that replay scripted batches of events, one batch per poll.
#[derive(Clone, Debug, Default)]
pub struct FakeSensors {
    batches: VecDeque<Vec<Event>>,
}

impl FakeSensors {
    pub fn new(batches: Vec<Vec<Event>>) -> Self {
        Self { batches: batches.into() }
    }
    /// True once every scripted batch has been polled.
    pub fn is_drained(&self) -> bool {
        self.batches.is_empty()
    }
}

impl Sensors for FakeSensors {
    fn poll(&mut self) -> Vec<Event> {
        self.batches.pop_front().unwrap_or_default()
    }
}

/// Actuators that record every applied action and can be forced to fail.
#[derive(Clone, Debug, Default)]
pub struct RecordingActuators {
    pub applied: Vec<ProposedAction>,
    pub fail: bool,
}

impl RecordingActuators {
    pub fn failing() -> Self {
        Self { applied: Vec::new(), fail: true }
    }
}

impl Actuators for RecordingActuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        if self.fail {
            return Err(ActuationError::Failed("injected".into()));
        }
        self.applied.push(action.clone());
        Ok(ActuationOutcome { note: format!("applied {action:?}") })
    }
}

/// Notifier that captures messages and permission requests.
#[derive(Clone, Debug, Default)]
pub struct CapturingNotifier {
    pub messages: Vec<String>,
    pub requests: Vec<PermissionRequest>,
}

impl Notifier for CapturingNotifier {
    fn notify(&mut self, message: &str) {
        self.messages.push(message.to_string());
    }
    fn request_permission(&mut self, request: &PermissionRequest) {
        self.requests.push(request.clone());
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-platform --features testkit`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-platform
git commit -m "Add testkit fake adapter: scripted sensors, recording actuators, capturing notifier"
```

### Task 2.3: The supervisor loop (`familiar-runtime`)

**Files:**
- Create: `crates/familiar-runtime/Cargo.toml`
- Create: `crates/familiar-runtime/src/lib.rs`
- Modify: `Cargo.toml` (add `crates/familiar-runtime` to `members`)

**Interfaces:**
- Consumes: `familiar_core::{advisor::Advisor, audit::*, permission::*, policy::*, Timestamp}`, `familiar_platform::{Actuators, Notifier, Sensors}`.
- Produces: `Supervisor<S, A, N, V>` with public `engine`, `ledger`, `audit`; `new(engine, sensors, actuators, notifier, advisor, default_timeout_ms)`; `notifier() -> &N`; `actuators() -> &A`; `drive_once(now)`; `resolve_permission(id, granted, now)`.

- [ ] **Step 1: Add the crate to members and write the manifest**

In `repos/familiar/Cargo.toml`, extend `members`:

```toml
members = [
    "crates/familiar-core",
    "crates/familiar-advisor",
    "crates/familiar-platform",
    "crates/familiar-runtime",
]
```

`crates/familiar-runtime/Cargo.toml`:

```toml
[package]
name = "familiar-runtime"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
familiar-core.workspace = true
familiar-platform.workspace = true

[dev-dependencies]
familiar-advisor.workspace = true
familiar-platform = { workspace = true, features = ["testkit"] }
```

- [ ] **Step 2: Write the failing golden-trace test**

`crates/familiar-runtime/src/lib.rs` — start with the module doc and the test:

```rust
#![forbid(unsafe_code)]
//! familiar-runtime — the OS-agnostic supervisor that drives the guardian loop.
//!
//! It owns the engine, the permission ledger, and the audit log, and drives them
//! over the `Sensors`/`Actuators`/`Notifier` seam. The daemon instantiates it
//! with the real Linux adapter; tests use the testkit fakes.

#[cfg(test)]
mod tests {
    use super::*;
    use familiar_advisor::NullAdvisor;
    use familiar_core::audit::AuditLog;
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
    use familiar_core::events::{Event, ProcessRef};
    use familiar_core::policy::{Engine, ExfilConfig, ExfilDetector, ProposedAction};
    use familiar_platform::testkit::{CapturingNotifier, FakeSensors, RecordingActuators};

    fn proc(pid: u32) -> ProcessRef {
        ProcessRef { pid, exe: "/usr/bin/curl".into() }
    }

    /// An engine with the sensor + detector + block-actuator capabilities on.
    pub(crate) fn armed_engine() -> Engine {
        let mut reg = CapabilityRegistry::new();
        let mut throwaway = AuditLog::new();
        for cap in [
            CapabilityId::SensorSensitiveRead,
            CapabilityId::SensorOutboundConn,
            CapabilityId::DetectorExfil,
            CapabilityId::ActuatorBlockConn,
        ] {
            reg.set(cap, true, 0, &mut throwaway);
        }
        let det = ExfilDetector::new(ExfilConfig {
            sensitive_prefixes: vec!["/home/u/.ssh".into()],
            ..ExfilConfig::default()
        });
        Engine::new(reg, det)
    }

    #[test]
    fn high_confidence_exfil_is_contained_audited_and_notified() {
        let sensors = FakeSensors::new(vec![vec![
            Event::FileRead { at: 1000, process: proc(7), path: "/home/u/.ssh/id_ed25519".into() },
            Event::OutboundConn { at: 1500, process: proc(7), dst_ip: "203.0.113.9".into(), dst_port: 443 },
        ]]);
        let mut sup = Supervisor::new(
            armed_engine(),
            sensors,
            RecordingActuators::default(),
            CapturingNotifier::default(),
            NullAdvisor,
            30_000,
        );
        sup.drive_once(2000);
        assert_eq!(sup.actuators().applied.len(), 1, "blocked reversibly");
        assert!(matches!(sup.actuators().applied[0], ProposedAction::BlockOutbound { .. }));
        assert!(!sup.notifier().messages.is_empty(), "notified");
        assert!(sup.audit.verify().is_ok(), "audit chain intact");
    }
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p familiar-runtime`
Expected: FAIL — `Supervisor` not found.

- [ ] **Step 4: Write the implementation**

Insert above the `#[cfg(test)]` block in `crates/familiar-runtime/src/lib.rs`:

```rust
use familiar_core::advisor::Advisor;
use familiar_core::audit::{AuditKind, AuditLog};
use familiar_core::permission::{PermissionLedger, RequestId};
use familiar_core::policy::{Decision, Disposition, Engine, ProposedAction};
use familiar_core::Timestamp;
use familiar_platform::{Actuators, Notifier, Sensors};

/// Wires the deterministic core to a platform adapter and an advisor, and drives
/// the detect -> decide -> act/ask -> audit -> notify loop.
pub struct Supervisor<S, A, N, V> {
    pub engine: Engine,
    pub ledger: PermissionLedger,
    pub audit: AuditLog,
    sensors: S,
    actuators: A,
    notifier: N,
    advisor: V,
    default_timeout_ms: u64,
}

impl<S, A, N, V> Supervisor<S, A, N, V>
where
    S: Sensors,
    A: Actuators,
    N: Notifier,
    V: Advisor,
{
    pub fn new(
        engine: Engine,
        sensors: S,
        actuators: A,
        notifier: N,
        advisor: V,
        default_timeout_ms: u64,
    ) -> Self {
        Self {
            engine,
            ledger: PermissionLedger::new(),
            audit: AuditLog::new(),
            sensors,
            actuators,
            notifier,
            advisor,
            default_timeout_ms,
        }
    }

    /// Borrow the notifier (tests inspect captured messages/requests).
    pub fn notifier(&self) -> &N {
        &self.notifier
    }
    /// Borrow the actuators (tests inspect applied actions).
    pub fn actuators(&self) -> &A {
        &self.actuators
    }

    /// One tick: expire overdue requests (timeout => deny), then poll sensors and
    /// handle each event's decision.
    pub fn drive_once(&mut self, now: Timestamp) {
        for (_outcome, req) in self.ledger.expire_due(now) {
            self.audit.append(
                now,
                AuditKind::PermissionResolved,
                format!("request {} timed out -> deny", req.id),
            );
            // A timeout is a denial: no action.
        }
        let events = self.sensors.poll();
        for ev in events {
            if let Some(decision) = self.engine.intake(&ev, &self.advisor, &mut self.audit) {
                self.dispatch(decision, now);
            }
        }
    }

    /// Resolve a pending request by explicit human decision. A grant acts; a
    /// denial records no-action.
    pub fn resolve_permission(&mut self, id: RequestId, granted: bool, now: Timestamp) {
        if let Some((outcome, req)) = self.ledger.resolve(id, granted) {
            self.audit.append(
                now,
                AuditKind::PermissionResolved,
                format!("request {id} -> {outcome:?}"),
            );
            if outcome.permits_action() {
                let action = req.detection.proposed.clone();
                self.act(&action, now);
            }
        }
    }

    fn dispatch(&mut self, decision: Decision, now: Timestamp) {
        match decision.disposition {
            Disposition::ActAutonomously => {
                let action = decision.detection.proposed.clone();
                self.act(&action, decision.detection.at);
            }
            Disposition::RequirePermission => {
                let id =
                    self.ledger.open(now, self.default_timeout_ms, decision.detection.clone());
                self.audit.append(
                    now,
                    AuditKind::PermissionRequested,
                    format!("request {id}: {}", decision.detection.rationale),
                );
                if let Some(req) = self.ledger.get(id).cloned() {
                    self.notifier.request_permission(&req);
                }
            }
            Disposition::Deny => {
                self.audit.append(decision.detection.at, AuditKind::NoAction, "denied by gate");
            }
        }
    }

    /// Carry out a reversible action. Fail-closed: an actuation error degrades to
    /// a recorded no-action, never to a silent pass.
    fn act(&mut self, action: &ProposedAction, at: Timestamp) {
        match self.actuators.apply(action) {
            Ok(outcome) => {
                self.audit.append(at, AuditKind::Actuation, format!("{action:?}: {}", outcome.note));
                self.notifier.notify(&format!("Contained {action:?} ({})", outcome.note));
            }
            Err(e) => {
                self.audit.append(
                    at,
                    AuditKind::NoAction,
                    format!("actuation failed for {action:?}: {e}"),
                );
                self.notifier.notify(&format!("Containment failed, no action taken: {e}"));
            }
        }
    }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p familiar-runtime`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/familiar-runtime
git commit -m "Add OS-agnostic supervisor that drives detect->decide->act->audit->notify"
```

### Task 2.4: Security-invariant tests through the full loop (`familiar-runtime/tests`)

These are §8's "security invariants as first-class property tests," exercised end-to-end through the supervisor and the fake adapter. They are the acceptance bar that any future change must keep green.

**Files:**
- Create: `crates/familiar-runtime/tests/common/mod.rs`
- Create: `crates/familiar-runtime/tests/security_invariants.rs`

**Interfaces:**
- Consumes: the public `familiar-runtime`, `familiar-core`, `familiar-advisor`, and `familiar-platform` (testkit) APIs.
- Produces: a reusable `common` test-support module and five invariant tests.

- [ ] **Step 1: Write the shared test-support module**

`crates/familiar-runtime/tests/common/mod.rs`:

```rust
#![allow(dead_code)]
use familiar_advisor::NullAdvisor;
use familiar_core::audit::AuditLog;
use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};
use familiar_core::events::{Event, ProcessRef};
use familiar_core::policy::{Engine, ExfilConfig, ExfilDetector};
use familiar_platform::testkit::{CapturingNotifier, FakeSensors, RecordingActuators};
use familiar_runtime::Supervisor;

pub type TestSupervisor =
    Supervisor<FakeSensors, RecordingActuators, CapturingNotifier, NullAdvisor>;

pub fn proc(pid: u32) -> ProcessRef {
    ProcessRef { pid, exe: "/usr/bin/curl".into() }
}
pub fn read(at: u64, pid: u32) -> Event {
    Event::FileRead { at, process: proc(pid), path: "/home/u/.ssh/id_ed25519".into() }
}
pub fn out(at: u64, pid: u32, ip: &str) -> Event {
    Event::OutboundConn { at, process: proc(pid), dst_ip: ip.into(), dst_port: 443 }
}

/// Which capabilities to arm. Default: everything on.
pub struct Caps {
    pub sensor_read: bool,
    pub sensor_out: bool,
    pub detector: bool,
    pub actuator_block: bool,
}
impl Default for Caps {
    fn default() -> Self {
        Self { sensor_read: true, sensor_out: true, detector: true, actuator_block: true }
    }
}

pub fn engine_with(caps: Caps) -> Engine {
    let mut reg = CapabilityRegistry::new();
    let mut throwaway = AuditLog::new();
    for (cap, on) in [
        (CapabilityId::SensorSensitiveRead, caps.sensor_read),
        (CapabilityId::SensorOutboundConn, caps.sensor_out),
        (CapabilityId::DetectorExfil, caps.detector),
        (CapabilityId::ActuatorBlockConn, caps.actuator_block),
    ] {
        if on {
            reg.set(cap, true, 0, &mut throwaway);
        }
    }
    let det = ExfilDetector::new(ExfilConfig {
        sensitive_prefixes: vec!["/home/u/.ssh".into()],
        ..ExfilConfig::default()
    });
    Engine::new(reg, det)
}

pub fn supervisor(
    engine: Engine,
    sensors: FakeSensors,
    actuators: RecordingActuators,
) -> TestSupervisor {
    Supervisor::new(engine, sensors, actuators, CapturingNotifier::default(), NullAdvisor, 30_000)
}
```

- [ ] **Step 2: Write the five failing invariant tests**

`crates/familiar-runtime/tests/security_invariants.rs`:

```rust
mod common;
use common::*;
use familiar_core::audit::AuditKind;
use familiar_platform::testkit::{FakeSensors, RecordingActuators};

/// §8: a disabled capability runs no detector and no actuator.
#[test]
fn disabled_detector_takes_no_action() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps { detector: false, ..Default::default() }),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert!(sup.actuators().applied.is_empty());
    assert!(!sup.audit.records().iter().any(|r| r.kind == AuditKind::Detection));
}

/// §8: no action without an explicit grant — the ask path acts only on a grant.
#[test]
fn ambiguous_exfil_acts_only_after_a_grant() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]); // confidence 50 => ask
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
    sup.drive_once(1000);
    assert!(sup.actuators().applied.is_empty(), "must not act before a grant");
    assert_eq!(sup.notifier().requests.len(), 1, "a prompt was raised");
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, true, 1100);
    assert_eq!(sup.actuators().applied.len(), 1, "acts only after the grant");
}

/// §8: a denied request never acts.
#[test]
fn denied_request_never_acts() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
    sup.drive_once(1000);
    let id = sup.notifier().requests[0].id;
    sup.resolve_permission(id, false, 1100);
    assert!(sup.actuators().applied.is_empty());
}

/// §8: a permission timeout resolves to deny — no action.
#[test]
fn timed_out_request_denies_and_takes_no_action() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
    sup.drive_once(1000); // raises an ask; default_timeout_ms = 30_000
    let id = sup.notifier().requests[0].id;
    sup.drive_once(1000 + 30_000); // next tick is past the deadline (sensors drained)
    assert!(!sup.ledger.is_open(id), "request expired");
    assert!(sup.actuators().applied.is_empty(), "timeout took no action");
    assert!(sup
        .audit
        .records()
        .iter()
        .any(|r| r.kind == AuditKind::PermissionResolved && r.detail.contains("timed out")));
}

/// §8: an actuator error degrades to a recorded no-action, never a silent pass.
#[test]
fn actuation_failure_degrades_to_recorded_no_action() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]); // autonomous
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::failing());
    sup.drive_once(2000);
    assert!(sup.actuators().applied.is_empty(), "the failing actuator recorded nothing");
    assert!(sup
        .audit
        .records()
        .iter()
        .any(|r| r.kind == AuditKind::NoAction && r.detail.contains("actuation failed")));
    assert!(sup.audit.verify().is_ok());
}
```

- [ ] **Step 3: Run the tests to verify they fail, then pass**

Run: `cargo test -p familiar-runtime --test security_invariants`
Expected first run: these compile but should already PASS against the Task 2.3 implementation — they assert behavior the supervisor already provides. If any fail, the supervisor logic is wrong; fix `familiar-runtime/src/lib.rs` (not the test) until green. This is the point of the suite: it is the regression bar.

- [ ] **Step 4: Commit**

```bash
git add crates/familiar-runtime/tests
git commit -m "Add §8 security-invariant tests through the full supervisor loop"
```

### Task 2.5: Golden scenario fixtures — the v0.1 acceptance bar (`familiar-runtime/tests`)

§8's scenario fixtures are "scripted exfil attempts ... the suite must detect, contain reversibly, and record." This task is the logic-level form of that bar: full traces with the exact audit-kind sequence asserted. The real-OS namespace/container version is Plan B; these golden traces pin the behavior the OS adapter must reproduce.

**Files:**
- Create: `crates/familiar-runtime/tests/scenarios.rs`

**Interfaces:**
- Consumes: the `common` module from Task 2.4.
- Produces: three golden end-to-end scenario tests (autonomous containment; ask→grant; ask→timeout→deny).

- [ ] **Step 1: Write the failing scenario tests**

`crates/familiar-runtime/tests/scenarios.rs`:

```rust
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
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
    sup.drive_once(2000);
    assert!(matches!(sup.actuators().applied[..], [ProposedAction::BlockOutbound { .. }]));
    assert_eq!(
        kinds(&sup),
        vec![AuditKind::Detection, AuditKind::Decision, AuditKind::Actuation]
    );
    assert!(sup.audit.verify().is_ok());
}

/// Scenario B — an ambiguous connection asks; the human grants; then it is
/// contained.
#[test]
fn scenario_ask_then_grant() {
    let sensors = FakeSensors::new(vec![vec![out(1000, 7, "203.0.113.9")]]);
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
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
    let mut sup = supervisor(engine_with(Caps::default()), sensors, RecordingActuators::default());
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
```

- [ ] **Step 2: Run the scenario tests**

Run: `cargo test -p familiar-runtime --test scenarios`
Expected: PASS (3 tests). If a trace mismatches, the supervisor's audit sequence is wrong — fix the supervisor, not the golden expectation, unless the spec's flow (§6) genuinely calls for a different order.

- [ ] **Step 3: Commit**

```bash
git add crates/familiar-runtime/tests/scenarios.rs
git commit -m "Add golden exfil scenario fixtures: autonomous, ask-grant, ask-timeout"
```

### Task 2.6: Whole-workspace verification gate

**Files:** none (verification only).

- [ ] **Step 1: The portability seam — core builds and tests with no adapter present**

Run: `cargo test -p familiar-core`
Expected: PASS. This is §10's criterion "the core compiles and its tests run with no platform adapter present" — `-p familiar-core` resolves none of `familiar-platform`/`familiar-runtime`.

- [ ] **Step 2: Full workspace test suite**

Run: `cargo test --workspace --all-features`
Expected: PASS — every crate, every test (unit + the property tests + the invariant and scenario integration tests).

- [ ] **Step 3: Lint and format gate across the workspace**

Run: `cargo clippy --workspace --all-targets --all-features -- -D warnings && cargo fmt --check`
Expected: clean. Fix findings before committing.

- [ ] **Step 4: Confirm no OS/clock/network leaked into the core**

Run: `! grep -rnE "std::(fs|net|process|thread)|SystemTime::now|Instant::now" crates/familiar-core/src`
Expected: exit 0 (no matches). The core stays pure; if this finds anything, a task introduced an OS dependency that must be moved behind the platform seam.

- [ ] **Step 5: Commit the green milestone**

```bash
git add -A
git commit -m "v0.1 deterministic spine green: core, advisor, platform seam, runtime, invariants"
```

(Still no push — the repo is local-only until the user authorizes a remote.)

---

## Self-review against the spec

**Spec coverage.** Every `familiar-core` responsibility in §4.1 has a task: events (1.1), audit (1.2), capabilities (1.3), authority envelope (1.4), advisor interface (1.5), exfil detector (1.7), permission (1.8), decision engine (1.9). The §4.2 two-gates-then-envelope control model is realized in the engine (1.9) and supervisor (2.3). The §6 data/control flow — including "every transition is appended to the audit log, including no-action outcomes, capability toggles, and permission decisions" — is the supervisor's job and is asserted by the golden traces (2.5). The §8 testing strategy is met in full at the logic level: pure-logic unit tests, security invariants as property tests (1.4/1.5/1.8) and as end-to-end tests (2.4), a fake adapter (2.2), and scenario fixtures (2.5). The §10 "core compiles and tests run with no adapter present" criterion is verified explicitly (2.6 step 1), and "no outbound traffic from the guardian" is structurally guaranteed by the no-OS core and checked by grep (2.6 step 4).

**Deliberate scope boundaries (not gaps).** The following spec items are intentionally **deferred to the follow-on plans**, consistent with §5's "ruthless YAGNI" and the planning rule against hand-writing OS/UI interfaces before a toolchain spike:

| Spec item | Where it lands |
|---|---|
| §5 Linux sensors + reversible actuators (nftables/netlink) | Plan B |
| §5 / §7 daemon, least-privilege (`CAP_NET_ADMIN`, systemd hardening) | Plan B |
| §8 real namespace/container red-team fixtures | Plan B |
| §5 / §10 Control Deck UI, live prompts, audit viewer, status feed | Plan C |
| §7 encrypted-at-rest persistent memory | v0.2 (advisor) |
| §4.1 `android/` adapter stub | v0.2c |

Each is named so the boundary is explicit. The `IntegrityAlert` audit kind and the `Irreversible` reversibility arm are implemented and tested now (the envelope is total) even though no v0.1 action exercises them — they guard every future actuator and the self-integrity check (§7) without a later refactor.

**Placeholder scan.** No `TBD`/`TODO`/"add error handling"/"similar to Task N" placeholders. Every code step carries complete, compile-ready Rust.

**Type consistency.** The public surface is consistent across tasks: `Event`/`ProcessRef` field names; `AuditKind`/`AuditLog::append(at, kind, detail)`; `CapabilityId` and its sensor/actuator mappings; `Confidence`/`HIGH_CONFIDENCE`; `Disposition`/`apply_caution`; `ProposedAction::{reversibility, actuator_capability}`; `Detection`/`Decision`; `PermissionLedger::{open, get, resolve, expire_due}` returning `(PermissionOutcome, PermissionRequest)`; the `Sensors`/`Actuators`/`Notifier` seam; and `Supervisor::{drive_once, resolve_permission}`. The `get` accessor the supervisor needs was added to the ledger interface (1.8).

---

## Follow-on plans (write each with this same skill, in order, once this plan is green)

These are **out of this plan**. They are listed so the v0.1 boundary is legible and the next planning pass has a starting point. Each will be a full bite-sized plan grounded in the *realized* core API (no guessing interfaces).

### Plan B — Linux adapter + daemon + real acceptance fixtures
- **Spike first (do not guess APIs):** prototype the chosen egress mechanism (nftables + netlink, per the settled decision) — confirm the exact crate APIs for adding/removing an nftables rule, receiving NFLOG/conntrack new-connection events, attributing a socket to a PID via `/proc`, and freezing a process via the cgroup-v2 freezer. Record the crate choices and the `/proc` attribution race as known limitations (eBPF/`aya` is the v0.2 upgrade).
- `familiar-platform/src/linux/`: implement `Sensors` (outbound-connection + sensitive-path read observation), `Actuators` (`BlockOutbound` via a removable nft rule; `FreezeProcess` via cgroup freezer), and a desktop `Notifier`. Localize all `unsafe`/FFI here, behind the same traits the fake adapter satisfies.
- `familiar-daemon`: wire `Engine` + linux adapter + `NullAdvisor` + `Supervisor`; run an async tick loop (tokio) and a least-privilege systemd unit (`AmbientCapabilities=CAP_NET_ADMIN`, `NoNewPrivileges`, `ProtectSystem=strict`, no network egress for the unit itself); persist capability snapshots and the audit log to disk.
- **Real red-team fixtures (§8 acceptance bar):** scripted exfil attempts in a throwaway network namespace/container that the suite must detect, reversibly contain (rule added then removed on resolution), and record — the same three scenarios as Task 2.5, now against the real OS.

### Plan C — Tauri Control Deck + IPC
- A local IPC surface on the daemon (Unix domain socket, newline-delimited JSON — no loopback TCP, preserving the no-self-egress posture): list/toggle capabilities, stream status + permission prompts, respond to a prompt, read the audit log.
- `familiar-ui` (Tauri, desktop): the Control Deck of per-capability switches (default-off, every toggle visible), live permission prompts wired to `resolve_permission`, the audit viewer (rendering the hash-chained log with a verify indicator), and a status feed.
- Tests: the UI cannot hold a gate — assert the IPC layer can only *toggle capabilities* and *answer prompts*, never bypass the envelope.

---

## Anomaly log

- The spec moved from `docs/superpowers/specs/` (the path in the original request) to `docs/design/` in `grimalkin` (commit `4ef1154`, per the memory index). Read at the `docs/design/` location; Task 0.1 copies it into the new `familiar` repo under `docs/superpowers/specs/` to match the skill's layout.
- `grimalkin`'s existing audit log (`grimalkin.py:610`) is a plain append-only SQLite table, **not** hash-chained — the spec's tamper-evident requirement (§7) is genuinely new in Familiar, not a port. Worth knowing when mining grimalkin for reference.
- `grimalkin` ships a committed `grimalkin.db` (+ `-shm`/`-wal`) and a `faiss_index/` in the repo root — live state tracked in git. Not this plan's concern; flagging since the rewrite should keep runtime state out of version control (the `.gitignore` in Task 0.1 already excludes `*.db`).

---

## Execution handoff

Plan complete and saved to `repos/familiar/docs/superpowers/plans/2026-06-17-familiar-guardian-v0.1.md`. It builds the deterministic spine — `familiar-core`, `familiar-advisor`, the `familiar-platform` seam + fake adapter, and `familiar-runtime` — fully tested with no OS and no UI, across 16 bite-sized TDD tasks in three phases. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task with two-stage review between tasks. Fast iteration, clean context per task.
2. **Inline Execution** — execute tasks in this session with checkpoints for review.

Which approach?
