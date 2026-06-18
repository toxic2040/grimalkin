# Familiar Guardian Plan C — Control Deck (egui UI) + control UDS IPC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the running Linux daemon a human-facing control surface — a daemon-side `/run/familiar/control.sock` IPC that lets the operator list/toggle capabilities at runtime, see and answer live permission prompts, lift containment blocks, and read the hash-chained audit log; plus a pure-Rust `familiar-ui` control deck that drives it — without ever giving the UI or the IPC a way to bypass the authority envelope.

**Architecture:** The daemon's tick loop stays the *single owner and only mutator* of the `Supervisor`. A background accept thread on a second Unix socket (`control.sock`, distinct from the helper's `fileread.sock`) authenticates the peer by uid (`SO_PEERCRED`), parses newline-delimited-JSON `ControlRequest`s, and hands each — paired with a one-shot reply channel — to the tick loop over an `mpsc`. The loop applies every command through a single trusted `apply_command` mapping that calls only the existing gated entrypoints (`Engine::set_capability`, `Supervisor::resolve_permission`, a new reverse-only `Supervisor::reverse_action`) and read accessors. There is no command — and no UI code path — that reaches `Actuators::apply`, so the only way a block or freeze installs is still sensor→detector→gates→(autonomous|granted). The UI depends only on the protocol crate; it never links the runtime or the actuators.

**Tech Stack:** Rust 1.95.0, edition 2024. New deps: `eframe`/`egui` (pure-Rust immediate-mode GUI, no webview, no JS — chosen over Tauri for auditability) in `familiar-ui` only; `rustix` (safe syscall wrappers, no `unsafe` exposed) in `familiar-daemon` for `SO_PEERCRED`. Persistence/wire stay `serde`/`serde_json`. The control transport is std `UnixListener`/`UnixStream` + `std::sync::mpsc` one-shots — no async runtime, consistent with Plan B.

## Global Constraints

These apply to **every** task. Exact values, copied from the v0.1 spec, the Plan B plan, and the workspace rules:

- **`familiar-core` gets exactly ONE additive change: a read-only `PermissionLedger::open_requests()` iterator (Task 1.1).** It adds no OS call, no clock, no mutation, and no behavior — it only lets the deck enumerate pending prompts. Nothing else in `familiar-core` is touched. If a task seems to need another core change, stop and surface it.
- **The UI can never hold or open a gate.** `ControlRequest` has no actuating variant; `apply_command` maps commands only to `set_capability` / `resolve_permission` / `reverse_action` / read accessors. `familiar-ui` depends on `familiar-ipc` only — never `familiar-runtime`, `familiar-linux`, or `familiar-platform` — so actuator types are not even in scope for UI code. This is asserted by a structural test (Task 2.2) and enforced by the dependency graph.
- **Reversal can only *reduce* containment.** The new `reverse_action` / `Actuators::reverse` path removes an nft DROP rule or thaws a frozen process. It can never install a block or freeze. The `Unblock` command therefore needs no permission gate (lifting containment is always safe), but it is still audited (`AuditKind::Actuation`, detail prefixed `reversed`).
- **`#![forbid(unsafe_code)]` stays at the top of every crate.** `familiar-ipc`, `familiar-ui`, and the daemon all keep the forbid. `SO_PEERCRED` is read through `rustix`'s safe API, not raw `libc`. (`familiar-fanotify-helper` remains the only `unsafe` crate; this plan does not touch it.)
- **Two privilege domains, unchanged.** The daemon still holds only `CAP_NET_ADMIN`; the helper still holds only `CAP_SYS_ADMIN`. `familiar-ui` runs unprivileged as the desktop operator. The control socket is the trust boundary between the unprivileged UI and the privileged daemon — authenticated by uid.
- **Control access is uid-pinned.** The daemon accepts a control connection only from `operator_uid` (config) or root (uid 0); every other peer is logged and dropped. The socket is created mode `0o660`. The in-process uid check is the authoritative gate; perms are defense-in-depth.
- **Single-owner Supervisor, no shared-state locks.** The tick loop owns the `Supervisor`; the IPC thread is pure transport. No `Arc<Mutex<Supervisor>>`. Control latency is bounded by `tick_ms` (200 ms default) — fine for a control deck.
- **Runtime toggling persists.** A `SetCapability` that succeeds rewrites `state_dir/capabilities.json` atomically, so the toggle survives a daemon restart. (Today the snapshot is loaded once at startup and never re-saved.)
- **Fail-closed everywhere.** A malformed request → `ControlResponse::Error`, never a crash and never a silent action. A failed reversal → recorded `NoAction` + `Error`. An unavailable control socket → the daemon logs and runs without the deck (degrades to Plan B behavior), never aborts.
- **No automation fingerprints** in commit messages, code comments, docs, or unit files. Write like a human engineer. No `Co-Authored-By`/AI-provenance trailers (the commit hook blocks them).
- **Repo is local-only.** `repos/familiar` is `private_local`: local commits OK, **no remote, no push** until the user explicitly authorizes.
- **Out of scope for Plan C (→ v0.2):** automatic un-blocking on `Event::ProcessExit` (needs a process-lifecycle sensor that does not exist yet), eBPF attribution/inline-drop, file+integrity detectors, the real candle advisor, Android.

---

## Red-team integration (this plan runs AFTER Plan B.1 hardening)

GPT's 2026-06-18 red-team (register: `~/.claude/.../memory/familiar-redteam-register.md`) is reconciled into the work as follows:

- **Prerequisite:** `docs/superpowers/plans/2026-06-18-familiar-guardian-plan-b1-hardening.md` lands first. It fixes F1 (disabled actuator denies — Plan C's `AnswerPrompt`/runtime-toggle paths depend on this), F2/F3 (audit reload + durable persistence — the deck's verify indicator is meaningless otherwise), F9 (`reverse_all` keeps sensing), and F7 (helper fd guard).
- **F9 already dodged here:** Plan C's `unblock_outbound` (Task 1.2) deletes one rule by handle and never touches the table, so the deck's unblock never blinds the sensor.
- **`run_nft` pre-exists:** Plan B.1 Task H5 adds the `run_nft` arg-runner to `nft.rs`. Plan C Task 1.2 **reuses it** — do not re-add it; `unblock_outbound`/`parse_handle` are the only new items there.
- **Folded into this plan:** F8 (authenticate the fileread socket — Task 2.5, reusing the control-socket peer-cred) and F4 (surface sensor health so a dead sensor under an enabled capability is loud, not silent — Task 2.6).
- **Documented as v0.2 (not built):** F5 (IPv6 egress is unsensed) and F6 (the post-SYN window lets a one-shot small exfil complete). Task 4.1 sharpens the limitations section to name both bypasses explicitly rather than implying full coverage.

---

## File structure (Plan C additions; the seven existing crates are unchanged except the four small seams noted)

```
repos/familiar/
├── Cargo.toml                         # MODIFY: add members familiar-ipc, familiar-ui; workspace deps eframe, rustix
├── crates/
│   ├── familiar-core/
│   │   └── src/permission.rs          # MODIFY (Task 1.1): add open_requests() read accessor + test
│   ├── familiar-platform/
│   │   └── src/lib.rs                 # MODIFY (Task 1.3): Actuators::reverse (default Unsupported)
│   │   └── src/testkit.rs             # MODIFY (Task 1.3): RecordingActuators::reverse records
│   ├── familiar-runtime/
│   │   └── src/lib.rs                 # MODIFY (Task 1.4): Supervisor::reverse_action (audited, reverse-only)
│   ├── familiar-linux/
│   │   └── src/nft.rs                 # MODIFY (Task 1.2): unblock_outbound + parse_handle + run_nft helpers
│   │   └── src/actuators.rs           # MODIFY (Task 1.3): LinuxActuators::reverse
│   │   └── tests/nft_netns.rs         # MODIFY (Task 1.2): block-two-unblock-one netns test
│   ├── familiar-ipc/                  # NEW — the control protocol crate (pure; no OS, no GUI)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                 # #![forbid(unsafe_code)]; ControlRequest/Response, DTOs, framing, ControlClient
│   │       └── client.rs              # ControlClient::connect/request over UnixStream
│   ├── familiar-daemon/
│   │   ├── Cargo.toml                 # MODIFY: dep familiar-ipc, rustix
│   │   └── src/
│   │       ├── config.rs              # MODIFY (Task 2.1): control_socket, operator_uid
│   │       ├── control.rs             # NEW (Task 2.2/2.3): apply_command + authorized + serve_control
│   │       ├── lib.rs                 # MODIFY: pub mod control
│   │       └── run.rs                 # MODIFY (Task 2.4): bind control socket, drain commands each tick
│   │   └── tests/control_ipc.rs       # NEW (Task 2.2/2.3): netns apply_command + socket round-trip + invariant
│   └── familiar-ui/                   # NEW — egui control deck (depends on familiar-ipc ONLY)
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                # eframe App: poll thread + render (Task 3.2)
│           └── deck.rs                # DeckModel: pure state + request-building (Task 3.1)
├── systemd/                           # (no unit change required; AF_UNIX already allowed) docs updated
└── docs/
    └── operating-familiar-linux.md    # MODIFY (Task 4.1): control deck section, operator_uid, build+run UI
```

**Dependency direction (acyclic):** `familiar-ipc → familiar-core + {serde, serde_json}`. `familiar-daemon → familiar-ipc + familiar-linux + familiar-runtime + familiar-core + familiar-advisor + {serde, serde_json, rustix, thiserror}`. `familiar-ui → familiar-ipc + {eframe}` only. Nothing depends on `familiar-ui`. Core/platform/runtime do not depend on any Plan C crate.

---

## Task 0.1: `familiar-ipc` crate — protocol types

**Files:**
- Create: `crates/familiar-ipc/Cargo.toml`
- Create: `crates/familiar-ipc/src/lib.rs`
- Modify: `Cargo.toml` (workspace `members`)

**Interfaces:**
- Consumes: `familiar_core::capabilities::{CapabilityId, CapabilitySnapshot}`, `familiar_core::audit::AuditRecord` (all already `Serialize`/`Deserialize`).
- Produces: `ControlRequest`, `ControlResponse`, `StatusSnapshot`, `PromptDto`, `BlockDto` — all `serde` round-trippable. Used by Tasks 0.2, 2.2, 3.1.

- [ ] **Step 1: Add the crate to the workspace and create its manifest**

In `Cargo.toml`, add `"crates/familiar-ipc"` to `members` and `familiar-ipc = { path = "crates/familiar-ipc" }` under `[workspace.dependencies]`.

Create `crates/familiar-ipc/Cargo.toml`:

```toml
[package]
name = "familiar-ipc"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
familiar-core.workspace = true
serde.workspace = true
serde_json.workspace = true
```

- [ ] **Step 2: Write the failing round-trip test**

Create `crates/familiar-ipc/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-ipc — the control protocol shared by the daemon and the UI.
//!
//! Newline-delimited JSON over a Unix socket. The protocol is deliberately
//! narrow: it can toggle capabilities, answer or read prompts, lift a block,
//! and read status/audit. It has NO variant that installs a block or freeze —
//! the authority envelope lives in the daemon and is never reachable here.

use familiar_core::audit::AuditRecord;
use familiar_core::capabilities::{CapabilityId, CapabilitySnapshot};
use serde::{Deserialize, Serialize};

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips_through_json() {
        let reqs = [
            ControlRequest::ListCapabilities,
            ControlRequest::SetCapability { id: CapabilityId::DetectorExfil, enabled: true },
            ControlRequest::AnswerPrompt { id: 3, granted: false },
            ControlRequest::Unblock { dst_ip: "203.0.113.9".into(), dst_port: 443 },
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
    fn status_response_round_trips_through_json() {
        let snap = StatusSnapshot {
            capabilities: CapabilitySnapshot { states: Default::default() },
            prompts: vec![PromptDto {
                id: 1, created_at: 1000, timeout_ms: 30_000, confidence: 50,
                kind: "ExfilSuspected".into(),
                proposed: "BlockOutbound".into(),
                rationale: "unestablished outbound".into(),
            }],
            active_blocks: vec![BlockDto { dst_ip: "203.0.113.9".into(), dst_port: 443 }],
            audit_ok: true,
            audit_head: "0".repeat(64),
            audit_len: 5,
        };
        let r = ControlResponse::Status(snap);
        let json = serde_json::to_string(&r).unwrap();
        let back: ControlResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `cargo test -p familiar-ipc -v`
Expected: PASS (2 tests). If `CapabilitySnapshot { states: Default::default() }` fails to construct, confirm `states` is `pub` in `familiar-core` (it is) and that `BTreeMap::default()` infers — annotate as `std::collections::BTreeMap::new()` if needed.

- [ ] **Step 4: Confirm the forbid and the workspace build**

Run: `cargo build -p familiar-ipc && cargo clippy -p familiar-ipc -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/familiar-ipc
git commit -m "Add familiar-ipc control protocol types"
```

---

## Task 0.2: `familiar-ipc` — NDJSON framing + `ControlClient`

**Files:**
- Create: `crates/familiar-ipc/src/client.rs`
- Modify: `crates/familiar-ipc/src/lib.rs` (add `pub mod client;` + framing fns)

**Interfaces:**
- Produces: `send<T: Serialize, W: Write>(&mut W, &T) -> io::Result<()>`, `recv<T: DeserializeOwned, R: BufRead>(&mut R) -> io::Result<T>`, and `client::ControlClient::{connect(path), request(&ControlRequest) -> io::Result<ControlResponse>}`. The daemon's `serve_control` (Task 2.3) uses `send`/`recv` with the types swapped; the UI (Task 3.2) uses `ControlClient`.

- [ ] **Step 1: Write the failing framing test**

Append to `crates/familiar-ipc/src/lib.rs` (above `#[cfg(test)]`):

```rust
pub mod client;

use serde::de::DeserializeOwned;
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
```

Add to the `tests` module in `lib.rs`:

```rust
#[test]
fn send_then_recv_round_trips_over_a_buffer() {
    use std::io::Cursor;
    let mut buf: Vec<u8> = Vec::new();
    let req = ControlRequest::AnswerPrompt { id: 9, granted: true };
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-ipc send_then_recv_round_trips_over_a_buffer -v`
Expected: FAIL — `client` module file does not exist yet (compile error).

- [ ] **Step 3: Implement the client**

Create `crates/familiar-ipc/src/client.rs`:

```rust
//! A blocking control client: connect to the daemon's control socket and issue
//! one request, read one response. The UI keeps one client and reuses it.
use crate::{ControlRequest, ControlResponse, recv, send};
use std::io::{self, BufReader};
use std::os::unix::net::UnixStream;
use std::path::Path;

pub struct ControlClient {
    stream: UnixStream,
    reader: BufReader<UnixStream>,
}

impl ControlClient {
    pub fn connect(path: &Path) -> io::Result<Self> {
        let stream = UnixStream::connect(path)?;
        let reader = BufReader::new(stream.try_clone()?);
        Ok(Self { stream, reader })
    }

    /// Send a request and block for the single response line.
    pub fn request(&mut self, req: &ControlRequest) -> io::Result<ControlResponse> {
        send(&mut self.stream, req)?;
        recv(&mut self.reader)
    }
}
```

- [ ] **Step 4: Run to verify pass + a live socket round-trip**

Add a socket-level test to `client.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{recv, send};
    use std::io::BufReader;
    use std::os::unix::net::UnixListener;

    #[test]
    fn client_request_round_trips_over_a_unix_socket() {
        let dir = std::env::temp_dir().join(format!("fam-ipc-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("t.sock");
        let _ = std::fs::remove_file(&sock);
        let listener = UnixListener::bind(&sock).unwrap();

        // A one-shot echo server: read a request, reply Ok.
        let h = std::thread::spawn(move || {
            let (conn, _) = listener.accept().unwrap();
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let _req: ControlRequest = recv(&mut r).unwrap();
            let mut w = conn;
            send(&mut w, &ControlResponse::Ok).unwrap();
        });

        let mut client = ControlClient::connect(&sock).unwrap();
        let resp = client.request(&ControlRequest::GetStatus).unwrap();
        assert_eq!(resp, ControlResponse::Ok);
        h.join().unwrap();
    }
}
```

Run: `cargo test -p familiar-ipc -v`
Expected: PASS (all tests, including the socket round-trip).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-ipc
git commit -m "Add familiar-ipc NDJSON framing and ControlClient"
```

---

## Task 1.1: core — `PermissionLedger::open_requests()` read accessor

**Files:**
- Modify: `crates/familiar-core/src/permission.rs`

**Interfaces:**
- Produces: `PermissionLedger::open_requests(&self) -> impl Iterator<Item = &PermissionRequest>`. The daemon's status assembly (Task 2.2) uses it to list pending prompts. This is the single permitted `familiar-core` change.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/familiar-core/src/permission.rs`:

```rust
#[test]
fn open_requests_lists_every_unresolved_request() {
    let mut led = PermissionLedger::new();
    let a = led.open(100, 5_000, sample(100));
    let b = led.open(200, 5_000, sample(200));
    led.resolve(a, true).expect("resolve a");
    let ids: Vec<RequestId> = led.open_requests().map(|r| r.id).collect();
    assert_eq!(ids, vec![b], "only the unresolved request remains");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-core open_requests_lists_every_unresolved_request -v`
Expected: FAIL — `no method named open_requests`.

- [ ] **Step 3: Add the accessor**

In `crates/familiar-core/src/permission.rs`, in `impl PermissionLedger`, after `get`:

```rust
    /// Borrow every still-open request, ordered by id (BTreeMap iteration). A
    /// read-only view for the control deck; it never mutates or expires.
    pub fn open_requests(&self) -> impl Iterator<Item = &PermissionRequest> {
        self.open.values()
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p familiar-core -v`
Expected: PASS (all core tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/permission.rs
git commit -m "Add PermissionLedger::open_requests read accessor for the control deck"
```

---

## Task 1.2: `familiar-linux` — per-block nft reversal (`unblock_outbound`)

**Files:**
- Modify: `crates/familiar-linux/src/nft.rs`
- Modify: `crates/familiar-linux/tests/nft_netns.rs`

**Interfaces:**
- Consumes: existing `nft::{ensure_table, block_outbound, TABLE, BLOCK_CHAIN}`.
- Produces: `nft::unblock_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError>` (removes exactly the one DROP rule), and a pure `parse_handle(listing: &str, dst: Ipv4Addr, dport: u16) -> Option<u64>`. Used by `LinuxActuators::reverse` (Task 1.3).

- [ ] **Step 1: Write the failing pure-parser test**

The rule is removed by handle: `nft -a list chain inet familiar egress-block` prints each rule with a trailing `# handle N`; we find ours by `daddr <ip>` + `dport <port>` + `drop`, then `nft delete rule ... handle N`. This mirrors how `install_queue_rule` already uses the `nft` binary for what rustables 0.8.7 cannot express. Add to the `tests` module of `crates/familiar-linux/src/nft.rs` (create the module if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
table inet familiar {
\tchain egress-block {
\t\ttype filter hook output priority filter; policy accept;
\t\tip daddr 203.0.113.9 tcp dport 443 drop # handle 4
\t\tip daddr 198.51.100.4 tcp dport 8443 drop # handle 7
\t}
}";

    #[test]
    fn parse_handle_finds_the_matching_rule() {
        assert_eq!(parse_handle(SAMPLE, "203.0.113.9".parse().unwrap(), 443), Some(4));
        assert_eq!(parse_handle(SAMPLE, "198.51.100.4".parse().unwrap(), 8443), Some(7));
    }

    #[test]
    fn parse_handle_does_not_confuse_a_port_prefix() {
        // dport 443 must not match a rule for dport 4430.
        let listing = "\t\tip daddr 203.0.113.9 tcp dport 4430 drop # handle 9";
        assert_eq!(parse_handle(listing, "203.0.113.9".parse().unwrap(), 443), None);
    }

    #[test]
    fn parse_handle_returns_none_when_absent() {
        assert_eq!(parse_handle(SAMPLE, "10.0.0.1".parse().unwrap(), 443), None);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-linux parse_handle -v`
Expected: FAIL — `cannot find function parse_handle`.

- [ ] **Step 3: Implement `parse_handle`, `unblock_outbound`, and the `nft` runner helpers**

In `crates/familiar-linux/src/nft.rs`, the `run_nft(args: &[&str]) -> Result<String, NftError>` arg-runner **already exists** (added by Plan B.1 Task H5). Reuse it; do not re-add it. Add only the parser and the reversal, near the top (after the existing helpers):

```rust
/// Find the kernel handle of the DROP rule for `dst:dport` in a `nft -a list`
/// chain dump. Token-exact on `daddr`/`dport` so a port prefix cannot collide.
fn parse_handle(listing: &str, dst: std::net::Ipv4Addr, dport: u16) -> Option<u64> {
    let ip = dst.to_string();
    let port = dport.to_string();
    for line in listing.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let pair = |k: &str, v: &str| toks.windows(2).any(|w| w[0] == k && w[1] == v);
        if pair("daddr", &ip) && pair("dport", &port) && toks.contains(&"drop") {
            // tokens: ... drop # handle N
            if let Some(i) = toks.iter().position(|t| *t == "handle") {
                return toks.get(i + 1).and_then(|n| n.parse().ok());
            }
        }
    }
    None
}
```

Add the public reversal at the end of the file:

```rust
/// Remove exactly the reversible DROP rule for `dst:dport` (by kernel handle),
/// leaving every other block intact. Errors if no such rule exists.
pub fn unblock_outbound(dst: std::net::Ipv4Addr, dport: u16) -> Result<String, NftError> {
    let listing = run_nft(&["-a", "list", "chain", "inet", TABLE, BLOCK_CHAIN])?;
    let handle = parse_handle(&listing, dst, dport)
        .ok_or_else(|| NftError::Send(format!("no drop rule for {dst}:{dport}")))?;
    let h = handle.to_string();
    run_nft(&["delete", "rule", "inet", TABLE, BLOCK_CHAIN, "handle", &h])?;
    Ok(format!("removed nft drop {dst}:{dport} (handle {handle})"))
}
```

(Optional, surgical: replace the hand-rolled `Command` block in `install_queue_rule` with the runner only if it keeps stdin handling intact — `install_queue_rule` pipes a batch to `nft -f -`, which `run_nft` does not cover, so **leave `install_queue_rule` as-is**; `run_nft` is only for argument-style invocations.)

- [ ] **Step 4a: Run the pure tests to verify pass**

Run: `cargo test -p familiar-linux parse_handle -v`
Expected: PASS (3 parser tests).

- [ ] **Step 4b: Add and run the netns integration test (block two, unblock one)**

Append to `crates/familiar-linux/tests/nft_netns.rs` a test that mirrors the file's existing netns/`unshare -Urn` self-exec harness (reuse the helper already in that file — do not duplicate it; if the file lacks one, copy the `reexec_in_netns` pattern verbatim from `crates/familiar-daemon/tests/redteam_network.rs`):

```rust
#[test]
fn block_two_then_unblock_one_leaves_the_other() {
    if reexec_in_netns("block_two_then_unblock_one_leaves_the_other") {
        return;
    }
    use familiar_linux::nft;
    use std::net::Ipv4Addr;
    use std::process::Command;

    let ruleset = || -> String {
        String::from_utf8(
            Command::new("nft").args(["list", "ruleset"]).output().unwrap().stdout,
        ).unwrap()
    };

    nft::ensure_table().expect("table");
    let a: Ipv4Addr = "203.0.113.9".parse().unwrap();
    let b: Ipv4Addr = "198.51.100.4".parse().unwrap();
    nft::block_outbound(a, 443).expect("block a");
    nft::block_outbound(b, 8443).expect("block b");
    let rs = ruleset();
    assert!(rs.contains("203.0.113.9") && rs.contains("198.51.100.4"), "both blocked:\n{rs}");

    nft::unblock_outbound(a, 443).expect("unblock a");
    let rs = ruleset();
    assert!(!rs.contains("203.0.113.9"), "a's drop must be gone:\n{rs}");
    assert!(rs.contains("198.51.100.4"), "b's drop must remain:\n{rs}");

    // Unblocking a non-existent rule is an error, not a silent success.
    assert!(nft::unblock_outbound(a, 443).is_err(), "second unblock has nothing to remove");
}
```

Run: `cargo test -p familiar-linux block_two_then_unblock_one_leaves_the_other -- --nocapture`
Expected: PASS (or `SKIP` line if unprivileged userns is unavailable — same convention as the other netns tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/nft.rs crates/familiar-linux/tests/nft_netns.rs
git commit -m "Add per-block nft reversal (unblock_outbound) with handle parsing"
```

---

## Task 1.3: `Actuators::reverse` trait method + `LinuxActuators`/testkit impls

**Files:**
- Modify: `crates/familiar-platform/src/lib.rs`
- Modify: `crates/familiar-platform/src/testkit.rs`
- Modify: `crates/familiar-linux/src/actuators.rs`

**Interfaces:**
- Consumes: `nft::unblock_outbound` (Task 1.2), `cgroup::Freezer::thaw`.
- Produces: `Actuators::reverse(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError>` (default: `Err(Unsupported)`); `LinuxActuators::reverse` (removes the matching block / thaws); `RecordingActuators::reverse` (records into `pub reversed: Vec<ProposedAction>`). Used by `Supervisor::reverse_action` (Task 1.4).

- [ ] **Step 1: Add the trait method with a safe default**

In `crates/familiar-platform/src/lib.rs`, inside `pub trait Actuators`, after `apply`:

```rust
    /// Reverse a previously-applied action (remove a block, thaw a process).
    /// Reversal can only *reduce* containment; it never installs anything.
    /// Default: unsupported, so existing fakes that never reverse are unaffected.
    fn reverse(&mut self, _action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        Err(ActuationError::Unsupported)
    }
```

- [ ] **Step 2: Write the failing impl tests**

In `crates/familiar-linux/src/actuators.rs` `tests` (create the module if absent) — note this needs nft, so it is a `#[cfg(test)]` unit test that only runs meaningfully under the netns harness; keep the assertion on the *recording* path here and prove the real nft path in Task 1.2's netns test. Instead, put the behavioral assertion in testkit. Add to `crates/familiar-platform/src/testkit.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_actuator_records_reversals_separately() {
        let mut a = RecordingActuators::default();
        let block = ProposedAction::BlockOutbound {
            process: familiar_core::events::ProcessRef { pid: 7, exe: "/x".into() },
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        };
        a.apply(&block).unwrap();
        a.reverse(&block).unwrap();
        assert_eq!(a.applied.len(), 1);
        assert_eq!(a.reversed.len(), 1);
        assert_eq!(a.reversed[0], block);
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p familiar-platform recording_actuator_records_reversals_separately -v`
Expected: FAIL — `RecordingActuators` has no field `reversed` / no `reverse`.

- [ ] **Step 4: Implement the two impls, run to pass**

In `crates/familiar-platform/src/testkit.rs`, extend `RecordingActuators`:

```rust
#[derive(Clone, Debug, Default)]
pub struct RecordingActuators {
    pub applied: Vec<ProposedAction>,
    pub reversed: Vec<ProposedAction>,
    pub fail: bool,
}
```

and add to `impl Actuators for RecordingActuators`:

```rust
    fn reverse(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        if self.fail {
            return Err(ActuationError::Failed("injected".into()));
        }
        self.reversed.push(action.clone());
        Ok(ActuationOutcome { note: format!("reversed {action:?}") })
    }
```

(The existing `RecordingActuators::failing()` and `applied` usages are unchanged — the new `reversed` field defaults to empty.)

In `crates/familiar-linux/src/actuators.rs`, add to `impl Actuators for LinuxActuators`:

```rust
    fn reverse(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        match action {
            ProposedAction::BlockOutbound { dst_ip, dst_port, .. } => {
                let ip: Ipv4Addr = dst_ip.parse().map_err(|_| {
                    ActuationError::Failed(format!("non-IPv4 dst {dst_ip}"))
                })?;
                let note = nft::unblock_outbound(ip, *dst_port)
                    .map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_blocks.retain(|(i, p)| !(*i == ip && *p == *dst_port));
                Ok(ActuationOutcome { note })
            }
            ProposedAction::FreezeProcess { pid } => {
                self.freezer.thaw(*pid).map_err(|e| ActuationError::Failed(e.to_string()))?;
                Ok(ActuationOutcome { note: format!("thawed pid {pid}") })
            }
        }
    }
```

Run: `cargo test -p familiar-platform -v && cargo build -p familiar-linux`
Expected: PASS; `familiar-linux` builds.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-platform/src/lib.rs crates/familiar-platform/src/testkit.rs crates/familiar-linux/src/actuators.rs
git commit -m "Add reverse-only Actuators::reverse with Linux and testkit impls"
```

---

## Task 1.4: runtime — `Supervisor::reverse_action` (audited, reverse-only)

**Files:**
- Modify: `crates/familiar-runtime/src/lib.rs`

**Interfaces:**
- Consumes: `Actuators::reverse` (Task 1.3).
- Produces: `Supervisor::reverse_action(&mut self, action: &ProposedAction, now: Timestamp) -> Result<(), ()>` — audits `AuditKind::Actuation` (`reversed …`) on success / `AuditKind::NoAction` on failure, notifies, and returns Ok/Err so the daemon can answer the IPC. Used by `apply_command` (Task 2.2).

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/familiar-runtime/src/lib.rs`:

```rust
#[test]
fn reverse_action_reverses_audits_and_notifies_but_never_installs() {
    use familiar_core::events::ProcessRef;
    let mut sup = Supervisor::new(
        armed_engine(),
        FakeSensors::new(vec![]),
        RecordingActuators::default(),
        CapturingNotifier::default(),
        NullAdvisor,
        30_000,
    );
    let block = ProposedAction::BlockOutbound {
        process: ProcessRef { pid: 7, exe: "/usr/bin/curl".into() },
        dst_ip: "203.0.113.9".into(),
        dst_port: 443,
    };
    sup.reverse_action(&block, 1000).expect("reverse ok");
    // It went through reverse(), NOT apply(): nothing was installed.
    assert!(sup.actuators().applied.is_empty(), "reverse must never install");
    assert_eq!(sup.actuators().reversed.len(), 1);
    assert!(
        sup.audit.records().iter().any(|r|
            r.kind == AuditKind::Actuation && r.detail.contains("reversed")),
        "the reversal is audited"
    );
    assert!(!sup.notifier().messages.is_empty(), "the reversal is surfaced");
    assert!(sup.audit.verify().is_ok());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-runtime reverse_action_reverses_audits_and_notifies_but_never_installs -v`
Expected: FAIL — `no method named reverse_action`.

- [ ] **Step 3: Implement `reverse_action`**

In `crates/familiar-runtime/src/lib.rs`, in `impl<S, A, N, V> Supervisor<…>`, after `act`:

```rust
    /// Lift a previously-applied containment (remove a block / thaw a process).
    /// This is the only public actuation path besides the gated `act` — and it
    /// can only *reduce* containment. Audited and surfaced like an actuation.
    pub fn reverse_action(&mut self, action: &ProposedAction, now: Timestamp) -> Result<(), ()> {
        match self.actuators.reverse(action) {
            Ok(outcome) => {
                self.audit.append(
                    now,
                    AuditKind::Actuation,
                    format!("reversed {action:?}: {}", outcome.note),
                );
                self.notifier.notify(&format!("Lifted containment {action:?} ({})", outcome.note));
                Ok(())
            }
            Err(e) => {
                self.audit.append(
                    now,
                    AuditKind::NoAction,
                    format!("reversal failed for {action:?}: {e}"),
                );
                self.notifier.notify(&format!("Could not lift containment: {e}"));
                Err(())
            }
        }
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p familiar-runtime -v`
Expected: PASS (all runtime tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-runtime/src/lib.rs
git commit -m "Add Supervisor::reverse_action: audited, reverse-only containment lift"
```

---

## Task 2.1: daemon config — `control_socket` + `operator_uid`

**Files:**
- Modify: `crates/familiar-daemon/src/config.rs`

**Interfaces:**
- Produces: `DaemonConfig.control_socket: PathBuf` (default `/run/familiar/control.sock`) and `DaemonConfig.operator_uid: u32` (default `1000`). Consumed by `serve_control` and `apply_command`.

- [ ] **Step 1: Extend the struct + default, update the round-trip test expectation**

In `crates/familiar-daemon/src/config.rs`, add to `DaemonConfig`:

```rust
    pub control_socket: PathBuf,
    pub operator_uid: u32,
```

and to `Default`:

```rust
            control_socket: PathBuf::from("/run/familiar/control.sock"),
            operator_uid: 1000,
```

The existing `config_round_trips_through_json` test covers the new fields automatically (it round-trips the whole default). Add one assertion to it:

```rust
        assert_eq!(back.operator_uid, 1000);
        assert_eq!(back.control_socket, PathBuf::from("/run/familiar/control.sock"));
```

- [ ] **Step 2: Run to verify pass**

Run: `cargo test -p familiar-daemon config_round_trips_through_json -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/familiar-daemon/src/config.rs
git commit -m "Add control_socket and operator_uid to DaemonConfig"
```

---

## Task 2.2: daemon — `apply_command` (the trusted command mapping)

**Files:**
- Create: `crates/familiar-daemon/src/control.rs`
- Modify: `crates/familiar-daemon/src/lib.rs` (`pub mod control;`)
- Modify: `crates/familiar-daemon/Cargo.toml` (dep `familiar-ipc`)
- Create: `crates/familiar-daemon/tests/control_ipc.rs`

**Interfaces:**
- Consumes: `familiar_ipc::{ControlRequest, ControlResponse, StatusSnapshot, PromptDto, BlockDto}`, `Supervisor` (concrete `LinuxActuators`/`LinuxNotifier`/`NullAdvisor`), `persistence::save_capabilities`.
- Produces: `pub fn apply_command<S: Sensors>(sup: &mut Supervisor<S, LinuxActuators, LinuxNotifier, NullAdvisor>, cfg: &DaemonConfig, req: ControlRequest, now: u64) -> ControlResponse`. Used by `serve_control` wiring (Task 2.4) and the tests.

- [ ] **Step 1: Add the dependency**

In `crates/familiar-daemon/Cargo.toml` `[dependencies]`, add `familiar-ipc.workspace = true`. In `crates/familiar-daemon/src/lib.rs`, add `pub mod control;`.

- [ ] **Step 2: Write the failing integration tests (netns)**

Create `crates/familiar-daemon/tests/control_ipc.rs`. Reuse the `reexec_in_netns`, `ruleset`, and `arm` helpers verbatim from `crates/familiar-daemon/tests/redteam_network.rs` (copy them into this file's top — the test files do not share a module). Then:

```rust
use familiar_core::capabilities::CapabilityId;
use familiar_core::events::{Event, ProcessRef};
use familiar_daemon::config::DaemonConfig;
use familiar_daemon::control::apply_command;
use familiar_daemon::run::build_supervisor_with_sensors;
use familiar_ipc::{ControlRequest, ControlResponse};

// --- paste reexec_in_netns, ruleset, arm from redteam_network.rs here ---

/// One unlinked outbound (confidence 50 => ask) on the first poll, then nothing.
struct OneOutbound(std::cell::Cell<bool>);
impl familiar_platform::Sensors for OneOutbound {
    fn poll(&mut self) -> Vec<Event> {
        if self.0.replace(true) { return vec![]; }
        vec![Event::OutboundConn {
            at: 1000,
            process: ProcessRef { pid: 7, exe: "/usr/bin/curl".into() },
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }]
    }
}

fn armed_sup(tag: &str) -> (DaemonConfig, familiar_runtime::Supervisor<
    OneOutbound, familiar_linux::LinuxActuators, familiar_linux::LinuxNotifier, familiar_advisor::NullAdvisor>) {
    let cfg = DaemonConfig {
        cgroup_root: "/sys/fs/cgroup".into(),
        state_dir: std::env::temp_dir().join(format!("fam-ctl-{}-{}", std::process::id(), tag)),
        ..DaemonConfig::default()
    };
    let _ = std::fs::remove_dir_all(&cfg.state_dir);
    arm(&cfg.state_dir);
    let sup = build_supervisor_with_sensors(&cfg, OneOutbound(std::cell::Cell::new(false))).expect("build");
    (cfg, sup)
}

#[test]
fn answer_prompt_grant_installs_the_block_via_ipc() {
    if reexec_in_netns("answer_prompt_grant_installs_the_block_via_ipc") { return; }
    let (cfg, mut sup) = armed_sup("grant");
    sup.drive_once(2000);
    // A prompt is now open; status shows it.
    let status = apply_command(&mut sup, &cfg, ControlRequest::GetStatus, 2000);
    let id = match status { ControlResponse::Status(s) => { assert_eq!(s.prompts.len(), 1); s.prompts[0].id }, o => panic!("{o:?}") };
    assert!(!ruleset().contains("drop"), "no block before the grant");
    let r = apply_command(&mut sup, &cfg, ControlRequest::AnswerPrompt { id, granted: true }, 2500);
    assert_eq!(r, ControlResponse::Ok);
    assert!(ruleset().contains("drop"), "grant installs the block");
}

#[test]
fn unblock_via_ipc_lifts_the_block() {
    if reexec_in_netns("unblock_via_ipc_lifts_the_block") { return; }
    let (cfg, mut sup) = armed_sup("unblock");
    sup.drive_once(2000);
    let status = apply_command(&mut sup, &cfg, ControlRequest::GetStatus, 2000);
    let id = match status { ControlResponse::Status(s) => s.prompts[0].id, o => panic!("{o:?}") };
    apply_command(&mut sup, &cfg, ControlRequest::AnswerPrompt { id, granted: true }, 2500);
    assert!(ruleset().contains("drop"));
    let r = apply_command(&mut sup, &cfg, ControlRequest::Unblock { dst_ip: "203.0.113.9".into(), dst_port: 443 }, 2600);
    assert_eq!(r, ControlResponse::Ok);
    assert!(!ruleset().contains("drop"), "unblock lifts containment");
}

#[test]
fn set_capability_toggles_and_persists() {
    if reexec_in_netns("set_capability_toggles_and_persists") { return; }
    let (cfg, mut sup) = armed_sup("toggle");
    let r = apply_command(&mut sup, &cfg, ControlRequest::SetCapability { id: CapabilityId::ActuatorFreezeProcess, enabled: true }, 100);
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
    if reexec_in_netns("no_command_can_install_containment") { return; }
    let (cfg, mut sup) = armed_sup("invariant");
    // Drain the one scripted outbound by answering its prompt with a DENY.
    sup.drive_once(2000);
    let status = apply_command(&mut sup, &cfg, ControlRequest::GetStatus, 2000);
    let id = match status { ControlResponse::Status(s) => s.prompts[0].id, o => panic!("{o:?}") };
    apply_command(&mut sup, &cfg, ControlRequest::AnswerPrompt { id, granted: false }, 2100);
    assert!(!ruleset().contains("drop"), "deny installs nothing");
    // Now hammer every other command; none may contain.
    for cmd in [
        ControlRequest::ListCapabilities,
        ControlRequest::GetStatus,
        ControlRequest::GetAudit { since_seq: 0 },
        ControlRequest::SetCapability { id: CapabilityId::ActuatorBlockConn, enabled: true },
        ControlRequest::Unblock { dst_ip: "203.0.113.9".into(), dst_port: 443 }, // nothing to remove
    ] {
        let _ = apply_command(&mut sup, &cfg, cmd, 3000);
        assert!(!ruleset().contains("drop"), "no non-grant command may install a block");
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cargo test -p familiar-daemon --test control_ipc -- --nocapture`
Expected: FAIL to compile — `familiar_daemon::control` does not exist.

- [ ] **Step 4: Implement `apply_command`, run to pass**

Create `crates/familiar-daemon/src/control.rs` (the `serve_control`/`authorized` parts come in Task 2.3; this step adds only `apply_command` + helpers):

```rust
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

fn status<S: Sensors>(sup: &Sup<S>) -> StatusSnapshot {
    StatusSnapshot {
        capabilities: sup.engine.registry().snapshot(),
        prompts: sup.ledger.open_requests().map(prompt_dto).collect(),
        active_blocks: sup
            .actuators()
            .active_blocks()
            .iter()
            .map(|(ip, p)| BlockDto { dst_ip: ip.to_string(), dst_port: *p })
            .collect(),
        audit_ok: sup.audit.verify().is_ok(),
        audit_head: sup.audit.head_hash().to_string(),
        audit_len: sup.audit.records().len() as u64,
    }
}

/// Apply one control request. Pure side-effects on `sup` + persistence; returns
/// the response to send back over the socket.
pub fn apply_command<S: Sensors>(
    sup: &mut Sup<S>,
    cfg: &DaemonConfig,
    req: ControlRequest,
    now: u64,
) -> ControlResponse {
    match req {
        ControlRequest::ListCapabilities => {
            ControlResponse::Capabilities(sup.engine.registry().snapshot())
        }
        ControlRequest::SetCapability { id, enabled } => {
            // Disjoint borrows of two distinct public fields.
            sup.engine.set_capability(id, enabled, now, &mut sup.audit);
            match persistence::save_capabilities(&cfg.state_dir, &sup.engine.registry().snapshot()) {
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
                process: familiar_core::events::ProcessRef { pid: 0, exe: String::new() },
                dst_ip,
                dst_port,
            };
            match sup.reverse_action(&action, now) {
                Ok(()) => ControlResponse::Ok,
                Err(()) => ControlResponse::Error("unblock failed (see audit log)".into()),
            }
        }
        ControlRequest::GetStatus => ControlResponse::Status(status(sup)),
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
```

Run: `cargo test -p familiar-daemon --test control_ipc -- --nocapture`
Expected: PASS (or SKIP lines if unprivileged userns is unavailable).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-daemon/Cargo.toml crates/familiar-daemon/src/lib.rs crates/familiar-daemon/src/control.rs crates/familiar-daemon/tests/control_ipc.rs
git commit -m "Add daemon apply_command control mapping with envelope-can't-be-bypassed test"
```

---

## Task 2.3: daemon — peer-uid auth + `serve_control` socket thread

**Files:**
- Modify: `crates/familiar-daemon/src/control.rs`
- Modify: `crates/familiar-daemon/Cargo.toml` (dep `rustix`)
- Modify: `crates/familiar-daemon/tests/control_ipc.rs` (auth fn test + socket round-trip)

**Interfaces:**
- Produces: `pub fn authorized(peer_uid: u32, operator_uid: u32) -> bool`; `pub type ControlEnvelope = (ControlRequest, std::sync::mpsc::Sender<ControlResponse>)`; `pub fn serve_control(socket: &Path, operator_uid: u32) -> std::io::Result<(Receiver<ControlEnvelope>, JoinHandle<()>)>`. Consumed by the run loop (Task 2.4).

- [ ] **Step 1: Add the dependency**

In `Cargo.toml` `[workspace.dependencies]`: `rustix = { version = "0.38", features = ["net"] }`. In `crates/familiar-daemon/Cargo.toml`: `rustix.workspace = true`.
(Confirm the current `rustix` peer-cred API path with `cargo doc -p rustix --open`; this plan uses `rustix::net::sockopt::socket_peercred(&fd) -> UCred` with `ucred.uid.as_raw()`. If the function is named differently in the resolved version, adjust the one call site in Step 3 — the rest is unaffected.)

- [ ] **Step 2: Write the failing auth test**

Add to `crates/familiar-daemon/tests/control_ipc.rs` (no netns needed — pure):

```rust
#[test]
fn only_operator_or_root_is_authorized() {
    use familiar_daemon::control::authorized;
    assert!(authorized(1000, 1000), "operator allowed");
    assert!(authorized(0, 1000), "root always allowed");
    assert!(!authorized(1001, 1000), "another user rejected");
    assert!(!authorized(33, 1000), "www-data rejected");
}
```

- [ ] **Step 3: Run to verify it fails, then implement**

Run: `cargo test -p familiar-daemon --test control_ipc only_operator_or_root_is_authorized -v`
Expected: FAIL — `authorized` not found.

Append to `crates/familiar-daemon/src/control.rs`:

```rust
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
    let cred = rustix::net::sockopt::socket_peercred(stream)?;
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
        let resp = reply_rx.recv().unwrap_or(ControlResponse::Error("daemon busy".into()));
        familiar_ipc::send(&mut writer, &resp)?;
    }
}
```

Run: `cargo test -p familiar-daemon --test control_ipc only_operator_or_root_is_authorized -v`
Expected: PASS.

- [ ] **Step 4: Add and run a socket round-trip integration test**

Add to `crates/familiar-daemon/tests/control_ipc.rs` (no netns; same-uid connect, so `authorized` passes; we feed responses from a stub loop thread):

```rust
#[test]
fn serve_control_round_trips_a_request() {
    use familiar_daemon::control::serve_control;
    use familiar_ipc::ControlClient;

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
    let resp = client.request(&ControlRequest::ListCapabilities).expect("request");
    assert_eq!(resp, ControlResponse::Ok);
    loop_h.join().unwrap();
}
```

(Imports `rustix::process::getuid` — already available via the `rustix` dep; if the path differs in the resolved version, use any available "current uid" source.)

Run: `cargo test -p familiar-daemon --test control_ipc -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-daemon/Cargo.toml crates/familiar-daemon/src/control.rs crates/familiar-daemon/tests/control_ipc.rs
git commit -m "Add uid-authenticated control socket server (serve_control)"
```

---

## Task 2.4: daemon — wire the control socket into the tick loop

**Files:**
- Modify: `crates/familiar-daemon/src/run.rs`

**Interfaces:**
- Consumes: `control::{serve_control, apply_command}`.
- Produces: the running daemon now serves the control deck. No new public fn (the loop is `-> !`; behavior is covered by Task 2.2/2.3 tests).

- [ ] **Step 1: Bind the control socket and drain commands each tick**

In `crates/familiar-daemon/src/run.rs`, inside `main_loop`, after the file-read source is spawned and before `build_supervisor`, add:

```rust
    // Control deck IPC. If the socket cannot be bound, log and run headless —
    // the guardian must not depend on the UI being present.
    let (ctl_rx, _ctl_handle) = match crate::control::serve_control(&cfg.control_socket, cfg.operator_uid) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[familiar] control socket unavailable ({e}); running without the deck");
            let (_tx, rx) = std::sync::mpsc::channel();
            (rx, std::thread::spawn(|| {}))
        }
    };
```

Then in the `loop { … }` body, after `sup.drive_once(now);` and before the audit-persist tail, drain the control channel so any audit records the commands produce get flushed in the same iteration:

```rust
        // Apply any queued control commands (single-owner: only this loop mutates sup).
        while let Ok((req, reply)) = ctl_rx.try_recv() {
            let resp = crate::control::apply_command(&mut sup, &cfg, req, now);
            let _ = reply.send(resp);
        }
```

- [ ] **Step 2: Build and run the full daemon test suite**

Run: `cargo build -p familiar-daemon && cargo test -p familiar-daemon -- --nocapture`
Expected: PASS (existing `loop_contains`, `redteam_network`, and new `control_ipc` tests; netns ones may SKIP unprivileged).

- [ ] **Step 3: Workspace-wide green + lint**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: clean. (`familiar-ui` is added in Phase 3; until then it is not a member yet, so this passes now and is re-run after Phase 3.)

- [ ] **Step 4: Commit**

```bash
git add crates/familiar-daemon/src/run.rs
git commit -m "Serve the control deck from the daemon tick loop"
```

---

## Task 2.5: F8 — authenticate the fileread (helper) socket; reject non-root

**Files:**
- Modify: `crates/familiar-daemon/src/filereads.rs`

**Interfaces:**
- `spawn_socket_source` now drops any connection whose peer uid is not 0. The legitimate client is the `CAP_SYS_ADMIN` helper, which runs as root; a non-root local process can therefore neither spoof `FileRead` events (which could manufacture a high-confidence linkage and trigger a real block) nor hold the connection to starve the real helper (it is rejected at accept).

- [ ] **Step 1: Add a peer-uid check to the accept loop**

In `crates/familiar-daemon/src/filereads.rs`, in the `listener.incoming().flatten()` loop, before constructing the `BufReader`, reject non-root peers (reuse the daemon's `rustix` dep):

```rust
        for stream in listener.incoming().flatten() {
            // F8: the only legitimate client is the CAP_SYS_ADMIN helper (root).
            // Reject anything else — a non-root process must not be able to spoof
            // FileRead events or hold the socket to starve the real helper.
            match rustix::net::sockopt::socket_peercred(&stream) {
                Ok(cred) if cred.uid.as_raw() == 0 => {}
                Ok(cred) => {
                    eprintln!("[familiar] fileread: rejecting non-root peer uid {}", cred.uid.as_raw());
                    continue;
                }
                Err(e) => {
                    eprintln!("[familiar] fileread: cannot read peer cred: {e}");
                    continue;
                }
            }
            let reader = BufReader::new(stream);
            // ... unchanged: read newline-JSON FileReadEvents onto the channel ...
        }
```

(Same `rustix` peer-cred call as Task 2.3; confirm the resolved API name once and use it in both sites.)

- [ ] **Step 2: Build + the existing fileread/loop tests**

Run: `cargo test -p familiar-daemon -- --nocapture`
Expected: PASS/SKIP as before — the netns tests connect as the same (root, under `unshare -Ur`) uid, so the check passes; the channel-based tests do not exercise the socket. Add an `authorized`-style note: the root-only decision is the same shape as `control::authorized`, already unit-tested.

- [ ] **Step 3: Commit**

```bash
git add crates/familiar-daemon/src/filereads.rs
git commit -m "Authenticate the fileread socket: accept only the root helper"
```

---

## Task 2.6: F4 — surface sensor health so a dead sensor is loud, not silent

**Files:**
- Modify: `crates/familiar-ipc/src/lib.rs` (two `StatusSnapshot` fields)
- Modify: `crates/familiar-daemon/src/run.rs` (health flags + blindness IntegrityAlert)
- Modify: `crates/familiar-daemon/src/control.rs` (`apply_command` takes sensor health into the status)

**Interfaces:**
- `StatusSnapshot` gains `network_sensor_ok: bool` and `file_sensor_ok: bool`. `apply_command` gains a `health: SensorHealth` parameter (a plain `{ network_ok: bool, file_ok: bool }` snapshot the loop reads from `Arc<AtomicBool>` flags). When a sensor backing an *enabled* capability is down, the loop records an `AuditKind::IntegrityAlert` once (not every tick) and the deck shows it red.

- [ ] **Step 1: Add the StatusSnapshot fields (and update the Task 0.1 round-trip test)**

In `crates/familiar-ipc/src/lib.rs`, add to `StatusSnapshot`:

```rust
    /// The NFQUEUE outbound sensor reader is alive.
    pub network_sensor_ok: bool,
    /// The fanotify helper file-read source is connected.
    pub file_sensor_ok: bool,
```

Update the `status_response_round_trips_through_json` test literal to set both (`network_sensor_ok: true, file_sensor_ok: true`).

Run: `cargo test -p familiar-ipc -v` — Expected: PASS.

- [ ] **Step 2: Health flags in the run loop (failing build until apply_command is updated)**

In `crates/familiar-daemon/src/run.rs`, define a tiny health snapshot and `Arc<AtomicBool>` flags the reader threads clear on exit:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
pub struct SensorHealth { pub network_ok: bool, pub file_ok: bool }
```

Make the NFQUEUE reader clear its flag on exit, and set `file_ok` from whether the helper source bound:

```rust
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
```

In the loop, build the health snapshot, raise a one-shot IntegrityAlert on blindness, and pass health into control:

```rust
        let health = SensorHealth {
            network_ok: network_ok.load(Ordering::Relaxed),
            file_ok: file_ok.load(Ordering::Relaxed),
        };
        // F4: a sensor backing an ENABLED capability being down is blindness —
        // record it once (track with a latch so we don't spam the chain).
        if sup.engine.registry().is_enabled(familiar_core::capabilities::CapabilityId::SensorOutboundConn)
            && !health.network_ok && !net_alerted {
            sup.audit.append(now, familiar_core::audit::AuditKind::IntegrityAlert,
                "outbound sensor (NFQUEUE reader) is down while SensorOutboundConn is enabled".into());
            net_alerted = true;
        }
        if sup.engine.registry().is_enabled(familiar_core::capabilities::CapabilityId::SensorSensitiveRead)
            && !health.file_ok && !file_alerted {
            sup.audit.append(now, familiar_core::audit::AuditKind::IntegrityAlert,
                "file-read sensor (fanotify helper) is unavailable while SensorSensitiveRead is enabled".into());
            file_alerted = true;
        }
        while let Ok((req, reply)) = ctl_rx.try_recv() {
            let resp = crate::control::apply_command(&mut sup, &cfg, req, now, health);
            let _ = reply.send(resp);
        }
```

(Declare `let mut net_alerted = false; let mut file_alerted = false;` before the loop.)

- [ ] **Step 3: Thread health into `apply_command`/`status`**

In `crates/familiar-daemon/src/control.rs`, add the parameter and populate the new fields:

```rust
pub fn apply_command<S: Sensors>(
    sup: &mut Sup<S>,
    cfg: &DaemonConfig,
    req: ControlRequest,
    now: u64,
    health: crate::run::SensorHealth,
) -> ControlResponse {
    // ... unchanged arms; GetStatus calls status(sup, health) ...
}

fn status<S: Sensors>(sup: &Sup<S>, health: crate::run::SensorHealth) -> StatusSnapshot {
    StatusSnapshot {
        // ... existing fields ...
        network_sensor_ok: health.network_ok,
        file_sensor_ok: health.file_ok,
    }
}
```

Update the Task 2.2 `control_ipc` tests' `apply_command(...)` calls to pass a healthy `SensorHealth { network_ok: true, file_ok: true }`.

- [ ] **Step 4: Show it in the deck (Task 3.2 addition)**

In `familiar-ui` `main.rs`, in the status render, add a sensor row:

```rust
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Sensors").strong());
                let chip = |ui: &mut egui::Ui, ok: bool, name: &str| {
                    let (c, t) = if ok { (egui::Color32::GREEN, "up") } else { (egui::Color32::RED, "DOWN") };
                    ui.colored_label(c, format!("{name}: {t}"));
                };
                chip(ui, status.network_sensor_ok, "network");
                chip(ui, status.file_sensor_ok, "file");
            });
```

- [ ] **Step 5: Workspace green + commit**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: clean.

```bash
git add crates/familiar-ipc/src/lib.rs crates/familiar-daemon/src/run.rs crates/familiar-daemon/src/control.rs crates/familiar-ui/src/main.rs
git commit -m "Surface sensor health and alert on blindness under an enabled capability"
```

(Stretch, same family as F7: the helper can send a wire sentinel on `FAN_Q_OVERFLOW` so `file_sensor_ok` flips and an IntegrityAlert fires on dropped reads — deferred unless time allows; the up/down signal above is the required deliverable.)

---

## Task 3.1: `familiar-ui` — `DeckModel` (pure state + request building)

**Files:**
- Create: `crates/familiar-ui/Cargo.toml`
- Create: `crates/familiar-ui/src/deck.rs`
- Create: `crates/familiar-ui/src/main.rs` (stub for now; filled in Task 3.2)
- Modify: `Cargo.toml` (workspace `members`)

**Interfaces:**
- Consumes: `familiar_ipc::{StatusSnapshot, ControlRequest}`, `familiar_core::capabilities::CapabilityId`.
- Produces: `DeckModel { status: Option<StatusSnapshot>, last_error: Option<String> }` with `toggle(id, enabled) -> ControlRequest`, `answer(id, granted) -> ControlRequest`, `unblock(&BlockDto) -> ControlRequest`. Pure; unit-tested. Used by the eframe App (Task 3.2).

- [ ] **Step 1: Create the crate + manifest**

In `Cargo.toml`, add `"crates/familiar-ui"` to `members`. Create `crates/familiar-ui/Cargo.toml`:

```toml
[package]
name = "familiar-ui"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
familiar-ipc.workspace = true
familiar-core.workspace = true
eframe = "0.31"   # confirm current release with `cargo add eframe`; uses only stable egui widgets
```

(`familiar-ui` depends on `familiar-ipc` + `familiar-core` only — never the runtime, linux, or platform crates. This is the structural guarantee that no UI code can reach an actuator.)

Create a minimal `crates/familiar-ui/src/main.rs` so the crate compiles:

```rust
#![forbid(unsafe_code)]
mod deck;

fn main() {
    eprintln!("familiar-ui placeholder — real UI lands in Task 3.2");
}
```

- [ ] **Step 2: Write the failing `DeckModel` test**

Create `crates/familiar-ui/src/deck.rs`:

```rust
//! The control deck's pure model: it holds the latest status the daemon sent and
//! turns user gestures into `ControlRequest`s. It contains no actuator types and
//! no socket I/O — the eframe App owns the client and feeds this model snapshots.
use familiar_core::capabilities::CapabilityId;
use familiar_ipc::{BlockDto, ControlRequest, StatusSnapshot};

#[derive(Default)]
pub struct DeckModel {
    pub status: Option<StatusSnapshot>,
    pub last_error: Option<String>,
}

impl DeckModel {
    pub fn toggle(&self, id: CapabilityId, enabled: bool) -> ControlRequest {
        ControlRequest::SetCapability { id, enabled }
    }
    pub fn answer(&self, id: u64, granted: bool) -> ControlRequest {
        ControlRequest::AnswerPrompt { id, granted }
    }
    pub fn unblock(&self, b: &BlockDto) -> ControlRequest {
        ControlRequest::Unblock { dst_ip: b.dst_ip.clone(), dst_port: b.dst_port }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gestures_map_to_the_narrow_requests() {
        let m = DeckModel::default();
        assert_eq!(
            m.toggle(CapabilityId::DetectorExfil, true),
            ControlRequest::SetCapability { id: CapabilityId::DetectorExfil, enabled: true }
        );
        assert_eq!(m.answer(5, false), ControlRequest::AnswerPrompt { id: 5, granted: false });
        let b = BlockDto { dst_ip: "203.0.113.9".into(), dst_port: 443 };
        assert_eq!(m.unblock(&b), ControlRequest::Unblock { dst_ip: "203.0.113.9".into(), dst_port: 443 });
    }
}
```

- [ ] **Step 3: Run to verify pass**

Run: `cargo test -p familiar-ui -v`
Expected: PASS (eframe compiles as a dependency; the test itself does not open a window).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/familiar-ui
git commit -m "Add familiar-ui DeckModel (pure request-building, ipc-only deps)"
```

---

## Task 3.2: `familiar-ui` — the eframe control deck

**Files:**
- Modify: `crates/familiar-ui/src/main.rs`

**Interfaces:**
- Consumes: `familiar_ipc::ControlClient`, `DeckModel`, `CapabilityId::ALL`.
- Produces: a runnable `familiar-ui` binary that polls `GetStatus` and renders capability switches, prompts, active blocks, and the audit verify chip. (Rendering is not unit-tested; verified by `cargo build` + manual launch in Task 4.)

- [ ] **Step 1: Implement the App**

Replace `crates/familiar-ui/src/main.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-ui — the local control deck. A thin egui front end over the daemon's
//! control socket. It can toggle capabilities, answer prompts, lift blocks, and
//! show the audit chain — nothing else: the protocol has no actuating verb and
//! this crate links no actuator code.
mod deck;

use deck::DeckModel;
use eframe::egui;
use familiar_core::capabilities::CapabilityId;
use familiar_ipc::{ControlClient, ControlRequest, ControlResponse};
use std::path::PathBuf;
use std::time::Duration;

struct App {
    client: Option<ControlClient>,
    socket: PathBuf,
    model: DeckModel,
}

impl App {
    fn new(socket: PathBuf) -> Self {
        Self { client: ControlClient::connect(&socket).ok(), socket, model: DeckModel::default() }
    }

    /// Issue one request, refreshing the connection if it dropped.
    fn send(&mut self, req: ControlRequest) -> Option<ControlResponse> {
        if self.client.is_none() {
            self.client = ControlClient::connect(&self.socket).ok();
        }
        let resp = self.client.as_mut()?.request(&req).ok();
        if resp.is_none() {
            self.client = None; // force reconnect next time
        }
        resp
    }

    fn refresh_status(&mut self) {
        match self.send(ControlRequest::GetStatus) {
            Some(ControlResponse::Status(s)) => { self.model.status = Some(s); self.model.last_error = None; }
            Some(ControlResponse::Error(e)) => self.model.last_error = Some(e),
            None => self.model.last_error = Some("daemon not reachable".into()),
            _ => {}
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the daemon ~2x/sec.
        self.refresh_status();
        ctx.request_repaint_after(Duration::from_millis(500));

        let mut pending: Vec<ControlRequest> = Vec::new();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Familiar — Control Deck");
            if let Some(err) = &self.model.last_error {
                ui.colored_label(egui::Color32::RED, format!("⚠ {err}"));
            }
            let status = self.model.status.clone();
            let Some(status) = status else { ui.label("waiting for the daemon…"); return; };

            ui.separator();
            ui.label(egui::RichText::new("Capabilities").strong());
            for id in CapabilityId::ALL {
                let on = status.capabilities.states.get(&id).copied().unwrap_or(false);
                let mut v = on;
                if ui.checkbox(&mut v, format!("{id:?}")).changed() {
                    pending.push(self.model.toggle(id, v));
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("Pending approvals").strong());
            if status.prompts.is_empty() { ui.weak("none"); }
            for p in &status.prompts {
                ui.horizontal(|ui| {
                    ui.label(format!("#{} [{}] {} — {}", p.id, p.confidence, p.proposed, p.rationale));
                    if ui.button("Allow").clicked() { pending.push(self.model.answer(p.id, true)); }
                    if ui.button("Deny").clicked() { pending.push(self.model.answer(p.id, false)); }
                });
            }

            ui.separator();
            ui.label(egui::RichText::new("Active containment").strong());
            if status.active_blocks.is_empty() { ui.weak("none"); }
            for b in &status.active_blocks {
                ui.horizontal(|ui| {
                    ui.label(format!("{}:{}", b.dst_ip, b.dst_port));
                    if ui.button("Lift").clicked() { pending.push(self.model.unblock(b)); }
                });
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Audit chain").strong());
                if status.audit_ok {
                    ui.colored_label(egui::Color32::GREEN, format!("✔ verified ({} records)", status.audit_len));
                } else {
                    ui.colored_label(egui::Color32::RED, "✘ TAMPERED");
                }
            });
            ui.monospace(format!("head {}", status.audit_head));
        });

        for req in pending {
            self.send(req);
        }
    }
}

fn main() -> eframe::Result<()> {
    let socket = std::env::args().nth(1).map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/run/familiar/control.sock"));
    eframe::run_native(
        "Familiar Control Deck",
        eframe::NativeOptions::default(),
        Box::new(move |_cc| Ok(Box::new(App::new(socket)))),
    )
}
```

- [ ] **Step 2: Build (the only automatable check for a GUI)**

Run: `cargo build -p familiar-ui`
Expected: builds. (If `eframe::run_native`'s closure signature differs in the resolved version — older eframe returns `Box<dyn App>` without `Ok(...)` — adjust the closure body; the App logic is unaffected.)

- [ ] **Step 3: Workspace green + lint with the UI included**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/familiar-ui/src/main.rs
git commit -m "Add familiar-ui eframe control deck"
```

---

## Task 4.1: docs — operating guide: the control deck

**Files:**
- Modify: `docs/operating-familiar-linux.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Document the control deck, operator_uid, and the build/run**

In `docs/operating-familiar-linux.md`, in the "Configuration" JSON example, add the two new keys:

```json
  "control_socket": "/run/familiar/control.sock",
  "operator_uid": 1000
```

Replace the "Runtime toggling — the control deck — is Plan C; until then…" sentence in "The capability model" with a real section:

```markdown
## The control deck (familiar-ui)

The deck is a local egui app that talks to the daemon over
`/run/familiar/control.sock`. It can, at runtime: toggle any capability (the
toggle is persisted to `capabilities.json` and survives a restart), see and
answer pending permission prompts, lift an active containment block, and view
the hash-chained audit log with a live verify indicator.

The deck holds no authority of its own. The control socket exposes only those
verbs — there is no command that installs a block or freeze, so the deck can
never bypass the authority envelope; containment still only happens through
sensor → detector → gates → (autonomous high-confidence | human grant).

Access is restricted to `operator_uid` (set it to your desktop user's uid) or
root; the socket is mode 0660. Build and run it as that user:

    cargo build --release -p familiar-ui
    ./target/release/familiar-ui            # or: familiar-ui /run/familiar/control.sock

Automatic un-blocking on process exit is v0.2 (it needs a process-lifecycle
sensor); v0.1 lifts blocks only on an explicit "Lift" in the deck.
```

- [ ] **Step 2: Verify the doc references match the shipped config + binary**

Run: `cargo run -p familiar-daemon -- --help 2>/dev/null; grep -n "control_socket\|operator_uid" crates/familiar-daemon/src/config.rs`
Expected: the two keys exist in `config.rs` (sanity check the doc matches code).

- [ ] **Step 3: Commit**

```bash
git add docs/operating-familiar-linux.md
git commit -m "Document the familiar-ui control deck and control socket"
```

---

## Task 4.2: acceptance — manual control-deck smoke + final sweep

**Files:** none (verification + optional note).

- [ ] **Step 1: Full workspace verification**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: all green. Record the test count.

- [ ] **Step 2: Privileged/manual smoke (operator-run, like Plan B's acceptance)**

This is a manual step the operator runs (the daemon needs `CAP_NET_ADMIN`; the deck needs to connect as `operator_uid`). Document the sequence; do not script an unattended privileged run:

1. Build: `cargo build --release -p familiar-daemon -p familiar-ui`.
2. In one terminal (root/caps): run the daemon against a test config whose `operator_uid` is your uid and `state_dir`/sockets are under `/run/familiar`.
3. In your desktop session: `familiar-ui /run/familiar/control.sock`.
4. Confirm: capabilities render and toggle (and persist across a daemon restart); a scripted/real ambiguous outbound raises a prompt that the deck shows and can Allow/Deny; an installed block appears under "Active containment" and "Lift" removes it (`nft list ruleset` confirms); the audit chip stays green and the head hash advances.

- [ ] **Step 3: Run the closure rituals (per workspace AGENTS.md)**

Run: `cd /home/toxic2040/work && ./bin/catalog-rescan` and address any STALE/MISSING for files this work touched.

- [ ] **Step 4: Final commit (if the smoke surfaced any doc/code fixes)**

```bash
git add -A
git commit -m "Plan C control deck: final verification fixes"
```

---

## Self-review

**Spec coverage (handoff lines 55–70):**
- Daemon-side UDS IPC (NDJSON on `/run/familiar`, no loopback TCP, no-self-egress preserved) → Tasks 0.1–0.2 (protocol), 2.2–2.4 (server + loop). ✔
- list/toggle capabilities → `ListCapabilities`/`SetCapability` (2.2). ✔
- stream status + permission prompts → `GetStatus`/`StatusSnapshot.prompts` via `open_requests` (1.1, 2.2). ✔
- answer a prompt → `AnswerPrompt` → `resolve_permission` (2.2). ✔
- read the audit log → `GetAudit` + status verify indicator (2.2). ✔
- runtime capability toggling (persisted) → `SetCapability` + `save_capabilities` (2.2). ✔
- automatic un-blocking — explicit user unblock → `Unblock` → `reverse_action` → `unblock_outbound` (1.2–1.4, 2.2); per-block `reverse` added (1.3). ProcessExit-trigger explicitly deferred to v0.2 (Global Constraints + 4.1). ✔ (scoped per the user's decision)
- familiar-ui: per-capability switches (default-off, visible), live prompts wired to resolve, hash-chained audit viewer with verify indicator, status feed → Task 3.2. ✔ (egui, not Tauri, per the user's decision)
- Tests: UI/IPC cannot hold a gate; mirror Plan B fixtures → `no_command_can_install_containment` (2.2), `reverse_action … never installs` (1.4), structural dep graph (3.1); grant/deny/unblock mirror `redteam_network.rs`. ✔
- Repo local-only → Global Constraints. ✔

**Placeholder scan:** the `familiar-ui/src/main.rs` stub in Task 3.1 is replaced wholesale in Task 3.2 (not a residual placeholder). Two version-sensitive call sites (`rustix` peer-cred fn name; `eframe::run_native` closure) carry explicit "confirm/adjust" notes with the fallback named — these are real APIs, not invented.

**Type consistency:** `apply_command`/`status`/`serve_control` all use `Sup<S> = Supervisor<S, LinuxActuators, LinuxNotifier, NullAdvisor>`. `ControlRequest`/`ControlResponse`/`StatusSnapshot`/`PromptDto`/`BlockDto` field names match across `familiar-ipc` (0.1), the daemon (2.2), and the UI (3.1/3.2). `open_requests()` (1.1) returns `&PermissionRequest`, consumed by `prompt_dto` (2.2). `Actuators::reverse` (1.3) ↔ `Supervisor::reverse_action` (1.4) ↔ `ControlRequest::Unblock` (2.2) agree on `&ProposedAction`.
