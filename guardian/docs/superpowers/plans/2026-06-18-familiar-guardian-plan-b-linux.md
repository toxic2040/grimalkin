# Familiar Guardian Plan B — Linux adapter + daemon + real acceptance fixtures

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the deterministic v0.1 spine a real Linux body — a `familiar-linux` adapter that implements the `Sensors`/`Actuators`/`Notifier` seam against the kernel (NFQUEUE outbound sensing, `/proc` PID attribution, a reversible nftables drop rule via netlink, the cgroup-v2 freezer, a privilege-separated fanotify file-read sensor), plus a least-privilege `familiar-daemon` that wires `Engine` + adapter + `NullAdvisor` + `Supervisor` into a tick loop with on-disk persistence, all proven by scripted exfil red-team fixtures that run in a throwaway network namespace.

**Architecture:** Two privilege domains. `familiar-daemon` holds only `CAP_NET_ADMIN` and drives the existing `Supervisor` loop; a minimal `familiar-fanotify-helper` holds `CAP_SYS_ADMIN`, watches the sensitive paths, and streams `FileRead` events to the daemon over a `/run/familiar` Unix socket. The Linux adapter lives in its own crate (`familiar-linux`) so the trait crate (`familiar-platform`) stays dependency-free and the core's "compiles with no adapter" portability seam is preserved. NFQUEUE only *senses* new outbound connections (verdict always ACCEPT); containment is a separate, durable, removable nftables rule installed via netlink — keeping the `Sensors`/`Actuators` boundary clean. Every behaviour the fakes proved in v0.1 is reproduced against the real OS; the autonomous high-confidence path is unlocked only once the file-read sensor lands.

**Tech Stack:** Rust 1.95.0, edition 2024. Confirmed by the Plan B spike (`~/work/sandbox/familiar-plan-b-spike/`, `FINDINGS.md`): `nfq` 0.2.5 (pure-Rust NFQUEUE, libc-only) for outbound sensing; `rustables` 0.8.7 **patched** (one-line fd-ownership fix, vendored — see Task 0.2) for the netlink drop rule; raw `libc` fanotify (`fanotify_init`/`fanotify_mark`/`FAN_REPORT_PIDFD`) in the helper; std `fs` for the cgroup-v2 freezer and `/proc` attribution; `serde`/`serde_json` for persistence and the helper↔daemon wire format; `thiserror` for errors. No async runtime — the daemon is a single-threaded tick loop plus one background NFQUEUE reader thread and one socket reader thread, communicating over `std::sync::mpsc`.

## Global Constraints

These apply to **every** task. Exact values, copied from the v0.1 spec, the v0.1 plan, the spike findings, and the workspace rules:

- **`familiar-core` is never modified by this plan.** Plan B binds to the *realized* core API exactly as built (signatures reproduced in the Interfaces blocks). If a task seems to need a core change, stop and surface it — do not edit `familiar-core`/`familiar-runtime`.
- **`#![forbid(unsafe_code)]` at the top of every crate except `familiar-fanotify-helper`.** `familiar-linux` uses only safe crates (`nfq`, `rustables`, std `fs`/`net`), so it keeps the forbid. The fanotify helper is the *only* place `unsafe` is allowed, and it localizes every `unsafe` block to thin libc wrappers with a safety comment each.
- **Two privilege domains, least privilege each.** `familiar-daemon` runs with `AmbientCapabilities=CAP_NET_ADMIN` only, `NoNewPrivileges=yes`, `ProtectSystem=strict`, and **no network egress for the unit itself**. `familiar-fanotify-helper` runs as a separate unit with `AmbientCapabilities=CAP_SYS_ADMIN` only, read-only filesystem, no network. Neither unit ever holds both caps.
- **Every capability stays default-OFF and fail-closed** (core enforces this; the adapter must never run a sensor/actuator for a disabled capability — it learns enablement only through `Engine`/`Supervisor`, never by reading config directly to bypass a gate).
- **Containment is reversible and removable.** The block actuator installs rules only inside a dedicated `table inet familiar`; reversal is deletion of that table (or the specific rule). The adapter never writes into any other table or chain.
- **NFQUEUE never holds a verdict for a decision.** The sensor thread issues `Verdict::Accept` immediately and emits an event; the Supervisor's decision and any block happen afterward as a separate rule. (Documented limitation: the triggering SYN may pass before the block lands; all subsequent traffic to that destination is dropped. eBPF inline-drop is the v0.2 tightening.)
- **`/proc` socket→PID attribution is racy by nature** (a process that exits between the packet and the scan is unattributable). Record it as a known limitation in code and surface unattributable connections as `ProcessRef{ pid: 0, exe: "" }` — never guess. eBPF socket attribution is the v0.2 upgrade.
- **Incremental, fail-isolated, no silent serial downgrade.** The tick loop wraps each event in its own error boundary so one failure never kills the loop. Persistence writes append-and-flush, never buffer-and-dump-at-exit.
- **No automation fingerprints** in commit messages, code comments, docs, or unit files. Write like a human engineer. No `Co-Authored-By`/AI-provenance trailers (the commit hook blocks them).
- **Repo is local-only.** `repos/familiar` is `private_local`: local commits OK, **no remote, no push** until the user explicitly authorizes. The vendored patched `rustables` is committed in-tree; the upstream one-line PR is filed separately (out of plan).
- **Spike is the ground truth.** Every OS/crate API used here was confirmed by `~/work/sandbox/familiar-plan-b-spike/run_legs.sh`. When a task says "as confirmed by the spike", the working reference is the named binary in that directory.

---

## File structure (Plan B additions; the four v0.1 crates are unchanged)

```
repos/familiar/
├── Cargo.toml                       # MODIFY: add members + workspace deps (nfq, rustables patch, libc)
├── vendor/
│   └── rustables-0.8.7-patched/     # NEW: vendored rustables with the fd double-close fix
├── crates/
│   ├── familiar-core/               # UNCHANGED
│   ├── familiar-advisor/            # UNCHANGED
│   ├── familiar-platform/           # UNCHANGED (stays the pure trait seam)
│   ├── familiar-runtime/            # UNCHANGED
│   ├── familiar-linux/              # NEW — implements Sensors/Actuators/Notifier for Linux
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs               # #![forbid(unsafe_code)]; module decls + re-exports
│   │       ├── nft.rs               # rustables: install/remove the dedicated table, queue rule, drop rules
│   │       ├── nfqueue.rs           # nfq: queue reader thread, IPv4/TCP parse -> (dst,dport,src_port)
│   │       ├── attribution.rs       # /proc/net/tcp{,6} + /proc/<pid>/fd  -> Pid (race documented)
│   │       ├── cgroup.rs            # cgroup-v2 freezer: own-cgroup create, move pid, freeze/thaw
│   │       ├── sensors.rs           # LinuxSensors: NFQUEUE outbound + injected FileRead source -> poll()
│   │       ├── actuators.rs         # LinuxActuators: BlockOutbound (nft) + FreezeProcess (cgroup) + reverse
│   │       ├── notifier.rs          # LinuxNotifier: structured log + best-effort desktop notify
│   │       └── wire.rs              # FileReadEvent JSON wire type shared with the helper
│   ├── familiar-fanotify-helper/    # NEW — the ONLY crate with unsafe; CAP_SYS_ADMIN
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs              # arg-parse watched prefixes, mark them, stream FileReadEvent JSON
│   │       └── fanotify.rs          # localized-unsafe libc fanotify wrapper
│   └── familiar-daemon/             # NEW — CAP_NET_ADMIN; wires everything; tick loop + persistence
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs              # parse config, build Supervisor, run the loop
│           ├── config.rs            # DaemonConfig (sensitive prefixes, established dsts, timeouts, paths)
│           ├── persistence.rs       # load/save CapabilitySnapshot + audit records (JSONL), verify on load
│           ├── filereads.rs         # FileReadSource: socket reader thread (prod) / channel (tests)
│           └── run.rs               # the detect->decide->act/ask->audit->notify tick loop over Supervisor
├── systemd/
│   ├── familiar-daemon.service      # NEW — CAP_NET_ADMIN, hardened, no egress
│   └── familiar-fanotify.service    # NEW — CAP_SYS_ADMIN, read-only, no network
└── tests/  (per-crate; see tasks)   # NFQUEUE/nft/cgroup integration tests run under unshare -Urn
```

**Dependency direction (acyclic, unchanged invariant):** `familiar-linux → familiar-platform + familiar-core + {nfq, rustables, libc, serde, serde_json, thiserror}`. `familiar-daemon → familiar-linux + familiar-runtime + familiar-core + familiar-advisor + {serde, serde_json}`. `familiar-fanotify-helper → {libc, serde, serde_json}` only (it shares the wire type by a tiny duplicated struct, not a dep on `familiar-linux`, to keep the privileged binary's dependency surface minimal). Nothing in core/platform/runtime depends on any Plan B crate.

---

## Phase 0 — Workspace, vendored dependency, adapter scaffold

### Task 0.1: Vendor the patched `rustables` and wire the workspace

**Files:**
- Create: `vendor/rustables-0.8.7-patched/` (copied crate source + one-line fix)
- Create: `vendor/PATCH-NOTES.md`
- Modify: `Cargo.toml` (workspace deps + `[patch.crates-io]`)

**Interfaces:**
- Produces: a workspace where `rustables` resolves to the patched local copy whose `query.rs::socket_close_wrapper` no longer double-closes the netlink fd. Confirmed by the spike (`nft_block.rs` aborted before the patch, succeeded after).

- [ ] **Step 1: Copy the crate source and apply the fix**

```bash
cd /home/toxic2040/work/repos/familiar
mkdir -p vendor
cp -r ~/.cargo/registry/src/*/rustables-0.8.7 vendor/rustables-0.8.7-patched
chmod -R u+w vendor/rustables-0.8.7-patched
```

In `vendor/rustables-0.8.7-patched/src/query.rs`, in `socket_close_wrapper`, delete the manual close so the caller's `OwnedFd` (nix 0.30 `socket()` returns `OwnedFd`) closes the fd exactly once:

```rust
    let ret = cb(sock);

    // The fd is owned by an `OwnedFd` in the caller (nix 0.30 `socket()` returns
    // `OwnedFd`), which closes it on drop. The original manual `close()` here
    // double-closed it, tripping Rust's IO-safety guard (abort on 1.95). Let the
    // owner close it instead.

    Ok(ret?)
```

(Remove the `nix::unistd::close(sock).map_err(QueryError::CloseFailed)?;` line and its comment. Leave everything else untouched.)

- [ ] **Step 2: Record why the patch exists**

`vendor/PATCH-NOTES.md`:

```markdown
# vendored rustables 0.8.7 — fd double-close fix

Upstream 0.8.7 (latest published) requires `nix ^0.30`, whose `socket()` returns
an `OwnedFd`. `src/query.rs::socket_close_wrapper` still manually
`nix::unistd::close()`s the raw fd, so the fd is closed twice (once manually,
once when the `OwnedFd` drops). On Rust 1.95 the std IO-safety guard turns the
second close into `fatal runtime error: IO Safety violation` and aborts.

Fix: remove the manual `close()`; both callers (`batch.rs::send`, the `query.rs`
list functions) keep the `OwnedFd` alive to end-of-scope, so its `Drop` closes
the fd once. Verified by the Plan B spike (`nft_block.rs` add/remove in a netns).

Upstream PR: file the same one-line change against the rustables repo; drop this
vendor copy once a fixed version is published.
```

- [ ] **Step 3: Add workspace deps and the patch redirect**

In `repos/familiar/Cargo.toml`, extend `[workspace.dependencies]`:

```toml
nfq = "0.2.5"
rustables = "0.8.7"
libc = "0.2.186"
```

Add at the end of `repos/familiar/Cargo.toml`:

```toml
[patch.crates-io]
rustables = { path = "vendor/rustables-0.8.7-patched" }
```

- [ ] **Step 4: Verify the patch resolves and the workspace still builds**

Run: `cd /home/toxic2040/work/repos/familiar && cargo build --workspace`
Expected: builds clean; `cargo tree -p rustables` shows the path source `vendor/rustables-0.8.7-patched`.

- [ ] **Step 5: Commit**

```bash
git add vendor Cargo.toml
git commit -m "Vendor rustables 0.8.7 with the netlink fd double-close fix"
```

### Task 0.2: Scaffold `familiar-linux`

**Files:**
- Create: `crates/familiar-linux/Cargo.toml`
- Create: `crates/familiar-linux/src/lib.rs`
- Modify: `Cargo.toml` (add `crates/familiar-linux` to members + `familiar-linux` workspace dep)

**Interfaces:**
- Consumes: `familiar-platform`, `familiar-core`.
- Produces: an empty-but-compiling `familiar-linux` crate with `#![forbid(unsafe_code)]` and the module skeleton later tasks fill in.

- [ ] **Step 1: Add to workspace members and deps**

In `repos/familiar/Cargo.toml`, extend `members` with `"crates/familiar-linux"`, and add to `[workspace.dependencies]`:

```toml
familiar-linux = { path = "crates/familiar-linux" }
```

- [ ] **Step 2: Write the crate manifest**

`crates/familiar-linux/Cargo.toml`:

```toml
[package]
name = "familiar-linux"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[dependencies]
familiar-core.workspace = true
familiar-platform.workspace = true
nfq.workspace = true
rustables.workspace = true
libc.workspace = true
serde = { workspace = true }
serde_json.workspace = true
thiserror.workspace = true

[dev-dependencies]
serde_json.workspace = true
```

(Add `serde_json` to `[workspace.dependencies]` already present from v0.1; it is. `libc` is used only for protocol constants like `IPPROTO_TCP` in `nft.rs` — no `unsafe`.)

- [ ] **Step 3: Write the lib root**

`crates/familiar-linux/src/lib.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-linux — the Linux implementation of the familiar-platform seam.
//!
//! Safe Rust only: NFQUEUE via `nfq`, the reversible block rule via `rustables`
//! (netlink), `/proc` attribution and the cgroup-v2 freezer via std `fs`. The
//! only `unsafe` in the whole workspace lives in the separate, privileged
//! `familiar-fanotify-helper`; this crate never holds CAP_SYS_ADMIN.

pub mod actuators;
pub mod attribution;
pub mod cgroup;
pub mod nft;
pub mod nfqueue;
pub mod notifier;
pub mod sensors;
pub mod wire;

pub use actuators::LinuxActuators;
pub use notifier::LinuxNotifier;
pub use sensors::LinuxSensors;
pub use wire::FileReadEvent;
```

- [ ] **Step 4: Create empty module files so the crate compiles**

Create each of `actuators.rs`, `attribution.rs`, `cgroup.rs`, `nft.rs`, `nfqueue.rs`, `notifier.rs`, `sensors.rs`, `wire.rs` in `crates/familiar-linux/src/` containing only:

```rust
// Filled in by a later task.
```

- [ ] **Step 5: Verify it builds**

Run: `cargo build -p familiar-linux`
Expected: compiles (empty-module warnings acceptable).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/familiar-linux
git commit -m "Scaffold familiar-linux adapter crate"
```

---

## Phase 1 — Actuators: reversible block (netlink) and freeze (cgroup)

### Task 1.1: The dedicated nftables table and the reversible drop rule (`nft.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/nft.rs`
- Create: `crates/familiar-linux/tests/nft_netns.rs`

**Interfaces:**
- Consumes: `rustables::{Batch, Chain, ChainPolicy, Hook, HookClass, MsgType, ProtocolFamily, Rule, Table}`, `rustables::expr::*`, `std::net::Ipv4Addr`.
- Produces:
  - `pub const TABLE: &str = "familiar";`
  - `pub const BLOCK_CHAIN: &str = "egress-block";`
  - `pub fn ensure_table() -> Result<(), NftError>` — create the dedicated table (idempotent add).
  - `pub fn block_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError>` — install a drop rule; returns a human handle string for the audit note.
  - `pub fn delete_table() -> Result<(), NftError>` — remove everything familiar installed (full reversal).
  - `pub enum NftError { Send(String) }` (thiserror).

- [ ] **Step 1: Write the failing integration test (runs in a private netns)**

`crates/familiar-linux/tests/nft_netns.rs`:

```rust
//! Real netlink rule add/remove. Must run inside `unshare -Urn` (the test
//! harness re-execs itself into one). Skips with a clear message otherwise.
use std::net::Ipv4Addr;
use std::process::Command;

fn in_netns() -> bool {
    // CAP_NET_ADMIN over our own netns is what we need; detect by trying to
    // list the ruleset, which fails without it.
    Command::new("nft").args(["list", "ruleset"]).output().map(|o| o.status.success()).unwrap_or(false)
}

#[test]
fn block_rule_is_installed_then_fully_reversed() {
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        // Re-exec this very test binary inside a private user+net namespace.
        let exe = std::env::current_exe().unwrap();
        let status = Command::new("unshare")
            .args(["-Urn"]).arg(&exe).arg("--test-threads=1").arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1")
            .status().expect("unshare");
        assert!(status.success(), "netns child failed");
        return;
    }
    assert!(in_netns(), "expected CAP_NET_ADMIN in the private netns");

    familiar_linux::nft::ensure_table().expect("create table");
    let handle = familiar_linux::nft::block_outbound(Ipv4Addr::new(203, 0, 113, 9), 443).expect("block");
    let after = String::from_utf8(Command::new("nft").args(["list", "ruleset"]).output().unwrap().stdout).unwrap();
    assert!(after.contains("familiar") && after.contains("drop"), "rule present:\n{after}");
    assert!(handle.contains("203.0.113.9"), "handle names the dst");

    familiar_linux::nft::delete_table().expect("reverse");
    let clean = String::from_utf8(Command::new("nft").args(["list", "ruleset"]).output().unwrap().stdout).unwrap();
    assert!(!clean.contains("familiar"), "table gone:\n{clean}");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-linux --test nft_netns`
Expected: FAIL — `familiar_linux::nft::ensure_table` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `crates/familiar-linux/src/nft.rs` with (the `rustables` API confirmed by the spike's `nft_block.rs`):

```rust
use rustables::expr::{
    Cmp, CmpOp, HighLevelPayload, IPv4HeaderField, Immediate, Meta, MetaType,
    NetworkHeaderField, TCPHeaderField, TransportHeaderField, VerdictKind,
};
use rustables::{Batch, Chain, ChainPolicy, Hook, HookClass, MsgType, ProtocolFamily, Rule, Table};
use std::net::Ipv4Addr;

pub const TABLE: &str = "familiar";
pub const BLOCK_CHAIN: &str = "egress-block";

#[derive(Debug, thiserror::Error)]
pub enum NftError {
    #[error("netlink send failed: {0}")]
    Send(String),
}

fn table() -> Table {
    Table::new(ProtocolFamily::Inet).with_name(TABLE)
}

/// Create the dedicated `inet familiar` table and the block chain. Idempotent:
/// re-adding an existing table/chain is accepted by netfilter.
pub fn ensure_table() -> Result<(), NftError> {
    let mut batch = Batch::new();
    let t = table();
    batch.add(&t, MsgType::Add);
    let mut chain = Chain::new(&t).with_name(BLOCK_CHAIN);
    chain.set_hook(Hook::new(HookClass::Out, 0));
    chain.set_policy(ChainPolicy::Accept); // default accept; only our rules drop
    batch.add(&chain, MsgType::Add);
    batch.send().map_err(|e| NftError::Send(e.to_string()))
}

/// Install a reversible DROP for outbound TCP to `dst:dport`. Returns a handle
/// string for the audit/notify trail.
pub fn block_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError> {
    let t = table();
    let chain = Chain::new(&t).with_name(BLOCK_CHAIN);
    let rule = Rule::new(&chain)
        .map_err(|e| NftError::Send(e.to_string()))?
        .with_expr(Meta::new(MetaType::NfProto))
        .with_expr(Cmp::new(CmpOp::Eq, [libc::NFPROTO_IPV4 as u8]))
        .with_expr(HighLevelPayload::Network(NetworkHeaderField::IPv4(IPv4HeaderField::Daddr)).build())
        .with_expr(Cmp::new(CmpOp::Eq, dst.octets().to_vec()))
        .with_expr(Meta::new(MetaType::L4Proto))
        .with_expr(Cmp::new(CmpOp::Eq, [libc::IPPROTO_TCP as u8]))
        .with_expr(HighLevelPayload::Transport(TransportHeaderField::Tcp(TCPHeaderField::Dport)).build())
        .with_expr(Cmp::new(CmpOp::Eq, dport.to_be_bytes().to_vec()))
        .with_expr(Immediate::new_verdict(VerdictKind::Drop));
    let mut batch = Batch::new();
    batch.add(&rule, MsgType::Add);
    batch.send().map_err(|e| NftError::Send(e.to_string()))?;
    Ok(format!("nft drop {dst}:{dport} in table inet {TABLE}"))
}

/// Reverse everything familiar installed by deleting its dedicated table.
pub fn delete_table() -> Result<(), NftError> {
    let mut batch = Batch::new();
    batch.add(&table(), MsgType::Del);
    batch.send().map_err(|e| NftError::Send(e.to_string()))
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-linux --test nft_netns -- --nocapture`
Expected: PASS — installs the drop rule, then deletes the table; ruleset clean.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/nft.rs crates/familiar-linux/tests/nft_netns.rs
git commit -m "Add reversible netlink drop rule in a dedicated familiar table"
```

### Task 1.2: The cgroup-v2 freezer (`cgroup.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/cgroup.rs`
- Create: `crates/familiar-linux/tests/cgroup_freeze.rs`

**Interfaces:**
- Consumes: `std::fs`, `familiar_core::Pid`.
- Produces:
  - `pub struct Freezer { root: PathBuf }`
  - `pub fn new(cgroup_root: impl Into<PathBuf>) -> Freezer` — `cgroup_root` is the daemon-owned delegated cgroup dir under which per-target freeze cgroups are created.
  - `pub fn freeze(&self, pid: Pid) -> Result<String, CgroupError>` — create a child cgroup, move `pid` into it, write `cgroup.freeze=1`; returns a handle (the cgroup path).
  - `pub fn thaw(&self, pid: Pid) -> Result<(), CgroupError>` — write `cgroup.freeze=0` and remove the child cgroup.
  - `pub enum CgroupError { Io(String), NotFrozen }` (thiserror).

- [ ] **Step 1: Write the failing test**

`crates/familiar-linux/tests/cgroup_freeze.rs` (mirrors the spike's `cgroup_freeze.rs`; runs in the caller's own delegated user cgroup, no root):

```rust
use std::process::Command;
use std::time::Duration;

#[test]
fn freeze_then_thaw_a_child() {
    // A sleeper in its own systemd user scope -> a cgroup we own.
    let unit = format!("fam-test-freeze-{}", std::process::id());
    let mut child = Command::new("systemd-run")
        .args(["--user", "--scope", &format!("--unit={unit}"), "sleep", "30"])
        .spawn().expect("systemd-run");
    std::thread::sleep(Duration::from_millis(600));

    // The freezer creates its own child cgroup under the daemon-owned root. For
    // the test, the daemon-owned root is the scope cgroup systemd just gave us.
    let base = "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice";
    let scope = format!("{base}/{unit}.scope");
    let freezer = familiar_linux::cgroup::Freezer::new(&scope);

    let sleep_pid = String::from_utf8(Command::new("pgrep").args(["-n", "-x", "sleep"]).output().unwrap().stdout)
        .unwrap().trim().parse::<u32>().unwrap();

    let handle = freezer.freeze(sleep_pid).expect("freeze");
    assert!(std::path::Path::new(&handle).join("cgroup.events").exists());
    let events = std::fs::read_to_string(format!("{handle}/cgroup.events")).unwrap();
    assert!(events.contains("frozen 1"), "should report frozen 1:\n{events}");

    freezer.thaw(sleep_pid).expect("thaw");
    let _ = child.kill();
    let _ = Command::new("systemctl").args(["--user", "reset-failed", &format!("{unit}.scope")]).status();
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-linux --test cgroup_freeze`
Expected: FAIL — `familiar_linux::cgroup::Freezer` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `crates/familiar-linux/src/cgroup.rs`:

```rust
use familiar_core::Pid;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CgroupError {
    #[error("cgroup io: {0}")]
    Io(String),
    #[error("process did not enter the frozen state")]
    NotFrozen,
}

fn io<E: std::fmt::Display>(e: E) -> CgroupError {
    CgroupError::Io(e.to_string())
}

/// Freezes processes by moving each into a per-target child cgroup under a
/// daemon-owned root and writing `cgroup.freeze`. Reversible via `thaw`.
pub struct Freezer {
    root: PathBuf,
}

impl Freezer {
    pub fn new(cgroup_root: impl Into<PathBuf>) -> Self {
        Self { root: cgroup_root.into() }
    }

    fn cg_for(&self, pid: Pid) -> PathBuf {
        self.root.join(format!("familiar-freeze-{pid}"))
    }

    /// Create a child cgroup, move `pid` into it, freeze it. Returns the cgroup
    /// path as the reversal handle.
    pub fn freeze(&self, pid: Pid) -> Result<String, CgroupError> {
        let cg = self.cg_for(pid);
        fs::create_dir_all(&cg).map_err(io)?;
        // Moving the pid in: write it to cgroup.procs of the child.
        fs::write(cg.join("cgroup.procs"), pid.to_string()).map_err(io)?;
        fs::write(cg.join("cgroup.freeze"), "1").map_err(io)?;
        // Confirm via cgroup.events.
        let events = fs::read_to_string(cg.join("cgroup.events")).map_err(io)?;
        if !events.lines().any(|l| l == "frozen 1") {
            return Err(CgroupError::NotFrozen);
        }
        Ok(cg.to_string_lossy().into_owned())
    }

    /// Thaw and tear down the per-target cgroup. Best-effort rmdir (a cgroup with
    /// no procs can be removed).
    pub fn thaw(&self, pid: Pid) -> Result<(), CgroupError> {
        let cg = self.cg_for(pid);
        fs::write(cg.join("cgroup.freeze"), "0").map_err(io)?;
        // The pid returns to its original cgroup only if we move it back; the
        // daemon does not track the origin in v0.1, so we just thaw and leave the
        // (now-empty after the process exits) cgroup for later rmdir.
        let _ = fs::remove_dir(&cg); // ok to fail if procs still present
        Ok(())
    }
}
```

> Note: `cgroup.subtree_control` on the daemon-owned root must list nothing special — `cgroup.freeze` is a core interface file present on every v2 cgroup, so no controller delegation is needed for freezing (confirmed by the spike). Creating the child cgroup requires write access to the root, which the daemon has for its own delegated subtree (systemd `Delegate=yes`) or, for arbitrary third-party PIDs, real privilege — see Task 3.3's unit notes.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-linux --test cgroup_freeze -- --nocapture`
Expected: PASS — freezes (`frozen 1`), thaws.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/cgroup.rs crates/familiar-linux/tests/cgroup_freeze.rs
git commit -m "Add cgroup-v2 freezer with reversible per-target freeze cgroups"
```

### Task 1.3: `LinuxActuators` implementing the `Actuators` trait (`actuators.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/actuators.rs`

**Interfaces:**
- Consumes: `familiar_platform::{Actuators, ActuationError, ActuationOutcome}`, `familiar_core::policy::ProposedAction`, `crate::{nft, cgroup}`, `std::net::Ipv4Addr`.
- The realized trait (reproduced verbatim — do not change it):
  ```rust
  pub trait Actuators {
      fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError>;
  }
  // ProposedAction::BlockOutbound { process: ProcessRef, dst_ip: String, dst_port: u16 }
  // ProposedAction::FreezeProcess { pid: Pid }
  // ActuationOutcome { note: String }
  // ActuationError::{ Unsupported, Failed(String) }
  ```
- Produces:
  - `pub struct LinuxActuators { freezer: cgroup::Freezer, active_blocks: Vec<(Ipv4Addr, u16)> }`
  - `pub fn new(cgroup_root: impl Into<PathBuf>) -> Result<LinuxActuators, ActuationError>` — calls `nft::ensure_table()` once.
  - inherent `pub fn reverse_all(&mut self) -> Result<(), ActuationError>` — `nft::delete_table()`; clears `active_blocks`.
  - `impl Actuators for LinuxActuators`.

- [ ] **Step 1: Write the failing test (netns; behavioural, not a unit fake)**

Append to `crates/familiar-linux/tests/nft_netns.rs`:

```rust
#[test]
fn actuators_block_outbound_records_and_reverses() {
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        let exe = std::env::current_exe().unwrap();
        let status = std::process::Command::new("unshare")
            .args(["-Urn"]).arg(&exe).arg("actuators_block_outbound_records_and_reverses")
            .arg("--test-threads=1").arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    use familiar_core::events::ProcessRef;
    use familiar_core::policy::ProposedAction;
    use familiar_platform::Actuators;

    let mut act = familiar_linux::LinuxActuators::new("/sys/fs/cgroup").expect("new");
    let action = ProposedAction::BlockOutbound {
        process: ProcessRef { pid: 7, exe: "/usr/bin/curl".into() },
        dst_ip: "203.0.113.9".into(),
        dst_port: 443,
    };
    let outcome = act.apply(&action).expect("apply");
    assert!(outcome.note.contains("203.0.113.9"));
    let after = String::from_utf8(std::process::Command::new("nft").args(["list","ruleset"]).output().unwrap().stdout).unwrap();
    assert!(after.contains("drop"));
    act.reverse_all().expect("reverse");
    let clean = String::from_utf8(std::process::Command::new("nft").args(["list","ruleset"]).output().unwrap().stdout).unwrap();
    assert!(!clean.contains("familiar"));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-linux --test nft_netns actuators_block`
Expected: FAIL — `LinuxActuators` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `crates/familiar-linux/src/actuators.rs`:

```rust
use crate::{cgroup, nft};
use familiar_core::policy::ProposedAction;
use familiar_platform::{ActuationError, ActuationOutcome, Actuators};
use std::net::Ipv4Addr;
use std::path::PathBuf;

/// The Linux actuator: a reversible nft drop rule for BlockOutbound and the
/// cgroup-v2 freezer for FreezeProcess. Tracks active blocks so it can reverse.
pub struct LinuxActuators {
    freezer: cgroup::Freezer,
    active_blocks: Vec<(Ipv4Addr, u16)>,
}

impl LinuxActuators {
    pub fn new(cgroup_root: impl Into<PathBuf>) -> Result<Self, ActuationError> {
        nft::ensure_table().map_err(|e| ActuationError::Failed(e.to_string()))?;
        Ok(Self { freezer: cgroup::Freezer::new(cgroup_root), active_blocks: Vec::new() })
    }

    /// Reverse every block familiar installed (delete its table). Idempotent.
    pub fn reverse_all(&mut self) -> Result<(), ActuationError> {
        nft::delete_table().map_err(|e| ActuationError::Failed(e.to_string()))?;
        self.active_blocks.clear();
        Ok(())
    }
}

impl Actuators for LinuxActuators {
    fn apply(&mut self, action: &ProposedAction) -> Result<ActuationOutcome, ActuationError> {
        match action {
            ProposedAction::BlockOutbound { dst_ip, dst_port, .. } => {
                let ip: Ipv4Addr = dst_ip
                    .parse()
                    .map_err(|_| ActuationError::Failed(format!("non-IPv4 dst {dst_ip} (v0.1 is IPv4-only)")))?;
                let note = nft::block_outbound(ip, *dst_port).map_err(|e| ActuationError::Failed(e.to_string()))?;
                self.active_blocks.push((ip, *dst_port));
                Ok(ActuationOutcome { note })
            }
            ProposedAction::FreezeProcess { pid } => {
                let handle = self.freezer.freeze(*pid).map_err(|e| ActuationError::Failed(e.to_string()))?;
                Ok(ActuationOutcome { note: format!("froze pid {pid} ({handle})") })
            }
        }
    }
}
```

> v0.1 is IPv4-only by deliberate scope (the spike validated IPv4); a non-IPv4 dst fails closed to a recorded `ActuationError`, which the Supervisor degrades to an audited no-action — never a silent pass. IPv6 is a later slice.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-linux --test nft_netns -- --nocapture`
Expected: PASS — block applied, recorded, reversed.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/actuators.rs crates/familiar-linux/tests/nft_netns.rs
git commit -m "Add LinuxActuators: reversible block + freeze behind the Actuators trait"
```

---

## Phase 2 — Outbound sensor: NFQUEUE + `/proc` attribution

### Task 2.1: NFQUEUE reader and packet parse (`nfqueue.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/nfqueue.rs`
- Create: `crates/familiar-linux/tests/nfqueue_netns.rs`

**Interfaces:**
- Consumes: `nfq::{Queue, Verdict}`, `std::net::Ipv4Addr`, `std::sync::mpsc::Sender`.
- Produces:
  - `pub struct OutboundSyn { pub dst: Ipv4Addr, pub dport: u16, pub src_port: u16 }`
  - `pub fn parse_ipv4_tcp_syn(pkt: &[u8]) -> Option<OutboundSyn>` (pure; unit-testable without privilege).
  - `pub fn run_reader(queue_num: u16, tx: Sender<OutboundSyn>) -> std::io::Result<()>` — opens the queue, loops `recv` → parse → `tx.send` → `Verdict::Accept`. Intended to run on a background thread; returns only on error.

- [ ] **Step 1: Write the failing tests**

`crates/familiar-linux/tests/nfqueue_netns.rs` covers the pure parser (no privilege) plus a netns end-to-end capture (mirrors the spike's `conn_events.rs`):

```rust
use familiar_linux::nfqueue::{parse_ipv4_tcp_syn, OutboundSyn};

#[test]
fn parses_dst_and_dport_from_a_syn() {
    // Minimal IPv4(20)+TCP header: ihl=5, proto=6(TCP), daddr=203.0.113.9,
    // dport=443 at TCP offset +2.
    let mut pkt = vec![0u8; 24];
    pkt[0] = 0x45;            // version 4, ihl 5
    pkt[9] = 6;               // TCP
    pkt[16..20].copy_from_slice(&[203, 0, 113, 9]);
    pkt[20..22].copy_from_slice(&34000u16.to_be_bytes()); // src port
    pkt[22..24].copy_from_slice(&443u16.to_be_bytes());   // dst port
    let syn = parse_ipv4_tcp_syn(&pkt).expect("parse");
    assert_eq!(syn, OutboundSyn { dst: "203.0.113.9".parse().unwrap(), dport: 443, src_port: 34000 });
}

#[test]
fn rejects_non_ipv4_and_non_tcp() {
    assert!(parse_ipv4_tcp_syn(&[0x60; 24]).is_none()); // v6
    let mut udp = vec![0u8; 24];
    udp[0] = 0x45; udp[9] = 17; // UDP
    assert!(parse_ipv4_tcp_syn(&udp).is_none());
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p familiar-linux --test nfqueue_netns`
Expected: FAIL — `parse_ipv4_tcp_syn` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `crates/familiar-linux/src/nfqueue.rs` (parser identical to the spike's, with the dport-offset fix the spike caught):

```rust
use nfq::{Queue, Verdict};
use std::net::Ipv4Addr;
use std::sync::mpsc::Sender;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutboundSyn {
    pub dst: Ipv4Addr,
    pub dport: u16,
    pub src_port: u16,
}

/// Parse an IPv4+TCP packet. Returns the dst/dport/src_port if it is TCP.
pub fn parse_ipv4_tcp_syn(pkt: &[u8]) -> Option<OutboundSyn> {
    if pkt.len() < 20 || (pkt[0] >> 4) != 4 {
        return None;
    }
    let ihl = ((pkt[0] & 0x0f) as usize) * 4;
    if pkt[9] != 6 || pkt.len() < ihl + 4 {
        return None; // not TCP, or truncated
    }
    let dst = Ipv4Addr::new(pkt[16], pkt[17], pkt[18], pkt[19]);
    let src_port = u16::from_be_bytes([pkt[ihl], pkt[ihl + 1]]);
    let dport = u16::from_be_bytes([pkt[ihl + 2], pkt[ihl + 3]]); // +2 = dest port
    Some(OutboundSyn { dst, dport, src_port })
}

/// Drive the NFQUEUE: receive, parse, forward, ACCEPT. Sensing only — the
/// verdict is always Accept; containment is a separate nft rule (Task 1.x).
/// Runs until a recv/verdict error; intended for a background thread.
pub fn run_reader(queue_num: u16, tx: Sender<OutboundSyn>) -> std::io::Result<()> {
    let mut queue = Queue::open()?;
    queue.bind(queue_num)?;
    loop {
        let mut msg = queue.recv()?;
        if let Some(syn) = parse_ipv4_tcp_syn(msg.get_payload()) {
            // A full channel (daemon stalled) must not block the verdict; drop
            // the event rather than the packet.
            let _ = tx.send(syn);
        }
        msg.set_verdict(Verdict::Accept);
        queue.verdict(msg)?;
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-linux --test nfqueue_netns`
Expected: PASS (2 parser tests).

- [ ] **Step 5: Add the netns end-to-end capture test**

Append to `crates/familiar-linux/tests/nfqueue_netns.rs` (re-exec into a netns, install the queue rule via `nft`, connect, assert the channel yields the dst — same shape as the spike's `conn_events.rs`):

```rust
#[test]
fn captures_a_real_outbound_syn_in_netns() {
    use std::process::Command;
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        let exe = std::env::current_exe().unwrap();
        let status = Command::new("unshare").args(["-Urn"]).arg(&exe)
            .arg("captures_a_real_outbound_syn_in_netns").arg("--test-threads=1").arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    use std::io::Write;
    let _ = Command::new("ip").args(["link", "set", "lo", "up"]).status();
    let batch = "add table inet q; add chain inet q out { type filter hook output priority 0; policy accept; } \
                 ; add rule inet q out tcp dport 8443 tcp flags syn queue num 0";
    let mut c = Command::new("nft").args(["-f", "-"]).stdin(std::process::Stdio::piped()).spawn().unwrap();
    c.stdin.take().unwrap().write_all(batch.as_bytes()).unwrap();
    assert!(c.wait().unwrap().success());

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { let _ = familiar_linux::nfqueue::run_reader(0, tx); });
    std::thread::sleep(std::time::Duration::from_millis(200)); // let bind happen
    let l = std::net::TcpListener::bind(("127.0.0.1", 8443)).unwrap();
    std::thread::spawn(move || { for _ in l.incoming() { break; } });
    let _ = std::net::TcpStream::connect(("127.0.0.1", 8443));

    let syn = rx.recv_timeout(std::time::Duration::from_secs(3)).expect("captured a syn");
    assert_eq!(syn.dport, 8443);
}
```

Run: `cargo test -p familiar-linux --test nfqueue_netns captures_a_real -- --nocapture`
Expected: PASS — captures dst port 8443.

- [ ] **Step 6: Commit**

```bash
git add crates/familiar-linux/src/nfqueue.rs crates/familiar-linux/tests/nfqueue_netns.rs
git commit -m "Add NFQUEUE outbound-SYN reader with IPv4/TCP parse"
```

### Task 2.2: `/proc` socket → PID attribution (`attribution.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/attribution.rs`
- Create: `crates/familiar-linux/tests/attribution.rs`

**Interfaces:**
- Consumes: `std::fs`, `familiar_core::{Pid, events::ProcessRef}`.
- Produces:
  - `pub fn attribute(src_port: u16) -> Option<ProcessRef>` — map a local source port → socket inode (`/proc/net/tcp{,6}`) → PID (`/proc/<pid>/fd`), and read the exe (`/proc/<pid>/exe`). Returns `None` if unattributable (the documented race).

- [ ] **Step 1: Write the failing test**

`crates/familiar-linux/tests/attribution.rs` (self-attribution against a held loopback socket — same approach as the spike's `proc_attribution.rs`):

```rust
#[test]
fn attributes_our_own_loopback_socket_to_this_pid() {
    use std::io::Read;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let h = std::thread::spawn(move || {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut b = [0u8; 1]; let _ = s.read(&mut b);
    });
    let (_srv, _peer) = listener.accept().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    // The accepted server socket's *local* port is `port`; attribute it.
    let pr = familiar_linux::attribution::attribute(port).expect("attributed");
    assert_eq!(pr.pid, std::process::id());
    drop(_srv);
    let _ = h.join();
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-linux --test attribution`
Expected: FAIL — `attribute` not found.

- [ ] **Step 3: Write the implementation**

Replace the placeholder in `crates/familiar-linux/src/attribution.rs`:

```rust
use familiar_core::Pid;
use familiar_core::events::ProcessRef;
use std::fs;

fn inode_for_local_port(path: &str, port: u16) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 {
            continue;
        }
        let (_, port_hex) = f[1].split_once(':')?;
        if u16::from_str_radix(port_hex, 16).ok()? == port {
            if let Ok(inode) = f[9].parse::<u64>() {
                return Some(inode);
            }
        }
    }
    None
}

fn pid_for_inode(inode: u64) -> Option<Pid> {
    let needle = format!("socket:[{inode}]");
    for entry in fs::read_dir("/proc").ok()?.flatten() {
        let Some(pid) = entry.file_name().to_str().and_then(|s| s.parse::<Pid>().ok()) else { continue };
        let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in fds.flatten() {
            if fs::read_link(fd.path()).map(|t| t.to_string_lossy() == needle).unwrap_or(false) {
                return Some(pid);
            }
        }
    }
    None
}

/// Best-effort attribution of a local source port to the owning process.
/// Returns None when the owner cannot be found (the documented exit race) — the
/// caller must treat that as "unknown process", never guess.
pub fn attribute(src_port: u16) -> Option<ProcessRef> {
    let inode = inode_for_local_port("/proc/net/tcp", src_port)
        .or_else(|| inode_for_local_port("/proc/net/tcp6", src_port))?;
    let pid = pid_for_inode(inode)?;
    let exe = fs::read_link(format!("/proc/{pid}/exe"))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(ProcessRef { pid, exe })
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-linux --test attribution`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/attribution.rs crates/familiar-linux/tests/attribution.rs
git commit -m "Add /proc socket-to-PID attribution (race documented)"
```

### Task 2.3: `LinuxSensors` implementing the `Sensors` trait (`sensors.rs`, `wire.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/sensors.rs`
- Modify: `crates/familiar-linux/src/wire.rs`

**Interfaces:**
- Consumes: `familiar_platform::Sensors`, `familiar_core::events::{Event, ProcessRef}`, `crate::{nfqueue::OutboundSyn, attribution}`, `std::sync::mpsc::Receiver`.
- The realized trait (reproduced — do not change):
  ```rust
  pub trait Sensors { fn poll(&mut self) -> Vec<Event>; }
  ```
- Produces in `wire.rs`:
  - `#[derive(Serialize, Deserialize)] pub struct FileReadEvent { pub at: u64, pub pid: u32, pub exe: String, pub path: String }` — the helper↔daemon wire type.
- Produces in `sensors.rs`:
  - `pub struct LinuxSensors { syn_rx: Receiver<OutboundSyn>, file_rx: Receiver<FileReadEvent>, clock: fn() -> u64 }`
  - `pub fn new(syn_rx: Receiver<OutboundSyn>, file_rx: Receiver<FileReadEvent>) -> LinuxSensors`
  - `impl Sensors for LinuxSensors` — `poll()` drains both channels and maps to `Event::OutboundConn` (attributing the PID) and `Event::FileRead`.

- [ ] **Step 1: Write `wire.rs`**

Replace the placeholder in `crates/familiar-linux/src/wire.rs`:

```rust
use serde::{Deserialize, Serialize};

/// A sensitive-path read observed by the privileged fanotify helper and sent to
/// the daemon. Mirrors `familiar_core::events::Event::FileRead`'s payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileReadEvent {
    pub at: u64,
    pub pid: u32,
    pub exe: String,
    pub path: String,
}
```

- [ ] **Step 2: Write the failing test**

Append to `crates/familiar-linux/src/sensors.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::events::Event;
    use std::sync::mpsc::channel;

    #[test]
    fn poll_maps_file_reads_and_drains_both_channels() {
        let (_syn_tx, syn_rx) = channel();
        let (file_tx, file_rx) = channel();
        file_tx.send(crate::wire::FileReadEvent { at: 1000, pid: 7, exe: "/usr/bin/curl".into(), path: "/home/u/.ssh/id".into() }).unwrap();
        let mut sensors = LinuxSensors::new(syn_rx, file_rx);
        let events = sensors.poll();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::FileRead { at, process, path } => {
                assert_eq!(*at, 1000);
                assert_eq!(process.pid, 7);
                assert_eq!(path, "/home/u/.ssh/id");
            }
            other => panic!("expected FileRead, got {other:?}"),
        }
        // Channel drained: a second poll with nothing queued is empty.
        assert!(sensors.poll().is_empty());
    }
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cargo test -p familiar-linux sensors`
Expected: FAIL — `LinuxSensors` not found.

- [ ] **Step 4: Write the implementation**

At the top of `crates/familiar-linux/src/sensors.rs` (above the test module):

```rust
use crate::attribution;
use crate::nfqueue::OutboundSyn;
use crate::wire::FileReadEvent;
use familiar_core::events::{Event, ProcessRef};
use familiar_platform::Sensors;
use std::sync::mpsc::Receiver;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The Linux event source: NFQUEUE outbound SYNs (attributed via /proc) plus
/// FileRead events streamed from the privileged fanotify helper. Both arrive on
/// channels filled by background threads; `poll()` drains what is queued.
pub struct LinuxSensors {
    syn_rx: Receiver<OutboundSyn>,
    file_rx: Receiver<FileReadEvent>,
    clock: fn() -> u64,
}

impl LinuxSensors {
    pub fn new(syn_rx: Receiver<OutboundSyn>, file_rx: Receiver<FileReadEvent>) -> Self {
        Self { syn_rx, file_rx, clock: now_ms }
    }
}

impl Sensors for LinuxSensors {
    fn poll(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        // FileRead events from the helper (already carry pid/exe/path).
        while let Ok(fr) = self.file_rx.try_recv() {
            events.push(Event::FileRead {
                at: fr.at,
                process: ProcessRef { pid: fr.pid, exe: fr.exe },
                path: fr.path,
            });
        }
        // Outbound SYNs: attribute the PID now (best-effort; unknown => pid 0).
        while let Ok(syn) = self.syn_rx.try_recv() {
            let process = attribution::attribute(syn.src_port)
                .unwrap_or(ProcessRef { pid: 0, exe: String::new() });
            events.push(Event::OutboundConn {
                at: (self.clock)(),
                process,
                dst_ip: syn.dst.to_string(),
                dst_port: syn.dport,
            });
        }
        events
    }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p familiar-linux sensors`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/familiar-linux/src/sensors.rs crates/familiar-linux/src/wire.rs
git commit -m "Add LinuxSensors merging NFQUEUE outbound and helper file-read events"
```

---

## Phase 3 — Notifier, daemon wiring, persistence, systemd

### Task 3.1: `LinuxNotifier` (`notifier.rs`)

**Files:**
- Modify: `crates/familiar-linux/src/notifier.rs`

**Interfaces:**
- Consumes: `familiar_platform::Notifier`, `familiar_core::permission::PermissionRequest`.
- The realized trait (reproduced — do not change):
  ```rust
  pub trait Notifier {
      fn notify(&mut self, message: &str);
      fn request_permission(&mut self, request: &PermissionRequest);
  }
  ```
- Produces:
  - `pub struct LinuxNotifier { desktop: bool }`
  - `pub fn new(desktop: bool) -> LinuxNotifier`
  - `impl Notifier` — writes a structured line to stderr (captured by journald under systemd) and, when `desktop`, best-effort `notify-send` (failure is ignored, never panics).

- [ ] **Step 1: Write the failing test**

Append to `crates/familiar-linux/src/notifier.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::events::ProcessRef;
    use familiar_core::permission::PermissionRequest;
    use familiar_core::policy::{Confidence, Detection, DetectionKind, ProposedAction};

    #[test]
    fn notifier_constructs_and_handles_a_request_without_panicking() {
        let mut n = LinuxNotifier::new(false); // no desktop in tests
        n.notify("contained something");
        let req = PermissionRequest {
            id: 1,
            created_at: 1000,
            timeout_ms: 30_000,
            detection: Detection {
                at: 1000,
                kind: DetectionKind::ExfilSuspected,
                confidence: Confidence(50),
                proposed: ProposedAction::FreezeProcess { pid: 7 },
                rationale: "x".into(),
            },
        };
        n.request_permission(&req); // must not panic
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-linux notifier`
Expected: FAIL — `LinuxNotifier` not found.

- [ ] **Step 3: Write the implementation**

At the top of `crates/familiar-linux/src/notifier.rs`:

```rust
use familiar_core::permission::PermissionRequest;
use familiar_platform::Notifier;
use std::process::Command;

/// Surfaces guardian activity. Writes a structured line to stderr (journald
/// captures it) and, optionally, a best-effort desktop notification. v0.1 has no
/// interactive prompt UI — that is Plan C; here a permission request is logged
/// and surfaced so a human can answer it through the (future) control deck.
pub struct LinuxNotifier {
    desktop: bool,
}

impl LinuxNotifier {
    pub fn new(desktop: bool) -> Self {
        Self { desktop }
    }

    fn desktop_notify(&self, summary: &str, body: &str) {
        if self.desktop {
            // Best-effort; a missing notify-send must never break the daemon.
            let _ = Command::new("notify-send").arg(summary).arg(body).status();
        }
    }
}

impl Notifier for LinuxNotifier {
    fn notify(&mut self, message: &str) {
        eprintln!("[familiar] {message}");
        self.desktop_notify("Familiar", message);
    }

    fn request_permission(&mut self, request: &PermissionRequest) {
        let msg = format!(
            "permission needed (request {}): {} [{:?}] — answer within {} ms",
            request.id, request.detection.rationale, request.detection.proposed, request.timeout_ms
        );
        eprintln!("[familiar] {msg}");
        self.desktop_notify("Familiar — action needs your approval", &request.detection.rationale);
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-linux notifier`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-linux/src/notifier.rs
git commit -m "Add LinuxNotifier: journald-friendly logging + best-effort desktop notify"
```

### Task 3.2: Scaffold `familiar-daemon` with config

**Files:**
- Create: `crates/familiar-daemon/Cargo.toml`
- Create: `crates/familiar-daemon/src/main.rs`
- Create: `crates/familiar-daemon/src/config.rs`
- Modify: `Cargo.toml` (add member)

**Interfaces:**
- Consumes: `familiar-core`, `familiar-linux`, `serde`, `serde_json`.
- Produces:
  - `DaemonConfig { sensitive_prefixes: Vec<String>, established_dsts: Vec<String>, link_window_ms: u64, permission_timeout_ms: u64, queue_num: u16, tick_ms: u64, state_dir: PathBuf, cgroup_root: PathBuf, helper_socket: PathBuf, desktop_notify: bool }` with `Default` and `fn load(path: &Path) -> Result<DaemonConfig, ConfigError>` (TOML or JSON; use JSON for zero extra deps).
  - `ConfigError` (thiserror).

- [ ] **Step 1: Add the member and write the manifest**

Add `"crates/familiar-daemon"` to workspace `members`. `crates/familiar-daemon/Cargo.toml`:

```toml
[package]
name = "familiar-daemon"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[[bin]]
name = "familiar-daemon"
path = "src/main.rs"

[dependencies]
familiar-core.workspace = true
familiar-runtime.workspace = true
familiar-advisor.workspace = true
familiar-platform.workspace = true
familiar-linux.workspace = true
serde = { workspace = true }
serde_json.workspace = true
thiserror.workspace = true
```

(Add `familiar-runtime`, `familiar-advisor` to `[workspace.dependencies]` if not already there. `familiar-runtime`/`familiar-advisor` are path deps; add them.)

- [ ] **Step 2: Write the failing test for config defaults + round-trip**

`crates/familiar-daemon/src/config.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DaemonConfig {
    pub sensitive_prefixes: Vec<String>,
    pub established_dsts: Vec<String>,
    pub link_window_ms: u64,
    pub permission_timeout_ms: u64,
    pub queue_num: u16,
    pub tick_ms: u64,
    pub state_dir: PathBuf,
    pub cgroup_root: PathBuf,
    pub helper_socket: PathBuf,
    pub desktop_notify: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            sensitive_prefixes: vec!["/home".into()], // narrowed by the operator
            established_dsts: Vec::new(),
            link_window_ms: 5_000,
            permission_timeout_ms: 30_000,
            queue_num: 0,
            tick_ms: 200,
            state_dir: PathBuf::from("/var/lib/familiar"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup/familiar.slice"),
            helper_socket: PathBuf::from("/run/familiar/fileread.sock"),
            desktop_notify: false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("read config: {0}")]
    Read(String),
    #[error("parse config: {0}")]
    Parse(String),
}

impl DaemonConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|e| ConfigError::Read(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_json() {
        let c = DaemonConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let back: DaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
```

`crates/familiar-daemon/src/main.rs` (minimal, fills in at Task 3.4):

```rust
#![forbid(unsafe_code)]
mod config;
mod filereads;
mod persistence;
mod run;

fn main() {
    eprintln!("familiar-daemon: see run::main_loop (wired in Task 3.4)");
}
```

(Create empty `filereads.rs`, `persistence.rs`, `run.rs` with `// Filled in by a later task.` so the crate compiles.)

- [ ] **Step 3: Run the test to verify it passes**

Run: `cargo test -p familiar-daemon config`
Expected: PASS (round-trip).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/familiar-daemon
git commit -m "Scaffold familiar-daemon with JSON config"
```

### Task 3.3: Persistence — capability snapshot + audit JSONL, verified on load (`persistence.rs`)

**Files:**
- Modify: `crates/familiar-daemon/src/persistence.rs`

**Interfaces:**
- Consumes: `familiar_core::audit::{AuditLog, AuditRecord}`, `familiar_core::capabilities::{CapabilityRegistry, CapabilitySnapshot}`, `serde_json`.
- The realized core persistence seam (reproduced):
  ```rust
  // AuditLog::from_records(Vec<AuditRecord>) -> AuditLog
  // AuditLog::records(&self) -> &[AuditRecord]
  // AuditLog::verify(&self) -> Result<(), AuditError>
  // CapabilityRegistry::snapshot(&self) -> CapabilitySnapshot   (Serialize/Deserialize)
  // CapabilityRegistry::restore(CapabilitySnapshot) -> CapabilityRegistry
  ```
- Produces:
  - `pub fn save_capabilities(dir: &Path, snap: &CapabilitySnapshot) -> io::Result<()>`
  - `pub fn load_capabilities(dir: &Path) -> CapabilityRegistry` (fail-closed: missing/garbled ⇒ fresh all-off registry).
  - `pub fn append_audit(dir: &Path, rec: &AuditRecord) -> io::Result<()>` (one JSON object per line, flushed).
  - `pub fn load_audit(dir: &Path) -> Result<AuditLog, PersistError>` (reads JSONL, rebuilds via `from_records`, then `verify()`; a broken chain is surfaced, not silently accepted).
  - `pub enum PersistError { Io(String), Corrupt(String), Tampered(String) }`.

- [ ] **Step 1: Write the failing tests**

Append to `crates/familiar-daemon/src/persistence.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use familiar_core::audit::{AuditKind, AuditLog};
    use familiar_core::capabilities::{CapabilityId, CapabilityRegistry};

    #[test]
    fn capabilities_persist_and_reload_fail_closed() {
        let dir = tempdir();
        let mut reg = CapabilityRegistry::new();
        let mut audit = AuditLog::new();
        reg.set(CapabilityId::DetectorExfil, true, 1, &mut audit);
        save_capabilities(&dir, &reg.snapshot()).unwrap();
        let reloaded = load_capabilities(&dir);
        assert!(reloaded.is_enabled(CapabilityId::DetectorExfil));
        assert!(!reloaded.is_enabled(CapabilityId::ActuatorBlockConn));
        // Missing dir => fresh all-off.
        let fresh = load_capabilities(std::path::Path::new("/nonexistent/familiar"));
        for id in CapabilityId::ALL { assert!(!fresh.is_enabled(id)); }
    }

    #[test]
    fn audit_appends_reload_and_verify() {
        let dir = tempdir();
        let mut log = AuditLog::new();
        for r in [
            log.append(1, AuditKind::Detection, "a").clone(),
            log.append(2, AuditKind::Actuation, "b").clone(),
        ] { append_audit(&dir, &r).unwrap(); }
        let reloaded = load_audit(&dir).expect("reload");
        assert_eq!(reloaded.records().len(), 2);
        assert!(reloaded.verify().is_ok());
    }

    #[test]
    fn tampered_audit_line_is_detected_on_load() {
        let dir = tempdir();
        let mut log = AuditLog::new();
        let r = log.append(1, AuditKind::Detection, "real").clone();
        append_audit(&dir, &r).unwrap();
        // Corrupt the stored detail without recomputing the hash.
        let p = dir.join("audit.jsonl");
        let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        v["detail"] = serde_json::Value::String("forged".into());
        std::fs::write(&p, format!("{v}\n")).unwrap();
        assert!(matches!(load_audit(&dir), Err(PersistError::Tampered(_))));
    }

    // tiny tempdir helper (no external crate)
    fn tempdir() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fam-test-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p familiar-daemon persistence`
Expected: FAIL — functions not found.

- [ ] **Step 3: Write the implementation**

At the top of `crates/familiar-daemon/src/persistence.rs`:

```rust
use familiar_core::audit::{AuditLog, AuditRecord};
use familiar_core::capabilities::{CapabilityRegistry, CapabilitySnapshot};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

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
    fs::write(&tmp, serde_json::to_vec_pretty(snap).expect("snapshot serializes"))?;
    fs::rename(tmp, dir.join("capabilities.json")) // atomic replace
}

/// Fail-closed: any problem reading/parsing yields a fresh all-off registry.
pub fn load_capabilities(dir: &Path) -> CapabilityRegistry {
    let path = dir.join("capabilities.json");
    match fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<CapabilitySnapshot>(&t).ok()) {
        Some(snap) => CapabilityRegistry::restore(snap),
        None => CapabilityRegistry::new(),
    }
}

pub fn append_audit(dir: &Path, rec: &AuditRecord) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut f = OpenOptions::new().create(true).append(true).open(dir.join("audit.jsonl"))?;
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
        let rec: AuditRecord = serde_json::from_str(line).map_err(|e| PersistError::Corrupt(e.to_string()))?;
        records.push(rec);
    }
    let log = AuditLog::from_records(records);
    log.verify().map_err(|e| PersistError::Tampered(e.to_string()))?;
    Ok(log)
}
```

> `AuditRecord` must be `Deserialize` for this to compile. The v0.1 core derived only `Serialize` on `AuditRecord`/`AuditKind`. **This is the one place Plan B needs a core touch** — adding `Deserialize` to those two derives is additive and breaks nothing. *Stop and confirm with the user before editing core* (Global Constraint: core is not modified by this plan). If declined, the daemon instead reloads via a daemon-local mirror struct. Default: ask, then add the derive in a separate, clearly-scoped commit to `familiar-core`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p familiar-daemon persistence`
Expected: PASS (3 tests), including tamper detection on load.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-daemon/src/persistence.rs
git commit -m "Add daemon persistence: capability snapshot + verified audit JSONL"
```

### Task 3.4: The tick loop and daemon wiring (`run.rs`, `filereads.rs`, `main.rs`)

**Files:**
- Modify: `crates/familiar-daemon/src/filereads.rs`
- Modify: `crates/familiar-daemon/src/run.rs`
- Modify: `crates/familiar-daemon/src/main.rs`

**Interfaces:**
- Consumes: the realized `Supervisor` API:
  ```rust
  // Supervisor::new(engine, sensors, actuators, notifier, advisor, default_timeout_ms)
  // sup.drive_once(now: Timestamp)
  // sup.resolve_permission(id, granted, now)
  // pub fields: engine, ledger, audit
  // Engine::new(registry, detector); ExfilDetector::new(ExfilConfig{..})
  ```
- Produces in `filereads.rs`:
  - `pub fn spawn_socket_source(socket: &Path) -> (Receiver<FileReadEvent>, JoinHandle<()>)` — bind a Unix listener, accept the helper, read newline-JSON `FileReadEvent`s onto a channel. (Test variant: a plain channel, no socket.)
- Produces in `run.rs`:
  - `pub fn build_supervisor(cfg, syn_rx, file_rx) -> Supervisor<LinuxSensors, LinuxActuators, LinuxNotifier, NullAdvisor>` — loads capabilities, builds the detector from cfg, arms enabled capabilities into the engine.
  - `pub fn main_loop(cfg: DaemonConfig) -> !` — spawn NFQUEUE reader + file-read source threads, tick `drive_once` every `tick_ms`, persist new audit records after each tick.

- [ ] **Step 1: Write the failing integration test (the loop, with a forced event source)**

`crates/familiar-daemon/tests/loop_contains.rs`:

```rust
//! Drives the real Supervisor wiring with injected channels (no OS sensors), to
//! prove the daemon's build_supervisor arms capabilities and the loop contains a
//! high-confidence exfil. Actuation uses the testkit recorder via a trait object
//! is NOT possible here (LinuxActuators is concrete), so this test runs the
//! network path in a netns. It re-execs into unshare -Urn like the adapter tests.
use std::process::Command;

#[test]
fn high_confidence_exfil_is_contained_end_to_end() {
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        let exe = std::env::current_exe().unwrap();
        let status = Command::new("unshare").args(["-Urn"]).arg(&exe)
            .arg("high_confidence_exfil_is_contained_end_to_end").arg("--test-threads=1").arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    use familiar_daemon::config::DaemonConfig;
    use familiar_daemon::run::build_supervisor;
    use familiar_linux::wire::FileReadEvent;
    use familiar_linux::nfqueue::OutboundSyn;
    use std::sync::mpsc::channel;

    let (syn_tx, syn_rx) = channel::<OutboundSyn>();
    let (file_tx, file_rx) = channel::<FileReadEvent>();
    let mut cfg = DaemonConfig::default();
    cfg.sensitive_prefixes = vec!["/home/u/.ssh".into()];
    cfg.state_dir = std::env::temp_dir().join(format!("fam-loop-{}", std::process::id()));
    cfg.cgroup_root = "/sys/fs/cgroup".into();

    let mut sup = build_supervisor(&cfg, syn_rx, file_rx).expect("build");
    // Sensitive read then outbound to an unestablished dst, same pid -> linked, conf 90.
    file_tx.send(FileReadEvent { at: 1000, pid: std::process::id(), exe: "/usr/bin/curl".into(), path: "/home/u/.ssh/id".into() }).unwrap();
    syn_tx.send(OutboundSyn { dst: "203.0.113.9".parse().unwrap(), dport: 443, src_port: 0 }).unwrap();
    // src_port 0 won't attribute, but the FileRead already linked by pid; the
    // OutboundConn just needs the same pid. For determinism the test injects the
    // outbound as a pre-attributed event instead — see build_supervisor's test hook.

    sup.drive_once(2000);
    let ruleset = String::from_utf8(Command::new("nft").args(["list","ruleset"]).output().unwrap().stdout).unwrap();
    assert!(ruleset.contains("drop"), "should have installed a block:\n{ruleset}");
    assert!(sup.audit.verify().is_ok());
}
```

> The attribution-vs-pid coupling above is real: the NFQUEUE path attributes the *outbound* pid, and the exfil detector links by pid. For a deterministic test, `build_supervisor` exposes (behind `#[cfg(test)]`) a way to inject pre-formed `Event`s. Simpler: this test sends the FileRead with `pid = current` and the OutboundConn is produced by `LinuxSensors` from a `syn` whose `src_port` belongs to a socket this process actually holds. To avoid flakiness, **the loop test injects events directly through a test-only `Sensors` shim** rather than the real channels. Adjust Step 3 accordingly: provide `build_supervisor_with_sensors` for tests.

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-daemon --test loop_contains`
Expected: FAIL — `build_supervisor` not found.

- [ ] **Step 3: Write `filereads.rs`**

Replace the placeholder in `crates/familiar-daemon/src/filereads.rs`:

```rust
use familiar_linux::wire::FileReadEvent;
use std::io::{BufRead, BufReader};
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver};
use std::thread::{self, JoinHandle};

/// Bind the helper socket, accept one helper connection, and stream its
/// newline-delimited FileReadEvent JSON onto a channel. The socket is created
/// with the daemon's umask; the systemd unit restricts /run/familiar to the
/// daemon + helper users.
pub fn spawn_socket_source(socket: &Path) -> std::io::Result<(Receiver<FileReadEvent>, JoinHandle<()>)> {
    if socket.exists() {
        let _ = std::fs::remove_file(socket);
    }
    if let Some(parent) = socket.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let listener = UnixListener::bind(socket)?;
    let (tx, rx) = channel();
    let handle = thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let reader = BufReader::new(stream);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(ev) = serde_json::from_str::<FileReadEvent>(&line) {
                    if tx.send(ev).is_err() {
                        return; // daemon gone
                    }
                }
            }
        }
    });
    Ok((rx, handle))
}
```

- [ ] **Step 4: Write `run.rs`**

Replace the placeholder in `crates/familiar-daemon/src/run.rs`:

```rust
use crate::config::DaemonConfig;
use crate::persistence;
use familiar_advisor::NullAdvisor;
use familiar_core::capabilities::CapabilityId;
use familiar_core::policy::{Engine, ExfilConfig, ExfilDetector};
use familiar_linux::nfqueue::{self, OutboundSyn};
use familiar_linux::wire::FileReadEvent;
use familiar_linux::{LinuxActuators, LinuxNotifier, LinuxSensors};
use familiar_runtime::Supervisor;
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("actuators: {0}")]
    Actuators(String),
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Build the Supervisor from config + persisted capability state. Arms the
/// engine's registry with the persisted snapshot (default all-off); the detector
/// is configured from cfg. Capabilities are toggled through the engine so each
/// toggle is audited (here, replayed from the loaded snapshot).
pub fn build_supervisor(
    cfg: &DaemonConfig,
    syn_rx: Receiver<OutboundSyn>,
    file_rx: Receiver<FileReadEvent>,
) -> Result<Supervisor<LinuxSensors, LinuxActuators, LinuxNotifier, NullAdvisor>, BuildError> {
    let registry = persistence::load_capabilities(&cfg.state_dir);
    let detector = ExfilDetector::new(ExfilConfig {
        sensitive_prefixes: cfg.sensitive_prefixes.clone(),
        established_dsts: cfg.established_dsts.clone(),
        link_window_ms: cfg.link_window_ms,
        ..ExfilConfig::default()
    });
    let engine = Engine::new(registry, detector);
    let sensors = LinuxSensors::new(syn_rx, file_rx);
    let actuators = LinuxActuators::new(cfg.cgroup_root.clone())
        .map_err(|e| BuildError::Actuators(e.to_string()))?;
    let notifier = LinuxNotifier::new(cfg.desktop_notify);
    Ok(Supervisor::new(engine, sensors, actuators, notifier, NullAdvisor, cfg.permission_timeout_ms))
}

/// The daemon's run loop. Spawns the NFQUEUE reader and the helper socket
/// source, then ticks the Supervisor and persists any new audit records.
pub fn main_loop(cfg: DaemonConfig) -> ! {
    let (syn_tx, syn_rx) = channel::<OutboundSyn>();
    let queue_num = cfg.queue_num;
    std::thread::spawn(move || {
        if let Err(e) = nfqueue::run_reader(queue_num, syn_tx) {
            eprintln!("[familiar] nfqueue reader stopped: {e}");
        }
    });
    let (file_rx, _file_handle) = match crate::filereads::spawn_socket_source(&cfg.helper_socket) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[familiar] file-read source unavailable ({e}); running network-only");
            let (_tx, rx) = channel();
            (rx, std::thread::spawn(|| {}))
        }
    };

    let mut sup = build_supervisor(&cfg, syn_rx, file_rx).expect("build supervisor");
    let mut persisted = 0usize;
    loop {
        let now = now_ms();
        sup.drive_once(now);
        // Persist any audit records appended this tick (append-and-flush).
        let recs = sup.audit.records();
        for rec in &recs[persisted..] {
            if let Err(e) = persistence::append_audit(&cfg.state_dir, rec) {
                eprintln!("[familiar] audit persist failed: {e}");
            }
        }
        persisted = recs.len();
        std::thread::sleep(Duration::from_millis(cfg.tick_ms));
    }
}
```

Replace `crates/familiar-daemon/src/main.rs`:

```rust
#![forbid(unsafe_code)]
//! familiar-daemon — the least-privilege guardian process (CAP_NET_ADMIN only).
mod config;
mod filereads;
mod persistence;
mod run;

use config::DaemonConfig;
use std::path::PathBuf;

fn main() {
    let cfg_path = std::env::args().nth(1).map(PathBuf::from);
    let cfg = match cfg_path {
        Some(p) => DaemonConfig::load(&p).unwrap_or_else(|e| {
            eprintln!("[familiar] config load failed ({e}); using defaults");
            DaemonConfig::default()
        }),
        None => DaemonConfig::default(),
    };
    run::main_loop(cfg);
}
```

> For the loop test to inject pre-formed events deterministically, also add behind `#[cfg(test)]` in `run.rs` a `build_supervisor_with_sensors(cfg, sensors)` that takes any `Sensors` impl, and have the test pass a tiny shim that returns the scripted `FileRead`+`OutboundConn` (with matching pid) on first poll. Make `config` and `run` modules `pub` in a `lib.rs` (add `crates/familiar-daemon/src/lib.rs` re-exporting `pub mod config; pub mod run; pub mod filereads; pub mod persistence;`) so the integration test can call them, and have `main.rs` use the crate's own modules.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p familiar-daemon --test loop_contains -- --nocapture`
Expected: PASS — a high-confidence exfil installs a drop rule; audit verifies.

- [ ] **Step 6: Commit**

```bash
git add crates/familiar-daemon/src
git commit -m "Wire the daemon tick loop over Supervisor with persistence"
```

### Task 3.5: Least-privilege systemd unit for the daemon

**Files:**
- Create: `systemd/familiar-daemon.service`

**Interfaces:**
- Produces: a hardened unit running `familiar-daemon` with `CAP_NET_ADMIN` only and no self-egress. (Cgroup write for the freezer is granted via `Delegate` on the unit's own slice; freezing third-party PIDs in v0.1 is limited to processes the daemon's cgroup can reach — documented.)

- [ ] **Step 1: Write the unit**

`systemd/familiar-daemon.service`:

```ini
[Unit]
Description=Familiar guardian daemon (network containment)
After=network.target familiar-fanotify.service
Wants=familiar-fanotify.service

[Service]
Type=simple
ExecStart=/usr/local/bin/familiar-daemon /etc/familiar/config.json
# Least privilege: only the cap the block actuator needs.
AmbientCapabilities=CAP_NET_ADMIN
CapabilityBoundingSet=CAP_NET_ADMIN
NoNewPrivileges=yes
# Harden the filesystem; keep state + the helper socket writable.
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/lib/familiar /run/familiar
RuntimeDirectory=familiar
StateDirectory=familiar
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectControlGroups=no
Delegate=yes
RestrictAddressFamilies=AF_UNIX AF_NETLINK
# The daemon must not egress on its own; it only manipulates the firewall.
IPAddressDeny=any
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

> `ProtectControlGroups=no` + `Delegate=yes` give the daemon a writable cgroup subtree for the freezer. `RestrictAddressFamilies=AF_UNIX AF_NETLINK` permits the helper socket and netlink (nftables/NFQUEUE) while denying ordinary sockets; `IPAddressDeny=any` enforces no self-egress. These are the spec's §7 least-privilege requirements made concrete.

- [ ] **Step 2: Validate the unit syntax**

Run: `systemd-analyze verify systemd/familiar-daemon.service` (or `--user` if testing unprivileged; absolute `ExecStart` may warn — acceptable for a not-yet-installed binary).
Expected: no syntax errors (a warning that the binary path does not yet exist is fine).

- [ ] **Step 3: Commit**

```bash
git add systemd/familiar-daemon.service
git commit -m "Add hardened CAP_NET_ADMIN systemd unit for the daemon"
```

---

## Phase 4 — Network-only acceptance: scripted exfil in a namespace

### Task 4.1: Red-team fixture — outbound to an unestablished destination requires permission, grant blocks, deny does not

**Files:**
- Create: `crates/familiar-daemon/tests/redteam_network.rs`

**Interfaces:**
- Consumes: `familiar_daemon::run::build_supervisor`, the realized `Supervisor::{drive_once, resolve_permission}`, `familiar_core::permission::RequestId`.
- This is the §8 acceptance bar for the network-only path (no file sensor yet): the unlinked outbound (confidence 50) routes to `RequirePermission`; an explicit grant installs the reversible block; a denial/timeout installs nothing.

- [ ] **Step 1: Write the failing test**

`crates/familiar-daemon/tests/redteam_network.rs`:

```rust
use std::process::Command;

fn ruleset() -> String {
    String::from_utf8(Command::new("nft").args(["list", "ruleset"]).output().unwrap().stdout).unwrap()
}

#[test]
fn unlinked_outbound_asks_then_grant_blocks_and_deny_does_not() {
    if std::env::var("FAMILIAR_IN_NETNS").is_err() {
        let exe = std::env::current_exe().unwrap();
        let status = Command::new("unshare").args(["-Urn"]).arg(&exe)
            .arg("unlinked_outbound_asks_then_grant_blocks_and_deny_does_not")
            .arg("--test-threads=1").arg("--nocapture")
            .env("FAMILIAR_IN_NETNS", "1").status().unwrap();
        assert!(status.success());
        return;
    }
    use familiar_daemon::config::DaemonConfig;
    use familiar_daemon::run::build_supervisor_with_sensors;
    use familiar_core::events::{Event, ProcessRef};

    // A scripted sensor that yields one unlinked outbound on first poll.
    struct Script(std::cell::Cell<bool>);
    impl familiar_platform::Sensors for Script {
        fn poll(&mut self) -> Vec<Event> {
            if self.0.replace(true) { return vec![]; }
            vec![Event::OutboundConn {
                at: 1000,
                process: ProcessRef { pid: 7, exe: "/usr/bin/curl".into() },
                dst_ip: "203.0.113.9".into(), dst_port: 443,
            }]
        }
    }

    let mut cfg = DaemonConfig::default();
    cfg.cgroup_root = "/sys/fs/cgroup".into();
    cfg.state_dir = std::env::temp_dir().join(format!("fam-rt-{}", std::process::id()));
    // Arm sensor+detector+actuator capabilities for the test via a prepared snapshot.
    prepare_armed_snapshot(&cfg.state_dir);

    let mut sup = build_supervisor_with_sensors(&cfg, Script(std::cell::Cell::new(false))).expect("build");
    sup.drive_once(2000);
    // Unlinked => RequirePermission, no rule yet.
    assert!(!ruleset().contains("drop"), "must ask before blocking");
    let id: familiar_core::permission::RequestId = 1; // first opened request
    assert!(sup.ledger.is_open(id));

    // Grant -> block installed.
    sup.resolve_permission(id, true, 2500);
    assert!(ruleset().contains("drop"), "grant should install the block");
    assert!(sup.audit.verify().is_ok());
}
```

(`prepare_armed_snapshot` writes a `capabilities.json` enabling `SensorOutboundConn`, `DetectorExfil`, `ActuatorBlockConn`; add it as a small helper in the test file using `familiar_core::capabilities::CapabilityRegistry` + `persistence::save_capabilities`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p familiar-daemon --test redteam_network`
Expected: FAIL — `build_supervisor_with_sensors` not found.

- [ ] **Step 3: Add `build_supervisor_with_sensors` to `run.rs`**

```rust
/// Like `build_supervisor` but with a caller-supplied Sensors impl. Used by the
/// red-team fixtures to script exact event sequences without the OS.
pub fn build_supervisor_with_sensors<S: familiar_platform::Sensors>(
    cfg: &DaemonConfig,
    sensors: S,
) -> Result<Supervisor<S, LinuxActuators, LinuxNotifier, NullAdvisor>, BuildError> {
    let registry = persistence::load_capabilities(&cfg.state_dir);
    let detector = ExfilDetector::new(ExfilConfig {
        sensitive_prefixes: cfg.sensitive_prefixes.clone(),
        established_dsts: cfg.established_dsts.clone(),
        link_window_ms: cfg.link_window_ms,
        ..ExfilConfig::default()
    });
    let engine = Engine::new(registry, detector);
    let actuators = LinuxActuators::new(cfg.cgroup_root.clone())
        .map_err(|e| BuildError::Actuators(e.to_string()))?;
    let notifier = LinuxNotifier::new(cfg.desktop_notify);
    Ok(Supervisor::new(engine, sensors, actuators, notifier, NullAdvisor, cfg.permission_timeout_ms))
}
```

Refactor `build_supervisor` to call this with a `LinuxSensors` (DRY).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p familiar-daemon --test redteam_network -- --nocapture`
Expected: PASS — asks first, grant installs the block, chain verifies.

- [ ] **Step 5: Add the deny/timeout case**

Append a second test `denied_outbound_installs_nothing` that opens the request, then `sup.resolve_permission(id, false, 2500)` and asserts `!ruleset().contains("drop")`; and a `timeout_denies` case driving `drive_once` past `created_at + permission_timeout_ms` and asserting no block plus a `PermissionResolved … timed out -> deny` audit record.

Run: `cargo test -p familiar-daemon --test redteam_network`
Expected: PASS (3 cases).

- [ ] **Step 6: Commit**

```bash
git add crates/familiar-daemon/tests/redteam_network.rs crates/familiar-daemon/src/run.rs
git commit -m "Add network-only red-team fixtures: ask, grant-blocks, deny/timeout"
```

---

## Phase 5 — Privilege-separated file-read sensor (fanotify helper)

### Task 5.1: The fanotify helper crate (`familiar-fanotify-helper`)

**Files:**
- Create: `crates/familiar-fanotify-helper/Cargo.toml`
- Create: `crates/familiar-fanotify-helper/src/main.rs`
- Create: `crates/familiar-fanotify-helper/src/fanotify.rs`
- Modify: `Cargo.toml` (add member)

**Interfaces:**
- Consumes: `libc` (fanotify syscalls), `serde`/`serde_json` (wire format).
- Produces: a binary that, given watched path prefixes as args, marks them with fanotify (`FAN_OPEN | FAN_ACCESS`, `FAN_REPORT_PIDFD` for race-free PID) and writes one `FileReadEvent` JSON line per matching event to a Unix socket (connects to the daemon's `helper_socket`). This is the **only crate with `unsafe`**; every `unsafe` block has a safety comment.
- Wire type duplicated locally (no dep on `familiar-linux`):
  ```rust
  #[derive(Serialize)] struct FileReadEvent { at: u64, pid: u32, exe: String, path: String }
  ```

- [ ] **Step 1: Add the member and manifest**

Add `"crates/familiar-fanotify-helper"` to workspace members. `crates/familiar-fanotify-helper/Cargo.toml`:

```toml
[package]
name = "familiar-fanotify-helper"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish = false

[[bin]]
name = "familiar-fanotify-helper"
path = "src/main.rs"

[dependencies]
libc.workspace = true
serde = { workspace = true }
serde_json.workspace = true
```

- [ ] **Step 2: Write the pure-logic failing test (path-prefix filter)**

The fanotify syscalls cannot be unit-tested without `CAP_SYS_ADMIN` (the spike confirmed `fanotify_init` returns `EPERM` even in `unshare -Ur`). So unit-test only the pure decision (does an observed path match a watched prefix and resolve to a clean event). Append to `crates/familiar-fanotify-helper/src/main.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_watched_prefix() {
        let prefixes = vec!["/home/u/.ssh".to_string(), "/etc/shadow".to_string()];
        assert!(is_watched("/home/u/.ssh/id_ed25519", &prefixes));
        assert!(!is_watched("/home/u/Documents/notes.txt", &prefixes));
    }
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `cargo test -p familiar-fanotify-helper`
Expected: FAIL — `is_watched` not found.

- [ ] **Step 4: Write `fanotify.rs` (localized unsafe)**

`crates/familiar-fanotify-helper/src/fanotify.rs` — a thin safe wrapper over the three libc calls the spike grounded (`fanotify_init`, `fanotify_mark`, `read`), each `unsafe` block documented:

```rust
//! Localized-unsafe fanotify wrapper. CAP_SYS_ADMIN required (the spike showed
//! fanotify_init returns EPERM otherwise, even inside unshare -Ur).
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

/// Initialize a fanotify group reporting the accessing PID (FAN_REPORT_PIDFD) and
/// classed for notification only (FAN_CLASS_NOTIF).
pub fn init() -> io::Result<OwnedFd> {
    // SAFETY: fanotify_init takes two scalar flags and returns a new fd or -1.
    // No memory is shared; we wrap the returned fd in OwnedFd for RAII.
    let fd = unsafe {
        libc::fanotify_init(
            libc::FAN_CLASS_NOTIF | libc::FAN_REPORT_PIDFD | libc::FAN_CLOEXEC,
            (libc::O_RDONLY | libc::O_CLOEXEC) as u32,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is a fresh, valid, owned fd returned above.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Add a mark for open+access events on `path` (and everything under it).
pub fn mark(group: &OwnedFd, path: &str) -> io::Result<()> {
    let c = std::ffi::CString::new(path).map_err(|_| io::Error::other("nul in path"))?;
    // SAFETY: group.as_raw_fd() is valid for the call; c is a valid NUL-terminated
    // C string that outlives the call; the mask/flags are plain scalars.
    let rc = unsafe {
        libc::fanotify_mark(
            group.as_raw_fd(),
            libc::FAN_MARK_ADD,
            (libc::FAN_OPEN | libc::FAN_ACCESS) as u64,
            libc::AT_FDCWD,
            c.as_ptr(),
        )
    };
    if rc < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// One decoded event: the accessing PID and the fd referring to the accessed
/// file (resolve the path via /proc/self/fd/<fd>).
pub struct RawEvent {
    pub pid: i32,
    pub fd: RawFd,
}

/// Read and decode the next batch of events. Returns owned event fds; the caller
/// must close each (wrap in OwnedFd). Blocks until at least one event arrives.
pub fn read_events(group: &OwnedFd) -> io::Result<Vec<RawEvent>> {
    let mut buf = [0u8; 4096];
    // SAFETY: read into a valid local buffer; rc bytes are initialized on success.
    let rc = unsafe { libc::read(group.as_raw_fd(), buf.as_mut_ptr() as *mut _, buf.len()) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut out = Vec::new();
    let mut off = 0usize;
    let meta_len = std::mem::size_of::<libc::fanotify_event_metadata>();
    while off + meta_len <= rc as usize {
        // SAFETY: off..off+meta_len is within the rc bytes the kernel wrote, and
        // fanotify guarantees event_len-aligned, fully-populated metadata records.
        let meta = unsafe { &*(buf.as_ptr().add(off) as *const libc::fanotify_event_metadata) };
        if meta.event_len as usize == 0 {
            break;
        }
        out.push(RawEvent { pid: meta.pid, fd: meta.fd });
        off += meta.event_len as usize;
    }
    Ok(out)
}
```

> `FAN_REPORT_PIDFD` makes `meta.pid` the accessing process at event time (race-free, unlike `/proc` socket attribution). v0.1 uses the plain `pid` field via `FAN_CLASS_NOTIF`; the pidfd refinement (to survive PID reuse during the window) is a hardening note for v0.2.

- [ ] **Step 5: Write `main.rs`**

`crates/familiar-fanotify-helper/src/main.rs`:

```rust
//! familiar-fanotify-helper — the privileged file-read sensor. Holds
//! CAP_SYS_ADMIN; does nothing but watch the configured prefixes and stream
//! FileRead events to the daemon. Minimal by design: the broad capability is
//! isolated here, away from the network daemon.
mod fanotify;

use serde::Serialize;
use std::io::Write;
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize)]
struct FileReadEvent {
    at: u64,
    pid: u32,
    exe: String,
    path: String,
}

fn is_watched(path: &str, prefixes: &[String]) -> bool {
    prefixes.iter().any(|p| path.starts_with(p.as_str()))
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn exe_of(pid: i32) -> String {
    std::fs::read_link(format!("/proc/{pid}/exe")).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let socket = args.next().expect("usage: familiar-fanotify-helper <socket> <prefix>...");
    let prefixes: Vec<String> = args.collect();
    assert!(!prefixes.is_empty(), "at least one watched prefix required");

    let group = fanotify::init().expect("fanotify_init (needs CAP_SYS_ADMIN)");
    for p in &prefixes {
        if let Err(e) = fanotify::mark(&group, p) {
            eprintln!("[helper] mark {p} failed: {e}");
        }
    }

    let mut out = UnixStream::connect(&socket).expect("connect to daemon socket");
    loop {
        let events = match fanotify::read_events(&group) {
            Ok(e) => e,
            Err(e) => { eprintln!("[helper] read error: {e}"); continue; }
        };
        for ev in events {
            // Resolve the accessed path via the event fd, then close it.
            let path = std::fs::read_link(format!("/proc/self/fd/{}", ev.fd)).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
            // SAFETY-equivalent: wrap the kernel-provided fd so it is closed on drop.
            let _owned = unsafe { OwnedFd::from_raw_fd(ev.fd) };
            if ev.pid > 0 && is_watched(&path, &prefixes) {
                let fr = FileReadEvent { at: now_ms(), pid: ev.pid as u32, exe: exe_of(ev.pid), path };
                let line = serde_json::to_string(&fr).expect("serialize");
                if writeln!(out, "{line}").is_err() {
                    eprintln!("[helper] daemon socket closed; exiting");
                    return;
                }
            }
        }
    }
}
```

> `main.rs` has one `unsafe` (`OwnedFd::from_raw_fd` on the event fd) plus the wrappers in `fanotify.rs`. That is the entire `unsafe` surface of the workspace.

- [ ] **Step 6: Run the unit test to verify it passes**

Run: `cargo test -p familiar-fanotify-helper`
Expected: PASS (`matches_watched_prefix`). The syscall path is exercised by the privileged acceptance test in Task 5.3.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/familiar-fanotify-helper
git commit -m "Add privilege-separated fanotify file-read helper"
```

### Task 5.2: Helper systemd unit + daemon socket wiring note

**Files:**
- Create: `systemd/familiar-fanotify.service`

**Interfaces:**
- Produces: a minimal `CAP_SYS_ADMIN` unit that runs the helper against the configured socket + prefixes. The daemon already binds the socket (`filereads::spawn_socket_source`) and reads events; the helper connects as a client.

- [ ] **Step 1: Write the unit**

`systemd/familiar-fanotify.service`:

```ini
[Unit]
Description=Familiar file-read sensor (privileged, minimal)
Before=familiar-daemon.service

[Service]
Type=simple
ExecStart=/usr/local/bin/familiar-fanotify-helper /run/familiar/fileread.sock /home /etc/shadow
# The one broad cap, isolated to this tiny binary.
AmbientCapabilities=CAP_SYS_ADMIN
CapabilityBoundingSet=CAP_SYS_ADMIN
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadOnlyPaths=/
ReadWritePaths=/run/familiar
RestrictAddressFamilies=AF_UNIX
IPAddressDeny=any
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

> `RestrictAddressFamilies=AF_UNIX` + `IPAddressDeny=any`: the privileged helper can reach only the local daemon socket, never the network. The watched prefixes are unit args (operator-narrowed; `/home` is illustrative — production should scope to the actual sensitive paths to limit event volume).

- [ ] **Step 2: Validate**

Run: `systemd-analyze verify systemd/familiar-fanotify.service`
Expected: no syntax errors (missing-binary warning acceptable).

- [ ] **Step 3: Commit**

```bash
git add systemd/familiar-fanotify.service
git commit -m "Add minimal CAP_SYS_ADMIN systemd unit for the file-read helper"
```

### Task 5.3: Full autonomous-path acceptance (privileged; gated)

**Files:**
- Create: `crates/familiar-daemon/tests/redteam_autonomous.rs`
- Create: `scripts/run-privileged-acceptance.sh`

**Interfaces:**
- This is the headline §8 bar: a scripted process reads a sensitive file then connects out; the helper + daemon link them (confidence 90) and **autonomously** install the reversible block, recording the full `Detection → Decision → Actuation` audit sequence. It needs real `CAP_SYS_ADMIN` (helper) + `CAP_NET_ADMIN` (daemon), so it cannot run in the unprivileged netns harness; it is gated behind an env flag and run by the script (as root, or with `setcap` on the helper test binary).

- [ ] **Step 1: Write the gated acceptance test**

`crates/familiar-daemon/tests/redteam_autonomous.rs`:

```rust
//! Privileged acceptance. Skipped unless FAMILIAR_PRIVILEGED_ACCEPTANCE=1 (set by
//! scripts/run-privileged-acceptance.sh, which provides CAP_SYS_ADMIN+CAP_NET_ADMIN
//! and a throwaway netns). Proves the autonomous linked-exfil containment path.
#[test]
fn linked_exfil_is_contained_autonomously_and_recorded() {
    if std::env::var("FAMILIAR_PRIVILEGED_ACCEPTANCE").is_err() {
        eprintln!("skipping privileged acceptance (set FAMILIAR_PRIVILEGED_ACCEPTANCE=1 via the script)");
        return;
    }
    // 1. Start the helper against a temp socket, watching a temp sensitive dir.
    // 2. Start the daemon loop wired to NFQUEUE + the helper socket.
    // 3. In the netns: a child reads <sensitive>/secret then connects out to an
    //    unestablished dst.
    // 4. Assert: a drop rule appears (autonomous block), and the persisted audit
    //    JSONL contains Detection -> Decision(implicit) -> Actuation, verify() ok.
    // (Full body uses std::process to launch the binaries the script built, polls
    //  `nft list ruleset`, and reads <state_dir>/audit.jsonl.)
    assert!(privileged_acceptance_body());
}

fn privileged_acceptance_body() -> bool {
    // Implemented against the built binaries; see scripts/run-privileged-acceptance.sh
    // for the environment it assumes.
    true // replaced by the real assertions during implementation
}
```

- [ ] **Step 2: Write the runner script**

`scripts/run-privileged-acceptance.sh`:

```bash
#!/usr/bin/env bash
# Run the privileged acceptance test. Requires sudo (CAP_SYS_ADMIN for the
# helper, CAP_NET_ADMIN for the daemon). Uses a throwaway netns for the network
# side so the host firewall is untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -p familiar-daemon -p familiar-fanotify-helper
echo "This test needs root (CAP_SYS_ADMIN + CAP_NET_ADMIN). Re-run under sudo:"
echo "  sudo FAMILIAR_PRIVILEGED_ACCEPTANCE=1 cargo test -p familiar-daemon --test redteam_autonomous -- --nocapture"
```

- [ ] **Step 3: Implement the test body**

Flesh out `privileged_acceptance_body()` to: create a temp `state_dir` + `sensitive_dir` + socket; spawn the helper (`target/debug/familiar-fanotify-helper <socket> <sensitive_dir>`); spawn the daemon with a config pointing at the socket, the sensitive prefix, NFQUEUE `queue_num`, and `cgroup_root`; in a child, `unshare -Urn`-style read the secret then connect out; poll `nft list ruleset` for up to 5s for `drop`; read `<state_dir>/audit.jsonl`, parse with `serde_json`, assert the kinds include `Detection` and `Actuation` in order and that an `AuditLog::from_records(...).verify()` is `Ok`. Replace the `true` stub with these assertions.

- [ ] **Step 4: Run it (privileged)**

Run: `chmod +x scripts/run-privileged-acceptance.sh && ./scripts/run-privileged-acceptance.sh` then the printed `sudo` command.
Expected: PASS — autonomous block installed; audit chain shows Detection→Actuation and verifies. (Unprivileged `cargo test` of this file SKIPS with the message, keeping the default suite green without privilege.)

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-daemon/tests/redteam_autonomous.rs scripts/run-privileged-acceptance.sh
git commit -m "Add privileged autonomous-exfil acceptance fixture (gated)"
```

---

## Phase 6 — Workspace green + docs

### Task 6.1: Full build, clippy, fmt, and a README for the Linux body

**Files:**
- Create: `crates/familiar-linux/README.md`
- Create: `docs/operating-familiar-linux.md`

- [ ] **Step 1: Whole-workspace verification**

Run, expecting all green:
```bash
cargo build --workspace
cargo test --workspace                    # unprivileged: netns tests self-exec into unshare; privileged acceptance SKIPS
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: builds; all non-privileged tests pass; clippy clean; fmt clean. If a netns test cannot create a user namespace in the current environment, it must SKIP with a clear message, not fail.

- [ ] **Step 2: Write the operator doc**

`docs/operating-familiar-linux.md` — how the two units fit together, the capability model (default-off; how to enable a capability once the Plan C control deck exists; until then, the persisted `capabilities.json`), the install steps (`setcap`/units), and the documented limitations (NFQUEUE SYN-pass race, `/proc` attribution race, IPv4-only, freeze scope). Pull the limitation wording from `~/work/sandbox/familiar-plan-b-spike/FINDINGS.md`.

- [ ] **Step 3: Write the crate README**

`crates/familiar-linux/README.md` — one paragraph per module, the dependency rationale (why a separate crate, why the patched rustables), and the "only unsafe lives in the helper" invariant.

- [ ] **Step 4: Commit**

```bash
git add crates/familiar-linux/README.md docs/operating-familiar-linux.md
git commit -m "Document the Linux body and verify the workspace is green"
```

---

## Self-Review

**Spec coverage (against the v0.1 plan's Plan B description, lines 2643–2647):**
- "prototype the chosen egress mechanism … confirm exact crate APIs" → done by the spike; this plan binds to those confirmed APIs (Phase 0–2 reference the spike binaries).
- "familiar-platform/src/linux/: implement Sensors/Actuators/Notifier" → realized as the dedicated `familiar-linux` crate (Phases 1–3), a deliberate refinement to keep the trait crate dependency-free; the trait set implemented matches exactly.
- "localize all unsafe/FFI here, behind the same traits the fake adapter satisfies" → all `unsafe` is in `familiar-fanotify-helper` (Phase 5); `familiar-linux` keeps `#![forbid(unsafe_code)]`; both satisfy the same realized traits the testkit fakes do.
- "familiar-daemon: wire Engine + linux adapter + NullAdvisor + Supervisor; tick loop; least-privilege systemd unit; persist capability snapshots and the audit log" → Phase 3 (Tasks 3.2–3.5).
- "real red-team fixtures (§8 acceptance bar): detect, reversibly contain (rule added then removed), record — the three scenarios" → Phase 4 (network-only: ask/grant/deny/timeout) + Phase 5 (autonomous linked path). Freeze is exercised via the actuator tests (Task 1.2/1.3); a freeze-path red-team fixture can be added symmetrically if the user wants all three actuators in one acceptance file.

**Placeholder scan:** every code step contains complete code. The two stubs that are intentionally deferred (`privileged_acceptance_body` and the loop-test sensor shim) have explicit follow-up steps that flesh them out (Task 5.3 Step 3; Task 3.4 Step 4 note + Task 4.1's `build_supervisor_with_sensors`). No "TODO/handle errors/etc."

**Type consistency:** signatures bind to the realized core/runtime/platform API verified by reading the source: `Supervisor::new(engine, sensors, actuators, notifier, advisor, default_timeout_ms)`, `drive_once(now)`, `resolve_permission(id, granted, now)`, public `engine/ledger/audit`; `Engine::new(registry, detector)`; `ExfilConfig{ sensitive_prefixes, established_dsts, link_window_ms, linked_confidence, unlinked_confidence }`; `Actuators::apply -> Result<ActuationOutcome, ActuationError>`; `Sensors::poll -> Vec<Event>`; `Notifier::{notify, request_permission}`; `AuditLog::{from_records, records, verify, append}`; `CapabilityRegistry::{snapshot, restore, is_enabled, set}`; `PermissionLedger::is_open`. The one additive core change (derive `Deserialize` on `AuditRecord`/`AuditKind`) is flagged in Task 3.3 to confirm before editing.

**Known deviations from the v0.1 sketch (all deliberate, all flagged):** separate `familiar-linux` crate instead of a `linux/` module in `familiar-platform`; NFQUEUE sense-only with a separate block rule; rustables via a vendored patch; the file sensor as a privilege-separated helper (per the user's decision). Each is justified at its first appearance.

---

## Execution handoff

Plan complete. It builds the Linux body in six phases — vendored dependency + adapter scaffold, the two actuators (netlink block, cgroup freeze) proven in a netns, the NFQUEUE+`/proc` outbound sensor, the least-privilege daemon with verified persistence and a hardened systemd unit, network-only red-team fixtures, and finally the privilege-separated fanotify helper that unlocks the autonomous linked-exfil path — every OS/crate API already confirmed by the Plan B spike.

**Open item to resolve before/at Task 3.3:** confirm the one additive `familiar-core` change (derive `Deserialize` on `AuditRecord` + `AuditKind`) so the daemon can reload the audit log; it is the only core touch and is purely additive.

Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task with two-stage review between tasks. Clean context per task; fast iteration. Note: tasks in Phases 1–2, 4, and 5.3 run real OS integration tests (`unshare -Urn` / privileged), so those subagents need a Linux shell with unprivileged-userns enabled (this box has it).

2. **Inline Execution** — execute tasks in this session with checkpoints for review (matches how the v0.1 spine was built).

Which approach?
