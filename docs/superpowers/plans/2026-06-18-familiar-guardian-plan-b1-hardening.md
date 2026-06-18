# Familiar Guardian Plan B.1 — Red-team hardening (spine + Linux body)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the confirmed red-team findings that are independent of the control deck, so Plan C lands on a sound base: make a disabled actuator truly unable to fire (even via a human grant), reload-and-verify the persisted audit chain on startup, never lose an audit record after the action it describes, stop `reverse_all` from blinding the sensor, and stop the fanotify helper from wrapping an invalid fd on queue overflow.

**Architecture:** Six surgical fixes across the existing crates — no new crates, no new dependencies. Two are core/runtime semantics (F1), two are daemon durability/integrity (F2, F3), one is the Linux actuator (F9), one is the privileged helper (F7). Each is its own TDD task with an independently testable deliverable. The two findings the control deck is the natural home for — silent sensor blindness (F4) and the unauthenticated fileread socket (F8) — are deferred to Plan C, which is being revised to fold them in. Two are honest scope cuts documented for v0.2 (F5 IPv6 sensing, F6 inline-drop of the SYN).

**Tech Stack:** Rust 1.95.0, edition 2024. Unchanged toolchain; no new deps. The fanotify helper stays the only `unsafe` crate.

## Global Constraints

- **Findings, verified.** Each fix targets a finding Claude verified against the code on 2026-06-18 (register: `~/.claude/.../memory/familiar-redteam-register.md`). Line numbers in tasks are from that review.
- **Fail-closed is the tie-breaker.** Where a fix has a "loud-but-keep-running" vs. "silently-degrade" choice, choose loud: record an `AuditKind::IntegrityAlert`/`NoAction`, never a silent pass and never a silent block.
- **`#![forbid(unsafe_code)]` stays on every crate except `familiar-fanotify-helper`.** F7's fix removes an unsafe-precondition violation; it does not add `unsafe`.
- **Surgical diffs.** Touch only the lines each finding requires. Do not restructure neighbouring code. Match existing style.
- **No automation fingerprints** in code, comments, commit messages, or docs. No AI-provenance trailers (the hook blocks them).
- **Repo is local-only.** Local commits OK; **no remote, no push** until the user authorizes.
- **Order:** H1→H2 (F1, together a behavior change + its tests), then H3, H4, H5, H6 are independent and may be done in any order. Run `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check` green before the final commit of each.

---

## File structure (all modifications; no new files except a rotation helper test)

```
crates/familiar-core/src/policy.rs           # H1: disabled actuator => Deny (+ test update)
crates/familiar-runtime/src/lib.rs           # H2: resolve_permission re-checks the actuator cap (+ test)
crates/familiar-runtime/tests/security_invariants.rs  # H1: update the disabled-actuator test
crates/familiar-daemon/src/persistence.rs    # H3: restore_audit + rotate_corrupt_audit (+ tests)
crates/familiar-daemon/src/run.rs            # H3+H4: reload/verify audit on start; durable persist loop
crates/familiar-linux/src/nft.rs             # H5: run_nft helper + flush_block_chain
crates/familiar-linux/src/actuators.rs       # H5: reverse_all flushes the block chain, keeps sensing
crates/familiar-linux/tests/nft_netns.rs     # H5: reverse_all-keeps-sensing netns test
crates/familiar-fanotify-helper/src/fanotify.rs  # H7: should_forward guard (+ test)
crates/familiar-fanotify-helper/src/main.rs  # H7: skip fd<0 overflow events before wrapping
```

---

## Task H1: F1 — a disabled actuator denies the action (no autonomous fire, no prompt)

**Files:**
- Modify: `crates/familiar-core/src/policy.rs` (the actuator-gate branch in `Engine::intake`, + the `disabled_actuator_*` test)
- Modify: `crates/familiar-runtime/tests/security_invariants.rs` (`high_confidence_with_disabled_actuator_*`)

**Interfaces:**
- Changes the disposition for any detection whose proposed action's actuator capability is disabled: from `RequirePermission` (current) to `Disposition::Deny`. No signature change. Downstream `dispatch` already turns `Deny` into a recorded `NoAction` with no prompt.

- [ ] **Step 1: Update the core test to the fail-closed expectation (failing)**

In `crates/familiar-core/src/policy.rs`, replace the body of `disabled_actuator_downgrades_autonomy_to_an_ask` (rename it) in `mod engine_tests`:

```rust
    #[test]
    fn disabled_actuator_denies_the_action() {
        // Fail-closed: ActuatorBlockConn is OFF, so the block cannot fire — not
        // autonomously and not via a later human grant. The engine denies.
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
        let decision = engine
            .intake(&out(1500, 7), &AbstainAdvisor, &mut audit)
            .expect("fires");
        assert_eq!(decision.disposition, Disposition::Deny);
    }
```

Run: `cargo test -p familiar-core disabled_actuator_denies_the_action -v`
Expected: FAIL — current code yields `RequirePermission`.

- [ ] **Step 2: Make the engine deny on a disabled actuator**

In `crates/familiar-core/src/policy.rs`, in `Engine::intake`, replace the actuator-gate block (currently the `if disposition == Disposition::ActAutonomously && !…is_enabled(...)` downgrade):

```rust
        // Authority envelope (rule-only).
        let mut disposition = classify(&detection);
        // Actuator capability gate, fail-closed: if the actuator this action
        // needs is disabled, the action can never be carried out — not
        // autonomously and not via a later human grant. Deny outright (the
        // toggle physically cuts the ability; it is not a "downgrade to ask").
        if !self
            .registry
            .is_enabled(detection.proposed.actuator_capability())
        {
            disposition = Disposition::Deny;
        }
        // Advisor (heighten-only; can never open a gate — and cannot relax Deny).
        let advice = advisor.assess(&detection);
        disposition = apply_caution(disposition, advice.caution);
```

Run: `cargo test -p familiar-core disabled_actuator_denies_the_action -v`
Expected: PASS.

- [ ] **Step 3: Update the runtime integration test (failing → passing)**

In `crates/familiar-runtime/tests/security_invariants.rs`, replace `high_confidence_with_disabled_actuator_asks_instead_of_acting`:

```rust
/// §7 fail-closed: a high-confidence detection with the actuator capability
/// disabled neither acts nor asks — a disabled actuator cannot fire at all.
#[test]
fn high_confidence_with_disabled_actuator_denies_with_no_prompt() {
    let sensors = FakeSensors::new(vec![vec![read(1000, 7), out(1500, 7, "203.0.113.9")]]);
    let mut sup = supervisor(
        engine_with(Caps { actuator_block: false, ..Default::default() }),
        sensors,
        RecordingActuators::default(),
    );
    sup.drive_once(2000);
    assert!(sup.actuators().applied.is_empty(), "a disabled actuator must not act");
    assert!(sup.notifier().requests.is_empty(), "and must not raise a prompt");
}
```

Run: `cargo test -p familiar-runtime --test security_invariants -v`
Expected: PASS (all, including the rewritten test).

- [ ] **Step 4: Workspace check**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean. (Confirm no other test depended on the old downgrade-to-ask behavior — grep `disabled_actuator` if a failure surfaces.)

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-core/src/policy.rs crates/familiar-runtime/tests/security_invariants.rs
git commit -m "Fail-closed: a disabled actuator denies the action instead of asking"
```

---

## Task H2: F1 — `resolve_permission` re-checks the actuator capability at grant time

**Files:**
- Modify: `crates/familiar-runtime/src/lib.rs` (`resolve_permission`)

**Interfaces:**
- `resolve_permission` now acts only if the action's actuator capability is still enabled at grant time. No signature change. Guards the race H1 cannot: actuator ON when the prompt was raised, toggled OFF (e.g. via the Plan C deck) before the grant arrives.

- [ ] **Step 1: Write the failing race test**

Add to the `tests` module in `crates/familiar-runtime/src/lib.rs`:

```rust
    #[test]
    fn a_grant_after_the_actuator_is_disabled_does_not_act() {
        use familiar_core::capabilities::CapabilityId;
        // Arm everything, raise an ambiguous prompt (confidence 50 => ask).
        let sensors = FakeSensors::new(vec![vec![Event::OutboundConn {
            at: 1000,
            process: proc(7),
            dst_ip: "203.0.113.9".into(),
            dst_port: 443,
        }]]);
        let mut sup = Supervisor::new(
            armed_engine(),
            sensors,
            RecordingActuators::default(),
            CapturingNotifier::default(),
            NullAdvisor,
            30_000,
        );
        sup.drive_once(1000);
        let id = sup.notifier().requests[0].id;
        // The operator now disables the block actuator, THEN grants the stale prompt.
        sup.engine
            .set_capability(CapabilityId::ActuatorBlockConn, false, 1100, &mut sup.audit);
        sup.resolve_permission(id, true, 1200);
        assert!(
            sup.actuators().applied.is_empty(),
            "a grant cannot resurrect a disabled actuator"
        );
        assert!(
            sup.audit.records().iter().any(|r| r.kind == AuditKind::NoAction
                && r.detail.contains("disabled")),
            "the ignored grant is recorded"
        );
    }
```

(`armed_engine()` already exists in the runtime tests module and arms all four capabilities.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-runtime a_grant_after_the_actuator_is_disabled_does_not_act -v`
Expected: FAIL — the block is currently applied on grant.

- [ ] **Step 3: Add the defensive re-check**

In `crates/familiar-runtime/src/lib.rs`, in `resolve_permission`, replace the grant arm:

```rust
            if outcome.permits_action() {
                let action = req.detection.proposed.clone();
                // Defensive re-check (spec §7, fail-closed): the actuator
                // capability may have been switched OFF between raising this
                // prompt and the grant. A disabled actuator cannot fire even on
                // an explicit grant — the toggle is physical.
                if self
                    .engine
                    .registry()
                    .is_enabled(action.actuator_capability())
                {
                    self.act(&action, now);
                } else {
                    self.audit.append(
                        now,
                        AuditKind::NoAction,
                        format!(
                            "grant ignored: {:?} actuator disabled",
                            action.actuator_capability()
                        ),
                    );
                    self.notifier
                        .notify("Grant ignored: the required capability is now disabled");
                }
            }
```

(`action.actuator_capability()` is already in scope via `ProposedAction`; `self.engine.registry()` is the existing public accessor.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p familiar-runtime -v`
Expected: PASS (all runtime tests).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-runtime/src/lib.rs
git commit -m "Re-check the actuator capability when resolving a permission grant"
```

---

## Task H3: F2 — reload and verify the persisted audit chain on startup

**Files:**
- Modify: `crates/familiar-daemon/src/persistence.rs` (`rotate_corrupt_audit`, `restore_audit` + tests)
- Modify: `crates/familiar-daemon/src/run.rs` (seed the Supervisor's audit from disk)

**Interfaces:**
- Produces: `persistence::rotate_corrupt_audit(dir: &Path) -> io::Result<()>` (renames `audit.jsonl` to the first free `audit.jsonl.corrupt-N`); `persistence::restore_audit(dir: &Path, now: u64) -> (AuditLog, usize)` (verified log if intact, else rotate aside + a fresh log carrying one `IntegrityAlert`; the `usize` is how many records are already on disk and must not be re-appended). `main_loop` assigns the log into `sup.audit` and seeds `persisted`.

- [ ] **Step 1: Write the failing persistence tests**

Add to the `tests` module in `crates/familiar-daemon/src/persistence.rs`:

```rust
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
        assert!(dir.join("audit.jsonl.corrupt-0").exists(), "bad file rotated aside");
        assert!(restored.verify().is_ok());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-daemon restore_audit -v`
Expected: FAIL — `restore_audit` / `rotate_corrupt_audit` not defined.

- [ ] **Step 3: Implement the helpers**

In `crates/familiar-daemon/src/persistence.rs`, add `AuditKind` to the imports (`use familiar_core::audit::{AuditKind, AuditLog, AuditRecord};`) and add:

```rust
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
            let _ = rotate_corrupt_audit(dir);
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
```

Run: `cargo test -p familiar-daemon restore_audit -v`
Expected: PASS.

- [ ] **Step 4: Wire it into the run loop**

In `crates/familiar-daemon/src/run.rs`, after `let mut sup = build_supervisor(…).expect("build supervisor");` and before the loop, replace `let mut persisted = 0usize;` with:

```rust
    // F2: reload + verify the persisted audit chain before driving. A tampered
    // or corrupt log is rotated aside and flagged, never silently trusted and
    // never appended-to (which would start a second genesis chain).
    let (audit_log, mut persisted) = persistence::restore_audit(&cfg.state_dir, now_ms());
    sup.audit = audit_log;
```

Run: `cargo build -p familiar-daemon && cargo test -p familiar-daemon -- --nocapture`
Expected: builds; existing daemon tests still pass (the netns ones may SKIP).

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-daemon/src/persistence.rs crates/familiar-daemon/src/run.rs
git commit -m "Reload and verify the audit chain on startup; rotate a tampered log aside"
```

---

## Task H4: F3 — never advance the persist cursor past a record that did not reach disk

**Files:**
- Modify: `crates/familiar-daemon/src/run.rs` (the per-tick persist loop)

**Interfaces:** none (loop-internal durability fix). Behaviour: a record whose append fails is retried next tick instead of being skipped forever.

- [ ] **Step 1: Replace the persist loop**

In `crates/familiar-daemon/src/run.rs`, replace the per-tick persistence block (currently `for rec in &recs[persisted..] { … } persisted = recs.len();`):

```rust
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
```

- [ ] **Step 2: Build + full daemon suite**

Run: `cargo build -p familiar-daemon && cargo test -p familiar-daemon -- --nocapture`
Expected: builds; tests pass/skip as before. (The fix is exercised end-to-end by the existing netns acceptance tests, which persist real records; a unit test would require an injectable failing `append`, out of scope for this surgical change — the logic is a straightforward retry-from-cursor.)

- [ ] **Step 3: Commit**

```bash
git add crates/familiar-daemon/src/run.rs
git commit -m "Retry audit persistence from the failed record instead of dropping it"
```

---

## Task H5: F9 — `reverse_all` flushes the block chain only, leaving sensing intact

**Files:**
- Modify: `crates/familiar-linux/src/nft.rs` (`run_nft` helper + `flush_block_chain`)
- Modify: `crates/familiar-linux/src/actuators.rs` (`reverse_all`)
- Modify: `crates/familiar-linux/tests/nft_netns.rs` (keeps-sensing test)

**Interfaces:**
- Produces: `nft::flush_block_chain() -> Result<(), NftError>` (`nft flush chain inet familiar egress-block` — removes every DROP rule, keeps the table and the `egress-sense` queue chain) and a private `run_nft(args: &[&str]) -> Result<String, NftError>` arg-style runner (reused by Plan C's `unblock_outbound`). `LinuxActuators::reverse_all` now calls `flush_block_chain`, not `delete_table`.

- [ ] **Step 1: Write the failing netns test (block, reverse_all, sensing survives)**

Append to `crates/familiar-linux/tests/nft_netns.rs` (reuse the file's existing `reexec_in_netns` helper):

```rust
#[test]
fn reverse_all_clears_blocks_but_keeps_the_sense_chain() {
    if reexec_in_netns("reverse_all_clears_blocks_but_keeps_the_sense_chain") {
        return;
    }
    use familiar_linux::{LinuxActuators, nft};
    use familiar_platform::Actuators;
    use familiar_core::policy::{ProposedAction};
    use familiar_core::events::ProcessRef;
    use std::process::Command;

    let ruleset = || -> String {
        String::from_utf8(Command::new("nft").args(["list", "ruleset"]).output().unwrap().stdout).unwrap()
    };

    let mut act = LinuxActuators::new("/sys/fs/cgroup").expect("actuators"); // ensures the table
    nft::install_queue_rule(0).expect("sense chain"); // the NFQUEUE divert lives in the same table
    act.apply(&ProposedAction::BlockOutbound {
        process: ProcessRef { pid: 7, exe: "/x".into() },
        dst_ip: "203.0.113.9".into(),
        dst_port: 443,
    }).expect("block");
    assert!(ruleset().contains("drop"), "block installed");

    act.reverse_all().expect("reverse_all");
    let rs = ruleset();
    assert!(!rs.contains("drop"), "blocks cleared:\n{rs}");
    assert!(rs.contains("queue"), "the sense (NFQUEUE) chain must survive reverse_all:\n{rs}");
}
```

Run: `cargo test -p familiar-linux reverse_all_clears_blocks_but_keeps_the_sense_chain -- --nocapture`
Expected: FAIL — current `reverse_all` deletes the whole table, so `queue` is gone (assert fails); or SKIP unprivileged.

- [ ] **Step 2: Add `run_nft` + `flush_block_chain`**

In `crates/familiar-linux/src/nft.rs`, add near the top (after the `NftError` enum):

```rust
use std::process::Command;

/// Run `nft <args...>` and return stdout, mapping any failure to `NftError::Send`.
/// (Argument form only; `install_queue_rule` keeps its `-f -` stdin pipe.)
fn run_nft(args: &[&str]) -> Result<String, NftError> {
    let out = Command::new("nft")
        .args(args)
        .output()
        .map_err(|e| NftError::Send(format!("spawn nft: {e}")))?;
    if !out.status.success() {
        return Err(NftError::Send(String::from_utf8_lossy(&out.stderr).into_owned()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Remove every DROP rule from the block chain while leaving the table and the
/// `egress-sense` NFQUEUE chain in place — so lifting containment never blinds
/// the sensor (the divert rule is installed only at startup).
pub fn flush_block_chain() -> Result<(), NftError> {
    run_nft(&["flush", "chain", "inet", TABLE, BLOCK_CHAIN]).map(|_| ())
}
```

In `crates/familiar-linux/src/actuators.rs`, change `reverse_all`:

```rust
    /// Reverse every block familiar installed by flushing the block chain. The
    /// table and the NFQUEUE sense chain are preserved, so sensing keeps running.
    /// Idempotent.
    pub fn reverse_all(&mut self) -> Result<(), ActuationError> {
        nft::flush_block_chain().map_err(|e| ActuationError::Failed(e.to_string()))?;
        self.active_blocks.clear();
        Ok(())
    }
```

(Leave `nft::delete_table` in place — it is still the correct full-teardown primitive for shutdown; `reverse_all` simply no longer uses it.)

- [ ] **Step 3: Run to verify pass**

Run: `cargo test -p familiar-linux reverse_all_clears_blocks_but_keeps_the_sense_chain -- --nocapture`
Expected: PASS (or SKIP unprivileged).

- [ ] **Step 4: Commit**

```bash
git add crates/familiar-linux/src/nft.rs crates/familiar-linux/src/actuators.rs crates/familiar-linux/tests/nft_netns.rs
git commit -m "reverse_all flushes the block chain only, preserving the sense chain"
```

---

## Task H6: F7 — the fanotify helper never wraps an overflow (`fd < 0`) event

**Files:**
- Modify: `crates/familiar-fanotify-helper/src/fanotify.rs` (`should_forward` + test)
- Modify: `crates/familiar-fanotify-helper/src/main.rs` (guard before `OwnedFd::from_raw_fd`)

**Interfaces:**
- Produces: `fanotify::should_forward(fd: i32) -> bool` (`fd >= 0`). The main loop skips and logs any event that fails it — a `FAN_Q_OVERFLOW`/`FAN_NOFD` (-1) record — before wrapping the fd, so `OwnedFd::from_raw_fd` is only ever called on a real fd. The unsafe-precondition violation is removed; queue overflow is logged loudly (richer fail-closed propagation to the daemon is Plan C / F4).

- [ ] **Step 1: Write the failing guard test**

Add to `crates/familiar-fanotify-helper/src/fanotify.rs`:

```rust
/// Whether an event's fd is real and forwardable. A fanotify queue overflow
/// yields `FAN_NOFD` (-1); such an event must never be wrapped in `OwnedFd`.
pub fn should_forward(fd: RawFd) -> bool {
    fd >= 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_or_nofd_is_not_forwarded() {
        assert!(!should_forward(-1), "FAN_NOFD must be dropped, never wrapped");
        assert!(should_forward(3), "a real fd is forwarded");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p familiar-fanotify-helper should_forward -v`
Expected: FAIL (compile) — `should_forward` not defined / wrong import; add it and re-run for PASS of this unit.

Run again: `cargo test -p familiar-fanotify-helper overflow_or_nofd_is_not_forwarded -v`
Expected: PASS.

- [ ] **Step 3: Guard the wrap in the main loop**

In `crates/familiar-fanotify-helper/src/main.rs`, in the per-event loop, insert the guard at the very top of the `for ev in events` body — before the `read_link`/`OwnedFd` lines — and tighten the SAFETY comment:

```rust
        for ev in events {
            if !fanotify::should_forward(ev.fd) {
                // FAN_Q_OVERFLOW / FAN_NOFD: events were dropped by the kernel
                // queue. Do NOT wrap fd<0 in OwnedFd. Log loudly — a lost read is
                // a sensing gap, not a silent success.
                eprintln!("[helper] fanotify overflow/no-fd (fd={}); file-read events were dropped", ev.fd);
                continue;
            }
            let path = std::fs::read_link(format!("/proc/self/fd/{}", ev.fd))
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            // SAFETY: should_forward guaranteed ev.fd >= 0; it is the valid fd the
            // kernel handed us in this event. Wrap it so it closes once on drop.
            let _owned = unsafe { OwnedFd::from_raw_fd(ev.fd) };
            if ev.pid > 0 && is_watched(&path, &prefixes) {
                // ... unchanged emit block ...
            }
        }
```

(Keep the existing emit block verbatim; only the guard + comment are new.)

- [ ] **Step 4: Build + test the helper**

Run: `cargo build -p familiar-fanotify-helper && cargo test -p familiar-fanotify-helper -v`
Expected: builds; tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/familiar-fanotify-helper/src/fanotify.rs crates/familiar-fanotify-helper/src/main.rs
git commit -m "Drop fanotify overflow (fd<0) events instead of wrapping an invalid fd"
```

---

## Task H7: full workspace verification

**Files:** none.

- [ ] **Step 1: Green + lint + fmt across the workspace**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: all green. Record the test count (it should be the prior count, with the two F1 tests rewritten in place and ~5 new tests added).

- [ ] **Step 2: Privileged acceptance still passes (operator-run, optional but recommended)**

Run (as the operator, like Plan B): `sudo ./scripts/run-privileged-acceptance.sh`
Expected: PASS — the hardening must not regress the autonomous-containment path. (If the script asserts the old disabled-actuator "ask" behavior anywhere, update it to the deny expectation.)

- [ ] **Step 3: Closure ritual**

Run: `cd /home/toxic2040/work && ./bin/catalog-rescan` and resolve any STALE/MISSING for files this work touched.

---

## Self-review

**Finding coverage:**
- F1 (grant fires disabled actuator) → H1 (engine denies) + H2 (resolve_permission re-check). ✔
- F2 (audit not reloaded/verified; double genesis) → H3 (restore_audit + rotate; wired into run loop). ✔
- F3 (append failure loses record) → H4 (retry-from-cursor persist loop). ✔
- F7 (overflow → from_raw_fd(-1) + silent loss) → H6 (should_forward guard + loud log). ✔
- F9 (reverse_all blinds sensing) → H5 (flush_block_chain, keep the table + sense chain). ✔
- F4 (silent sensor blindness), F8 (unauthenticated fileread socket) → **deferred to Plan C** (the deck surfaces sensor health; Plan C's peer-cred machinery applies to the fileread socket). Tracked there, not here.
- F5 (IPv6 silent), F6 (post-SYN one-shot) → **v0.2**, documented honestly in `docs/operating-familiar-linux.md` (sharpen the "IPv4 only" / "NFQUEUE senses, does not block" limitations to name the bypass explicitly). Add that doc edit as the final step of whichever plan runs last.

**Placeholder scan:** none — every step shows the actual code.

**Type/name consistency:** `restore_audit`/`rotate_corrupt_audit` (H3) signatures match their call site in `run.rs`. `run_nft`/`flush_block_chain` (H5) are reused by Plan C's `unblock_outbound` — Plan C Task 1.2 must be updated to note `run_nft` already exists (do not re-add it). `should_forward` (H6) takes `RawFd` matching `RawEvent.fd`. The F1 disposition change (`Deny`) is already handled by the existing `dispatch` `Deny` arm (records `NoAction`, no prompt) — no new runtime arm needed.
