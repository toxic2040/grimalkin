# Slice 1 — Guardian Fold-In & Relicense: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Rust guardian a genuinely MIT-licensed component of the public `grimalkin` repo — remove its only copyleft dependency, relicense it, and fold it in under `guardian/` without breaking the working Python companion and without pushing anything.

**Architecture:** Two repos collapse into one during this slice. First the guardian (currently `repos/familiar`, AGPL, links GPL-3.0 `rustables`) is cleaned *in place*: `rustables` is replaced by `nft`-binary calls (it already shells out to `nft` for everything else), then the workspace is relicensed AGPL→MIT. The cleaned tree is then squash-imported into `grimalkin/guardian/` so grimalkin's (eventually public) history never contains the GPL/AGPL past. The Python companion stays exactly where it is at the repo root; this slice only *adds* `guardian/`.

**Tech Stack:** Rust 1.95 (guardian workspace), the `nft` userspace binary (already a runtime dependency), Python 3.10+ (companion, untouched), `git subtree` for the import.

## Global Constraints

- **Uniformly MIT is the headline gate.** After this slice there must be NO AGPL or GPL anywhere in the guardian — not in a license field, not in a vendored dependency, not in a source header. `rustables` (GPL-3.0-or-later) is removed entirely, not relicensed.
- **Do not break the companion.** The Python app at the repo root (`grimalkin.py` + `grimalkin_*.py` + `test_grimalkin.py` + `scripts/`) is not moved or edited in this slice. Its tests must still pass and it must still import/run.
- **Nothing is pushed.** All work is local commits. `grimalkin` work happens on the `guardian-integration` branch (already created). The public push is Slice 7, gated separately.
- **Behavior parity for the guardian.** Replacing `rustables` with `nft` must not change observable containment behavior: the same `inet familiar` table, `egress-block` chain, and a drop rule matching outbound TCP to `dst:dport`. The existing netns tests are the parity check.
- **No automation fingerprints** in code, comments, commit messages, or docs (grimalkin runs pre-commit hooks; commits must pass them). No AI-provenance trailers.
- **The guardian ships opt-in / disarmed by default** is a *later* slice (Slice 3) — not this one. This slice changes no runtime arming behavior.

---

## Deviations from the spec (confirm at plan review)

The grounding pass surfaced three refinements to the spec's Slice 1 wording. Two are scope *reductions* and need a nod:

1. **`rustables` removal is folded in (already approved).** The spec's "relicense to MIT" is impossible while the GPL-3.0 `rustables` is linked; this slice removes it. (Approved.)
2. **Crate rename `familiar-*` → `guardian-*` is DEFERRED out of this slice.** Reason: the rename ripples into binary names, systemd units, scripts, and the `/run/familiar` runtime paths — all of which are Slice 6 (packaging) territory. A half-rename here (crates renamed, binaries/paths still `familiar`) is worse style than a clean rename done with packaging. The crate names are internal; the product is "grimalkin / the guardian" regardless. **Recommend: rename in Slice 6.**
3. **The Python companion is NOT moved into `companion/` this slice.** Reason: moving the root Python files breaks import paths, launch (`python grimalkin.py`), `test_grimalkin.py`, CI, pre-commit config, and `DEPLOY.md` for zero functional gain and real risk to the working public app. This slice only *adds* `guardian/` at the top level. **Recommend: the `companion/` reorg is a separate optional cosmetic task, later or never.**

If you accept 2 and 3, I'll update spec §6/§12 to match after the slice.

---

## File structure (Slice 1)

In `repos/familiar` (cleaned in place, then imported and retired as a local provenance mirror):
- Modify: `crates/familiar-linux/src/nft.rs` — replace the three `rustables` functions with `nft`-binary calls.
- Modify: `crates/familiar-linux/Cargo.toml` — drop `rustables` (and `libc` if it falls unused).
- Modify: `Cargo.toml` (workspace) — drop the `rustables` workspace dep and the `[patch.crates-io]` entry.
- Delete: `vendor/rustables-0.8.7-patched/` (the GPL tree).
- Modify: `Cargo.toml` (workspace) — `license` AGPL→MIT.
- Create: `LICENSE` (MIT) in the guardian repo.

In `repos/grimalkin` (branch `guardian-integration`):
- Create: `guardian/` — the squash-imported cleaned guardian workspace.
- Modify: `.gitignore` — ignore `guardian/target/`.
- Modify: `README.md` — describe the two-part product (companion + guardian, MIT, Linux-first).
- Modify (if needed): `.pre-commit-config.yaml` / `.secrets.baseline` — so commits pass with the Rust tree present.
- Create: `guardian/PROVENANCE.md` — points to the retained local guardian dev history.

---

## Task 1: Replace `rustables` with the `nft` binary (remove the GPL-3.0 dependency)

**Working dir:** `repos/familiar` (branch `master`).

**Files:**
- Modify: `crates/familiar-linux/src/nft.rs:1-136` (module doc, imports, `ensure_table`, `table()`, `block_outbound`, `delete_table`)
- Modify: `crates/familiar-linux/Cargo.toml`
- Modify: `Cargo.toml` (workspace deps + `[patch.crates-io]`)
- Delete: `vendor/rustables-0.8.7-patched/`
- Test: `crates/familiar-linux/tests/nft_netns.rs` (existing — parity check, no edits expected)

**Interfaces:**
- Produces (unchanged signatures): `nft::ensure_table() -> Result<(), NftError>`, `nft::block_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError>`, `nft::delete_table() -> Result<(), NftError>`. Callers (`LinuxActuators`, `run.rs`) are untouched.

- [ ] **Step 1: Confirm the existing netns parity tests cover block/unblock/reverse_all and run them on the current (rustables) code**

Run: `cd repos/familiar && cargo test -p familiar-linux --test nft_netns -- --nocapture`
Expected: PASS (they assert `nft list ruleset` contains `drop`/`queue` and that per-block unblock removes one rule). These are mechanism-agnostic and become the parity gate for the rewrite. Record the pass.

- [ ] **Step 2: Rewrite the three `rustables` functions to use `run_nft`**

In `crates/familiar-linux/src/nft.rs`: replace the module doc (lines 1-5), delete the `rustables` imports (lines 6-10) and the `table()` helper (lines 47-49), and replace `ensure_table`, `block_outbound`, `delete_table`. Keep `use std::net::Ipv4Addr;`, `use std::process::Command;`, `run_nft`, `flush_block_chain`, `install_queue_rule`, `parse_handle`, `unblock_outbound`, and the tests as-is.

New module doc (lines 1-5):
```rust
//! The dedicated `inet familiar` nftables table and the reversible drop rule.
//! Every nft operation goes through the `nft` userspace binary (no netlink
//! library), keeping the crate dependency-light and free of copyleft deps.
//! Per-block reversal removes a single rule by handle; `reverse_all` flushes the
//! block chain; `delete_table` is the full-teardown primitive.
use std::net::Ipv4Addr;
```

New `ensure_table` (replacing lines 47-62, i.e. drop `table()` and rewrite):
```rust
/// Create the dedicated `inet familiar` table and the block chain. Idempotent:
/// `nft add` of an existing table/chain is accepted.
pub fn ensure_table() -> Result<(), NftError> {
    run_nft(&["add", "table", "inet", TABLE])?;
    run_nft(&[
        "add", "chain", "inet", TABLE, BLOCK_CHAIN,
        "{ type filter hook output priority 0; policy accept; }",
    ])?;
    Ok(())
}
```

New `block_outbound` (replacing lines 105-129):
```rust
/// Install a reversible DROP for outbound TCP to `dst:dport`. Returns a note for
/// the audit/notify trail. The rule renders as `ip daddr <dst> tcp dport <dport>
/// drop`, which is what `parse_handle`/`unblock_outbound` match on.
pub fn block_outbound(dst: Ipv4Addr, dport: u16) -> Result<String, NftError> {
    let ip = dst.to_string();
    let port = dport.to_string();
    run_nft(&[
        "add", "rule", "inet", TABLE, BLOCK_CHAIN,
        "ip", "daddr", &ip, "tcp", "dport", &port, "drop",
    ])?;
    Ok(format!("nft drop {dst}:{dport} in table inet {TABLE}"))
}
```

New `delete_table` (replacing lines 131-136):
```rust
/// Reverse everything by deleting the dedicated table.
pub fn delete_table() -> Result<(), NftError> {
    run_nft(&["delete", "table", "inet", TABLE]).map(|_| ())
}
```

- [ ] **Step 3: Drop the `rustables` dependency and the vendored tree**

In `Cargo.toml` (workspace): delete the `rustables = "0.8.7"` line under `[workspace.dependencies]` and the entire `[patch.crates-io]` block (the only entry is the vendored rustables).
In `crates/familiar-linux/Cargo.toml`: delete `rustables.workspace = true`.
Delete the vendored tree:
```bash
cd repos/familiar && git rm -r vendor/rustables-0.8.7-patched
```

- [ ] **Step 4: Build; remove `libc` from `familiar-linux` if it is now unused**

Run: `cd repos/familiar && cargo build -p familiar-linux 2>&1 | tail -20`
Expected: builds. If it fails on `libc::NFPROTO_IPV4`/`libc::IPPROTO_TCP`, those were only in the old `block_outbound` and are gone — re-check Step 2. Then check whether `familiar-linux` still uses `libc`:
Run: `grep -rn "libc::" crates/familiar-linux/src/`
- If zero matches: remove `libc.workspace = true` from `crates/familiar-linux/Cargo.toml`. (Leave the workspace `libc` dep — the fanotify helper uses it.)
- If matches remain: leave `libc.workspace = true`.
Re-run `cargo build -p familiar-linux` — expected: clean.

- [ ] **Step 5: Verify `rustables` is fully gone and the parity tests still pass**

Run: `cd repos/familiar && grep -rni "rustables" --include='*.rs' --include='*.toml' . | grep -v target; echo "exit:$?"`
Expected: no matches (grep exit 1 = "no matches" is the pass).
Run: `cargo test -p familiar-linux --test nft_netns -- --nocapture` and `cargo test --workspace`
Expected: the netns parity tests PASS (same `drop`/`queue`/unblock assertions as Step 1); full workspace green. If netns tests SKIP (no userns), note it and run the privileged path manually or accept the SKIP as the environment's limit.
Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --all --check`
Expected: clean (the 2 pre-existing vendored-rustables warnings are GONE now that the vendor tree is deleted — a bonus).

- [ ] **Step 6: Commit**

```bash
cd repos/familiar
git add -A
git commit -m "Replace rustables with the nft binary; drop the vendored GPL crate"
```

---

## Task 2: Relicense the guardian AGPL-3.0 → MIT

**Working dir:** `repos/familiar` (branch `master`).

**Files:**
- Modify: `Cargo.toml:18` (workspace `license`)
- Create: `LICENSE` (MIT)

**Interfaces:** none (metadata only).

- [ ] **Step 1: Flip the workspace license to MIT**

In `repos/familiar/Cargo.toml`, change line 18:
```toml
license = "MIT"
```
(All nine crates inherit via `license.workspace = true`; no per-crate edits needed.)

- [ ] **Step 2: Add the MIT LICENSE file (matching grimalkin's)**

Create `repos/familiar/LICENSE` with the standard MIT text and the same copyright line grimalkin uses:
```
MIT License

Copyright (c) 2026 toxic2040

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
(Copy the exact text from `repos/grimalkin/LICENSE` to guarantee a byte-identical license body.)

- [ ] **Step 3: Verify no copyleft remains anywhere in the guardian**

Run: `cd repos/familiar && grep -rniE 'agpl|gpl|affero' --include='*.rs' --include='*.toml' --include='*.md' . | grep -v target; echo "exit:$?"`
Expected: no matches (the vendored GPL crate is already deleted from Task 1; the workspace license is now MIT). grep exit 1 = pass.
Run: `cargo build --workspace` — expected: builds (license change is metadata; no code impact).

- [ ] **Step 4: Commit**

```bash
cd repos/familiar
git add -A
git commit -m "Relicense the guardian to MIT"
```

---

## Task 3: Squash-import the cleaned guardian into `grimalkin/guardian/`

**Working dir:** `repos/grimalkin` (branch `guardian-integration`).

**Files:**
- Create: `guardian/` (the imported tree)
- Modify: `.gitignore`

**Interfaces:** none (repository restructure).

- [ ] **Step 1: Confirm the source is clean and record its commit**

Run: `cd repos/familiar && git log --oneline -3 && grep -rni 'rustables\|agpl\|gpl' --include='*.toml' . | grep -v target; echo "clean-exit:$?"`
Expected: the Task 1 + Task 2 commits are at the top; the grep finds nothing (exit 1). Record the source HEAD short SHA (call it `<SRC_SHA>`) for the import commit message.

- [ ] **Step 2: Squash-import the guardian under `guardian/`**

`git subtree add --squash` imports only the *current* tree as a single squashed commit, so grimalkin's history never carries the GPL/AGPL past. Run from `repos/grimalkin`:
```bash
cd repos/grimalkin
git remote add guardian-src ../familiar          # local path remote
git fetch guardian-src
git subtree add --prefix=guardian --squash guardian-src master
git remote remove guardian-src                   # the remote was only needed for the import
# Prune the guardian's internal dev docs from the PUBLIC tree: the superpowers
# plans/specs are verbose internal-process artifacts that reference the removed
# rustables and use agentic-process language, and the operating guide is tied to
# pre-rename paths/binaries (renamed in Slice 6). They stay in the retained
# familiar repo as provenance (see guardian/PROVENANCE.md, Task 4). grimalkin
# documents the guardian in its own README/docs going forward.
git rm -r guardian/docs
git commit -m "Drop the guardian's internal dev docs from the public tree (provenance retained)"
```
Expected: `guardian/` contains the Rust workspace (`crates/`, `Cargo.toml`, `systemd/`, `scripts/`, `LICENSE`) and NO `docs/`; one squashed import commit plus the docs-prune commit on `guardian-integration`.

- [ ] **Step 3: Ignore the Rust build output**

Append to `repos/grimalkin/.gitignore`:
```
# Rust guardian build output
guardian/target/
```

- [ ] **Step 4: Verify the guardian builds and tests in its new home, and the companion is untouched**

Run:
```bash
cd repos/grimalkin/guardian && cargo test --workspace 2>&1 | grep -E "test result:|error" | tail -20
cd repos/grimalkin/guardian && cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -3
cd repos/grimalkin/guardian && cargo fmt --all --check && echo "fmt clean"
```
Expected: guardian tests green (same count as before the move), clippy clean (no more vendored-rustables warnings), fmt clean.
Run (companion unaffected):
```bash
cd repos/grimalkin && git status --short | grep -vE '^\?\?|guardian/' ; echo "companion files changed above (expect none)"
cd repos/grimalkin && python -c "import ast; ast.parse(open('grimalkin.py').read())" && echo "grimalkin.py parses"
```
Expected: no tracked companion files modified by the import; `grimalkin.py` still parses. (Full companion test run happens in Task 4 after pre-commit reconciliation.)

- [ ] **Step 5: Commit**

```bash
cd repos/grimalkin
git add .gitignore
git commit -m "Import the guardian under guardian/ (squashed from <SRC_SHA>); ignore its build output"
```

---

## Task 4: Two-part README + provenance note + pre-commit reconciliation (Slice 1 gate)

**Working dir:** `repos/grimalkin` (branch `guardian-integration`).

**Files:**
- Modify: `README.md`
- Create: `guardian/PROVENANCE.md`
- Modify (only if needed): `.pre-commit-config.yaml`, `.secrets.baseline`

**Interfaces:** none.

- [ ] **Step 1: Reconcile pre-commit with the Rust tree**

Run: `cd repos/grimalkin && pre-commit run --all-files 2>&1 | tail -30` (if `pre-commit` is installed; else skip and note).
Expected: passes. If a Python-oriented hook (e.g. `detect-secrets`, file-size, EOL) flags files under `guardian/`, scope that hook to exclude `guardian/` in `.pre-commit-config.yaml` (add `exclude: '^guardian/'` to the offending hook) OR update `.secrets.baseline` via `detect-secrets scan > .secrets.baseline` if it is a benign new-baseline. Do the minimal change that makes `pre-commit run --all-files` pass without weakening a hook for the companion code.

- [ ] **Step 2: Write the provenance note**

Create `repos/grimalkin/guardian/PROVENANCE.md`:
```markdown
# Guardian provenance

The guardian was developed in a standalone Rust repository (`Familiar`) and
folded into grimalkin as a squashed import, so this repo's history starts from
a clean, MIT-licensed tree. The full development history — the deterministic
core, the Linux body, the red-team hardening pass, and the control deck — is
preserved in that original local repository and is not reproduced here.

The guardian was relicensed from AGPL-3.0-or-later to MIT and had its only
copyleft dependency (the GPL-3.0 `rustables` netlink crate) removed in favor of
the `nft` userspace binary before this import.
```

- [ ] **Step 3: Rewrite the top-level README as the two-part product**

In `repos/grimalkin/README.md`, update the description and add a guardian section. Keep grimalkin's existing companion documentation; add (after the "What It Does" / stack sections) a section like:
```markdown
## The Guardian (Linux)

grimalkin includes a local security guardian: a separate, privileged Rust daemon
that watches for hostile activity against your files and system and contains it
within a strict, auditable authority envelope — reversible, fail-closed
containment for high-confidence threats, and an explicit ask for anything
ambiguous. The security guarantees live in a small, deterministic, memory-safe
harness, not in a model.

The guardian is **off until you turn it on** and is Linux-only (it uses NFQUEUE,
nftables, cgroup-v2, and fanotify). It lives under `guardian/` and is built
separately (`cd guardian && cargo build --release`). The companion drives it
over a local Unix socket; wiring the Control Deck to it is the next slice.

The whole project — companion and guardian — is MIT licensed.
```
Update the top-of-README one-liner/badges so the MIT + "companion and guardian" framing is accurate (the model line becomes Gemma in Slice 2; leave model wording alone here unless trivially consistent).

- [ ] **Step 4: Slice 1 gate — verify the whole slice**

Run, from `repos/grimalkin`:
```bash
# 1. Uniformly MIT, no copyleft, no rustables in the guardian's CODE/CONFIG:
#    (scope to source + manifests + lockfile — that is the license-relevant surface.
#     guardian/PROVENANCE.md prose legitimately explains that rustables/AGPL were
#     removed, so .md prose is intentionally NOT grepped here.)
grep -rniE 'agpl|affero|rustables' guardian --include='*.rs' --include='*.toml' --include='Cargo.lock' | grep -v 'guardian/target'; echo "copyleft-grep-exit:$?"   # expect exit 1 (none)
grep -n '"MIT"' guardian/Cargo.toml                                  # expect the workspace license = "MIT"
test -f guardian/LICENSE && head -1 guardian/LICENSE                  # expect "MIT License"
# 2. Guardian builds + tests + lint clean in-place:
( cd guardian && cargo test --workspace >/tmp/g.txt 2>&1; grep -E "test result:" /tmp/g.txt | awk '{p+=$4;f+=$6} END{print "guardian: "p" passed, "f" failed"}' )
( cd guardian && cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1 && echo "guardian clippy clean" )
( cd guardian && cargo fmt --all --check && echo "guardian fmt clean" )
# 3. Companion still works:
python -m pytest test_grimalkin.py -q 2>&1 | tail -5   # or the project's documented test command
# 4. Nothing pushed:
git status --short --branch | head -1                  # expect "## guardian-integration" with no "ahead of origin"
```
Expected: copyleft grep finds nothing (exit 1); license is MIT; guardian tests pass with 0 failures and clippy/fmt clean; companion tests pass; on `guardian-integration`, not pushed.

- [ ] **Step 5: Commit**

```bash
cd repos/grimalkin
git add README.md guardian/PROVENANCE.md .pre-commit-config.yaml .secrets.baseline 2>/dev/null
git commit -m "Document grimalkin as a two-part product (companion + guardian); add guardian provenance"
```

---

## Self-review

**Spec coverage (spec §10 Slice 1):** relicense AGPL→MIT → Task 2 ✓; subtree-merge under `guardian/` → Task 3 (squash, for clean public history) ✓; reconcile the build → Task 3 Step 4 + Task 4 Step 1 ✓; top-level README → Task 4 ✓; gate (uniformly MIT, both halves build, history preserved, not pushed) → Task 4 Step 4 ✓. **Deviations from spec wording:** `rustables` removal added (Task 1, approved); crate rename deferred to Slice 6; `companion/` move dropped (guardian added at root) — all flagged in the Deviations section for confirmation.

**Placeholder scan:** none — Task 1 shows the full replacement code; the MIT text is given verbatim; all commands are exact. The one conditional (remove `libc` if unused) has both branches specified. The pre-commit reconciliation names the exact minimal fixes (`exclude:` or baseline rescan) rather than "handle it."

**Type/consistency:** `ensure_table`/`block_outbound`/`delete_table` keep their exact signatures, so `LinuxActuators` and `run.rs` callers are untouched. The new `block_outbound` rule (`ip daddr <ip> tcp dport <port> drop`) matches what `parse_handle`/`unblock_outbound` already grep for (`daddr <ip>`, `dport <port>`, `drop`) — verified against the current `nft.rs:140-165`. The netns tests assert `drop`/`queue` presence, which both the old and new rules satisfy → genuine parity check.

**Risk note:** Task 1's behavior parity rests on the netns tests actually running (they self-exec under `unshare -Urn`). If the execution environment lacks user namespaces they SKIP — in that case the executor must run the privileged path manually (or flag it) before trusting parity, since the rustables→nft swap is the one real behavior change in this slice.
