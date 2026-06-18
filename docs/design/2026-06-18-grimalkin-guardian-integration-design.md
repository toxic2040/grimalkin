# Grimalkin Guardian — Integration Design

Date: 2026-06-18
Status: approved design, ready for slice-by-slice implementation planning
Lineage: folds the Rust `Familiar` guardian into the public `grimalkin` familiar as a co-equal, persona-unified component

## 1. Summary

grimalkin is a privacy-first, fully-local AI familiar: chat, a searchable vault,
a knowledge graph, and a persona that sharpens over time. It also ships a
"Control Deck" that today only *displays* a privacy posture — it does not enforce
anything.

This design folds the Rust guardian (built separately as `Familiar` — a
deterministic, fail-closed security harness with real Linux containment) into
grimalkin as a first-class, co-equal half of one product. The familiar that
knows your files now also guards your machine, and the same persona that
organizes your downloads tells you what it caught and why. grimalkin's weakest
feature — a cosmetic security panel — becomes its strongest: a real, auditable,
opt-in guardian.

The two halves run as two processes in one product, bridged by a Unix-socket
control protocol that already exists. The companion (Python: Ollama, FAISS,
Gradio) stays in the ecosystem where local-AI evolves fastest; the guardian
(Rust: privileged, network-isolated, `#![forbid(unsafe_code)]`) stays in the
substrate where rigor is non-negotiable. Neither is rewritten in the other's
language — the seam is the product.

## 2. Locked decisions

These were settled before this document and are not reopened here:

1. **Architecture: two runtimes, one product** (not a single-language rewrite).
   The companion stays Python/Ollama/FAISS/Gradio; the guardian stays Rust. They
   communicate over the guardian's control UDS.
2. **Fold into the public `grimalkin` repo** as a polyglot monorepo — one repo,
   one product, one release. (Not two repos, not a fresh v2 repo.)
3. **Publish the guardian source.** The guardian becomes public as part of
   grimalkin. (A defensive, auditable security tool gains trust by being open.)
4. **Relicense the guardian crates AGPL-3.0-or-later → MIT** to match grimalkin.
   The whole product is MIT.
5. **Name: "the guardian."** The product is grimalkin; the Rust component is
   *grimalkin's guardian*. No separate brand.
6. **Co-equal, persona-unified.** Both halves are first-class; the persona is the
   seam (it narrates the guardian and can heighten its caution).
7. **Model: Gemma 4B** (`gemma3:4b` via Ollama) replaces `qwen3:8b` as the
   default brain, for both chat and guardian narration/advice.
8. **Linux-first**, with the cross-platform door left open by construction (see
   §4). The guardian's enforcement is Linux-kernel-specific.
9. **Ship in slices**, each with its own spec → plan → implementation and a hard
   gate. No rush; accuracy and style over speed.

## 3. What this supersedes

The v0.1 guardian design (`docs/design/2026-06-17-familiar-guardian-v0.1-design.md`)
framed the Rust work as a *complete rewrite that replaces* the Python grimalkin,
keeping grimalkin only as a reference to mine. **That positioning is reversed
here.** grimalkin is the product and keeps evolving; the guardian folds into it
as the enforcement half. The v0.1 document remains accurate about the guardian's
*internal* architecture (core, envelope, audit, Linux adapters) — only its
product positioning is updated by this document.

## 4. Architecture

Two processes, one product:

- **Companion (Python):** grimalkin as it exists — Gradio UI, Ollama (Gemma 4B),
  FAISS vault, knowledge graph, persona, bond. Runs **unprivileged**. This is the
  face and the brain.
- **Guardian (Rust):** the daemon (`CAP_NET_ADMIN`) + the privileged fanotify
  helper (`CAP_SYS_ADMIN`), deterministic core, authority envelope, hash-chained
  audit, real Linux containment (NFQUEUE sensing, reversible nftables block,
  cgroup-v2 freeze). Runs **privileged** and **network-isolated**
  (`IPAddressDeny=any` — it must never egress).
- **The bridge:** the guardian's control socket (`/run/familiar/control.sock`,
  to be renamed under grimalkin) speaking newline-delimited JSON — the
  `familiar-ipc` protocol already built and reviewed. The companion is a client
  of this socket.

**Why the split is the design, not a compromise.** The two halves have opposite
optimal substrates. The companion must *evolve fast* — new models, RAG, agent
techniques land in Python/Ollama first, and swapping the model (Gemma today,
others later) is a one-line change on Ollama. The guardian must be *trustworthy*
— and the trust boundary is the guardian, which is privileged and already
rigorous Rust. An unprivileged Python companion talking to a privileged,
audited, network-isolated Rust guardian is a *good* security architecture:
privilege separation is a feature.

**Where the model runs.** The LLM lives **companion-side only** (Ollama). The
guardian daemon never calls a model — it stays network-isolated, and an LLM call
is too slow for the hot containment path anyway. The model does two out-of-band
things: it **narrates** what the guardian did (persona), and it sends
**heighten-only advice** to the guardian. The guardian's hot path stays
rule-based and deterministic. (This supersedes the earlier v0.2 idea of an
in-daemon `candle` advisor; the brain is companion-side.)

**Cross-platform later.** The guardian's deterministic core is OS-agnostic; the
`platform` trait seam is the swap point. A future macOS/Windows tier implements
those traits (or runs advisory/posture-only) without touching the core or the
protocol. Nothing in this design forecloses it.

## 5. The opt-in model (the guardian is off until you turn it on)

The guardian is opt-in at four layers, smallest blast radius first:

1. **Install-level.** The guardian is a separate component. grimalkin runs
   companion-only without it ever installed.
2. **Service-level.** It is a separate daemon. Don't enable the service → no
   guardian.
3. **Master switch — `armed` / `disarmed`, default *disarmed*.** A first-class
   daemon state. **Disarmed = dormant:** the daemon installs no NFQUEUE divert
   rule, sets no fanotify marks, and intercepts nothing — zero kernel footprint
   beyond an idle process. **Armed:** sensing infrastructure is installed and the
   user's enabled capabilities take effect. This is the one obvious switch at the
   top of grimalkin's deck. Arming is the explicit opt-in; disarming is always
   permitted (it only ever *reduces* protection, like an unblock). Every
   arm/disarm is written to the audit log.
4. **Per-capability.** The existing capability registry — every sensor, detector,
   and actuator default-OFF and fail-closed — for fine control once armed.

The product ships **disarmed and all-capabilities-off**: installed, inert, and
explicitly opt-in at every level. This is a precondition of the public push
(§10, Slice 7).

## 6. Repository layout (polyglot monorepo)

Target layout after the fold-in (exact directory names finalized in Slice 1):

```
grimalkin/                      # the public repo, MIT
├── grimalkin.py, grimalkin_*.py, scripts/, …   # Python companion (stays at repo root)
├── guardian/                   # Rust workspace: the folded-in guardian
│   ├── crates/…                # core, platform, runtime, linux, daemon, ipc, helper, ui
│   ├── systemd/                # guardian units
│   └── Cargo.toml
├── docs/                       # design specs (this file), operating guide
└── README.md                   # the two-part product
```

**Slice 1 adds `guardian/` only and leaves the Python companion at the repo root
untouched.** Moving the companion into a `companion/` subdirectory would break the
working app's import/launch/CI/pre-commit paths for no functional gain, so that
reorg is a deferred, optional cosmetic step (later or never).

The guardian is **squash-imported** (`git subtree add --squash`) so grimalkin's
published history starts from a clean, MIT tree — it never carries the GPL/AGPL
past. The full guardian development history (the v0.1 spine, the red-team
hardening pass, the control-deck build) is retained in the original local
repository as provenance, pointed to from `guardian/PROVENANCE.md`.

## 7. Naming and relicense

- **Naming.** Product-facing: "grimalkin" and "the guardian." The internal Rust
  crates are currently `familiar-*`; renaming them to `guardian-*` (or
  `grimalkin-guardian-*`) is a low-stakes, mechanical call made in Slice 1.
- **Relicense.** The guardian workspace is `AGPL-3.0-or-later`; grimalkin is MIT.
  Slice 1 relicenses the guardian crates to MIT (workspace `license`, any
  file headers/notices) so the whole repo is uniformly MIT before any public
  push. This is a hard gate: **no public push until the tree is consistently MIT.**

## 8. Control protocol

The existing `familiar-ipc` protocol (NDJSON over the UDS, uid-authenticated)
already provides: `ListCapabilities`, `SetCapability`, `AnswerPrompt`, `Unblock`,
`GetStatus`, `GetAudit`. The integration adds three things:

- **`SetArmed(bool)`** + an `armed` field on the status snapshot — the master
  opt-in (§5).
- **An event stream (guardian → companion)** — detections, containments, and
  audit appends pushed/surfaced so the companion (and the persona) learn what the
  guardian did, rather than only polling status.
- **A heighten-only advice channel (companion → guardian)** — feeds the
  guardian's `Advisor` seam. Structurally constrained: advice can only move a
  disposition toward caution (`ActAutonomously → RequirePermission`), never open
  a gate. (See §9.)

All three preserve the protocol's defining property: **no verb installs
containment.** Containment still only happens through sensor → detector → gates
(autonomous) or an explicit human grant.

## 9. Trust and security model

The fold-in must not weaken any guarantee the guardian already proved:

- **The envelope holds end-to-end.** The companion is a client of the narrow
  protocol; it can toggle capabilities, answer prompts, lift blocks, arm/disarm,
  and read state — it cannot install a block or freeze. This "no command installs
  containment" property is re-proven at the Python-client layer (Slice 3 gate).
- **Advice is heighten-only.** The companion's Gemma-driven advice can make the
  guardian *more* cautious, never less — so it is safe for an unprivileged,
  model-driven companion to advise a privileged daemon. Proven by property test
  (Slice 5 gate).
- **The daemon stays network-isolated.** The model lives companion-side; the
  daemon never gains egress. Advice and events cross the UDS, not the network.
- **Master opt-in is fail-closed.** Disarmed means dormant and intercepting
  nothing; the default is disarmed.
- **Licensing is clean.** Uniformly MIT before publish.

## 10. Slice plan

Each slice is an independent spec → plan → implementation with a hard gate. The
gate is a testable stop: the slice is not done until it passes.

### Slice 1 — Fold-in & relicense (foundation)
- Remove the GPL-3.0 `rustables` dependency: do all nftables work via the `nft`
  binary (parity-checked by the existing netns tests). This is the precondition
  for a genuine MIT relicense.
- Relicense the guardian crates AGPL → MIT (workspace `license` + MIT `LICENSE`).
- Squash-import the cleaned guardian into `grimalkin/guardian/` (clean public
  history; full dev history retained as provenance — §6).
- Add `guardian/` at the repo root; the Python companion stays where it is (no
  `companion/` move this slice).
- Reconcile the build (Rust workspace under `guardian/`; `.gitignore`; pre-commit)
  and rewrite the top-level README as the two-part product.
- **Gate:** the guardian tree is consistently MIT with NO copyleft (no AGPL/GPL,
  no `rustables`); the Rust guardian builds and tests green in its new home; the
  Python companion is unchanged and its tests pass; **nothing is pushed.** (Crate
  rename and the `/run/familiar` path rename are Slice 6.)

### Slice 2 — Gemma 4B as the default model (small, early)
- Switch the companion default `qwen3:8b` → `gemma3:4b`; handle Gemma-family
  prompt format and any think-artifact cleanup (mirroring the existing Qwen3
  path); update config and docs.
- **Gate:** companion chat works on Gemma 4B by default; model is still
  configurable; existing companion tests pass.

### Slice 3 — Live bridge: Control Deck becomes a real UDS client (+ master toggle)
- A Python client for the control protocol (connect, the existing verbs, plus
  `SetArmed` and the `armed` status field).
- Guardian-side: add the `armed`/`disarmed` master state (default disarmed),
  gating sensing-infrastructure installation on `armed`; `SetArmed` verb; `armed`
  in the status snapshot; audit each transition.
- grimalkin's Gradio Control Deck goes live: the master Armed/Disarmed switch at
  the top, real capability toggles, live Allow/Deny prompts, active blocks with
  Lift, the hash-chained audit viewer + verify chip, sensor health.
- **Gate:** the deck drives a running guardian (arm/disarm, toggle, answer a
  prompt, lift a block, read audit) **and** the Python client structurally
  cannot bypass the envelope — it speaks only the narrow verbs; a test re-proves
  "no command installs containment" at the Python layer. Disarmed = verified
  dormant (no divert rule, no marks).

### Slice 4 — Persona awareness: guardian events flow up and the familiar narrates
- Protocol event stream (guardian → companion): detections, containments, audit
  appends.
- The persona (Gemma 4B) turns a containment into plain language ("I blocked
  curl reaching 203.0.113.9 after it read your SSH key"), surfaced in Scratch
  Post / Whispers / the bond.
- **Gate:** a guardian containment produces an accurate, audit-faithful
  persona-narrated message in grimalkin (tested against scripted events).

### Slice 5 — Heighten-only advice: the companion's brain raises the guardian's caution
- Companion → guardian advice channel into the `Advisor` seam (replaces
  `NullAdvisor` with a companion-fed advisor); Gemma assesses detections
  out-of-band (never in the hot path).
- **Gate:** advice can only move `ActAutonomously → RequirePermission`, never the
  reverse and never open a gate (property test); the daemon remains
  network-isolated.

### Slice 6 — One-install packaging (the unified UX)
- A unified setup that lays down the companion + guardian: binaries, systemd
  units, capabilities, `operator_uid`, control-socket permissions, and the
  polkit/systemd privilege handoff so the unprivileged companion can manage the
  privileged daemon's lifecycle.
- **Gate:** a clean-machine install brings up both halves, the deck connects, and
  the privileged end-to-end acceptance passes (now grimalkin's, run by the
  operator).

### Slice 7 — Public push + release (final gate)
- Push the folded, MIT, ships-disarmed repo; changelog/release notes; update the
  public README and screenshots.
- **Gate:** explicit operator authorization to push; the tree is MIT and ships
  disarmed; all prior slice gates green.

## 11. Gates summary

| Gate | Where | Hard stop |
|------|-------|-----------|
| Uniformly MIT | Slice 1 | No public push until the whole tree is MIT |
| Both halves build/run; history preserved; not pushed | Slice 1 | — |
| Chat works on Gemma 4B | Slice 2 | — |
| Deck drives the guardian; Python client cannot bypass the envelope; disarmed = dormant | Slice 3 | — |
| Containment → accurate persona narration | Slice 4 | — |
| Advice is heighten-only (property-tested); daemon stays isolated | Slice 5 | — |
| Clean-machine install → end-to-end acceptance passes | Slice 6 | — |
| Operator authorizes the push; MIT; ships disarmed | Slice 7 | The only public-exposure step |

## 12. Cross-cutting decisions and todos

- **Standalone egui deck (`familiar-ui`):** keep as an optional, minimal headless
  fallback (for running the guardian without grimalkin), not retired. grimalkin's
  Gradio deck becomes the primary control surface.
- **Crate rename** `familiar-*` → `guardian-*`: **deferred to Slice 6 (packaging)**,
  because it ripples into binary names, systemd units, scripts, and the
  `/run/familiar` runtime paths — best done in one pass with packaging, not
  half-done earlier. The crate names are internal; the product is "grimalkin /
  the guardian" regardless.
- **Socket/path naming:** `/run/familiar/*` paths reviewed for rename to a
  grimalkin-namespaced path during Slice 1/3.
- **Persona voice for security events:** the tone for guardian narration (calm,
  factual, the familiar's character) is a content decision refined in Slice 4.

## 13. Out of scope (separate tracks)

- **The guardian's own v0.2 hardening** — eBPF inline-drop (closing the post-SYN
  one-shot-exfil window), eBPF socket attribution (replacing the racy `/proc`
  scan), IPv6 sensing, `ProcessExit`-triggered auto-unblock, file/integrity
  detectors. Each is its own future spec, independent of this integration.
- **macOS/Windows guardian backends** — future, via the `platform` trait seam.
- **Upstreaming the vendored `rustables` fd-double-close patch** — a standalone
  loose end carried from the guardian's Plan B.

## 14. Open questions

- Final top-level directory names (`companion/` vs keeping Python at root with a
  `guardian/` subtree) — resolved in Slice 1 against the actual `grimalkin.py`
  structure.
- Whether the event stream is push (the daemon writes events as they happen) or
  pull (the companion polls a since-cursor) — resolved in Slice 4 against the
  single-client UDS model; pull is simpler and matches the current `GetAudit`.
