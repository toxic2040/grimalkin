# Familiar — Local Defensive Guardian: Design (v0.1)

Date: 2026-06-17
Status: approved design, ready for implementation planning
Lineage: ground-up Rust successor to the `grimalkin` Python familiar

## 1. Summary

A local guardian daemon that watches a machine for hostile activity against the
user's data, files, and system integrity, and responds within a strict authority
envelope: reversible, fail-safe containment on its own for high-confidence
threats, and an explicit ask for anything ambiguous or irreversible.

The security guarantees live in a small, auditable, deterministic harness — not
in a model. A language model is an optional advisor that informs decisions but
never holds a gate. The harness is portable by construction: a platform-neutral
core with thin per-OS adapters. Linux is the first target; Android is the next
slice on the same core.

This is a complete rewrite in Rust. The existing Python `grimalkin` is kept as a
working, hardened reference to mine for persona, policy logic, and the
quarantine model — not as code to extend.

## 2. Locked decisions

- **North star:** trustworthy guardian first (v0.1), custom-trained model later
  (v0.2+), sequenced.
- **Substrate:** Linux PC daemon for v0.1; portable harness is a first-class
  goal; Android is the v0.2 port slice.
- **Threat model:** data exfiltration / privacy leaks, malicious files /
  software, integrity / tampering. These define what the guardian is *for*. They
  are three detection policies over one shared event spine, not three subsystems;
  v0.1 implements exfiltration first (see §5) and the other two follow on the
  same spine.
- **Authority envelope:** on high-confidence threats the guardian may take
  reversible, fail-safe containment autonomously and notify; anything ambiguous,
  or any irreversible action, requires explicit confirmation.
- **Every ability is gated:** each sensor, detector, and actuator is a named
  capability — default-off, fail-closed, user-toggleable, every toggle audited.
- **Architecture:** all-Rust, single language, portable core + platform
  adapters; Tauri UI targeting desktop now and Android later.
- **Model posture:** the model is an advisor behind an interface; the v0.1 spine
  runs rule-only with no model.
- **Distillation / RL:** out of scope for v0.1; a separate v0.3 spec, undertaken
  only once real use shows which behaviors a custom model must improve.

## 3. Open decisions (settle at implementation start)

- **Repo structure:** new local-only repo `familiar` (recommended — keep
  `grimalkin` intact as the hardened reference) vs. in-place rewrite of the
  `grimalkin` repo. New repos stay local-only until the user authorizes release.
- **v0.1 egress mechanism:** eBPF vs. nftables + netlink for outbound
  observation and blocking. Pick by implementation effort during planning; the
  `Sensors`/`Actuators` traits hide the choice from the core.
- **Advisor backend (deferred to v0.2):** candle (pure Rust) vs. llama.cpp-rs
  bindings.

## 4. Architecture

A Rust workspace of single-purpose crates. The invariant that everything
respects: **`familiar-core` makes no OS calls and runs no model.** It is pure
policy and bookkeeping, identical on every platform.

### 4.1 Crates

- **`familiar-core`** — the portable spine:
  - `events` — the normalized, platform-neutral event vocabulary
    (`FileEvent`, `ProcessEvent`, `NetEvent`, …). Adapters translate OS specifics
    into this; the core only ever sees this.
  - `capabilities` — the Capability Registry. Every sensor/detector/actuator
    registers as a named capability: default-off, fail-closed, persisted state,
    every toggle audited.
  - `policy` — the detectors expressed as rules over events, the decision
    engine, and the authority envelope that classifies each proposed action as
    reversible or irreversible.
  - `permission` — the permission protocol: request → user decision →
    grant/deny. A request that times out resolves to deny.
  - `audit` — an append-only, hash-chained, tamper-evident log.
  - `advisor` — an interface (trait) the core may call for fuzzy judgment or
    explanation. The advisor can inform a decision; it can never hold a gate.
- **`familiar-platform`** — the adapter seam: traits `Sensors`, `Actuators`,
  `Notifier`, with `linux/` implemented in v0.1 and `android/` stubbed for v0.2.
- **`familiar-advisor`** — the optional AI brain behind the `advisor` trait:
  local model, encrypted vector memory, persona. Pluggable; v0.1 ships a
  null advisor.
- **`familiar-ui`** — Tauri app (desktop now, Android later): the Control Deck of
  capability switches, live permission prompts, the audit viewer, and a status
  feed.
- **`familiar-daemon`** — the Linux host binary that wires core + linux adapter +
  advisor, runs least-privilege, and lives in the tray.

### 4.2 Control model — two gates in series, then the envelope

For any proposed response to an event:

1. **Capability gate** — is this sensor/detector/actuator enabled at all? Off by
   default; if off, nothing runs and nothing is recorded beyond the toggle
   history.
2. **Permission gate** — for an enabled capability, is the action permitted now,
   or pre-authorized to run autonomously?
3. **Authority envelope** — reversible and high-confidence: act, then notify.
   Irreversible or ambiguous: require explicit confirmation; deny on timeout.

Every gate is deterministic. The advisor, when present, is consulted between
detection and decision to supply confidence and a human-readable explanation; it
never moves a gate.

## 5. v0.1 scope boundary

v0.1 proves the entire loop end-to-end on Linux with the harness spine and a
single detector family fully wired. Ruthless YAGNI: one detector, no model.

### In scope

- `familiar-core` complete: events, capability registry, policy/decision engine,
  authority envelope, permission protocol, audit.
- `familiar-platform/linux`: one sensor family and its matching reversible
  actuators for the **exfiltration** detector — chosen as the v0.1 marquee
  because it exercises both a real sensor and a real reversible actuator, it is
  the strongest privacy-guardian demonstration, and `grimalkin` cannot do it at
  all today.
  - Sensors: outbound-connection observation, plus file-read observation on a
    user-configured sensitive-path set (e.g. `~/.ssh`, browser profiles,
    keyrings).
  - Reversible actuators: block/hold an outbound connection (e.g. a removable
    firewall rule) and/or freeze the offending process pending a decision.
- Rule-based exfil detection: a process reading a sensitive path and/or opening
  an outbound connection it has no established basis for. Confidence is
  rule-derived; no model.
- `familiar-ui` (Tauri, desktop): Control Deck with per-capability switches, live
  permission prompts, an audit viewer, and a status feed.
- `familiar-daemon`: wires it together, runs least-privilege.

### Deferred (explicitly, to hold the line)

- The malicious-file and integrity/tampering detectors. They are fast-follows on
  the same spine; the file detector largely ports `grimalkin`'s hunt, isolated
  parse, and Pyre quarantine logic.
- The AI advisor: local model serving, encrypted vector memory, persona, voice.
- The Android adapter (v0.2 slice).
- The distillation / RL custom-model pipeline (v0.3, separate spec).
- Any self-improvement loop.

## 6. Data and control flow

```
OS event
  → linux adapter normalizes into a core event
  → core event intake
  → enabled detectors evaluate (capability-gated; disabled ⇒ skipped)
  → detector proposes an action with a confidence
  → [advisor consulted for confidence/explanation, if present]  (never a gate)
  → authority envelope classifies the action
      reversible + high-confidence + permission allows
        → actuator acts → audit → notify
      irreversible or ambiguous or permission required
        → permission request to UI (timeout ⇒ deny)
            granted ⇒ act → audit → notify
            denied/timed-out ⇒ audit, no action
```

Every transition is appended to the hash-chained audit log, including
no-action outcomes, capability toggles, and permission decisions.

## 7. The guardian's own threat model

The guardian is the most privileged and most data-aware process on the machine,
so it must be designed not to become the attack surface it defends against.

- **Least privilege.** Isolate the privileged sensor/actuator work into the
  smallest possible component using Linux capabilities rather than full root;
  everything else runs unprivileged.
- **No self-egress.** The guardian makes no outbound network calls; the advisor
  and any model run locally. A privacy guardian that phones home is a
  contradiction.
- **Tamper-evidence and self-integrity.** The audit log is hash-chained; the
  guardian checks the integrity of its own binary and configuration and treats a
  mismatch as a high-severity event.
- **Fail-closed everywhere.** Capabilities default off; permission requests deny
  on timeout; an adapter or detector error degrades to no-action, never to an
  unguarded silent pass.
- **User sovereignty over capabilities.** The on/off gates physically cut a
  capability; they are not a model preference the system can override.
- **Small, reviewable core.** The security-critical surface is the deterministic
  Rust core, kept minimal and auditable; the large, fast-moving AI and UI code
  cannot hold a gate.
- **Encrypted at rest.** Any persistent memory (a v0.2 feature) is encrypted;
  the principle is stated now so it is not retrofitted.

## 8. Testing and verification

- **Core as pure logic.** Unit-test the decision engine and authority envelope
  with synthetic event streams and golden detect→decide→act traces.
- **Security invariants as first-class property tests:**
  - no irreversible action ever fires without an explicit grant;
  - a disabled capability runs no detector and no actuator;
  - a permission timeout resolves to deny;
  - an adapter/detector error degrades to no-action.
- **Adapter tests** against recorded and synthetic OS events.
- **Scenario fixtures** — scripted exfil attempts run in a throwaway namespace or
  container that the suite must detect, contain reversibly, and record. These
  red-team fixtures are the acceptance bar for v0.1.

## 9. Roadmap after v0.1

- **v0.2a** — file and integrity detectors on the same spine (port `grimalkin`'s
  quarantine and isolated-parse logic).
- **v0.2b** — the AI advisor: local model behind the `advisor` trait, encrypted
  vector memory, persona, voice.
- **v0.2c** — the Android adapter: `Sensors`/`Actuators`/`Notifier` against
  Android APIs, Tauri mobile UI, same core.
- **v0.3** — the custom-model pipeline (distillation, then RL on reward signals
  drawn from real harness behavior), as a separate spec. The self-improvement
  loop, if built, proposes policy and memory edits for user approval; weights
  stay frozen and versioned; any retrain is offline and gated by the v0.1
  scenario suite. Autonomous weight self-modification is explicitly excluded.

## 10. Success criteria for v0.1

- The full loop runs on Linux against a real exfil scenario: detect → classify →
  reversibly contain or ask → act → audit → notify.
- Every ability is visible and individually switchable in the Control Deck, and
  defaults off.
- The security-invariant property tests and the red-team scenario fixtures pass.
- No outbound network traffic originates from the guardian itself.
- The core compiles and its tests run with no platform adapter present, proving
  the portability seam holds for the Android slice.
