# Operating the Familiar Linux body

Familiar runs as two least-privilege systemd units that talk over a `/run/familiar`
Unix socket:

- **`familiar-daemon`** — `CAP_NET_ADMIN` only. Runs the detect → decide →
  act/ask → audit → notify loop, installs the reversible nftables block rule,
  freezes processes via the cgroup-v2 freezer, persists state. No network egress
  of its own (`IPAddressDeny=any`).
- **`familiar-fanotify-helper`** — `CAP_SYS_ADMIN` only. A minimal binary that
  watches the configured sensitive paths with fanotify and streams `FileRead`
  events to the daemon. It is the only component with the broad capability and
  the only code in the project that uses `unsafe`.

Neither unit holds both capabilities. Splitting the privileged file-watch out of
the network daemon is the whole point of the design.

## Install

```bash
cargo build --release -p familiar-daemon -p familiar-fanotify-helper
sudo install -m755 target/release/familiar-daemon /usr/local/bin/
sudo install -m755 target/release/familiar-fanotify-helper /usr/local/bin/
sudo install -d /etc/familiar
sudo install -m644 systemd/familiar-daemon.service systemd/familiar-fanotify.service /etc/systemd/system/
# Write /etc/familiar/config.json (see "Configuration"); narrow the helper's
# watched prefixes in familiar-fanotify.service to your actual sensitive paths.
sudo systemctl daemon-reload
sudo systemctl enable --now familiar-fanotify familiar-daemon
```

## Configuration (`/etc/familiar/config.json`)

```json
{
  "sensitive_prefixes": ["/home/you/.ssh", "/home/you/.gnupg"],
  "established_dsts": ["10.0.0.0/8 hosts you trust, as exact IPs"],
  "link_window_ms": 5000,
  "permission_timeout_ms": 30000,
  "queue_num": 0,
  "tick_ms": 200,
  "state_dir": "/var/lib/familiar",
  "cgroup_root": "/sys/fs/cgroup/familiar.slice",
  "helper_socket": "/run/familiar/fileread.sock",
  "control_socket": "/run/familiar/control.sock",
  "operator_uid": 1000,
  "desktop_notify": false
}
```

## The capability model

Every sensor, detector, and actuator is a capability that is **default-OFF and
fail-closed**. v0.1 reads the enabled set from `state_dir/capabilities.json`
(written atomically). A disabled capability runs no sensor/detector/actuator
and records nothing beyond toggle history.

The authority envelope is unchanged from the core: a reversible, high-confidence
detection (a sensitive read linked to an outbound connection within
`link_window_ms`) is contained autonomously; anything ambiguous or unlinked is
routed to a human and **denied on timeout**. The advisor (NullAdvisor in v0.1)
can only ever heighten caution, never authorize.

## The control deck (familiar-ui)

The deck is a local egui app that talks to the daemon over
`/run/familiar/control.sock`. At runtime it can: toggle any capability (the
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

## Verifying it works

- Unprivileged, no host impact: `cargo test --workspace` (the netns tests
  self-isolate in `unshare -Urn`).
- Privileged end-to-end (fanotify + autonomous containment in a throwaway netns):
  `sudo ./scripts/run-privileged-acceptance.sh`.

## Known limitations (v0.1)

- **NFQUEUE senses, it does not block.** The NFQUEUE sense path accepts the
  triggering SYN before the block rule lands. A one-shot exfil — read a secret,
  open one connection, send a small payload — can complete inside that window;
  blocking subsequent traffic is not sufficient against a single short burst.
  eBPF inline-drop is the v0.2 fix.
- **`/proc` attribution is racy.** A process that exits between its SYN and the
  scan is unattributable (surfaced as pid 0). eBPF socket attribution is v0.2.
- **IPv4 only — IPv6 egress is unmonitored.** The NFQUEUE sense path parses only
  IPv4 packets. On a host with IPv6 enabled, IPv6 TCP egress generates no event
  and no audit entry; it is an unmonitored exfil path in v0.1 until v0.2 adds
  IPv6 sensing. Do not rely on Familiar for containment on dual-stack hosts in v0.1.
- **Freeze scope.** Freezing an arbitrary third-party PID requires the daemon to
  move it into a cgroup it owns; that needs cgroup write access to the target's
  subtree (the unit gets a delegated subtree via `Delegate=yes`).
- **Automatic un-blocking** (on process exit, or a user "unblock") is Plan C; v0.1
  installs reversible blocks and exposes `reverse_all`, but does not yet lift
  them on its own.
- **NFQUEUE 0 only** for the divert rule's queue number is not a limitation — the
  divert uses the `nft` binary and supports any `queue_num`; the rustables block
  path is unaffected.

These are recorded in full in the Plan B spike findings
(`~/work/sandbox/familiar-plan-b-spike/FINDINGS.md`).
