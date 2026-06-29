# familiar-linux

The Linux implementation of the `familiar-platform` seam — the guardian's body.
It satisfies the same `Sensors` / `Actuators` / `Notifier` traits the v0.1
testkit fakes do, so the deterministic core and the supervisor loop are unchanged.

## Why a separate crate

`familiar-platform` stays a dependency-free trait crate so the core's "compiles
and tests with no platform adapter present" portability seam holds. All the OS
machinery and the heavier dependencies (`nfq`, `libc`) live here.

## Modules

- `nft` — the dedicated `inet familiar` nftables table. `ensure_table` +
  `block_outbound` install a reversible DROP rule through the `nft` userspace
  binary; `delete_table` reverses everything in one operation. `install_queue_rule`
  adds the sense chain that diverts outbound TCP to NFQUEUE.
- `nfqueue` — the NFQUEUE reader (`nfq`, pure Rust). Senses only: it ACCEPTs
  every packet and forwards connection-opening SYNs. The SYN filter lives in the
  parser, so this is new-connection sensing, not per-packet noise.
- `attribution` — best-effort `/proc` socket → PID mapping. Racy by nature; an
  unattributable connection surfaces as pid 0, never a guess.
- `cgroup` — the cgroup-v2 freezer: move a pid into a daemon-owned child cgroup
  and write `cgroup.freeze`. Reversible via `thaw`.
- `sensors` / `actuators` / `notifier` — the three trait impls. `LinuxSensors`
  merges NFQUEUE outbound events with FileRead events from the privileged helper;
  `LinuxActuators` installs the reversible block and the freeze; `LinuxNotifier`
  logs to journald and, optionally, the desktop.
- `wire` — the `FileReadEvent` JSON type shared (by mirroring) with the helper.

## The one unsafe invariant

This crate keeps `#![forbid(unsafe_code)]`. Every `unsafe` line in the whole
workspace lives in the separate `familiar-fanotify-helper` binary, which holds
`CAP_SYS_ADMIN` and does nothing but watch files and stream events. The daemon
holds only `CAP_NET_ADMIN`.

## Tests

Integration tests under `tests/` exercise the real kernel inside a private
user+net namespace (`unshare -Urn`); they self-exec into one and skip cleanly if
unprivileged user namespaces are unavailable. The fanotify path needs real
`CAP_SYS_ADMIN` and is covered by the gated privileged test in
`familiar-fanotify-helper` plus `scripts/run-privileged-acceptance.sh`.
