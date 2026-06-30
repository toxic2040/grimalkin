# Security

Grimalkin runs entirely on your own machine. There is no backend service and no
telemetry; the only outbound network traffic is to the local Ollama endpoint you
configure. The two surfaces worth thinking about are the files it ingests and the
Gradio web UI it serves.

## Reporting a vulnerability

Report privately through the repository's GitHub Security Advisories
("Report a vulnerability" under the Security tab). Please don't open a public
issue for anything exploitable. Include the version (`VERSION` in `grimalkin.py`),
how to reproduce, and the impact you have in mind.

## Deployment posture

- Grimalkin binds to `127.0.0.1` by default. Keep it there unless you have a
  reason not to.
- A non-loopback bind (e.g. `--host 0.0.0.0`) refuses to start unless
  `GRIM_AUTH_TOKEN` is set. Gradio has no native TLS, so put a reverse proxy
  (nginx, caddy, …) with a real certificate in front before exposing it.
- The Gradio file route is scoped to the avatar images only — it does not serve
  the database, the vault, or your sorted files.

## Document ingestion

Anything you drop into the chat, or that lands in the hunting-grounds folder, is
parsed by third-party loaders (`pypdf`, `unstructured`, `lxml`, `python-docx`).
These are the most exposed dependencies, because they process files you may have
received from someone else.

Two guards apply:

1. A size and type gate rejects unsupported extensions and files larger than
   `GRIM_MAX_INGEST_MB` (default 25 MB) before any parser runs.
2. Parsing happens in a separate worker process (`grimalkin_parse.py`) with an
   address-space cap (`GRIM_PARSE_MEM_MB`), a CPU-time cap, and a wall-clock
   timeout (`GRIM_PARSE_TIMEOUT`). A file that hangs, loops, or tries to exhaust
   memory takes down the worker, not Grimalkin; the file is simply skipped.

Isolation reduces blast radius but is not a substitute for patched parsers. Keep
the dependencies current.

## Guardian threat model

The Rust guardian is an optional local Linux component. It is not a remote
service. It senses two things — new outbound connections and reads of configured
sensitive paths — and when a detector links them into likely exfiltration it
proposes a reversible containment action (an nftables drop, or a cgroup freeze)
that a human approves before it takes effect. Its security boundary is the local
machine account model plus Linux capabilities.

### Components and privilege

- `familiar-daemon` owns the supervisor, the audit chain, the nftables rules, and
  the cgroup freezer. It runs as uid 0 but holds only `CAP_NET_ADMIN`
  (`AmbientCapabilities` and `CapabilityBoundingSet` in the unit), because the
  block actuator needs to edit the firewall and nothing else needs more.
- `familiar-fanotify-helper` is the only component with `CAP_SYS_ADMIN`, isolated
  to that one small binary. It decides no policy and installs no containment; it
  only streams file-read events to the daemon over a Unix socket.
- Both units run with `NoNewPrivileges=yes`, `ProtectSystem=strict`, and
  `IPAddressDeny=any`: neither process can gain privileges, write outside its
  declared state directories, or send IP traffic of its own.

### Control socket

The control deck talks to the daemon over a newline-delimited JSON Unix socket.
Only the configured operator uid or root may drive it: the daemon checks the peer
with `SO_PEERCRED` on every connection, and as defense in depth the socket is
created mode `0600` and chowned to the operator uid, so the filesystem rejects
everyone else before the credential check even runs. Each JSON line is
length-capped and read incrementally, so neither the operator nor the root helper
can exhaust memory with one oversized frame.

Control authority is split further by effect. The configured operator uid may read
status/audit state, enable sensor/detector capabilities, and deny prompts. Root is
required for privileged posture changes or actuation: arm/disarm the guardian,
enable or disable actuator capabilities, grant a prompt, or unblock a contained
destination. This is deliberate for the public build — malware already running as
the desktop user should not be able to
switch the guardian off through the sanctioned IPC.

### Safety properties

- Dormant by default. Nothing is sensed or contained until the guardian is armed;
  disarmed is the persisted default.
- Disarm is a full teardown. It reverses every containment the daemon installed —
  flushing the block chain, thawing any frozen process, and deleting the dedicated
  nftables table — so lifting the guardian never leaves a rule or a suspended
  process behind.
- Capability toggles are physical and fail-closed. Every sensor, detector, and
  actuator is a named capability that defaults off; an unknown or missing one reads
  as disabled, and the switch is a registry entry the model cannot override. Every
  toggle is written to the audit chain.
- The audit log is a tamper-evident hash chain. On startup the persisted chain is
  re-verified; a tampered or truncated log is rotated aside and flagged rather than
  silently trusted or appended to.

### Boundary

This boundary does not claim to defeat a fully compromised root account or a
kernel-level attacker, and it is not designed to withstand malware that has
already achieved operator-uid code execution and specifically targets the
guardian. Such a process cannot disarm, unblock, or grant through the IPC — those
need root — but it can deny the containment prompts meant to stop it, and it can
still read whatever files the user can read or abuse the Python/Gradio surface if
the UI is exposed beyond loopback. Even then the detection still fires and the
audit chain still records it. The guardian defends against exfiltration and
against unprivileged containment, not against an attacker who already owns the
account it runs for.

Current v0.1 limits:

- Network sensing and blocking are IPv4/TCP only.
- `FreezeProcess` is implemented end to end, including the active-freeze tracking
  that lets disarm thaw it, but no v0.1 detector proposes it yet. Before one does,
  the freeze path still needs process-identity revalidation to avoid pid-reuse
  races.
- The control and helper protocols cap each JSON line, but a local privileged peer
  can still deny service by flooding connections or events.

## Dependencies

`requirements-lock.txt` is the canonical install set, with hashes:

```bash
pip install --require-hashes -r requirements-lock.txt
```

`requirements.txt` holds the direct dependencies and minimum safe version floors.
After changing it, regenerate the lock and audit the result:

```bash
pip-compile --generate-hashes --allow-unsafe \
    --output-file requirements-lock.txt requirements.txt
pip-audit -r requirements-lock.txt
```

### Watch list

These parse untrusted input or terminate the network connection; check them first
when a new advisory lands:

- `pypdf`, `unstructured`, `lxml`, `python-docx` — document parsers
- `gradio`, `starlette`, `python-multipart`, `uvicorn` — the web layer
- `aiohttp`, `requests`, `urllib3` — HTTP clients

### Known residual

- `gradio` 6.12.0 — PYSEC-2026-211, a weak hash in the local Audio cache key
  handler. No fixed release is available yet. The issue is local-only and rated
  hard to exploit; the loopback default and the auth-token requirement on network
  binds limit the exposure. Revisit when a patched Gradio ships.
