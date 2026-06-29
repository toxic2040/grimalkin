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
service. Its security boundary is the local machine account model plus Linux
capabilities:

- `familiar-daemon` owns the supervisor, audit chain, nftables rules, and cgroup
  freezer. It runs as uid 0 with a bounded capability set because nftables and
  cgroup containment need elevated kernel authority.
- `familiar-fanotify-helper` is the only component with `CAP_SYS_ADMIN`. It does
  not decide policy or install containment; it only streams file-read events to
  the daemon over a Unix socket.
- The control deck talks to the daemon over a newline-delimited JSON Unix socket.
  The socket is owned by the configured operator uid and mode `0600`; the daemon
  also verifies peer credentials with `SO_PEERCRED`.

Control authority is split by effect. The configured operator uid may read
status/audit state, arm the guardian, enable capabilities, and deny prompts.
Root is required to lower protection or grant actuation: disarm, disable a
capability, grant a prompt, or unblock a contained destination. This is deliberate
for the public build: malware already running as the desktop user should not be
able to switch the guardian off through the sanctioned IPC path.

This boundary does not claim to defeat a fully compromised root account or a
kernel-level attacker. Same-uid malware can still read files the user can read,
interfere with unprivileged UI processes, or abuse the Python/Gradio surface if
the UI is exposed beyond loopback. The guardian's job is narrower: detect and
contain selected local exfiltration behavior, make control decisions auditable,
and keep broad kernel privileges out of the Python companion.

Current v0.1 limits:

- Network sensing and blocking are IPv4/TCP only.
- `FreezeProcess` is present as an actuator but no v0.1 detector proposes it.
  Before it is enabled, disarm needs active-freeze tracking and the freeze path
  needs process-identity revalidation to avoid pid-reuse races.
- The control and helper protocols cap each JSON line, but a local privileged
  peer can still deny service by flooding connections or events.

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
