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
