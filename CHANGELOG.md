# Grimalkin — Changelog

---

## Unreleased

### Security

- Mark retrieved vault context and conversation summaries as untrusted reference
  data, frame document text separately from the current question, and instruct
  the model not to follow data-borne commands or role changes.
- Raise the Gradio floor to 6.15.0 and SoupSieve to 2.8.4, then refresh the
  hash-pinned install set past the current fixed-version advisories.

---

## v5.1.0 — Companion + Guardian

Two-part product: the local AI companion and a privileged Linux security
guardian, shipped under one MIT tree.

### Guardian (new)

| Feature | Detail |
|---------|--------|
| Rust guardian under `guardian/` | Optional, opt-in, dormant-by-default daemon (NFQUEUE + fanotify + nftables + cgroup freeze path) |
| Master arm switch | Persisted armed/disarmed; disarmed = no sensing infrastructure |
| Control protocol | NDJSON Unix socket with `SO_PEERCRED` + mode `0600`; root required for arm/disarm, grants, unblock, privileged actuators |
| Standalone control deck | egui deck in `guardian/` drives the socket; Gradio Control Deck remains **posture-only** (known residual — see below) |
| Threat model | Documented in `SECURITY.md` + `guardian` docs |

### Companion

| Change | Detail |
|--------|--------|
| Default model | `gemma4:12b-it-qat` with runtime model swap in Settings |
| Shadow backend removed | Dead parallel `grimalkin_core` path cut; single redaction engine |
| Model eval harness | `eval/` for base-model swap decisions |
| STT | Local speech-to-text with engine fallbacks |
| Scheduler | Background scheduler opt-in |

### Known residuals (documented, not fixed in this tag)

These were pre-push flags; they ship as explicit limits, not silent gaps.

1. **Gradio Control Deck is posture-only.** It does not open the guardian
   control socket. Arm, capability toggles, and prompts use the standalone
   Rust deck. A Guardian card on the Gradio deck states this plainly.
2. **Same-uid DoS / autonomous-block shape.** Once root has armed the
   guardian and enabled sensing, a process running as the operator uid can
   (a) deny containment prompts and (b) trigger high-confidence
   read-then-outbound autonomous nft blocks against itself or peers. The
   root-only IPC barrier does not cover that shape. Documented under
   SECURITY.md Boundary. Defeat requires not arming on a fully compromised
   operator account — out of v0.1 scope.
3. **Parser / dep residuals from 5.0.3.** Isolated parse workers and
   hash-pinned lock are in; Gradio PYSEC-2026-211 remains until upstream
   ships a fix. Loopback default + auth token on non-loopback binds.

### License

Companion and guardian are MIT. The former GPL netlink dependency was
removed during fold-in; nftables ops go through the `nft` binary.

---

## v5.0.3 — Ingestion Isolation & Dependency Refresh

### Security

| Fix | Detail |
|-----|--------|
| Isolated document parsing | Files are now parsed in a separate worker process (`grimalkin_parse.py`) with an address-space cap (`RLIMIT_AS`), a CPU-time cap, and a wall-clock timeout. A hostile or malformed file can no longer hang or exhaust the main process; on any failure ingestion returns no chunks. |
| Ingestion gate | Unsupported file types and files over `GRIM_MAX_INGEST_MB` (default 25 MB) are rejected before any parser runs. |
| Dependency refresh | Regenerated the hash-pinned lock and raised security floors, moving `pypdf` (10 CVEs), `lxml`, `starlette`, `python-multipart`, `langchain-core`, `langchain-text-splitters`, `langsmith`, `urllib3`, `idna`, and `aiohttp` past known advisories. |
| Install policy | `requirements-lock.txt` is now the canonical hash-verified install set; added `SECURITY.md` with the parser CVE-watch list and lock-refresh procedure. |

### Config

| Variable | Default | Detail |
|----------|---------|--------|
| `GRIM_MAX_INGEST_MB` | 25 | Max file size accepted for parsing |
| `GRIM_PARSE_TIMEOUT` | 60 | Wall-clock seconds per file in the parse worker |
| `GRIM_PARSE_MEM_MB` | 1024 | Address-space cap for the parse worker |

### Tests

| Change | Detail |
|--------|--------|
| Ingestion tests | Added coverage for the size/type gate, the worker JSON contract on a real file, and fail-closed behaviour under a memory cap too small to parse |

### Known residual

- `gradio` 6.12.0 carries PYSEC-2026-211 (weak hash in the local Audio cache key handler). No fixed release exists yet; the local-only, high-complexity issue is mitigated by the loopback-default bind and the auth token required on network binds. Tracked in `SECURITY.md`.

---

## v5.0.2 — Security Hardening

### Security

| Fix | Detail |
|-----|--------|
| Gradio file serving | Restricted `allowed_paths` and static paths to avatar image files instead of the whole app directory |
| Non-loopback launch | Refuses network binds without `GRIM_AUTH_TOKEN`; warning now states the auth requirement |
| Auth comparison | Uses constant-time token comparison |
| Pyre/Loom HTML | Escapes attacker-controlled filenames and entity names in HTML-rendered surfaces |

### Tests

| Change | Detail |
|--------|--------|
| Security regression tests | Added coverage for file-serving scope, non-loopback auth enforcement, Pyre escaping, and Loom fallback escaping |

---

## v5.0.1 — Privacy Control Deck

### New Features

| Feature | Description |
|---------|-------------|
| **🛡️ Control Deck** | Live local posture cards for network endpoint, voice adapters, memory store, audit trail, file access, and source state |
| **Push-to-talk hooks** | Optional local STT/TTS command templates with default microphone temp-file cleanup |
| **Local voice adapter** | Repo-local command wrapper for Whisper/Vosk STT and Piper/espeak/flite/spd-say TTS |
| **Metadata audit trail** | Local action metadata is recorded without prompts, transcripts, or file contents |
| **Qwen3 no-think handling** | Appends `/no_think` for Qwen3 models and strips visible think artifacts from responses |

### Docs and Tests

| Change | Detail |
|--------|--------|
| Voice adapter docs | README and `.env.example` document `GRIM_STT_COMMAND`, `GRIM_TTS_COMMAND`, and `GRIM_KEEP_VOICE_AUDIO` |
| Control helper tests | Added tests for no-think handling, command status, audit writes, HTML escaping, and the local voice adapter |

---

## v4.1 — The Mirror Wakes

**The Mirror now has a tab.** It was documented. Now it exists.

### New Features

| Feature | Description |
|---------|-------------|
| **🪞 Mirror tab** | View the latest reflection and weave new ones on demand — no longer scheduler-only |
| **⚙️ Settings tab** | Change familiar name and address title directly in the UI |
| `mirror` command | Read the latest Mirror reflection from Scratch Post |
| `address` command | Change how Grimalkin addresses you (`address captain`, `address old friend`, etc.) |
| Opening line | Random flavor quote shown on each session start |
| 7 new easter eggs | `meow`, `purr`, `sleep`, `feed me`, `who made you`, `bad cat`, `thank you` |

### Bug Fixes

| Fix | Detail |
|-----|--------|
| Dependency pre-flight check | Missing packages now print a clean install message and exit instead of a raw traceback |
| `describe_node` hardcoded "Seven" | Thread count now reflects actual relationships |
| `find_clusters` hardcoded intro | No longer says "Seventeen names dance" regardless of data |
| `generate_weekly_reflection` system prompt | Now uses current `pet_name` instead of hardcoded "Grimalkin" |
| `OllamaEmbeddings` import | Tries `langchain_ollama` first, falls back to `langchain_community` for forward compatibility |
| `PERSONA_SYSTEM` global removed | Was a stale module-level variable; replaced with inline `build_persona()` call at usage site |

---

## v4.0 — The Veil Lifts

The Loom came alive. The Mirror began forming. The cat got a name.

### New Features

| Feature | Description |
|---------|-------------|
| **🪞 The Mirror** | Weekly reflections generated from vault activity and stored in `reflections` table |
| **`pet_name` setting** | Rename your familiar with `name <new_name>` — persists across sessions |
| **`build_persona(name)`** | Dynamic system prompt builder using current familiar name |
| **🕸️ The Loom** | Force-directed knowledge graph visualization with Plotly + HTML fallback |
| `spring_layout()` | Pure NumPy force-directed layout — no networkx dependency |
| `describe_node()` | Entity deep-dive: type, sightings, all connected threads |
| `find_clusters()` | Surface the most densely connected entity pairs |
| `export_loom_markdown()` | Dump the full web to a markdown file in sorted/ |
| `merge_entity()` | Canonicalize duplicate entities, dedup relationships |
| `set_entity_importance()` | Flag entities as important (★ in listings) |
| `forget_entity()` | Remove an entity and all its relationships |
| `recall` command | Cross-source synthesis: files + graph + past reflections |
| `proactive_whispers()` | Bond ≥ 60 unlocks proactive entity insights in Whispers |
| `migrate_v4()` | Adds `reflections` table and `importance` column via safe ALTER TABLE |

---

## v3.0 — The Pyre and The Web

Fire and memory. Files could be destroyed. Entities could be known.

### New Features

| Feature | Description |
|---------|-------------|
| **🔥 The Pyre** | Ritual file deletion: 3-step ceremony, bond gate (≥ 30), name confirmation |
| 7-day ash cooling | Files move to `sorted/PYRE/` and sit for 7 days before permanent deletion |
| `unburn` command | Rescue files from the Pyre before cremation |
| **Knowledge Graph** | Entity + relationship extraction via LLM during nightly groom |
| `entities`, `stats` commands | Graph visibility from Scratch Post |
| Nightly groom | Automated tag/note/entity extraction on a 24h scheduler |
| `groom` command | Trigger nightly groom manually |
| Custom categories | User-defined sort categories stored in settings |
| `migrate_v3()` | Adds `entities`, `relationships` tables; `burned_at` column on `file_memory` |
| `PYRE` + `DUPLICATES` folders | Created automatically by `ensure_dirs()` |

---

## v2.1 — Hybrid Search

The vault got smarter. Keyword blindness was fixed.

### Changes

| Change | Description |
|--------|-------------|
| `keyword_search()` | Multi-term OR matching — "ARGUS whitepaper" now finds files with either word |
| `hybrid_vault_rag()` | Merged FAISS semantic + keyword results with score boosting |
| `ingest` command | Discover and index orphan files already present in sorted/ |
| `index` command | Re-index files that failed or were skipped on first pass |
| Removed `route_vault_query` | Was a passthrough wrapper; inlined into callers |

---

## v2.0 — The Full Rebuild

Single file. One database. One cat. Everything that came before was practice.

**Source:** `grimalkin.py` (722 LOC) + `indexer.py` (67 LOC) → `grimalkin.py` (983 LOC)

### Bug Fixes

| # | Fix | Severity |
|---|-----|----------|
| 1 | UTC/local time mismatch in `generate_whispers` | HIGH |
| 2 | 6 bare `except:` → `except Exception:` across all modules | HIGH |
| 3 | Duplicate files moved to `sorted/DUPLICATES/` instead of silently deleted | HIGH |
| 4 | Qwen3 `<think>` tag stripping in `scrub_corporate()` | HIGH |
| 5 | `_call_openai` URL double-`/v1` bug | MEDIUM |
| 6 | `on_upload` hashes source before copy | MEDIUM |
| 7 | `log_interaction` bond bump now atomic within `_DB_LOCK` | MEDIUM |
| 8 | WAL + `synchronous=NORMAL` set per-connection, not just at init | MEDIUM |
| 9 | Symlinks skipped in `run_hunt` | LOW |
| 10 | `hash_file` size gate: files >500 MB use fast fingerprint | LOW |

### Structural Changes

- Lazy FAISS index loading — loads on first query, not at startup
- `DEFAULTS` dict: all magic numbers in one place
- `INDEXABLE_EXTENSIONS` set: explicit declaration of what the vault can read
- Bond system: every interaction increments bond (0–100)
- Whispers: daily briefings with bond-gated detail levels
- `idx_fm_indexed` and `idx_briefing_date` indexes for query performance
- `_VS_LOCK` as `RLock` to prevent deadlock on nested lock acquisition
