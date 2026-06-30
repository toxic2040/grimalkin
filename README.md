![Grimalkin](grimalkin_avatar.jpg)

**Your private AI familiar and local security guardian — fully local, fully yours**

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-000000?logo=faiss&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-FF8C00?logo=gradio&logoColor=white)](https://gradio.app)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![X](https://img.shields.io/badge/X-%40toxic2040-000000?logo=x&logoColor=white)](https://x.com/toxic2040)

---

**~5200 LOC companion · Rust guardian · 100% offline · Ollama + FAISS + Knowledge Graph + Local Voice**

Grimalkin is two things: a local AI familiar that sorts your files, indexes them for instant Q&A, builds a knowledge graph of everything it finds, and develops a personality that sharpens the longer you spend together; and a privileged Rust guardian that watches for hostile activity and contains threats within a strict, auditable authority envelope. No cloud, no API keys, no network tracking. One database, one bond, one cat.

## Latest: v5.0.3

**Ingestion isolation** parses documents in a capped worker process, rejects
unsupported or oversized files before parsing, and keeps the hash-pinned
dependency lock above the current security floors.

**Security hardening** narrows Gradio file serving to avatar assets only, requires `GRIM_AUTH_TOKEN` for non-loopback binds, and escapes attacker-controlled filenames/entity names in Pyre/Loom HTML surfaces.

**Privacy Control Deck** shows the live local posture: Ollama endpoint, voice adapter readiness, plaintext memory footprint, metadata-only audit trail, file access mode, and git source state.

**Local voice dock templates** wire Scratch Post to local STT/TTS commands. The repo-local adapter detects Whisper/Vosk for transcription and Piper/espeak/flite/spd-say for speech, with browser microphone temp files discarded by default.

**Qwen3 local path** adds `/no_think` handling and visible think-artifact cleanup for Qwen3-family Ollama models.

**PII redaction (options 1+2)**: deterministic regex+validators + optional hybrid with an explicitly configured local GLiNER-style model, applied to user prompts and document chunks before they reach the LLM or FAISS. Stable scoped placeholders stay redacted by default; local rehydration is opt-in with `GRIM_PII_REVEAL=true`. Config via `GRIM_PII_REDACTION=deterministic|hybrid|off`.

Prototype training script (`scripts/train_gemma_pii.py`) for small local Gemma-class models on synthetic PII redaction or persona data. Remote model downloads require `--allow-downloads`, and remote code loading requires `--trust-remote-code`.

## What It Does

**🐾 Scratch Post** — Chat with Grimalkin. Ask questions, run commands, use optional local push-to-talk, or just talk. It remembers.

**🏹 The Hunt** — Scans your Downloads folder, classifies files by type, copies them into organized folders, and indexes the contents for search. Manual trigger — no background watchers.

**🌙 Whispers** — Daily briefings on your vault: file counts, top categories, graph insights, bond level. Gets more detailed as your bond grows.

**📚 The Vault** — Hybrid search (FAISS semantic + keyword matching) over everything Grimalkin has indexed. Ask by content or by name — both work.

**🔥 The Pyre** — Ritual file deletion with a 3-step confirmation ceremony. Bond-gated (must reach level 30). Burned files cool in ashes for 7 days before true cremation. Reversible until then.

**🕸️ The Loom** — Interactive knowledge graph visualization. Entities and relationships extracted from your files, rendered as a force-directed network. Filter by type, search by name, export to markdown.

**🪞 The Mirror** — Weekly reflections generated from vault activity and graph growth. View past reflections and weave new ones directly from the tab. Grimalkin develops memory across time.

**🛡️ Control Deck** — Local privacy posture for network endpoint, voice adapter readiness, plaintext memory, metadata-only audit trail, file access mode, and source state.

**⚙️ Settings** — Name your familiar, set your title, upload an avatar, toggle sandbox mode, serious mode, graph injection, and custom categories. All runtime — no restarts.

## Stack

| Component | Role |
|-----------|------|
| Python 3.10+ | Runtime |
| [Ollama](https://ollama.com) | Local LLM inference (qwen3:8b by default, configurable) |
| FAISS | Vector similarity search |
| LangChain + langchain-ollama | Document loading, text splitting, chat integration |
| Gradio 6.x | Web UI |
| SQLite (WAL) | Persistent storage — files, entities, relationships, settings |
| Plotly | Interactive graph visualization in The Loom |

## The Guardian (Linux)

grimalkin includes a local security guardian: a separate, privileged Rust daemon
that watches for hostile activity against your files and system and contains it
within a strict, auditable authority envelope — reversible, fail-closed
containment for high-confidence threats, and an explicit ask for anything
ambiguous. The security guarantees live in a small, deterministic, memory-safe
harness, not in a model.

The guardian is **off until you turn it on** and is Linux-only (it uses NFQUEUE,
nftables, cgroup-v2, and fanotify). It persists a master armed/disarmed switch
and defaults to disarmed; in that state the daemon does not install the NFQUEUE
divert rule or bind the file-read helper socket. The Unix-socket protocol
exposes the master switch, capability toggles, prompts, active blocks, sensor
health, and the hash-chained audit tail. The standalone guardian deck can drive
that protocol now; wiring grimalkin's Gradio Control Deck to it remains a later
integration slice.

It lives under `guardian/` and is built separately:

```bash
cd guardian
cargo build --release
```

The whole project — companion and guardian — is MIT licensed.

## Quick Start

### 1. Install Ollama

Download from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull qwen3:8b             # reasoning model
ollama pull nomic-embed-text     # embedding model
```

### 2. Clone and install

```bash
git clone https://github.com/toxic2040/grimalkin.git
cd grimalkin
pip install --require-hashes -r requirements-lock.txt   # reproducible, hash-pinned
cp .env.example .env
```

`requirements-lock.txt` is the canonical, hash-verified install set. `requirements.txt`
holds the human-readable direct dependencies and minimum safe version floors; regenerate
the lock from it with `pip-compile` after changing a dependency (see `SECURITY.md`).

### 3. Run

```bash
python grimalkin.py
```

Open **http://localhost:7860** in your browser. The cat is waiting.

### Optional: local voice adapters

The Scratch Post voice dock is disabled until local command templates are set. Grimalkin does not ship cloud STT/TTS bindings. The repo includes a small local adapter at `scripts/grim_voice.py`; it detects Whisper/Vosk for STT and Piper/espeak/flite/spd-say for TTS.

```bash
# STT receives {audio}; transcript text is written to {out}.
export GRIM_STT_COMMAND='"{python}" "{app}/scripts/grim_voice.py" stt --audio "{audio}" --out "{out}"'

# TTS receives response text through {text_file}; playable audio is written to {out}.
export GRIM_TTS_COMMAND='"{python}" "{app}/scripts/grim_voice.py" tts --text-file "{text_file}" --out "{out}"'

# Default false: discard browser microphone temp files after transcription.
export GRIM_KEEP_VOICE_AUDIO=false

# Optional engine/model overrides:
export GRIM_STT_ENGINE=auto
export GRIM_TTS_ENGINE=auto
export GRIM_WHISPER_MODEL=base
export GRIM_PIPER_MODEL=/path/to/piper-voice.onnx
```

## Scratch Post Commands

| Command | What it does |
|---------|-------------|
| `hunt` | Scan Downloads, sort and index new files |
| `whispers` | Generate today's briefing |
| `groom` | Run the nightly groom cycle manually (tags, notes, entity extraction) |
| `index` | Re-index any unindexed files in file_memory |
| `ingest` | Discover orphan files in sorted/ not yet tracked |
| `bond` | Check your bond level |
| `stats` | Vault statistics |
| `entities` | List top entities in the knowledge graph |
| `mirror` | Read the latest Mirror reflection |
| `categories` | List all file categories |
| `unburn <hash>` | Restore a file from the Pyre |
| `name <new_name>` | Rename your familiar |
| `address <title>` | Change how Grimalkin addresses you |
| `merge "A" "B"` | Canonicalize two entity names into one |
| `important Name` | Flag an entity as important |
| `forget Name` | Delete an entity and all its relationships |
| `recall Name` | Deep cross-source recall (files + graph + reflections) |
| `help` | Show all commands |

## How It Works

**File sorting:** The Hunt scans `~/Downloads` for new files (by SHA-256 hash), classifies them by extension into categories (FINANCIAL, PERSONAL, RESEARCH, MEDIA, MISC), copies them to `sorted/<CATEGORY>/`, and indexes supported formats into FAISS.

**Hybrid search:** Vault queries run both FAISS semantic search and keyword matching against filenames, tags, and notes. Results are merged and boosted — files matching by name get priority, but semantic matches still surface. Graph connections are injected into context when relevant.

**Knowledge graph:** The nightly groom extracts entities (people, orgs, dates, locations, amounts, topics) and relationships from file contents via LLM. These populate the `entities` and `relationships` tables, visualized in The Loom.

**Bond system:** Every interaction increments your bond level (0–100). Higher bond unlocks features: Pyre access at 30, graph stats in Whispers at 40, proactive insights at 60. Bond level also shapes personality — at Stranger she's aloof and feline; by Bonded she's sharp, opinionated, and fully present.

**Personality:** Tiered persona system with bond-scaled voice, situational mood injection (time of day, vault state, burn history), and anti-corporate scrubbing. She progresses from cat to companion — not by getting bigger, but by waking up.

**The Mirror:** Weekly reflections are generated automatically during the nightly groom cycle, or on demand from The Mirror tab. Each reflection synthesizes vault activity, graph growth, and top entities into a 2–3 sentence entry in Grimalkin's voice, stored permanently in the database.

**Control Deck:** The deck renders the current local posture: Ollama endpoint classification, STT/TTS command availability, SQLite/vector-store footprint, audit event count, file access mode, and git source state. The audit trail stores action metadata only, not prompts, transcripts, or file contents.

## Supported File Types

**Full indexing:** PDF, TXT, MD, HTML, CSV, DOCX, DOC, Python, JavaScript, TypeScript, Shell, C/C++, Java, Go, Rust, Ruby, Perl, Lua, Swift, Kotlin, TOML, JSON, YAML, XML, INI, CFG, RTF, LOG

**Sorted but not indexed:** Images (JPG, PNG, GIF), audio/video (MP3, MP4, WAV), archives

## File Structure

```
grimalkin/
├── grimalkin.py              # Main application (~5200 LOC)
├── grimalkin_core.py         # Engine — config, DB, bond, persona, search
├── grimalkin_features.py     # Feature handlers — hunt, groom, pyre, mirror
├── grimalkin_interfaces.py   # Gradio UI builder
├── scripts/grim_voice.py     # Local STT/TTS command adapter
├── test_grimalkin.py         # Test suite
├── grimalkin.jpg             # Hero image
├── grimalkin_avatar.jpg      # Avatar / social icon
├── grimalkin.db              # SQLite database (created on first run)
├── requirements.txt
├── LICENSE
├── guardian/                 # Rust security guardian (Linux-only, built separately)
│   ├── Cargo.toml
│   ├── crates/               # familiar-core, familiar-linux, familiar-daemon, ...
│   └── PROVENANCE.md
├── sorted/                   # Organized files (created on first run)
│   ├── FINANCIAL/
│   ├── PERSONAL/
│   ├── RESEARCH/
│   ├── MEDIA/
│   ├── MISC/
│   ├── PYRE/                 # Burned files awaiting cremation
│   └── DUPLICATES/
├── faiss_index/              # FAISS vector index (created on first run)
└── vault/                    # Reserved for future use
```

## Requirements

- Python 3.10+
- Ollama running locally with `qwen3:8b` and `nomic-embed-text`
- ~8 GB RAM recommended for `qwen3:8b`; larger local models need more
- Works on Linux, macOS, Windows (tested on Pop!_OS)

## License

MIT — do whatever you want with it.

---

*I sort your files, guard your vault, and judge you silently.*
*— Grimalkin*
