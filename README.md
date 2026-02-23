![Grimalkin](grimalkin.jpg)

**Your private AI familiar — fully local, fully yours**

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-000000?logo=faiss&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-FF8C00?logo=gradio&logoColor=white)](https://gradio.app)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

**Single-file · ~1760 LOC · 100% offline · Ollama + FAISS + Knowledge Graph**

Grimalkin is a local AI familiar that sorts your files, indexes them for instant Q&A, builds a knowledge graph of everything it finds, and judges you silently. No cloud, no API keys, no tracking. One Python file, one database, one cat.

## What It Does

**🐾 Scratch Post** — Chat with Grimalkin. Ask questions, run commands, or just talk. It remembers.

**🏹 The Hunt** — Scans your Downloads folder, classifies files by type, copies them into organized folders, and indexes the contents for search. Manual trigger — no background watchers.

**🌙 Whispers** — Daily briefings on your vault: file counts, top categories, graph insights, bond level. Gets more detailed as your bond grows.

**📚 The Vault** — Hybrid search (FAISS semantic + keyword matching) over everything Grimalkin has indexed. Ask by content or by name — both work.

**🔥 The Pyre** — Ritual file deletion with a 3-step confirmation ceremony. Bond-gated (must reach level 30). Burned files cool in ashes for 7 days before true cremation. Reversible until then.

**🕸️ The Loom** — Interactive knowledge graph visualization. Entities and relationships extracted from your files, rendered as a force-directed network. Filter by type, search by name, export to markdown.

**🪞 The Mirror** — Weekly reflections generated from vault activity and graph growth. Grimalkin develops memory across time.

## Stack

| Component | Role |
|-----------|------|
| Python 3.10+ | Runtime |
| [Ollama](https://ollama.com) | Local LLM inference (qwen3:8b) |
| FAISS | Vector similarity search |
| LangChain | Document loading + text splitting |
| Gradio 6.x | Web UI |
| SQLite (WAL) | Persistent storage — files, entities, relationships, settings |
| Plotly *(optional)* | Interactive graph visualization in The Loom |

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
pip install -r requirements.txt
```

### 3. Run

```bash
python grimalkin.py
```

Open **http://localhost:7860** in your browser. The cat is waiting.

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
| `merge "A" "B"` | Canonicalize two entity names into one |
| `important Name` | Flag an entity as important |
| `forget Name` | Delete an entity and all its relationships |
| `recall Name` | Deep cross-source recall (files + graph + reflections) |
| `help` | Show all commands |

## How It Works

**File sorting:** The Hunt scans `~/Downloads` for new files (by SHA-256 hash), classifies them by extension into categories (FINANCIAL, PERSONAL, RESEARCH, MEDIA, MISC), copies them to `sorted/<CATEGORY>/`, and indexes supported formats into FAISS.

**Hybrid search:** Vault queries run both FAISS semantic search and keyword matching against filenames, tags, and notes. Results are merged and boosted — files matching by name get priority, but semantic matches still surface. Graph connections are injected into context when relevant.

**Knowledge graph:** The nightly groom extracts entities (people, orgs, dates, locations, amounts, topics) and relationships from file contents via LLM. These populate the `entities` and `relationships` tables, visualized in The Loom.

**Bond system:** Every interaction increments your bond level (0–100). Higher bond unlocks features: Pyre access at 30, graph stats in Whispers at 40, proactive insights at 60.

## Supported File Types

**Full indexing:** PDF, TXT, MD, HTML, CSV, DOCX, DOC, Python, JavaScript, TypeScript, Shell, C/C++, Java, Go, Rust, Ruby, Perl, Lua, Swift, Kotlin, TOML, JSON, YAML, XML, INI, CFG, RTF, LOG

**Sorted but not indexed:** Images (JPG, PNG, GIF), audio/video (MP3, MP4, WAV), archives

## File Structure

```
grimalkin/
├── grimalkin.py          # The entire application (~1760 LOC)
├── grimalkin.jpg         # Hero image
├── grimalkin.db          # SQLite database (created on first run)
├── requirements.txt
├── sorted/               # Organized files (created on first run)
│   ├── FINANCIAL/
│   ├── PERSONAL/
│   ├── RESEARCH/
│   ├── MEDIA/
│   ├── MISC/
│   ├── PYRE/             # Burned files awaiting cremation
│   └── DUPLICATES/
├── faiss_index/          # FAISS vector index (created on first run)
└── vault/                # Reserved for future use
```

## Requirements

- Python 3.10+
- Ollama running locally with `qwen3:8b` and `nomic-embed-text`
- ~8 GB RAM recommended (for the 8B model)
- Works on Linux, macOS, Windows (tested on Pop!_OS)

## License

MIT — do whatever you want with it.

---

*I sort your files, guard your vault, and judge you silently.*
*— Grimalkin*
