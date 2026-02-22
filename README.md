<p align="center">
  <img src="grimalkin.jpg" alt="grimalkin" width="400">
</p>

<h1 align="center">grimalkin</h1>

<p align="center">
  <strong>Your private AI familiar — fully local, fully yours</strong><br>
  Single-file • 100% offline • Ollama + FAISS RAG
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-Local-green" alt="Ollama">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

Turn your personal documents into daily briefings, smart file organization, and instant Q&A — no cloud, no API keys, no tracking.

## ✨ Features

- **Daily Briefings** — Structured markdown reports (Summary → Key Data → Outlook) generated from your documents
- **Ask Anything** — RAG-powered Q&A over your entire local knowledge base
- **Smart File Sorting** — LLM classifies and auto-organizes your Downloads (RESEARCH / NOTES / MEETING / PERSONAL / misc)
- **Auto-Indexing** — Sorted files are automatically embedded into the search index
- **Background Watcher** — Monitors your Downloads folder and sorts new files in real time
- **Local Web UI** — Clean Gradio interface with tabs for every feature
- **Runs on Consumer Hardware** — AMD / NVIDIA / Apple Silicon — no GPU required (but it helps)

---

## 🚀 Quick Start (5 minutes)

### 1. Install Ollama

Download from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull qwen3:8b             # main reasoning model
ollama pull nomic-embed-text     # embedding model
```

> **Lighter alternatives:** `llama3.2:3b` or `phi3:3.8b` if you're short on VRAM/RAM.

### 2. Clone & Install

```bash
git clone https://github.com/yourusername/grimalkin.git
cd grimalkin
pip install -r requirements.txt
```

### 3. Add Your Knowledge

```bash
mkdir -p knowledge
# Drop your .md, .txt, .pdf, .csv files into knowledge/
```

### 4. Build the Index

```bash
python indexer.py
```

### 5. Run It

```bash
# Web UI (recommended)
python agent.py --web

# Or use the CLI
python agent.py --briefing --dry-run
python agent.py --ask "What are my top open items?"
python agent.py --keep
```

---

## 📖 Usage

### CLI Modes

| Command | What it does |
|---------|-------------|
| `python agent.py --briefing` | Generate & save daily briefing |
| `python agent.py --briefing --dry-run` | Preview briefing without saving |
| `python agent.py --ask "your question"` | Ask anything against your knowledge base |
| `python agent.py --keep` | Sort all files in Downloads |
| `python agent.py --watch` | Background watcher — auto-sorts new Downloads |
| `python agent.py --reindex` | Rebuild the full FAISS index |
| `python agent.py --web` | Launch the local web UI |

### Web UI

```bash
python agent.py --web
```

Opens at [http://127.0.0.1:7860](http://127.0.0.1:7860) with tabs for:

- **📅 Daily Briefing** — Generate and preview briefings
- **❓ Ask Anything** — Chat with your knowledge base
- **📁 File Keeper** — One-click Downloads sorting
- **📤 Upload to Knowledge** — Drag & drop files to add and auto-reindex
- **🛠 Tools** — Rebuild index on demand

Everything stays 100% local. No data ever leaves your machine.

---

## 📁 Project Structure

```
grimalkin/
├── agent.py              # Main script — all features (CLI + Web UI)
├── indexer.py            # Build/rebuild FAISS index
├── knowledge/            # ← YOUR DOCUMENTS GO HERE
├── faiss_index/          # Auto-generated search index
├── briefings/            # Saved daily briefings (.md)
├── organized/            # Auto-sorted files from Downloads
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Edit the top of `agent.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3:8b` | Ollama model for reasoning |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `PERSONA` | (see file) | Customize your assistant's tone and style |
| `DOWNLOADS_DIR` | `~/Downloads` | Folder to scan/watch |
| `ORGANIZED_DIR` | `~/organized` | Where sorted files go |
| `BRIEFING_DIR` | `~/briefings` | Where briefings are saved |

---

## 🤖 Automation

### Daily briefing at 7 AM (cron)

```bash
crontab -e
# Add this line:
0 7 * * * cd /path/to/grimalkin && python agent.py --briefing >> briefings/log.txt 2>&1
```

### Background file watcher (always-on sorting)

```bash
# Run in a terminal or as a background service:
python agent.py --watch
```

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| "FAISS index not found" | Run `python agent.py --reindex` (make sure `knowledge/` has files) |
| Ollama connection error | Start Ollama with `ollama serve` |
| Slow first response | Normal — model loads into memory on first call |
| Out of memory | Switch to a smaller model: `llama3.2:3b` |

---

## 📋 Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- 8GB+ RAM (16GB recommended for 8B+ models)
- No GPU required (but CUDA/ROCm/Metal will speed things up)

---

## 🗺️ Roadmap

- [x] Daily briefings with RAG
- [x] LLM-powered file sorting
- [x] Auto-indexing after sort
- [x] Background file watcher
- [x] Local web UI (Gradio)
- [ ] Conversation history / multi-turn chat
- [ ] Incremental indexing (only new/changed files)
- [ ] Scheduled briefing from web UI
- [ ] Plugin system for custom tools

---

## 📄 License

MIT — do whatever you want.

---

<p align="center">
  <em>Made with 🐱 for people who value privacy and local-first tools.</em><br>
  <strong>by toxic2040</strong>
</p>
