<p align="center">
  <img src="grimalkin.jpg" width="320" alt="Grimalkin">
</p>

<p align="center">
  <strong>Your private AI familiar — fully local, fully yours</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/FAISS-000000?logo=faiss&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/Gradio-FF8C00?logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

---

**Single-file • 100% offline • Ollama + FAISS RAG**

Turn your personal documents into daily briefings, smart file organization, and instant Q&A — no cloud, no API keys, no tracking.

## ✨ Features

- **Daily Briefings** — Structured markdown reports (Summary → Key Data → Outlook) generated from your documents
- **Ask Anything** — RAG-powered Q&A over your entire local knowledge base
- **Smart File Sorting** — LLM classifies and auto-organizes your Downloads (RESEARCH / NOTES / MEETING / PERSONAL / misc)
- **Auto-Indexing** — Sorted files are automatically embedded into the search index
- **Background Watcher** — Monitors your Downloads folder and sorts new files in real time
- **Local Web UI** — Clean Gradio interface with tabs for every feature
- **Runs on Consumer Hardware** — AMD / NVIDIA / Apple Silicon — no GPU required (but it helps)

## 🚀 Quick Start (5 minutes)

### 1. Install Ollama
Download from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull qwen3:8b             # main reasoning model
ollama pull nomic-embed-text     # embedding model