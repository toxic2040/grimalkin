#!/usr/bin/env python3
"""
grimalkin — v1.2 SHIP-READY
Fully local AI assistant (Ollama + FAISS RAG) with CLI + Web UI

Features:
  --briefing    Daily structured markdown briefing
  --ask         RAG-powered Q&A from your local knowledge
  --keep        Auto-sort Downloads using LLM classification
  --watch       Background watcher for Downloads folder
  --reindex     Rebuild FAISS index from knowledge/
  --web         Launch local Gradio web UI
  --dry-run     Preview briefing without saving
"""

import argparse
import re
import time
import logging
import shutil
from datetime import datetime, date
from pathlib import Path

import tenacity
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("local-agent")

# ====================== CONFIG ======================
MODEL = "qwen3:8b"                    # or llama3.2:3b, gemma2:9b, etc.
EMBEDDING_MODEL = "nomic-embed-text"
FAISS_INDEX_DIR = "faiss_index"
BRIEFING_DIR = Path.home() / "briefings"
DOWNLOADS_DIR = Path.home() / "Downloads"
ORGANIZED_DIR = Path.home() / "organized"
KNOWLEDGE_DIR = Path("knowledge")         # matches indexer.py (Path, not str)

PERSONA = """You are a private, concise, numbers-first personal AI assistant.
When local documents are relevant, prioritize them and cite them.
When local documents are NOT relevant to the question, ignore them entirely
and answer from your general knowledge instead.
Always indicate whether your answer comes from [LOCAL DOCS] or [GENERAL KNOWLEDGE].
Tone: direct, actionable, zero fluff.
Lead with the single most important insight.
Current date and time: {today}."""

# ====================== HELPERS ======================
def make_llm(temp=0.5):
    return ChatOllama(model=MODEL, num_ctx=32768, temperature=temp, timeout=300)


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(min=2, max=15))
def safe_complete(llm, prompt):
    return llm.invoke(prompt).content


def strip_thinking(text):
    """Strip <think>...</think> tags (Qwen, DeepSeek, etc). Passes through cleanly for other models."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return (m.group(1).strip(), text[m.end():].strip()) if m else ("", text.strip())


def get_vectorstore():
    """Load the FAISS index. Raises RuntimeError with clear message if missing."""
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        raise RuntimeError(
            "FAISS index not found. Run: python agent.py --reindex\n"
            "(Make sure you have files in knowledge/ first)"
        )


def get_rag_sample(query="latest updates", k=10, max_chars=700, score_threshold=0.4):
    """Retrieve relevant chunks from the FAISS index with relevance filtering."""
    try:
        vs = get_vectorstore()
        docs_and_scores = vs.similarity_search_with_relevance_scores(query, k=k)
        # Filter out low-relevance chunks
        relevant = [(d, s) for d, s in docs_and_scores if s >= score_threshold]
        if not relevant:
            return "(No relevant documents found in your knowledge base for this query.)"
        return "\n---\n".join(
            f"[relevance: {s:.2f}] {d.page_content[:max_chars]}"
            for d, s in relevant
        )
    except Exception:
        return "(No RAG context available — index may not exist yet)"


def _load_single_file(path: Path):
    """Load a single file using the appropriate loader. Returns list of Documents."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    elif ext == ".csv":
        return CSVLoader(str(path)).load()
    elif ext in {".md", ".txt", ".json"}:
        return TextLoader(str(path)).load()
    else:
        return TextLoader(str(path)).load()  # fallback


# ====================== CORE LOGIC (returns strings — used by both CLI and Web UI) ======================

def generate_briefing(dry_run=False):
    """Core briefing logic — always returns a markdown string."""
    try:
        now = datetime.now()
        today_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        sample = get_rag_sample(k=12, max_chars=700)

        prompt = f"""{PERSONA.format(today=today_str)}

Recent excerpts from local documents:
{sample}

Write a daily briefing in exactly this structure (under 450 words):

## Summary
Most important things today.

## Key Data
- bullets or table

## Outlook
Risks, opportunities, what to watch.

## Topics Covered
General updates

BRIEFING:"""

        raw = safe_complete(make_llm(0.5), prompt)
        _, content = strip_thinking(raw)

        if not dry_run:
            BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
            filepath = BRIEFING_DIR / f"{date.today():%Y-%m-%d}.md"
            header = f"# Daily Briefing — {today_str}\n\n> grimalkin v1.2 • {datetime.now():%H:%M}\n\n"
            filepath.write_text(header + content, encoding="utf-8")
            content += f"\n\n---\n**Saved:** `{filepath}`"

        return content

    except Exception as e:
        return f"❌ Error generating briefing: {e}\n\n(Is Ollama running? Is the index built?)"


def generate_answer(query):
    """Core ask logic — always returns a string."""
    try:
        if not query or not query.strip():
            return "Please enter a question."

        now = datetime.now()
        today_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        sample = get_rag_sample(query, k=6, max_chars=800)

        prompt = f"""{PERSONA.format(today=today_str)}

Query: {query}

Local knowledge excerpts (may or may not be relevant — check before using):
{sample}

Instructions:
- If the excerpts above are relevant to the query, use them and cite [LOCAL DOCS].
- If the excerpts are NOT relevant (e.g. the query is about general knowledge but
  the excerpts are about unrelated topics), IGNORE them completely and answer from
  your own knowledge. Cite [GENERAL KNOWLEDGE].
- Never apologize for lack of local data when you can answer from general knowledge.

Answer concisely, numbers-first."""

        raw = safe_complete(make_llm(0.6), prompt)
        _, ans = strip_thinking(raw)
        return ans

    except Exception as e:
        return (
            f"❌ Error: {e}\n\n"
            "Make sure Ollama is running (`ollama serve`) and the index exists "
            "(`python agent.py --reindex`)."
        )


# ====================== FILE KEEPER ======================

def process_file(path: Path):
    """Classify, move, and auto-index a single file. Returns a status string."""
    if path.suffix.lower() not in {".pdf", ".md", ".txt", ".csv", ".json"}:
        return f"⏭️  Skipped (unsupported type): {path.name}"
    if not path.exists():
        return f"⏭️  Skipped (file gone): {path.name}"

    # --- Classify ---
    llm = make_llm(0.3)
    prompt = f"""{PERSONA.format(today="today")}
Classify filename into ONE category only: RESEARCH, NOTES, MEETING, PERSONAL, OTHER.
Filename: {path.name}
Return ONLY the category word."""

    cat = safe_complete(llm, prompt).strip().upper()

    target_map = {
        "RESEARCH": ORGANIZED_DIR / "research",
        "NOTES":    ORGANIZED_DIR / "notes",
        "MEETING":  ORGANIZED_DIR / "meeting",
        "PERSONAL": ORGANIZED_DIR / "personal",
    }
    target = target_map.get(cat, ORGANIZED_DIR / "misc")
    target.mkdir(parents=True, exist_ok=True)

    new_path = target / path.name
    if new_path.exists():
        log.info(f"⚠️  Skipped (already exists): {path.name}")
        return f"⚠️  Already exists: {path.name} → {target.name}/"

    # --- Move ---
    shutil.move(str(path), str(new_path))
    log.info(f"✅ Sorted: {path.name} → {target.name}")
    status = f"✅ {path.name} → {target.name}/"

    # --- Auto-index into FAISS ---
    try:
        docs = _load_single_file(new_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        vs = get_vectorstore()
        vs.add_documents(chunks)
        vs.save_local(FAISS_INDEX_DIR)
        log.info(f"✅ Auto-indexed: {path.name}")
        status += " (indexed)"
    except Exception as e:
        log.warning(f"Auto-index failed for {path.name}: {e}")
        status += " (index failed — run --reindex later)"

    return status


def run_keep_core():
    """Core keep logic — scans Downloads, returns per-file status report."""
    try:
        results = []
        files = [f for f in DOWNLOADS_DIR.iterdir() if f.is_file()]

        if not files:
            return "📂 Downloads folder is empty — nothing to sort."

        for f in files:
            result = process_file(f)
            results.append(result)

        summary = f"**Processed {len(files)} file(s):**\n\n" + "\n".join(results)
        return summary

    except Exception as e:
        return f"❌ Keep error: {e}"


class FileKeeperHandler(FileSystemEventHandler):
    """Watchdog handler for background file monitoring."""
    def on_created(self, event):
        if event.is_directory:
            return
        process_file(Path(event.src_path))


# ====================== CLI WRAPPERS ======================

def run_briefing(dry_run=False):
    """CLI wrapper — prints briefing to stdout."""
    print("=" * 75)
    print(f"  grimalkin v1.2 | {date.today():%A, %B %d, %Y} | fully offline")
    print("=" * 75)
    content = generate_briefing(dry_run)
    print(content)


def run_ask(query):
    """CLI wrapper — prints answer to stdout."""
    ans = generate_answer(query)
    print("\n" + "=" * 75)
    print(ans)
    print("=" * 75)


def run_keep():
    """CLI wrapper — prints keep results to stdout."""
    print("Scanning Downloads...")
    result = run_keep_core()
    print(result)
    print("\nKeep complete.")


def run_reindex():
    """Rebuild the full FAISS index from knowledge/. Returns status string."""
    from indexer import build_index
    build_index()
    return "✅ Index rebuilt successfully"


def run_watch():
    """Foreground file watcher — Ctrl+C to stop. CLI only."""
    print("👁️  Watching Downloads for new files... (Ctrl+C to stop)")
    observer = Observer()
    observer.schedule(FileKeeperHandler(), str(DOWNLOADS_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    observer.join()


# ====================== WEB UI ======================

def run_web():
    """Launch local Gradio web interface. Everything stays on your machine."""
    import gradio as gr

    print("🐱 Starting grimalkin Web UI → http://127.0.0.1:7860")

    with gr.Blocks(title="grimalkin v1.2") as demo:
        gr.Markdown("# 🐱 grimalkin v1.2\n**Fully local • Private • Yours**")

        with gr.Tabs():

            # --- Briefing ---
            with gr.Tab("📅 Daily Briefing"):
                dry = gr.Checkbox(label="Dry run (preview only)", value=True)
                btn_brief = gr.Button("Generate Briefing", variant="primary")
                out_brief = gr.Markdown()
                btn_brief.click(generate_briefing, inputs=[dry], outputs=out_brief)

            # --- Ask ---
            with gr.Tab("❓ Ask Anything"):
                q = gr.Textbox(label="Question", placeholder="What are my top open items?")
                btn_ask = gr.Button("Ask", variant="primary")
                out_ask = gr.Markdown()
                btn_ask.click(generate_answer, inputs=[q], outputs=out_ask)

            # --- File Keeper ---
            with gr.Tab("📁 File Keeper"):
                btn_keep = gr.Button("Scan & Sort Downloads Now", variant="primary")
                out_keep = gr.Markdown()
                btn_keep.click(run_keep_core, outputs=out_keep)

            # --- Knowledge Upload ---
            with gr.Tab("📤 Upload to Knowledge"):
                gr.Markdown("Drag & drop files to add them to your knowledge base and rebuild the index.")
                file_input = gr.File(label="Drop files here", file_count="multiple")
                btn_upload = gr.Button("Add to Knowledge & Reindex", variant="primary")
                out_upload = gr.Textbox(label="Status", interactive=False)

                def handle_upload(files):
                    if not files:
                        return "No files selected."
                    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
                    names = []
                    for f in files:
                        src = Path(f)                       # Gradio v5 passes temp file paths
                        dest = KNOWLEDGE_DIR / src.name
                        shutil.copy2(str(src), str(dest))
                        names.append(src.name)
                    run_reindex()
                    return f"✅ Added {len(names)} file(s): {', '.join(names)} — index rebuilt"

                btn_upload.click(handle_upload, inputs=[file_input], outputs=out_upload)

            # --- Tools ---
            with gr.Tab("🛠 Tools"):
                gr.Markdown(
                    "**Background watcher:** Run `python agent.py --watch` in your terminal (CLI only).\n\n"
                    "Use the button below to rebuild the full search index from your `knowledge/` folder."
                )
                btn_reindex = gr.Button("Rebuild Full Index")
                out_tools = gr.Textbox(label="Status", interactive=False)
                btn_reindex.click(run_reindex, outputs=out_tools)

    demo.launch(server_name="127.0.0.1", share=False)


# ====================== CLI ======================

def main():
    parser = argparse.ArgumentParser(
        description="grimalkin v1.2 — Fully local AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python agent.py --briefing --dry-run    Preview daily briefing
  python agent.py --briefing              Generate & save briefing
  python agent.py --ask "open items?"     Ask a question
  python agent.py --keep                  Sort Downloads folder
  python agent.py --watch                 Background watcher
  python agent.py --reindex               Rebuild FAISS index
  python agent.py --web                   Launch web UI
        """,
    )
    parser.add_argument("--briefing", action="store_true", help="Generate daily briefing")
    parser.add_argument("--keep", action="store_true", help="Sort Downloads folder")
    parser.add_argument("--ask", nargs="*", help="Ask a question")
    parser.add_argument("--reindex", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--watch", action="store_true", help="Watch Downloads (foreground)")
    parser.add_argument("--web", action="store_true", help="Launch web UI")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    if args.web:
        run_web()
    elif args.reindex:
        result = run_reindex()
        print(result)
    elif args.watch:
        run_watch()
    elif args.keep:
        run_keep()
    elif args.ask is not None:
        q = " ".join(args.ask) if args.ask else input("Ask: ").strip()
        run_ask(q)
    else:
        run_briefing(args.dry_run)


if __name__ == "__main__":
    main()
