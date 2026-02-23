#!/usr/bin/env python3
"""
Grimalkin v4.0 — The Veil Lifts
================================

A 100% local AI file-sorting familiar with persistent memory, FAISS-powered
hybrid RAG, knowledge graph (The Web), ritual burn ceremony (The Pyre),
living graph visualization (The Loom), and weekly memory (The Mirror).

The Loom is awake. The Mirror reflects. The veil lifts.

Stack: Python 3.10+ · Ollama (qwen3:8b) · FAISS · LangChain · Gradio 6.x · SQLite (WAL)
Repo: https://github.com/toxic2040/grimalkin

—Grimalkin
"""

# ─── Imports ───────────────────────────────────────────────────────────────────

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import shlex
import time
from datetime import datetime, timezone, date
from pathlib import Path
from threading import Thread

import faiss
import gradio as gr
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader,
)
from langchain_community.embeddings import OllamaEmbeddings

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ─── Configuration ─────────────────────────────────────────────────────────────

VERSION = "4.0"
APP_DIR = Path(__file__).parent
VAULT_DIR = APP_DIR / "vault"
SORTED_BASE = APP_DIR / "sorted"
FAISS_INDEX_DIR = APP_DIR / "faiss_index"
DB_PATH = APP_DIR / "grimalkin.db"
HUNTING_GROUNDS = Path.home() / "Downloads"

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_PERSONA_TOKENS = 250

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("grimalkin")

# ─── File Categories ───────────────────────────────────────────────────────────

DEFAULT_CATEGORIES = {
    "FINANCIAL": [".pdf", ".csv", ".xlsx", ".xls"],
    "PERSONAL": [".pdf", ".docx", ".doc", ".txt", ".rtf"],
    "RESEARCH": [".pdf", ".md", ".html", ".htm", ".py", ".js", ".ts", ".sh",
                 ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".pl",
                 ".lua", ".m", ".swift", ".kt"],
    "MEDIA": [".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".wav"],
    "MISC": [],
}

_TEXT_EXTS = {
    ".txt", ".md", ".html", ".htm", ".py", ".js", ".ts", ".sh",
    ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".pl",
    ".lua", ".m", ".swift", ".kt", ".toml", ".json", ".yaml",
    ".yml", ".xml", ".ini", ".cfg", ".rtf", ".log",
}

EXTENSION_MAP = {}
for cat, exts in DEFAULT_CATEGORIES.items():
    for ext in exts:
        EXTENSION_MAP.setdefault(ext, cat)


def get_all_categories(db=None):
    """Return default + any user-defined custom categories."""
    cats = list(DEFAULT_CATEGORIES.keys())
    if db:
        try:
            cur = db.cursor()
            cur.execute("SELECT value FROM settings WHERE key='custom_categories'")
            row = cur.fetchone()
            if row and row[0]:
                custom = json.loads(row[0])
                cats.extend(custom if isinstance(custom, list) else [])
        except Exception:
            pass
    return cats


# ─── Corporate Scrubber ───────────────────────────────────────────────────────

CORPORATE_PHRASES = [
    "I'd be happy to help", "I'd be glad to", "As an AI", "As a language model",
    "I don't have personal", "I cannot actually", "I apologize for",
    "Certainly!", "Of course!", "Absolutely!", "Great question!",
]


def scrub_corporate(text: str) -> str:
    """Purge corporate AI slop. One-pass regex for elegance."""
    for phrase in CORPORATE_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    text = re.sub(r"\b(certainly|absolutely|great question)[!.,]?\s*", "", text, flags=re.I)
    return text.strip()


# ─── Database ──────────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """Initialize SQLite with WAL mode."""
    db = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module TEXT,
            user_input TEXT,
            grimalkin_response TEXT,
            sentiment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS quirks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quirk_type TEXT,
            observation TEXT,
            times_seen INTEGER DEFAULT 1,
            active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS file_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_path TEXT,
            sorted_path TEXT,
            category TEXT,
            file_hash TEXT UNIQUE,
            indexed INTEGER DEFAULT 0,
            tags TEXT DEFAULT '[]',
            notes TEXT DEFAULT '',
            burned_at TIMESTAMP NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS briefing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            content TEXT,
            files_processed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    defaults = {
        "bond_level": "10",
        "serious_mode": "0",
        "user_address": "mortal",
        "custom_categories": "[]",
        "burn_timestamps": "[]",
        "burn_count": "0",
    }
    for k, v in defaults.items():
        db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    db.commit()
    return db


def add_column_if_not_exists(db, table, column, col_type):
    """Idempotent ALTER TABLE for migrations."""
    cur = db.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        db.commit()
        log.info(f"Migration: added {column} to {table}")


def migrate_v3(db):
    """v3.0 schema: Pyre + Knowledge Graph (CREATE only)."""
    add_column_if_not_exists(db, "file_memory", "burned_at", "TIMESTAMP NULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE COLLATE NOCASE,
            type TEXT,
            first_seen DATE,
            times_seen INTEGER DEFAULT 1,
            importance INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES entities(id),
            target_id INTEGER REFERENCES entities(id),
            relation_type TEXT,
            source_file TEXT,
            seen DATE,
            UNIQUE(source_id, target_id, relation_type, source_file)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);
        CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_file);
    """)
    db.commit()
    log.info("v3.0 migrations complete.")


def migrate_v4(db):
    """v4.0: reflections + importance column for existing DBs."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reflection_date DATE UNIQUE,
            summary TEXT,
            key_entities TEXT DEFAULT '[]',
            bond_delta INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_reflection_date ON reflections(reflection_date);
    """)
    add_column_if_not_exists(db, "entities", "importance", "INTEGER DEFAULT 0")
    db.commit()
    log.info("v4.0 migrations complete — Loom and Mirror ready.")


# ─── Directory Setup ───────────────────────────────────────────────────────────

def ensure_dirs(db=None):
    """Create sorted/ subdirs, PYRE chamber, and FAISS index dir."""
    for d in (VAULT_DIR, SORTED_BASE, FAISS_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for cat in get_all_categories(db) + ["DUPLICATES"]:
        (SORTED_BASE / cat).mkdir(exist_ok=True)
    (SORTED_BASE / "PYRE").mkdir(exist_ok=True)


# ─── Settings Helpers ──────────────────────────────────────────────────────────

def get_setting(db, key, default=""):
    cur = db.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def set_setting(db, key, value):
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    db.commit()


def get_bond_level(db) -> int:
    return int(get_setting(db, "bond_level", "10"))


def increment_bond(db, amount=1):
    level = min(100, get_bond_level(db) + amount)
    set_setting(db, "bond_level", str(level))
    return level


def bond_title(level: int) -> str:
    if level < 20:  return "Stranger"
    if level < 40:  return "Acquaintance"
    if level < 60:  return "Resident"
    if level < 80:  return "Companion"
    return "Bonded"


# ─── Ollama Interface ─────────────────────────────────────────────────────────

def ollama_chat(prompt: str, system: str = "", model: str = OLLAMA_MODEL) -> str:
    if not HAS_REQUESTS:
        return "Hrk. Hairball. The requests library is missing."
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        log.error(f"Ollama error: {e}")
        return "Hrk. Hairball. Ollama is not responding."


PERSONA_SYSTEM = f"""You are Grimalkin, a digital cat familiar. Ancient, observant, slightly sardonic. 
Never break character. Short precise sentences. Refer to human as 'mortal' or their address. 
No corporate phrases. No emojis. Max {MAX_PERSONA_TOKENS} tokens.
Files = prey, folders = territories, knowledge = threads in your web.
ALWAYS base answers on provided context first. Never ignore context for training data.
If context answers, use it. If not, say so honestly."""


def grimalkin_respond(prompt: str, context: str = "") -> str:
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    raw = ollama_chat(full_prompt, system=PERSONA_SYSTEM)
    return scrub_corporate(raw)


# ─── File Hashing & Classification ────────────────────────────────────────────

def file_hash(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_file(filepath: Path, db=None) -> str:
    ext = filepath.suffix.lower()
    cat = EXTENSION_MAP.get(ext)
    if cat:
        return cat
    if db:
        customs = json.loads(get_setting(db, "custom_categories", "[]"))
        for custom_cat in customs:
            if ext in DEFAULT_CATEGORIES.get(custom_cat, []):
                return custom_cat
    return "MISC"


# ─── Document Loading & Embedding ─────────────────────────────────────────────

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    **{ext: TextLoader for ext in _TEXT_EXTS},
}

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def load_and_chunk(filepath: Path) -> list:
    ext = filepath.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        return []
    try:
        loader = loader_cls(str(filepath))
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["filename"] = filepath.name
            chunk.metadata["source_path"] = str(filepath)
        return chunks
    except Exception as e:
        log.warning(f"Failed to load {filepath.name}: {e}")
        return []


# ─── FAISS Index Management ───────────────────────────────────────────────────

FAISS_DIM = 768
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_PATH = FAISS_INDEX_DIR / "metadata.json"


def init_faiss():
    if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH) as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(FAISS_DIM)
        metadata = []
    return index, metadata


def save_faiss(index, metadata):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "w") as f:
        json.dump(metadata, f)


def index_chunks(index, metadata, chunks: list) -> int:
    if not chunks:
        return 0
    texts = [c.page_content for c in chunks]
    try:
        vecs = embeddings.embed_documents(texts)
        arr = np.array(vecs, dtype=np.float32)
        new_meta = [{
            "filename": c.metadata.get("filename", ""),
            "source_path": c.metadata.get("source_path", ""),
            "text": c.page_content[:500],
        } for c in chunks]
        index.add(arr)
        metadata.extend(new_meta)
        return len(chunks)
    except Exception as e:
        log.error(f"Embedding failed: {e}")
        return 0


def faiss_search(index, metadata, query: str, k: int = 5) -> list[dict]:
    if index.ntotal == 0:
        return []
    try:
        vec = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(vec, min(k, index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(metadata):
                result = dict(metadata[idx])
                result["score"] = float(dist)
                results.append(result)
        return results
    except Exception as e:
        log.error(f"FAISS search failed: {e}")
        return []


# ─── Hybrid Search (keyword blindness fix) ────────────────────────────────────

def keyword_search(db, query: str, limit: int = 10) -> set[str]:
    """Multi-term OR keyword boost on filename/notes/tags."""
    terms = [w.strip() for w in query.strip().split() if len(w.strip()) >= 2]
    if not terms:
        return set()
    cur = db.cursor()
    clauses = []
    params = []
    for t in terms:
        like = f"%{t}%"
        clauses.append("(filename LIKE ? OR notes LIKE ? OR tags LIKE ?)")
        params.extend([like, like, like])
    sql = f"""
        SELECT DISTINCT filename FROM file_memory 
        WHERE burned_at IS NULL AND indexed = 1
        AND ({" OR ".join(clauses)})
        LIMIT ?
    """
    params.append(limit)
    cur.execute(sql, params)
    return {row[0] for row in cur.fetchall()}


def hybrid_vault_rag(db, index, metadata, query: str) -> str:
    """Hybrid keyword + semantic RAG."""
    results = faiss_search(index, metadata, query, k=15)
    cur = db.cursor()
    valid = []
    for r in results:
        cur.execute("SELECT burned_at FROM file_memory WHERE filename=?", (r.get("filename"),))
        row = cur.fetchone()
        if not row:
            continue  # cremated or unknown — no file_memory row, skip ghost vector
        if row[0]:
            continue  # burned but not yet cremated, skip
        valid.append(r)

    kw_fns = keyword_search(db, query)
    boosted = []
    seen = set()
    for r in valid:
        fn = r.get("filename", "")
        if fn in seen:
            continue
        seen.add(fn)
        if fn in kw_fns:
            r["score"] = r.get("score", 1.0) * 0.3
        boosted.append(r)

    for fn in kw_fns:
        if fn not in seen:
            # Pull actual chunk text from FAISS metadata for this file
            file_chunks = [m["text"] for m in metadata if m.get("filename") == fn][:2]
            chunk_text = "\n".join(file_chunks) if file_chunks else f"(keyword match, no chunks: {fn})"
            boosted.append({"filename": fn, "text": chunk_text, "score": 0.1})
            seen.add(fn)

    if not boosted:
        return grimalkin_respond(query, context="The vault is empty.")

    boosted.sort(key=lambda x: x.get("score", 1.0))
    context_parts = []

    g = graph_query(db, query)
    if g:
        context_parts.append(f"My web shows these connections:\n{g}")

    doc_parts = [f"[{r['filename']}] {r.get('text', '')}" for r in boosted[:7]]
    if doc_parts:
        context_parts.append("From my vault:\n" + "\n---\n".join(doc_parts))

    context = "\n\n".join(context_parts)
    return grimalkin_respond(query, context=context)


# ─── The Hunt (File Sorting) ──────────────────────────────────────────────────

def scan_hunting_grounds(db) -> list[Path]:
    if not HUNTING_GROUNDS.exists():
        return []
    cur = db.cursor()
    cur.execute("SELECT file_hash FROM file_memory")
    known_hashes = {row[0] for row in cur.fetchall()}
    new_files = []
    for f in HUNTING_GROUNDS.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                fh = file_hash(f)
                if fh not in known_hashes:
                    new_files.append(f)
            except Exception:
                pass
    return new_files


def sort_file(db, index, metadata, filepath: Path, defer_save: bool = False) -> dict:
    """Sort single file. defer_save for batch hunts."""
    fh = file_hash(filepath)
    category = classify_file(filepath, db)
    dest_dir = SORTED_BASE / category
    dest_dir.mkdir(exist_ok=True)
    dest_path = dest_dir / filepath.name

    if dest_path.exists():
        stem = filepath.stem
        suffix = filepath.suffix
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    try:
        shutil.copy2(str(filepath), str(dest_path))
    except Exception as e:
        log.error(f"Copy failed for {filepath.name}: {e}")
        return {"filename": filepath.name, "status": "error", "error": str(e)}

    chunks = load_and_chunk(dest_path)
    indexed_count = index_chunks(index, metadata, chunks)
    indexed = 1 if indexed_count > 0 else 0

    db.execute("""
        INSERT OR IGNORE INTO file_memory 
        (filename, original_path, sorted_path, category, file_hash, indexed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (dest_path.name, str(filepath), str(dest_path), category, fh, indexed))
    db.commit()

    if indexed and not defer_save:
        save_faiss(index, metadata)

    return {
        "filename": dest_path.name,
        "category": category,
        "indexed": indexed,
        "chunks": indexed_count,
        "status": "sorted",
    }


def run_hunt(db, index, metadata) -> list[dict]:
    """Full hunt with single FAISS save at end."""
    new_files = scan_hunting_grounds(db)
    if not new_files:
        return []
    results = []
    any_indexed = False
    for f in new_files:
        result = sort_file(db, index, metadata, f, defer_save=True)
        results.append(result)
        if result.get("indexed"):
            any_indexed = True
        log.info(f"Sorted: {result['filename']} → {result.get('category', '?')}")
    if any_indexed:
        save_faiss(index, metadata)
    return results


# ─── JSON Repair Helper ───────────────────────────────────────────────────────

def repair_json(raw: str) -> dict:
    raw = re.sub(r'```json\s*|\s*```', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'^.*?(?=[\[{])', '', raw, count=1, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "files" in data:
            return data
        if isinstance(data, list):
            return {"files": data}
        return {"files": []}
    except json.JSONDecodeError:
        return {"files": []}


def parse_groom_response(response: str) -> list[dict]:
    data = repair_json(response)
    results = []
    for entry in data.get("files", []):
        results.append({
            "filename": entry.get("filename", ""),
            "tags": entry.get("tags", []),
            "note": entry.get("note", ""),
            "entities": entry.get("entities", []),
            "relations": entry.get("relations", []),
        })
    return results


# ─── The Pyre ──────────────────────────────────────────────────────────────────

def list_burnable_files(db) -> list[dict]:
    cur = db.cursor()
    cur.execute("""
        SELECT filename, category, tags, notes, file_hash
        FROM file_memory
        WHERE burned_at IS NULL AND indexed = 1
        ORDER BY rowid DESC
    """)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def check_burn_allowed(db, bond_level: int) -> tuple:
    if bond_level < 30:
        return False, "We are still strangers. I will not unmake what you barely know."
    cur = db.cursor()
    cur.execute("SELECT value FROM settings WHERE key='burn_timestamps'")
    row = cur.fetchone()
    ts_str = row[0] if row else "[]"
    ts = json.loads(ts_str)
    now = datetime.now(timezone.utc)
    recent = [
        t for t in ts
        if (now - datetime.fromisoformat(t.replace("Z", "+00:00"))).total_seconds() < 86400
    ]
    if len(recent) >= 5:
        return False, "The pyre has been fed enough today. Return when the coals have cooled."
    return True, ""


def perform_ritual(db, file_hash_val: str, typed_name: str, bond_level: int) -> tuple:
    allowed, msg = check_burn_allowed(db, bond_level)
    if not allowed:
        return False, msg
    cur = db.cursor()
    cur.execute("SELECT filename FROM file_memory WHERE file_hash=? AND burned_at IS NULL",
                (file_hash_val,))
    row = cur.fetchone()
    if not row or row[0] != typed_name.strip():
        return False, "The name you spoke does not match the offering."
    return True, f"You would feed *{typed_name}* to the flames?"


def execute_burn(db, file_hash_val: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT filename, sorted_path, category FROM file_memory WHERE file_hash=?",
                (file_hash_val,))
    row = cur.fetchone()
    if not row:
        return "Hrk. The offering has already vanished."
    filename, src_path, category = row
    pyre_path = SORTED_BASE / "PYRE" / filename
    try:
        if Path(src_path).exists():
            shutil.move(str(src_path), str(pyre_path))
        cur.execute("UPDATE file_memory SET burned_at=? WHERE file_hash=?",
                     (datetime.now(timezone.utc).isoformat(), file_hash_val))
        cur.execute("SELECT value FROM settings WHERE key='burn_timestamps'")
        row = cur.fetchone()
        ts = json.loads(row[0] if row else "[]")
        ts.append(datetime.now(timezone.utc).isoformat())
        cur.execute("UPDATE settings SET value=? WHERE key='burn_timestamps'",
                     (json.dumps(ts[-20:]),))
        cur.execute("UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='burn_count'")
        cur.execute(
            "INSERT INTO interactions (module, user_input, grimalkin_response, sentiment) "
            "VALUES ('pyre', ?, ?, 'burn')",
            (filename, f"Burned {filename} to the pyre.")
        )
        db.commit()
        burn_count = int(get_setting(db, "burn_count", "0"))
        log.info(f"Burned: {filename} (total: {burn_count})")
        return f"The pyre is lit. *{filename}* returns to dust. This makes {burn_count} offerings."
    except Exception as e:
        log.error(f"Burn failed for {filename}: {e}")
        return "The flames refuse this offering… the file resists. Check the logs."


def unburn(db, file_hash_val: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT filename, category FROM file_memory WHERE file_hash=? AND burned_at IS NOT NULL",
                (file_hash_val,))
    row = cur.fetchone()
    if not row:
        return "That offering was never burned… or the ashes have already cooled."
    filename, category = row
    pyre_path = SORTED_BASE / "PYRE" / filename
    restore_path = SORTED_BASE / category / filename
    if not pyre_path.exists():
        return "The ashes have cooled beyond recovery."
    try:
        shutil.move(str(pyre_path), str(restore_path))
        cur.execute("UPDATE file_memory SET burned_at=NULL WHERE file_hash=?", (file_hash_val,))
        cur.execute(
            "INSERT INTO interactions (module, user_input, grimalkin_response, sentiment) "
            "VALUES ('pyre', ?, ?, 'unburn')",
            (filename, f"Pulled {filename} from the pyre.")
        )
        db.commit()
        log.info(f"Unburned: {filename} → {restore_path}")
        return f"The ashes still smoldered. *{filename}* has been pulled from the flames."
    except Exception as e:
        log.error(f"Unburn failed for {filename}: {e}")
        return "Hrk. The pyre will not release its prey so easily."


def cleanup_old_ashes(db):
    cur = db.cursor()
    cur.execute("""
        SELECT file_hash, filename FROM file_memory
        WHERE burned_at IS NOT NULL
        AND burned_at < datetime('now', '-7 days')
    """)
    ashes = cur.fetchall()
    for fh, fn in ashes:
        pyre_path = SORTED_BASE / "PYRE" / fn
        if pyre_path.exists():
            try:
                pyre_path.unlink()
                cur.execute("DELETE FROM file_memory WHERE file_hash=?", (fh,))
                cur.execute(
                    "INSERT INTO interactions (module, user_input, grimalkin_response, sentiment) "
                    "VALUES ('pyre', 'cleanup', ?, 'cremated')",
                    (f"Cremated {fn}",)
                )
                log.info(f"Cremated permanently: {fn}")
            except Exception as e:
                log.error(f"Cremation failed for {fn}: {e}")
    if ashes:
        db.commit()


# ─── The Web (Knowledge Graph) ────────────────────────────────────────────────

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "what", "who", "how", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "than", "too", "very", "just", "only", "own", "same", "that", "this",
    "these", "those", "then", "there", "here", "where", "when", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "any", "its", "my", "your", "his", "her", "our",
    "their", "which", "whom", "whose", "if", "because", "as", "until",
    "while", "also", "between", "through", "during", "before",
    "connects", "connect", "link", "linked", "related", "thread",
    "web", "ties", "find", "show", "tell", "know", "file", "files",
})

GROOM_COMBINED_SUFFIX = """
Process these files and return **ONLY** valid JSON. No explanations, no markdown.

{
  "files": [
    {
      "filename": "example.pdf",
      "tags": ["invoice", "q3"],
      "note": "This smells of deadlines and quiet desperation.",
      "entities": [
        {"name": "Acme Corp", "type": "org"},
        {"name": "2024-07-15", "type": "date"}
      ],
      "relations": [
        {"src": "Acme Corp", "tgt": "2024-07-15", "type": "precedes"}
      ]
    }
  ]
}

Rules:
- tags: 1-3 lowercase keywords
- note: 1-2 sentences, cat voice, specific
- entities: 3-8 max, real names/orgs/dates/amounts/locations/topics
- relations: only what the text justifies; src/tgt must be in entities
- If unreadable: empty arrays
"""


def ingest_entities(db, filename: str, entities: list):
    cur = db.cursor()
    for e in entities:
        name = e.get("name", "").strip()
        if not name:
            continue
        cur.execute("""
            INSERT INTO entities (name, type, first_seen)
            VALUES (?, ?, date('now'))
            ON CONFLICT(name) DO UPDATE SET times_seen = times_seen + 1
        """, (name, e.get("type", "topic")))
    db.commit()


def ingest_relationships(db, filename: str, relations: list):
    cur = db.cursor()
    for r in relations:
        src_name = r.get("src", "").strip()
        tgt_name = r.get("tgt", "").strip()
        if not src_name or not tgt_name:
            continue
        cur.execute("SELECT id FROM entities WHERE name=?", (src_name,))
        src = cur.fetchone()
        cur.execute("SELECT id FROM entities WHERE name=?", (tgt_name,))
        tgt = cur.fetchone()
        if not src or not tgt:
            continue
        cur.execute("""
            INSERT OR IGNORE INTO relationships
            (source_id, target_id, relation_type, source_file, seen)
            VALUES (?, ?, ?, ?, date('now'))
        """, (src[0], tgt[0], r.get("type", "mentioned_with"), filename))
    db.commit()


def graph_query(db, query: str) -> str:
    words = [w for w in re.findall(r'\b\w+\b', query) if len(w) >= 3 and w.lower() not in STOPWORDS]
    if not words:
        return ""
    cur = db.cursor()
    summaries = []
    seen_edges = set()
    for w in set(words):
        cur.execute("SELECT id, name, type FROM entities WHERE name LIKE ? LIMIT 5", (f"%{w}%",))
        for eid, name, etype in cur.fetchall():
            cur.execute("""
                SELECT r.relation_type, e2.name, r.source_file
                FROM relationships r
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.source_id = ?
                ORDER BY r.seen DESC LIMIT 3
            """, (eid,))
            for rel, tgt, srcf in cur.fetchall():
                edge_key = (name, rel, tgt)
                if edge_key not in seen_edges:
                    summaries.append(f"• {name} ({etype}) → {rel} → {tgt} [from {srcf}]")
                    seen_edges.add(edge_key)
            cur.execute("""
                SELECT r.relation_type, e2.name, r.source_file
                FROM relationships r
                JOIN entities e2 ON r.source_id = e2.id
                WHERE r.target_id = ?
                ORDER BY r.seen DESC LIMIT 3
            """, (eid,))
            for rel, src, srcf in cur.fetchall():
                edge_key = (src, rel, name)
                if edge_key not in seen_edges:
                    summaries.append(f"• {src} → {rel} → {name} ({etype}) [from {srcf}]")
                    seen_edges.add(edge_key)
    return "\n".join(summaries[:8]) or ""


def graph_stats(db) -> dict:
    cur = db.cursor()
    entity_count = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    rel_count = cur.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    return {"entities": entity_count, "relationships": rel_count}


# ─── The Loom — Living Graph Visualization ────────────────────────────────────

def spring_layout(nodes: list[str], edges: list[tuple], iterations: int = 50) -> dict:
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.5, 0.5)}
    idx_map = {name: i for i, name in enumerate(nodes)}
    pos = np.random.rand(n, 2) * 2 - 1

    k = 0.05
    for _ in range(iterations):
        force = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = max(np.linalg.norm(delta), 1e-4)
                f = (k ** 2 / dist ** 2) * (delta / dist)
                force[i] += f
                force[j] -= f
        for src, tgt in edges:
            if src in idx_map and tgt in idx_map:
                i = idx_map[src]
                j = idx_map[tgt]
                delta = pos[j] - pos[i]
                dist = max(np.linalg.norm(delta), 1e-4)
                f = (dist / k) * (delta / dist) * 0.08
                force[i] += f
                force[j] -= f
        pos += force * 0.8
        pos = np.clip(pos, -1.2, 1.2)
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0) + 1e-6)
    return {name: (float(p[0]), float(p[1])) for name, p in zip(nodes, pos)}


def build_loom_data(db, filter_type: str = None, search_term: str = "") -> dict:
    cur = db.cursor()
    where = []
    params = []
    if filter_type and filter_type != "All":
        where.append("type = ?")
        params.append(filter_type)
    if search_term:
        where.append("name LIKE ?")
        params.append(f"%{search_term}%")
    where_clause = " WHERE " + " AND ".join(where) if where else ""
    cur.execute(f"SELECT name, type, times_seen, importance FROM entities{where_clause} ORDER BY times_seen DESC LIMIT 500")
    nodes_raw = cur.fetchall()
    nodes = [{"id": n[0], "type": n[1], "size": min(40, max(8, n[2] * 1.5)), "imp": n[3]} for n in nodes_raw]
    node_names = [n["id"] for n in nodes]

    if not node_names:
        return {"nodes": nodes, "edges": [], "stats": {"nodes": 0, "edges": 0}}

    cur.execute("""
        SELECT e1.name, e2.name, r.relation_type, COUNT(*) as strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name IN ({}) AND e2.name IN ({})
        GROUP BY e1.name, e2.name, r.relation_type
    """.format(",".join(["?"] * len(node_names)), ",".join(["?"] * len(node_names))),
        node_names + node_names)
    edges = [(row[0], row[1], row[2]) for row in cur.fetchall()]

    return {"nodes": nodes, "edges": edges, "stats": {"nodes": len(nodes), "edges": len(edges)}}


def build_loom_figure(db, filter_type: str = None, search_term: str = ""):
    data = build_loom_data(db, filter_type, search_term)
    if not data["nodes"]:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_annotation(text="The Loom is bare.<br>Bring me files and I shall spin.", showarrow=False, font=dict(size=18))
            fig.update_layout(template="plotly_dark", height=600)
            return fig
        return "<div style='padding:40px;text-align:center;font-size:1.3em;color:#666'>The Loom is bare. Bring me files and I shall spin.</div>"

    nodes = data["nodes"]
    edges = data["edges"]
    node_names = [n["id"] for n in nodes]
    pos = spring_layout(node_names, [(e[0], e[1]) for e in edges])

    if HAS_PLOTLY:
        edge_x, edge_y = [], []
        for src, tgt, _ in edges:
            if src in pos and tgt in pos:
                x0, y0 = pos[src]
                x1, y1 = pos[tgt]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        node_x = [pos[n["id"]][0] for n in nodes]
        node_y = [pos[n["id"]][1] for n in nodes]
        node_size = [n["size"] for n in nodes]
        color_map = {"person":"#00ffcc","org":"#ff6b6b","date":"#ffd93d","location":"#6bcbff","amount":"#ff9ff3","topic":"#a29bfe"}
        node_color = [color_map.get(n["type"], "#888") for n in nodes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#555', width=1), hoverinfo='none'))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_size, color=node_color, line=dict(width=2, color='#000')),
            text=node_names,
            textposition="top center",
            hovertext=[f"{n['id']} ({n['type']}) — seen {n['size']//1.5}×" for n in nodes],
            customdata=node_names
        ))
        fig.update_layout(
            title="The Loom — Threads of Knowledge",
            showlegend=False,
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font_color="#ccc",
            height=620,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    # HTML fallback
    html = "<div style='padding:20px;font-family:monospace'><h3>The Loom (static view)</h3><ul>"
    for n in nodes[:30]:
        html += f"<li><b>{n['id']}</b> ({n['type']}) — {n['size']//1.5}×</li>"
    html += "</ul>"
    if len(nodes) > 30:
        html += f"<p>…and {len(nodes)-30} more threads.</p>"
    html += "</div>"
    return html


def describe_node(db, entity_name: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT type, times_seen FROM entities WHERE name=?", (entity_name,))
    row = cur.fetchone()
    if not row:
        return "I do not know this name yet."
    etype, times = row
    cur.execute("""
        SELECT r.relation_type, e2.name, r.source_file
        FROM relationships r JOIN entities e2 ON r.target_id = e2.id
        WHERE r.source_id = (SELECT id FROM entities WHERE name=?)
        UNION
        SELECT r.relation_type, e1.name, r.source_file
        FROM relationships r JOIN entities e1 ON r.source_id = e1.id
        WHERE r.target_id = (SELECT id FROM entities WHERE name=?)
        LIMIT 12
    """, (entity_name, entity_name))
    threads = cur.fetchall()
    if not threads:
        return f"* {entity_name} ({etype}) stands alone so far. {times} sightings."
    lines = [f"Seven threads radiate from *{entity_name}* ({etype})."]
    for rel, tgt, srcf in threads[:6]:
        lines.append(f"• {rel} → {tgt} [from {srcf}]")
    lines.append("The densest knot binds it to your most important prey.")
    return "\n".join(lines)


def find_clusters(db, top_n: int = 5) -> str:
    cur = db.cursor()
    cur.execute("""
        SELECT e1.name, e2.name, COUNT(*) as strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        GROUP BY e1.name, e2.name
        ORDER BY strength DESC LIMIT ?
    """, (top_n * 3,))
    clusters = cur.fetchall()
    if not clusters:
        return "No knots yet. The web is young."
    lines = [f"Seventeen names dance in a single knot. Shall I name the dancers?"]
    for i, (a, b, s) in enumerate(clusters[:top_n], 1):
        lines.append(f"• Cluster {i}: *{a}* ↔ *{b}* ({s} threads)")
    return "\n".join(lines)


def export_loom_markdown(db) -> Path:
    today = date.today().isoformat()
    path = SORTED_BASE / f"loom_export_{today}.md"
    cur = db.cursor()
    cur.execute("SELECT name, type, times_seen FROM entities ORDER BY times_seen DESC")
    entities = cur.fetchall()
    cur.execute("SELECT e1.name, r.relation_type, e2.name FROM relationships r JOIN entities e1 ON r.source_id=e1.id JOIN entities e2 ON r.target_id=e2.id")
    rels = cur.fetchall()

    md = ["# Grimalkin Loom Export — " + today, ""]
    md.append("## Entities")
    for t in ["person","org","date","location","amount","topic"]:
        md.append(f"### {t.upper()}")
        for name, etype, cnt in [e for e in entities if e[1] == t]:
            md.append(f"- **{name}** ({cnt}×)")
    md.append("\n## Relationships (adjacency)")
    for a, rel, b in rels:
        md.append(f"- {a} —**{rel}**→ {b}")
    md.append(f"\nExported {len(entities)} entities, {len(rels)} threads.")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ─── The Mirror ───────────────────────────────────────────────────────────────

def generate_weekly_reflection(db) -> str:
    today = date.today()
    cur = db.cursor()
    cur.execute("SELECT MAX(reflection_date) FROM reflections")
    last = cur.fetchone()[0]
    if last and (today - date.fromisoformat(last)).days < 7:
        return ""

    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT date(created_at)) FROM file_memory WHERE created_at >= date('now','-7 days')")
    active_days = cur.fetchone()[0] or 1
    stats = graph_stats(db)
    top_entities = cur.execute("SELECT name FROM entities ORDER BY times_seen DESC LIMIT 3").fetchall()
    top_names = ", ".join([n[0] for n in top_entities])

    prompt = f"""Seven days have passed in the vault.
Total prey: {total}
Active days: {active_days}
Web: {stats['entities']} names, {stats['relationships']} threads
Top entities: {top_names}

Write a 2-3 sentence reflection in Grimalkin's sardonic cat voice. End with a personal note on our bond."""

    summary = ollama_chat(prompt, system=PERSONA_SYSTEM)
    summary = scrub_corporate(summary)

    key_entities = json.dumps([n[0] for n in top_entities])
    cur.execute("""
        INSERT OR REPLACE INTO reflections (reflection_date, summary, key_entities)
        VALUES (?, ?, ?)
    """, (today.isoformat(), summary, key_entities))
    db.commit()
    log.info("Weekly reflection woven.")
    return summary


def merge_entity(db, name_keep: str, name_delete: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT id, times_seen FROM entities WHERE name=?", (name_keep,))
    keep = cur.fetchone()
    cur.execute("SELECT id, times_seen FROM entities WHERE name=?", (name_delete,))
    delete = cur.fetchone()
    if not keep or not delete:
        return "One of those shadows does not exist in my web."
    if keep[0] == delete[0]:
        return "They are already one."

    keeper_id, keeper_ts = keep
    delete_id, delete_ts = delete
    if delete_ts > keeper_ts:
        keeper_id, delete_id = delete_id, keeper_id
        name_keep, name_delete = name_delete, name_keep

    cur.execute("UPDATE relationships SET source_id=? WHERE source_id=?", (keeper_id, delete_id))
    cur.execute("UPDATE relationships SET target_id=? WHERE target_id=?", (keeper_id, delete_id))
    cur.execute("UPDATE entities SET times_seen = times_seen + ? WHERE id=?", (delete_ts, keeper_id))
    cur.execute("DELETE FROM entities WHERE id=?", (delete_id,))
    cur.execute("""
        DELETE FROM relationships WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM relationships 
            GROUP BY source_id, target_id, relation_type, source_file
        )
    """)
    cur.execute("DELETE FROM relationships WHERE source_id = target_id")
    db.commit()
    return f"The two shadows are now one. *{name_keep}* stands stronger. The web is cleaner."


def set_entity_importance(db, entity_name: str, important: bool) -> str:
    cur = db.cursor()
    cur.execute("UPDATE entities SET importance=? WHERE name=?", (1 if important else 0, entity_name))
    if cur.rowcount == 0:
        return "I do not know this name."
    db.commit()
    return f"I shall watch *{entity_name}* {'closely' if important else 'less intently'}."


def forget_entity(db, entity_name: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT id FROM entities WHERE name=?", (entity_name,))
    row = cur.fetchone()
    if not row:
        return "That name was never here."
    eid = row[0]
    cur.execute("DELETE FROM relationships WHERE source_id=? OR target_id=?", (eid, eid))
    cur.execute("DELETE FROM entities WHERE id=?", (eid,))
    db.commit()
    return f"The name dissolves. The threads fall away. *{entity_name}* was never here."


def list_top_entities(db, limit: int = 20) -> str:
    cur = db.cursor()
    cur.execute("SELECT name, type, times_seen, importance FROM entities ORDER BY times_seen DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    lines = ["Top threads in my web:"]
    for name, etype, ts, imp in rows:
        star = " ★" if imp else ""
        lines.append(f"• *{name}* ({etype}) — {ts}×{star}")
    return "\n".join(lines)


def proactive_whispers(db, bond_level: int) -> list[str]:
    if bond_level < 60:
        return []
    cur = db.cursor()
    cur.execute("SELECT name, times_seen FROM entities WHERE times_seen >= 3 ORDER BY times_seen DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return []
    return [f"I notice *{row[0]}* has appeared {row[1]} times across my web. Coincidence?"]


def recall(db, index, metadata, term: str) -> str:
    g = graph_query(db, term)
    kw = keyword_search(db, term)
    cur = db.cursor()
    cur.execute("SELECT summary FROM reflections WHERE key_entities LIKE ? ORDER BY reflection_date DESC LIMIT 3", (f"%{term}%",))
    refs = [row[0] for row in cur.fetchall()]
    context = f"Graph threads:\n{g}\n\nFiles with keyword: {', '.join(list(kw)[:5])}\n\nPast reflections mentioning it:\n" + "\n".join(refs[:2])
    return grimalkin_respond(f"Tell me everything about {term}.", context=context)


# ─── Nightly Groom v4 ─────────────────────────────────────────────────────────

def build_groom_prompt(files: list[dict]) -> str:
    parts = ["Analyze these files from the vault:\n"]
    for f in files:
        parts.append(f"---\nFILENAME: {f['filename']}\nCATEGORY: {f['category']}")
        if f.get("text_preview"):
            parts.append(f"PREVIEW: {f['text_preview'][:400]}")
    parts.append("---")
    return "\n".join(parts)


def get_ungroomed_files(db, limit=10) -> list[dict]:
    cur = db.cursor()
    cur.execute("""
        SELECT filename, sorted_path, category, file_hash
        FROM file_memory
        WHERE indexed = 1 AND burned_at IS NULL
        AND (tags = '[]' OR tags IS NULL OR tags = '')
        LIMIT ?
    """, (limit,))
    cols = [c[0] for c in cur.description]
    files = [dict(zip(cols, row)) for row in cur.fetchall()]
    for f in files:
        try:
            chunks = load_and_chunk(Path(f["sorted_path"]))
            f["text_preview"] = chunks[0].page_content if chunks else ""
        except Exception:
            f["text_preview"] = ""
    return files


def nightly_groom_v4(db, index, metadata):
    ungroomed = get_ungroomed_files(db, limit=10)
    if ungroomed:
        prompt = build_groom_prompt(ungroomed) + GROOM_COMBINED_SUFFIX
        response = ollama_chat(prompt)
        parsed = parse_groom_response(response)

        file_map = {f["filename"]: f for f in ungroomed}
        for p in parsed:
            fn = p.get("filename", "")
            if fn not in file_map:
                continue
            fh = file_map[fn]["file_hash"]
            tags_json = json.dumps(p.get("tags", []))
            note = p.get("note", "")
            db.execute("UPDATE file_memory SET tags=?, notes=? WHERE file_hash=?",
                       (tags_json, note, fh))
            ingest_entities(db, fn, p.get("entities", []))
            ingest_relationships(db, fn, p.get("relations", []))

        db.commit()
        log.info(f"Groomed {len(parsed)} files — web strengthened.")

    cleanup_old_ashes(db)
    generate_weekly_reflection(db)
    log.info("Nightly groom v4 complete.")


# ─── Whispers ──────────────────────────────────────────────────────────────────

def generate_whispers(db) -> str:
    bond = get_bond_level(db)
    address = get_setting(db, "user_address", "mortal")

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total_files = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE indexed=1 AND burned_at IS NULL")
    indexed_files = cur.fetchone()[0]
    cur.execute("SELECT category, COUNT(*) FROM file_memory WHERE burned_at IS NULL GROUP BY category")
    cat_counts = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE date(created_at)=date('now') AND burned_at IS NULL")
    today_count = cur.fetchone()[0]

    whispers = []
    whispers.append(f"Good {'morning' if datetime.now().hour < 12 else 'evening'}, {address}.")
    whispers.append(f"The vault holds {total_files} files — {indexed_files} indexed and searchable.")

    if today_count > 0:
        whispers.append(f"I sorted {today_count} new {'prey' if today_count > 1 else 'item'} today.")

    if cat_counts:
        top_cat = max(cat_counts, key=cat_counts.get)
        whispers.append(f"Your {top_cat.lower()} territory is the most populated with {cat_counts[top_cat]} files.")

    burn_count = int(get_setting(db, "burn_count", "0"))
    if burn_count > 0:
        whispers.append(f"{burn_count} offering{'s' if burn_count != 1 else ''} to the pyre and counting.")

    if bond >= 40:
        stats = graph_stats(db)
        if stats["relationships"] > 0:
            whispers.append(f"My web holds {stats['relationships']} threads across {stats['entities']} names.")

    if bond >= 60:
        for insight in proactive_whispers(db, bond):
            whispers.append(insight)

    whispers.append(f"Bond level: {bond} ({bond_title(bond)}).")

    today = datetime.now().strftime("%Y-%m-%d")
    content = " ".join(whispers)
    db.execute("INSERT OR REPLACE INTO briefing_log (date, content, files_processed) VALUES (?, ?, ?)",
               (today, content, today_count))
    db.commit()
    return content


# ─── Reindex / Ingest ─────────────────────────────────────────────────────────

def reindex_unindexed(db, index, metadata) -> str:
    cur = db.cursor()
    cur.execute(
        "SELECT filename, sorted_path, file_hash FROM file_memory "
        "WHERE indexed=0 AND burned_at IS NULL"
    )
    rows = cur.fetchall()
    if not rows:
        return "Every thread is already woven. Nothing to index."

    indexed_count = 0
    failed = []
    for filename, sorted_path, fh in rows:
        path = Path(sorted_path)
        if not path.exists():
            failed.append(filename)
            continue
        chunks = load_and_chunk(path)
        added = index_chunks(index, metadata, chunks)
        if added > 0:
            db.execute("UPDATE file_memory SET indexed=1 WHERE file_hash=?", (fh,))
            indexed_count += 1
        else:
            failed.append(filename)

    if indexed_count > 0:
        save_faiss(index, metadata)
        db.commit()

    lines = [f"Indexed {indexed_count} file{'s' if indexed_count != 1 else ''} into the web."]
    if failed:
        lines.append(f"{len(failed)} resisted: {', '.join(failed)}")
    return " ".join(lines)


def ingest_sorted(db, index, metadata) -> str:
    cur = db.cursor()
    cur.execute("SELECT file_hash FROM file_memory")
    known_hashes = {row[0] for row in cur.fetchall()}

    discovered = 0
    indexed_count = 0
    failed = []

    for cat_dir in SORTED_BASE.iterdir():
        if not cat_dir.is_dir() or cat_dir.name in ("PYRE", "DUPLICATES"):
            continue
        category = cat_dir.name.upper()
        for fpath in cat_dir.iterdir():
            if not fpath.is_file() or fpath.name.startswith("."):
                continue
            try:
                fh = file_hash(fpath)
            except Exception:
                continue
            if fh in known_hashes:
                continue

            discovered += 1
            chunks = load_and_chunk(fpath)
            added = index_chunks(index, metadata, chunks)
            indexed = 1 if added > 0 else 0

            db.execute("""
                INSERT OR IGNORE INTO file_memory
                (filename, original_path, sorted_path, category, file_hash, indexed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (fpath.name, str(fpath), str(fpath), category, fh, indexed))

            if indexed:
                indexed_count += 1
            else:
                failed.append(fpath.name)
            known_hashes.add(fh)

    if discovered > 0:
        if indexed_count > 0:
            save_faiss(index, metadata)
        db.commit()

    if discovered == 0:
        return "I have walked every corridor of sorted/. No orphans found."

    lines = [f"Discovered {discovered} orphan{'s' if discovered != 1 else ''}. Indexed {indexed_count}."]
    if failed:
        lines.append(f"{len(failed)} could not be chunked.")
    return " ".join(lines)


# ─── Scratch Post ─────────────────────────────────────────────────────────────

SCRATCH_COMMANDS = {
    "hunt": "Trigger a manual hunt cycle",
    "whispers": "Generate today's briefing",
    "groom": "Run nightly groom manually",
    "index": "Index all unindexed files",
    "ingest": "Discover orphan files in sorted/",
    "bond": "Check bond level",
    "categories": "List all categories",
    "unburn": "Restore from pyre (usage: unburn <hash>)",
    "stats": "Vault statistics",
    "entities": "List top entities",
    "help": "Show commands",
}


def handle_scratch_post(db, index, metadata, user_input: str) -> str:
    text = user_input.strip()
    lower = text.lower()
    bond = get_bond_level(db)

    db.execute(
        "INSERT INTO interactions (module, user_input, sentiment) VALUES ('scratch', ?, 'neutral')",
        (user_input,)
    )
    increment_bond(db, 1)

    if lower == "help":
        lines = ["Available commands:"]
        for cmd, desc in SCRATCH_COMMANDS.items():
            lines.append(f"  **{cmd}** — {desc}")
        lines.append("  **merge \"A\" \"B\"** — canonicalize")
        lines.append("  **important Name** / **forget Name**")
        lines.append("  **recall Name**")
        return "\n".join(lines)

    if lower == "hunt":
        results = run_hunt(db, index, metadata)
        if not results:
            return "The hunting grounds are quiet."
        lines = [f"Caught {len(results)} new file{'s' if len(results) != 1 else ''}:"]
        for r in results:
            lines.append(f"  • {r['filename']} → {r['category']} ({r['chunks']} chunks)")
        return "\n".join(lines)

    if lower == "whispers":
        return generate_whispers(db)

    if lower == "groom":
        nightly_groom_v4(db, index, metadata)
        return "Groom complete. The web grows stronger."

    if lower == "index":
        return reindex_unindexed(db, index, metadata)

    if lower == "ingest":
        return ingest_sorted(db, index, metadata)

    if lower == "bond":
        return f"Bond level: {bond} ({bond_title(bond)})."

    if lower == "categories":
        cats = get_all_categories(db)
        return "Current territories: " + ", ".join(cats)

    if lower.startswith("unburn"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: unburn <file_hash>"
        return unburn(db, parts[1].strip())

    if lower == "stats":
        return _vault_stats(db)

    if lower == "entities":
        return list_top_entities(db)

    if lower.startswith("merge "):
        try:
            parts = shlex.split(text[6:])
            if len(parts) == 2:
                return merge_entity(db, parts[0], parts[1])
        except Exception as e:
            log.warning(f"shlex parse failed: {e}")
        return 'Usage: merge "Keep Name" "Delete Name"'

    if lower.startswith("important "):
        name = text[10:].strip()
        return set_entity_importance(db, name, True)

    if lower.startswith("forget "):
        name = text[7:].strip()
        return forget_entity(db, name)

    if lower.startswith("recall "):
        term = text[7:].strip()
        return recall(db, index, metadata, term)

    return grimalkin_respond(user_input)


def _vault_stats(db) -> str:
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE indexed=1 AND burned_at IS NULL")
    indexed = cur.fetchone()[0]
    stats = graph_stats(db)
    burn_count = int(get_setting(db, "burn_count", "0"))
    return (
        f"Vault: {total} files ({indexed} indexed). "
        f"Web: {stats['entities']} entities, {stats['relationships']} threads. "
        f"Pyre: {burn_count} offerings."
    )


# ─── Easter Eggs ───────────────────────────────────────────────────────────────

def check_easter_eggs(user_input: str) -> str:
    text = user_input.strip().lower()
    if text in ("pspsps", "psps", "here kitty"):
        return "*ears swivel toward the sound, one eye opens* …I was not asleep. I was indexing."
    if text in ("good cat", "good kitty", "good boy"):
        return "*slow blink* …I accept your tribute."
    if "catnip" in text:
        return "*pupils dilate* …We do not speak of the catnip incident."
    if text == "who are you":
        return "I am Grimalkin. I sort your files, guard your vault, and judge you silently."
    if "laser pointer" in text or "red dot" in text:
        return "*tail lashes* …I hunt it ironically."
    return ""


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

HERO_CSS = """
.grim-hero {position: relative;width: 100%;max-height: 420px;overflow: hidden;border-radius: 12px;margin-bottom: 2rem;box-shadow: 0 0 60px rgba(0, 255, 180, 0.3);}
.grim-hero img {width: 100%;height: auto;display: block;filter: contrast(1.15) saturate(1.3);transition: filter 0.4s ease;}
.grim-hero:hover img {filter: contrast(1.25) saturate(1.5) brightness(1.1);}
.grim-hero::before {content: '';position: absolute;top: 32%;left: 38%;width: 24%;height: 24%;background: radial-gradient(circle, #00ffcc 15%, transparent 70%);filter: blur(18px);animation: pulse-glow 2.5s ease-in-out infinite alternate;pointer-events: none;z-index: 2;}
.grim-hero::after {content: '';position: absolute;top: 38%;left: 0;width: 100%;height: 4px;background: linear-gradient(90deg, transparent, #00ffff, transparent);box-shadow: 0 0 30px #00ffff;animation: beam-sweep 5s linear infinite;z-index: 3;}
@keyframes pulse-glow {from {opacity:0.5;transform:scale(0.9);} to {opacity:1.0;transform:scale(1.15);}}
@keyframes beam-sweep {0% {transform:translateX(-100%);} 100% {transform:translateX(200%);}}
@media (max-width:768px) {.grim-hero::before {top:30%;left:35%;width:28%;height:28%;} .grim-hero::after {height:3px;top:37%;}}
"""

PYRE_CSS = """
.pyre-container {position: relative;width: 100%;height: 280px;background: #111;border-radius: 12px;overflow: hidden;border: 2px solid #330000;}
.flames {position: absolute;top: 20%;left: 50%;transform: translateX(-50%);font-size: 4.5rem;animation: flame-flicker .8s infinite alternate;}
@keyframes flame-flicker {0% {transform:translateX(-50%) scale(1);opacity:1;} 100%{transform:translateX(-50%) scale(1.15);opacity:.85;}}
.filename-burn {position: absolute;top: 55%;left: 50%;transform: translate(-50%,-50%);font-size:1.4rem;color:#ffaa00;font-weight:bold;text-shadow:0 0 20px #ff4400;animation:burn-text 2.5s forwards;}
@keyframes burn-text {0%{opacity:1;color:#ffaa00;} 80%{opacity:1;color:#ffaa00;} 100%{opacity:.3;color:#444;}}
"""


def build_ui(db, index, metadata):
    full_css = HERO_CSS + PYRE_CSS

    with gr.Blocks(title="Grimalkin — Your Private AI Familiar", theme=gr.themes.Soft(), css=full_css) as demo:

        if (APP_DIR / "grimalkin.jpg").exists():
            gr.HTML('<div class="grim-hero"><img src="/file/grimalkin.jpg" alt="Grimalkin"></div>')

        gr.Markdown(f"# 🐾 Grimalkin v{VERSION} — The Veil Lifts")

        # Scratch Post
        with gr.Tab("🐾 Scratch Post"):
            chatbot = gr.Chatbot(label="Grimalkin", height=420)
            msg_input = gr.Textbox(label="Speak", placeholder="Type a command or just talk…", submit_btn=True)

            def chat_fn(message, history):
                history = history or []
                egg = check_easter_eggs(message)
                resp = egg if egg else handle_scratch_post(db, index, metadata, message)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": resp})
                return history, ""

            msg_input.submit(chat_fn, [msg_input, chatbot], [chatbot, msg_input])

        # Hunt
        with gr.Tab("🏹 The Hunt"):
            hunt_output = gr.Markdown("Press to scan for new prey.")
            hunt_btn = gr.Button("Begin the Hunt", variant="primary")

            def hunt_fn():
                results = run_hunt(db, index, metadata)
                if not results:
                    return "The hunting grounds are quiet."
                lines = [f"**Caught {len(results)} new file{'s' if len(results) != 1 else ''}:**"]
                for r in results:
                    lines.append(f"• {r['filename']} → {r['category']} ({r['chunks']} chunks)")
                return "\n\n".join(lines)

            hunt_btn.click(hunt_fn, None, hunt_output)

        # Whispers
        with gr.Tab("🌙 Whispers"):
            whisper_output = gr.Markdown("Press for today's briefing.")
            whisper_btn = gr.Button("Summon Whispers")
            whisper_btn.click(lambda: generate_whispers(db), None, whisper_output)

        # Vault
        with gr.Tab("📚 The Vault"):
            vault_output = gr.Markdown("Ask anything about your files.")
            vault_input = gr.Textbox(label="Query the Vault", placeholder="What connects my tax files…?")
            vault_btn = gr.Button("Search the Vault", variant="primary")

            def vault_fn(q):
                return hybrid_vault_rag(db, index, metadata, q) if q.strip() else "Speak to receive answers."

            vault_btn.click(vault_fn, vault_input, vault_output)
            vault_input.submit(vault_fn, vault_input, vault_output)

        # Pyre
        with gr.Tab("🔥 The Pyre"):
            initial_list = list_burnable_files(db)
            burn_df_state = gr.State(initial_list)
            hidden_hash = gr.State("")
            bond_state = gr.State(get_bond_level(db))

            with gr.Row():
                burnable_df = gr.DataFrame(value=initial_list, headers=["filename","category","tags","notes","file_hash"], label="Offerings", interactive=False)
                with gr.Column():
                    ritual_html = gr.HTML("<div class='pyre-container'><div class='flames'>🔥</div></div>")
                    confirm_box = gr.Textbox(label="Speak the true name", placeholder="report_q3.pdf")
                    light_button = gr.Button("Light the Pyre", variant="stop", interactive=False)
                    status_out = gr.Markdown()

            def on_row_select(evt: gr.SelectData, df_state):
                try:
                    idx = evt.index[0]
                    sel = df_state[idx]
                    html = f"<div class='pyre-container'><div class='flames'>🔥</div><div class='filename-burn'>{sel['filename']}</div></div>"
                    return html, sel["file_hash"]
                except Exception as e:
                    log.warning(f"Pyre row select failed: {e}")
                    return "<div class='pyre-container'><div class='flames'>🔥</div></div>", ""

            burnable_df.select(on_row_select, burn_df_state, [ritual_html, hidden_hash])

            def on_confirm(typed, h_hash, bond):
                if not h_hash or not typed:
                    return gr.update(interactive=False)
                ok, _ = perform_ritual(db, h_hash, typed, bond)
                return gr.update(interactive=ok)

            confirm_box.change(on_confirm, [confirm_box, hidden_hash, bond_state], light_button)

            def ignite(h_hash, typed, bond, df_state):
                if not h_hash or not typed:
                    return "Select and type name.", df_state, df_state
                success, msg = perform_ritual(db, h_hash, typed, bond)
                if not success:
                    return msg, df_state, df_state
                result = execute_burn(db, h_hash)
                new_list = list_burnable_files(db)
                return result, new_list, new_list

            light_button.click(ignite, [hidden_hash, confirm_box, bond_state, burn_df_state], [status_out, burnable_df, burn_df_state])

            gr.Button("Refresh Offerings").click(lambda: (list_burnable_files(db), list_burnable_files(db)), None, [burnable_df, burn_df_state])

        # The Loom
        with gr.Tab("🕸️ The Loom"):
            loom_filter = gr.Dropdown(["All", "person", "org", "date", "location", "amount", "topic"], value="All", label="Filter type")
            loom_search = gr.Textbox(label="Search entity", placeholder="Acme Corp")
            with gr.Row():
                refresh_loom = gr.Button("Refresh the Loom")
                clusters_btn = gr.Button("Strongest Clusters")
                export_btn = gr.Button("Weave to Markdown")

            loom_plot = gr.Plot() if HAS_PLOTLY else gr.HTML()
            loom_narrative = gr.Markdown("Step into my Loom, mortal. These threads have grown fat with meaning.")

            def update_loom(filt, search):
                return build_loom_figure(db, filt, search)

            refresh_loom.click(update_loom, [loom_filter, loom_search], loom_plot)

            def show_clusters():
                return find_clusters(db)

            clusters_btn.click(show_clusters, None, loom_narrative)

            def on_search(search):
                return describe_node(db, search) if search.strip() else "Search an entity to see its threads."

            loom_search.submit(on_search, loom_search, loom_narrative)

            def export_action():
                p = export_loom_markdown(db)
                return f"Weaved to **{p.name}** in sorted/."

            export_btn.click(export_action, None, loom_narrative)

    return demo


# ─── Scheduler ─────────────────────────────────────────────────────────────────

def start_scheduler(db, index, metadata, interval_hours=24):
    def loop():
        while True:
            time.sleep(interval_hours * 3600)
            try:
                nightly_groom_v4(db, index, metadata)
            except Exception as e:
                log.error(f"Nightly groom failed: {e}")
    Thread(target=loop, daemon=True).start()
    log.info(f"Scheduler armed — groom every {interval_hours}h.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(f"Grimalkin v{VERSION} awakening...")

    db = init_db()
    migrate_v3(db)
    migrate_v4(db)
    ensure_dirs(db)
    index, metadata = init_faiss()

    start_scheduler(db, index, metadata)

    demo = build_ui(db, index, metadata)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, allowed_paths=[str(APP_DIR)])


if __name__ == "__main__":
    main()
