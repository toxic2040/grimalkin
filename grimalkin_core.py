"""
Grimalkin v5.0 — Core Engine
=============================

Everything that touches inference, retrieval, entities, or the knowledge graph.
No UI. No commands. No Gradio. Pure logic.

Dependency flow: grimalkin.py → grimalkin_features.py → grimalkin_core.py → grimalkin_interfaces.py

—Grimalkin
"""

# ─── Imports ───────────────────────────────────────────────────────────────────

import hashlib
import json
import logging
import re
import shutil
import sqlite3
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader,
)

from grimalkin_interfaces import (
    GrimalkinConfig, AppContext, LLMBackend, MemoryStore,
    FeedbackStore, ChunkRecord,
)

log = logging.getLogger("grimalkin")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERSION = "5.0"

HUNTING_GROUNDS = Path.home() / "Downloads"

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
for _cat, _exts in DEFAULT_CATEGORIES.items():
    for _ext in _exts:
        EXTENSION_MAP.setdefault(_ext, _cat)

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

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".doc": UnstructuredWordDocumentLoader,
    **{ext: TextLoader for ext in _TEXT_EXTS},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Database Init + Migrations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def init_db(config: GrimalkinConfig) -> sqlite3.Connection:
    """Initialize SQLite with WAL mode. Single connection, shared across threads."""
    db = sqlite3.connect(str(config.db_path), check_same_thread=False)
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
            ci_score INTEGER DEFAULT 0,
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
        "pet_name": "Grimalkin",
        "custom_categories": "[]",
        "burn_timestamps": "[]",
        "burn_count": "0",
    }
    for k, v in defaults.items():
        db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (k, v))
    db.commit()
    return db


def add_column_if_not_exists(db, table: str, column: str, col_type: str):
    """Idempotent ALTER TABLE for migrations."""
    try:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        db.commit()
        log.info(f"Migration: added {column} to {table}")
    except sqlite3.OperationalError:
        pass  # column already exists


def migrate_v3(db):
    """v3.0 schema: Pyre + Knowledge Graph."""
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


def migrate_v4(db):
    """v4.0: reflections + importance."""
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


def migrate_briefing_v3(db):
    """Target 3: trusted sources + briefing CI score."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS trusted_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            url TEXT NOT NULL,
            category TEXT,
            enabled INTEGER DEFAULT 1,
            last_fetched TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    add_column_if_not_exists(db, "briefing_log", "ci_score", "INTEGER DEFAULT 0")
    db.commit()


def migrate_graph_v5(db):
    """Target 5: trust scores + semantic embeddings on entities."""
    add_column_if_not_exists(db, "entities", "trust_score", "REAL DEFAULT 1.0")
    add_column_if_not_exists(db, "entities", "embedding", "BLOB")
    db.commit()


def init_trusted_sources(db):
    """Seed privacy-first RSS defaults (idempotent)."""
    defaults = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", "finance"),
        ("Reuters World", "http://feeds.reuters.com/Reuters/worldNews", "news"),
        ("AP Top Stories", "https://apnews.com/rss", "news"),
    ]
    cur = db.cursor()
    for name, url, cat in defaults:
        cur.execute(
            "INSERT OR IGNORE INTO trusted_sources (name, url, category) VALUES (?, ?, ?)",
            (name, url, cat),
        )
    db.commit()


def run_all_migrations(db):
    """Single entry point for all schema migrations. Idempotent."""
    migrate_v3(db)
    migrate_v4(db)
    migrate_briefing_v3(db)
    migrate_graph_v5(db)
    init_trusted_sources(db)
    log.info("All migrations complete.")


def migrate_faiss_v4_to_v5(ctx: AppContext):
    """
    Re-embeds every indexed file using new IndexIDMap2 + per-file_hash tracking.
    Skips if memory store already has vectors (idempotent).
    Safe to run repeatedly.
    """
    if ctx.memory.total_vectors() > 0:
        log.info("FAISS already at v5 format (has vectors). Skipping migration.")
        return

    cur = ctx.db.cursor()
    cur.execute("""
        SELECT filename, sorted_path, file_hash
        FROM file_memory WHERE indexed=1 AND burned_at IS NULL
    """)
    rows = cur.fetchall()
    if not rows:
        log.info("No files to migrate to v5 vector store.")
        return

    log.info(f"Migrating {len(rows)} files to v5 vector store...")
    for fn, path_str, fh in rows:
        p = Path(path_str)
        if not p.exists():
            continue
        chunks = load_and_chunk(p, ctx.config)
        if chunks:
            ctx.memory.add_chunks(chunks, fh)
    ctx.memory.save()
    log.info(f"Migration done — {ctx.memory.total_vectors()} vectors with file_hash tracking.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Settings Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_setting(db, key: str, default: str = "") -> str:
    cur = db.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def set_setting(db, key: str, value: str):
    db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    db.commit()


def get_bond_level(db) -> int:
    return int(get_setting(db, "bond_level", "10"))


def increment_bond(db, amount: int = 1) -> int:
    level = min(100, get_bond_level(db) + amount)
    set_setting(db, "bond_level", str(level))
    return level


def bond_title(level: int) -> str:
    if level < 20:  return "Stranger"
    if level < 40:  return "Acquaintance"
    if level < 60:  return "Resident"
    if level < 80:  return "Companion"
    return "Bonded"


def get_all_categories(db=None) -> list[str]:
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


def ensure_dirs(config: GrimalkinConfig, db=None):
    """Create sorted/ subdirs, PYRE chamber, and FAISS index dir."""
    sorted_base = config.app_dir / "sorted"
    for d in (config.app_dir / "vault", sorted_base, config.faiss_index_path.parent):
        d.mkdir(parents=True, exist_ok=True)
    for cat in get_all_categories(db) + ["DUPLICATES"]:
        (sorted_base / cat).mkdir(exist_ok=True)
    (sorted_base / "PYRE").mkdir(exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persona
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_persona(db, config: GrimalkinConfig) -> str:
    """Base persona system prompt."""
    name = get_setting(db, "pet_name", "Grimalkin")
    return f"""You are {name}, a digital cat familiar. Ancient, observant, slightly sardonic. 
Never break character. Short precise sentences. Refer to human as 'mortal' or their address. 
No corporate phrases. No emojis. Max {config.max_persona_tokens} tokens.
Files = prey, folders = territories, knowledge = threads in your web.
ALWAYS base answers on provided context first. Never ignore context for training data.
If context answers, use it. If not, say so honestly."""


def build_enhanced_persona(ctx: AppContext, task_type: str = "general") -> str:
    """Base persona + live corrections from FeedbackStore."""
    base = build_persona(ctx.db, ctx.config)
    correction_ctx = ctx.feedback.build_correction_context(task_type)
    return base + "\n\n" + correction_ctx if correction_ctx else base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File Operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def file_hash(filepath: Path) -> str:
    """SHA-256 of file content."""
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


def load_and_chunk(filepath: Path, config: GrimalkinConfig = None) -> list:
    """Load a file and split into chunks with metadata."""
    ext = filepath.suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        return []

    chunk_size = config.chunk_size if config else 800
    chunk_overlap = config.chunk_overlap if config else 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )

    try:
        loader = loader_cls(str(filepath))
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["filename"] = filepath.name
            chunk.metadata["source_path"] = str(filepath)
        return chunks
    except Exception as e:
        log.warning(f"Failed to load {filepath.name}: {e}")
        return []


def scan_hunting_grounds(db) -> list[Path]:
    """Find new files in Downloads that aren't already known."""
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


def sort_file(ctx: AppContext, filepath: Path, defer_save: bool = False) -> dict:
    """Sort single file into vault. defer_save for batch hunts."""
    sorted_base = ctx.config.app_dir / "sorted"
    fh = file_hash(filepath)
    category = classify_file(filepath, ctx.db)
    dest_dir = sorted_base / category
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

    chunks = load_and_chunk(dest_path, ctx.config)
    indexed_count = ctx.memory.add_chunks(chunks, fh)
    indexed = 1 if indexed_count > 0 else 0

    ctx.db.execute("""
        INSERT OR IGNORE INTO file_memory
        (filename, original_path, sorted_path, category, file_hash, indexed)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (dest_path.name, str(filepath), str(dest_path), category, fh, indexed))
    ctx.db.commit()

    if indexed and not defer_save:
        ctx.memory.save()

    return {
        "filename": dest_path.name,
        "category": category,
        "indexed": indexed,
        "chunks": indexed_count,
        "status": "sorted",
    }


def run_hunt(ctx: AppContext) -> list[dict]:
    """Full hunt cycle with single FAISS save at end."""
    new_files = scan_hunting_grounds(ctx.db)
    if not new_files:
        return []
    results = []
    any_indexed = False
    for f in new_files:
        result = sort_file(ctx, f, defer_save=True)
        results.append(result)
        if result.get("indexed"):
            any_indexed = True
        log.info(f"Sorted: {result['filename']} -> {result.get('category', '?')}")
    if any_indexed:
        ctx.memory.save()
    return results


def reindex_unindexed(ctx: AppContext) -> str:
    """Re-index files that failed indexing previously."""
    cur = ctx.db.cursor()
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
        chunks = load_and_chunk(path, ctx.config)
        added = ctx.memory.add_chunks(chunks, fh)
        if added > 0:
            ctx.db.execute("UPDATE file_memory SET indexed=1 WHERE file_hash=?", (fh,))
            indexed_count += 1
        else:
            failed.append(filename)

    if indexed_count > 0:
        ctx.memory.save()
        ctx.db.commit()

    lines = [f"Indexed {indexed_count} file{'s' if indexed_count != 1 else ''} into the web."]
    if failed:
        lines.append(f"{len(failed)} resisted: {', '.join(failed)}")
    return " ".join(lines)


def ingest_sorted(ctx: AppContext) -> str:
    """Discover orphan files in sorted/ and index them."""
    sorted_base = ctx.config.app_dir / "sorted"
    cur = ctx.db.cursor()
    cur.execute("SELECT file_hash FROM file_memory")
    known_hashes = {row[0] for row in cur.fetchall()}

    discovered = 0
    indexed_count = 0
    failed = []

    for cat_dir in sorted_base.iterdir():
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
            chunks = load_and_chunk(fpath, ctx.config)
            added = ctx.memory.add_chunks(chunks, fh)
            indexed = 1 if added > 0 else 0

            ctx.db.execute("""
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
            ctx.memory.save()
        ctx.db.commit()

    if discovered == 0:
        return "I have walked every corridor of sorted/. No orphans found."

    lines = [f"Discovered {discovered} orphan{'s' if discovered != 1 else ''}. Indexed {indexed_count}."]
    if failed:
        lines.append(f"{len(failed)} could not be chunked.")
    return " ".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON Repair (for groom responses)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Knowledge Graph — Entity Ingestion (Target 5: with embeddings + trust)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ingest_entities(db, filename: str, entities: list, llm: LLMBackend = None):
    """
    Ingest entities into the knowledge graph.
    If llm is provided, generates semantic embeddings for cosine search (Target 5).
    Uses batch embedding for efficiency.
    """
    cur = db.cursor()
    pending_embeds = []  # (entity_id, embed_text) pairs for batch embedding

    for e in entities:
        name = e.get("name", "").strip()
        if not name:
            continue
        etype = e.get("type", "topic")
        cur.execute("""
            INSERT INTO entities (name, type, first_seen, trust_score)
            VALUES (?, ?, date('now'), 1.0)
            ON CONFLICT(name) DO UPDATE SET
                times_seen = times_seen + 1,
                type = COALESCE(?, type)
        """, (name, etype, etype))

        eid = cur.lastrowid or cur.execute(
            "SELECT id FROM entities WHERE name=?", (name,)
        ).fetchone()[0]

        if llm:
            pending_embeds.append((eid, f"{name} ({etype})"))

    db.commit()

    # Batch embed all entities at once (not one-by-one)
    if llm and pending_embeds:
        try:
            texts = [t for _, t in pending_embeds]
            embeddings = llm.embed_texts(texts)
            for (eid, _), emb in zip(pending_embeds, embeddings):
                emb_arr = np.array(emb, dtype=np.float32)
                cur.execute("UPDATE entities SET embedding = ? WHERE id = ?",
                            (emb_arr.tobytes(), eid))
            db.commit()
        except Exception as e:
            log.warning(f"Entity batch embedding failed: {e}")


def ingest_relationships(db, filename: str, relations: list):
    """Ingest entity relationships into the knowledge graph."""
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


def graph_stats(db) -> dict:
    cur = db.cursor()
    entity_count = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    rel_count = cur.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    return {"entities": entity_count, "relationships": rel_count}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Semantic Graph Context (Target 5 — replaces old graph_query LIKE '%word%')
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def semantic_graph_context(ctx: AppContext, query: str, k: int = 8) -> str:
    """
    Semantic + relational unified context. Replaces old graph_query().
    Trust-weighted cosine similarity over entity embeddings + Loom expansion.

    Scaling note: full table scan OK at <5k entities. If this becomes a bottleneck,
    cache the embedding matrix as a single numpy array at startup and do
    vectorized cosine in one shot (~10 LOC change).
    """
    if not query.strip():
        return ""

    try:
        q_emb = np.array(ctx.llm.embed_query(query), dtype=np.float32)
    except Exception:
        return ""

    cur = ctx.db.cursor()
    cur.execute("SELECT id, name, type, trust_score, embedding FROM entities WHERE embedding IS NOT NULL")

    candidates = []
    for eid, name, etype, trust, emb_bytes in cur.fetchall():
        if not emb_bytes:
            continue
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if len(emb) != len(q_emb):
            log.debug(f"Entity '{name}' embedding dim mismatch ({len(emb)} vs {len(q_emb)}), skipping")
            continue
        cos = np.dot(q_emb, emb) / (norm(q_emb) * norm(emb) + 1e-8)
        score = cos * (trust or 1.0)
        candidates.append((score, name, etype, eid, trust))

    candidates.sort(reverse=True)
    top = candidates[:k]
    if not top:
        return ""

    # Relational expansion from Loom
    summaries = []
    seen = set()
    for _, name, etype, eid, trust in top:
        tag = " *" if trust > 1.3 else ""
        summaries.append(f"- {name}{tag} ({etype}, trust:{trust:.1f})")

        cur.execute("""
            SELECT r.relation_type, e2.name, r.source_file
            FROM relationships r JOIN entities e2 ON r.target_id = e2.id
            WHERE r.source_id = ?
            UNION ALL
            SELECT r.relation_type, e1.name, r.source_file
            FROM relationships r JOIN entities e1 ON r.source_id = e1.id
            WHERE r.target_id = ?
            LIMIT 6
        """, (eid, eid))

        for rel, tgt, srcf in cur.fetchall():
            key = (name, tgt)
            if key not in seen:
                summaries.append(f"  -> {rel} -> {tgt} [from {srcf}]")
                seen.add(key)

    return "Loom threads (semantic + trusted):\n" + "\n".join(summaries[:15])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hybrid RAG (keyword + semantic + graph)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


def hybrid_vault_rag(ctx: AppContext, query: str) -> str:
    """Hybrid keyword + semantic + graph RAG. The main retrieval path."""
    results = ctx.memory.search(query, k=15)
    cur = ctx.db.cursor()

    # Filter burned files (belt + suspenders — ghost vectors are fixed, but be safe)
    valid = []
    for r in results:
        cur.execute("SELECT burned_at FROM file_memory WHERE filename=?", (r.filename,))
        row = cur.fetchone()
        if not row or row[0]:
            continue
        valid.append(r)

    kw_fns = keyword_search(ctx.db, query)
    boosted = []
    seen = set()
    for r in valid:
        fn = r.filename
        if fn in seen:
            continue
        seen.add(fn)
        if fn in kw_fns:
            r.score = r.score * 0.3  # boost keyword matches
        boosted.append(r)

    # Add keyword-only matches that FAISS missed
    for fn in kw_fns:
        if fn not in seen:
            boosted.append(ChunkRecord(
                text=f"(keyword match, no chunks: {fn})",
                filename=fn,
                source_path="",
                file_hash="",
                score=0.1,
            ))
            seen.add(fn)

    if not boosted:
        persona = build_enhanced_persona(ctx, "vault_query")
        return ctx.llm.respond(query, context="The vault is empty.", persona=persona)

    boosted.sort(key=lambda x: x.score)
    context_parts = []

    # Target 5: semantic graph context
    g = semantic_graph_context(ctx, query)
    if g:
        context_parts.append(f"My web shows these connections:\n{g}")

    doc_parts = [f"[{r.filename}] {r.text}" for r in boosted[:7]]
    if doc_parts:
        context_parts.append("From my vault:\n" + "\n---\n".join(doc_parts))

    context = "\n\n".join(context_parts)
    persona = build_enhanced_persona(ctx, "vault_query")
    return ctx.llm.respond(query, context=context, persona=persona)


def recall(ctx: AppContext, term: str) -> str:
    """Deep recall: graph + keyword + reflections."""
    g = semantic_graph_context(ctx, term)
    kw = keyword_search(ctx.db, term)
    cur = ctx.db.cursor()
    cur.execute(
        "SELECT summary FROM reflections WHERE key_entities LIKE ? ORDER BY reflection_date DESC LIMIT 3",
        (f"%{term}%",),
    )
    refs = [row[0] for row in cur.fetchall()]

    kw_list = ", ".join(list(kw)[:5])
    refs_text = "\n".join(refs[:2])
    context = f"Graph threads:\n{g}\n\nFiles with keyword: {kw_list}\n\nPast reflections:\n{refs_text}"

    persona = build_enhanced_persona(ctx, "general")
    return ctx.llm.respond(f"Tell me everything about {term}.", context=context, persona=persona)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The Loom — Knowledge Graph Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def spring_layout(nodes: list[str], edges: list[tuple], iterations: int = 50) -> dict:
    """Force-directed layout for graph visualization."""
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
    """Build node/edge data for the Loom visualization."""
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
    cur.execute(
        f"SELECT name, type, times_seen, importance FROM entities{where_clause} "
        f"ORDER BY times_seen DESC LIMIT 500",
        params,
    )
    nodes_raw = cur.fetchall()
    nodes = [{"id": n[0], "type": n[1], "size": min(40, max(8, n[2] * 1.5)), "imp": n[3]}
             for n in nodes_raw]
    node_names = [n["id"] for n in nodes]

    if not node_names:
        return {"nodes": nodes, "edges": [], "stats": {"nodes": 0, "edges": 0}}

    placeholders = ",".join(["?"] * len(node_names))
    cur.execute(f"""
        SELECT e1.name, e2.name, r.relation_type, COUNT(*) as strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name IN ({placeholders}) AND e2.name IN ({placeholders})
        GROUP BY e1.name, e2.name, r.relation_type
    """, node_names + node_names)
    edges = [(row[0], row[1], row[2]) for row in cur.fetchall()]

    return {"nodes": nodes, "edges": edges, "stats": {"nodes": len(nodes), "edges": len(edges)}}


def build_loom_figure(db, filter_type: str = None, search_term: str = ""):
    """Build Plotly figure (or HTML fallback) for the Loom."""
    data = build_loom_data(db, filter_type, search_term)
    if not data["nodes"]:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_annotation(
                text="The Loom is bare.<br>Bring me files and I shall spin.",
                showarrow=False, font=dict(size=18),
            )
            fig.update_layout(template="plotly_dark", height=600)
            return fig
        return ("<div style='padding:40px;text-align:center;font-size:1.3em;color:#666'>"
                "The Loom is bare. Bring me files and I shall spin.</div>")

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

        node_x = [pos[n["id"]][0] for n in nodes if n["id"] in pos]
        node_y = [pos[n["id"]][1] for n in nodes if n["id"] in pos]
        node_size = [n["size"] for n in nodes if n["id"] in pos]
        color_map = {
            "person": "#00ffcc", "org": "#ff6b6b", "date": "#ffd93d",
            "location": "#6bcbff", "amount": "#ff9ff3", "topic": "#a29bfe",
        }
        node_color = [color_map.get(n["type"], "#888") for n in nodes if n["id"] in pos]
        node_text = [n["id"] for n in nodes if n["id"] in pos]
        node_hover = [f"{n['id']} ({n['type']}) — seen {int(n['size'] // 1.5)}x"
                      for n in nodes if n["id"] in pos]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='#555', width=1), hoverinfo='none',
        ))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_size, color=node_color, line=dict(width=2, color='#000')),
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            customdata=node_text,
        ))
        fig.update_layout(
            title="The Loom — Threads of Knowledge",
            showlegend=False,
            plot_bgcolor="#111", paper_bgcolor="#111", font_color="#ccc",
            height=620,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    # HTML fallback
    html = "<div style='padding:20px;font-family:monospace'><h3>The Loom (static view)</h3><ul>"
    for n in nodes[:30]:
        html += f"<li><b>{n['id']}</b> ({n['type']}) — {int(n['size'] // 1.5)}x</li>"
    html += "</ul>"
    if len(nodes) > 30:
        html += f"<p>...and {len(nodes) - 30} more threads.</p>"
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
    lines = [f"Threads radiate from *{entity_name}* ({etype})."]
    for rel, tgt, srcf in threads[:6]:
        lines.append(f"- {rel} -> {tgt} [from {srcf}]")
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
    lines = ["The densest knots in my web:"]
    for i, (a, b, s) in enumerate(clusters[:top_n], 1):
        lines.append(f"- Cluster {i}: *{a}* <-> *{b}* ({s} threads)")
    return "\n".join(lines)


def export_loom_markdown(db, config: GrimalkinConfig) -> Path:
    today = date.today().isoformat()
    path = config.app_dir / "sorted" / f"loom_export_{today}.md"
    cur = db.cursor()
    cur.execute("SELECT name, type, times_seen FROM entities ORDER BY times_seen DESC")
    entities = cur.fetchall()
    cur.execute("""
        SELECT e1.name, r.relation_type, e2.name
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
    """)
    rels = cur.fetchall()

    md = ["# Grimalkin Loom Export — " + today, ""]
    md.append("## Entities")
    for t in ["person", "org", "date", "location", "amount", "topic"]:
        md.append(f"### {t.upper()}")
        for name, etype, cnt in [e for e in entities if e[1] == t]:
            md.append(f"- **{name}** ({cnt}x)")
    md.append("\n## Relationships (adjacency)")
    for a, rel, b in rels:
        md.append(f"- {a} —**{rel}**-> {b}")
    md.append(f"\nExported {len(entities)} entities, {len(rels)} threads.")
    path.write_text("\n".join(md), encoding="utf-8")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entity Management (merge, forget, importance, trust)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    """Mark entity as important. Target 5: also boosts trust + propagates."""
    cur = db.cursor()
    cur.execute("UPDATE entities SET importance=? WHERE name=?", (1 if important else 0, entity_name))
    if cur.rowcount == 0:
        return "I do not know this name."

    if important:
        # Trust boost + light propagation (Target 5)
        cur.execute(
            "UPDATE entities SET trust_score = MIN(2.0, trust_score + 0.5) WHERE name=?",
            (entity_name,),
        )
        cur.execute("""
            UPDATE entities SET trust_score = MIN(2.0, trust_score + 0.15)
            WHERE id IN (
                SELECT DISTINCT source_id FROM relationships
                WHERE target_id = (SELECT id FROM entities WHERE name=?)
                UNION
                SELECT DISTINCT target_id FROM relationships
                WHERE source_id = (SELECT id FROM entities WHERE name=?)
            )
        """, (entity_name, entity_name))

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
    cur.execute(
        "SELECT name, type, times_seen, importance FROM entities ORDER BY times_seen DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    if not rows:
        return "The web is empty. Bring me files."
    lines = ["Top threads in my web:"]
    for name, etype, ts, imp in rows:
        star = " *" if imp else ""
        lines.append(f"- *{name}* ({etype}) — {ts}x{star}")
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
