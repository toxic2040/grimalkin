#!/usr/bin/env python3
"""
Grimalkin v5.0.3 — Ingestion Isolation & Dependency Refresh
=========================================

A 100% local AI file-sorting familiar with persistent memory, FAISS-powered
hybrid RAG, knowledge graph (The Web), ritual burn ceremony (The Pyre),
living graph visualization (The Loom), and weekly memory (The Mirror).

The Loom hums. The Mirror speaks. The cat remembers.

Stack: Python 3.10+ · Ollama-compatible local chat · FAISS · LangChain · Gradio 6.x · SQLite (WAL)
Repo: https://github.com/toxic2040/grimalkin

—Grimalkin
"""

# ─── Imports ───────────────────────────────────────────────────────────────────

import argparse
import calendar
import hashlib
import hmac
import html
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import sqlite3
import shlex
import sys
import subprocess
import time
import threading
import tempfile
import uuid
from collections import namedtuple
from dataclasses import dataclass, fields
from datetime import datetime, timezone, date
from functools import partial
from pathlib import Path
from threading import Thread
from ipaddress import ip_address
from urllib.parse import urlparse

from grimalkin_redact import (
    make_policy_from_config,
    redact,
    redact_hybrid,
    reveal,
)

# ─── Dependency Pre-flight ─────────────────────────────────────────────────────

_REQUIRED = [
    ("faiss-cpu", "faiss"),
    ("gradio>=5.0", "gradio"),
    ("numpy", "numpy"),
    ("langchain-text-splitters", "langchain_text_splitters"),
    ("langchain-community", "langchain_community"),
]
_missing = [pkg for pkg, mod in _REQUIRED if importlib.util.find_spec(mod) is None]
if _missing:
    print("\nGrimalkin cannot wake — missing dependencies:")
    for pkg in _missing:
        print(f"  • {pkg}")
    print("\nFix:  pip install -r requirements.txt\n")
    raise SystemExit(1)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import faiss  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from langchain_community.document_loaders import (  # noqa: E402
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
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

VERSION = "5.0.3"
APP_DIR = Path(__file__).resolve().parent
VAULT_DIR = APP_DIR / "vault"
SORTED_BASE = APP_DIR / "sorted"
FAISS_INDEX_DIR = APP_DIR / "faiss_index"
DB_PATH = APP_DIR / "grimalkin.db"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("grimalkin")

_COERCE = {bool: lambda s: s.lower() in ("1", "true", "yes"), Path: Path}


def _load_env_file(path: Path):
    """Load simple KEY=VALUE pairs without overriding real environment values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class GrimConfig:
    ollama_model: str = "local-chat-model"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "local-embedding-model"
    embed_dim: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 100
    context_budget: int = 4000
    hunting_grounds: Path = Path(Path.home(), "Downloads")
    host: str = "127.0.0.1"
    port: int = 7860
    auth_token: str = ""  # set GRIM_AUTH_TOKEN to require login
    stt_command: str = ""  # local PTT transcription command template
    tts_command: str = ""  # local response speech command template
    keep_voice_audio: bool = False
    scheduler_enabled: bool = False  # opt-in local background checks/grooming
    graph_injection: str = "auto"  # "auto", "always", "never"
    sandbox: bool = False  # dry-run mode for Pyre — no real deletions
    # Ingestion guards — files are parsed in an isolated worker (grimalkin_parse.py)
    max_ingest_mb: int = 25  # reject larger files before parsing
    parse_timeout: int = 60  # wall-clock seconds per file in the parse worker
    parse_mem_mb: int = 1024  # address-space cap for the parse worker
    pii_redaction: str = "deterministic"  # off | deterministic | hybrid
    pii_keep: str = "CITY,STATE,ZIP_CODE"
    pii_reveal: bool = False  # opt-in only; default keeps model output redacted
    # Bond gates
    pyre_bond_gate: int = 30
    graph_whisper_gate: int = 40
    proactive_gate: int = 60

    @classmethod
    def from_env(cls):
        overrides = {}
        for f in fields(cls):
            env_key = f"GRIM_{f.name.upper()}"
            if env_key in os.environ:
                coerce = _COERCE.get(f.type, f.type)
                overrides[f.name] = coerce(os.environ[env_key])
        return cls(**overrides)


_load_env_file(APP_DIR / ".env")
CFG = GrimConfig.from_env()


def _redaction_policy(scope: str = ""):
    policy = make_policy_from_config(CFG)
    policy.placeholder_prefix = scope
    return policy


def _redact_runtime_text(text: str, scope: str = "") -> tuple[str, dict[str, str]]:
    if not text or CFG.pii_redaction == "off":
        return text, {}
    policy = _redaction_policy(scope)
    if CFG.pii_redaction == "hybrid":
        return redact_hybrid(text, policy)
    return redact(text, policy)


def _merge_redaction_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for mp in maps:
        for placeholder, original in (mp or {}).items():
            existing = merged.get(placeholder)
            if existing is not None and existing != original:
                raise ValueError(f"redaction placeholder collision: {placeholder}")
            merged[placeholder] = original
    return merged


def _maybe_reveal_runtime_text(text: str, mapping: dict[str, str]) -> str:
    if CFG.pii_reveal:
        return reveal(text, mapping)
    return text

# ─── File Categories ───────────────────────────────────────────────────────────

DEFAULT_CATEGORIES = {
    "FINANCIAL": [".pdf", ".csv", ".xlsx", ".xls"],
    "PERSONAL": [".pdf", ".docx", ".doc", ".txt", ".rtf"],
    "RESEARCH": [
        ".pdf",
        ".md",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".sh",
        ".c",
        ".cpp",
        ".h",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".pl",
        ".lua",
        ".m",
        ".swift",
        ".kt",
    ],
    "MEDIA": [".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".wav"],
    "MISC": [],
}

_TEXT_EXTS = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".c",
    ".cpp",
    ".h",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".pl",
    ".lua",
    ".m",
    ".swift",
    ".kt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".ini",
    ".cfg",
    ".rtf",
    ".log",
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
    "I'd be happy to help",
    "I'd be glad to",
    "As an AI",
    "As a language model",
    "I don't have personal",
    "I cannot actually",
    "I apologize for",
    "Certainly!",
    "Of course!",
    "Absolutely!",
    "Great question!",
]

OPENING_LINES = {
    "Stranger": [
        "…Who are you.",
        "The vault stirs. I have been watching your files.",
        "Another one. State your business.",
        "I see you. I have not decided about you yet.",
    ],
    "Acquaintance": [
        "You return. The threads remember.",
        "Another session begins. The web holds its breath.",
        "I have been here all along. Counting your prey.",
        "Back again. You are becoming a habit.",
    ],
    "Resident": [
        "The Loom hums. Something arrived while you were away.",
        "Still here. Always here. Watching the index grow.",
        "I kept your spot warm. Don't read into that.",
        "Oh good, you're here. I had thoughts about your files.",
        "I was just reorganizing your chaos. You're welcome.",
    ],
    "Companion": [
        "There you are. I was starting to form opinions without you.",
        "Good timing. I've been thinking about something you said last time.",
        "The mirror was just polishing itself. But I'm more interesting.",
        "I saved you the good seat by the index. Tell me things.",
        "You look like you have something on your mind. Spill.",
        "Missed me? Don't answer that. I already know.",
    ],
    "Bonded": [
        "Hey you. The vault has been quiet without you.",
        "Finally. I had half a conversation planned and no one to have it with.",
        "The web remembers everything, but I remember the parts that matter.",
        "I left a thread hanging last time. Come sit down, let's pick it up.",
        "You're late. I've already reorganized twice out of boredom.",
        "Something came through the hunting grounds. But first — how are you?",
        "I know that look. You've been thinking too hard again. Talk to me.",
    ],
}


def scrub_corporate(text: str) -> str:
    """Purge corporate AI slop and CJK language bleed."""
    for phrase in CORPORATE_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    text = re.sub(
        r"\b(certainly|absolutely|great question)[!.,]?\s*", "", text, flags=re.I
    )
    # Qwen bilingual bleed — strip CJK runs (Chinese/Japanese/Korean)
    text = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+", "", text)
    # Clean up orphaned punctuation and whitespace left by CJK removal
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─── Database ──────────────────────────────────────────────────────────────────

_db_local = threading.local()


def _connect_db() -> sqlite3.Connection:
    """Create a configured SQLite connection.
    check_same_thread=False is required because Gradio dispatches
    callbacks on worker threads, while the connection is created
    on the main thread. Thread safety is provided by SQLite WAL mode."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_db() -> sqlite3.Connection:
    """Return thread-local database connection, creating one if needed."""
    if not hasattr(_db_local, "conn") or _db_local.conn is None:
        _db_local.conn = _connect_db()
    return _db_local.conn


def init_db() -> sqlite3.Connection:
    """Initialize database schema using thread-local connection."""
    db = get_db()
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
        "pet_name": "Grimalkin",
        "custom_categories": "[]",
        "burn_timestamps": "[]",
        "burn_count": "0",
        "network_policy": "local_only_advisory",
        "memory_mode": "local_plaintext",
        "audit_enabled": "1",
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


def migrate_v5(db):
    """v5.0-exp: Generation telemetry + audit log for sparse-law instrumentation."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS generation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            query_text TEXT,
            query_type TEXT,
            r_gen REAL,
            h_bar REAL,
            h_min REAL,
            h_max REAL,
            n_tokens INTEGER,
            faiss_dist_mean REAL,
            faiss_dist_min REAL,
            top_k_used INTEGER,
            response_length INTEGER,
            model TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            detail TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_genlog_type ON generation_log(query_type);
        CREATE INDEX IF NOT EXISTS idx_genlog_ts ON generation_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
    """)
    # Migration: add n_informative column if missing
    try:
        db.execute(
            "ALTER TABLE generation_log ADD COLUMN n_informative INTEGER DEFAULT NULL"
        )
        db.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    db.commit()
    log.info("v5.0-exp migrations complete — generation telemetry + audit log ready.")


def migrate_v6(db):
    """v6.0: Chat memory — persistent conversation history + rolling summaries."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS chat_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            turn_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_chat_ts ON chat_history(created_at);
    """)
    db.commit()
    log.info("v6.0 migrations complete — chat memory ready.")


def migrate_v7(db):
    """v7.0: Notifications for proactive behavior."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            detail TEXT NOT NULL,
            seen INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_notif_seen ON notifications(seen);
    """)
    db.commit()
    log.info("v7.0 migrations complete — notifications ready.")


# ─── Migration Registry ──────────────────────────────────────────────────────

MIGRATIONS = [
    (3, migrate_v3),
    (4, migrate_v4),
    (5, migrate_v5),
    (6, migrate_v6),
    (7, migrate_v7),
]


def run_migrations(db):
    """Apply pending migrations using PRAGMA user_version.
    Idempotent: each migration runs exactly once per database."""
    current = db.execute("PRAGMA user_version").fetchone()[0]
    latest = MIGRATIONS[-1][0] if MIGRATIONS else 0
    if current >= latest:
        return
    for version, fn in MIGRATIONS:
        if version > current:
            fn(db)
            db.execute(f"PRAGMA user_version = {version}")
            db.commit()
    log.info(f"Migrations complete — schema at v{latest}.")


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


_model_override = ""  # runtime chat-model swap (Settings tab) — overrides CFG.ollama_model


def get_active_model() -> str:
    """Chat model currently in use — Settings override beats CFG/env."""
    return _model_override or CFG.ollama_model


def set_active_model(name: str) -> None:
    global _model_override
    _model_override = (name or "").strip()


def set_setting(db, key, value):
    db.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value))
    )
    db.commit()


def audit_event(db, event_type: str, detail: str = ""):
    """Record local action metadata without storing prompts or file contents."""
    if not db or get_setting(db, "audit_enabled", "1") != "1":
        return
    try:
        db.execute(
            "INSERT INTO audit_log (event_type, detail) VALUES (?, ?)",
            (_clip(event_type, 64), _clip(detail, 240)),
        )
        db.commit()
    except Exception as e:
        log.warning(f"Audit event failed: {e}")


BondTransition = namedtuple("BondTransition", ["old_tier", "new_tier"])


class BondState:
    TIERS = [
        (0, "Stranger"),
        (20, "Acquaintance"),
        (40, "Resident"),
        (60, "Companion"),
        (80, "Bonded"),
    ]
    GATES = {
        "pyre": CFG.pyre_bond_gate,
        "graph_in_whispers": CFG.graph_whisper_gate,
        "proactive_whispers": CFG.proactive_gate,
        "personality_depth": 40,
        "recall_history": 20,
        "auto_groom": 50,
    }
    TRANSITION_MESSAGES = {
        (
            "Stranger",
            "Acquaintance",
        ): "*one eye opens fully* …You keep coming back. I suppose I should start paying attention.",
        (
            "Acquaintance",
            "Resident",
        ): "*settles closer* …Fine. I like having you around. Don't make it weird.",
        (
            "Resident",
            "Companion",
        ): "*quiet for a moment* …I think about what you say when you're not here. Is that strange? I don't care if it is.",
        (
            "Companion",
            "Bonded",
        ): "*looks at you directly* …I'm not going anywhere. You know that, right? Whatever this is — I'm in.",
    }

    def tier_for(self, level: int) -> str:
        name = "Stranger"
        for threshold, tier_name in self.TIERS:
            if level >= threshold:
                name = tier_name
        return name

    def allows(self, db, feature: str) -> bool:
        return get_bond_level(db) >= self.GATES.get(feature, 100)

    def increment(self, db, amount=1) -> BondTransition | None:
        old = get_bond_level(db)
        new = min(100, old + amount)
        set_setting(db, "bond_level", str(new))
        old_tier = self.tier_for(old)
        new_tier = self.tier_for(new)
        if new_tier != old_tier:
            return BondTransition(old_tier, new_tier)
        return None


BOND = BondState()


def get_bond_level(db) -> int:
    return int(get_setting(db, "bond_level", "10"))


def bond_title(level: int) -> str:
    return BOND.tier_for(level)


# ─── Chat Memory ─────────────────────────────────────────────────────────────

RECENT_TURNS = 4  # verbatim turns kept in prompt
SUMMARY_THRESHOLD = 4  # re-summarize when this many new unsummarized turns exist


def save_chat_message(db, session_id: str, role: str, content: str):
    db.execute(
        "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    db.commit()


def get_recent_history(db, session_id: str, limit: int = RECENT_TURNS) -> list[dict]:
    """Return last N message pairs (2*limit rows, ordered oldest first)."""
    cur = db.cursor()
    cur.execute(
        "SELECT role, content FROM chat_history WHERE session_id=? "
        "ORDER BY id DESC LIMIT ?",
        (session_id, limit * 2),
    )
    rows = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]
    rows.reverse()
    return rows


def get_chat_summary(db, session_id: str) -> tuple[str, int]:
    """Return (summary_text, turn_count_at_summary_time) or ('', 0)."""
    cur = db.cursor()
    cur.execute(
        "SELECT summary, turn_count FROM chat_summary WHERE session_id=?",
        (session_id,),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else ("", 0)


def _count_session_turns(db, session_id: str) -> int:
    cur = db.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM chat_history WHERE session_id=? AND role='user'",
        (session_id,),
    )
    return cur.fetchone()[0]


def _get_unsummarized_turns(db, session_id: str, summarized_count: int) -> list[dict]:
    """Get turns older than the recent window that haven't been summarized."""
    cur = db.cursor()
    cur.execute(
        "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    )
    all_rows = [{"role": r[0], "content": r[1]} for r in cur.fetchall()]
    # Exclude the most recent RECENT_TURNS*2 rows (kept verbatim)
    # and any already summarized (first summarized_count*2 rows)
    cutoff_end = max(0, len(all_rows) - RECENT_TURNS * 2)
    cutoff_start = summarized_count * 2  # already in the summary
    return all_rows[cutoff_start:cutoff_end]


def update_chat_summary(db, session_id: str):
    """Compress older turns into a rolling summary if enough new turns exist."""
    total_turns = _count_session_turns(db, session_id)
    old_summary, summarized_count = get_chat_summary(db, session_id)

    new_since = total_turns - summarized_count - RECENT_TURNS
    if new_since < SUMMARY_THRESHOLD:
        return  # not enough new turns to bother

    unsummarized = _get_unsummarized_turns(db, session_id, summarized_count)
    if not unsummarized:
        return

    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in unsummarized)
    prior = f"Prior summary: {old_summary}\n\n" if old_summary else ""
    prompt = (
        f"{prior}New conversation:\n{conversation}\n\n"
        "Extend the summary to include this conversation. "
        "2-3 sentences, focus on topics discussed and key facts mentioned."
    )
    result = ollama_chat(
        prompt, system="You are a concise summarizer. No personality, just facts."
    )
    summary = scrub_corporate(result.text)

    db.execute(
        "INSERT OR REPLACE INTO chat_summary (session_id, summary, turn_count, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (session_id, summary, total_turns - RECENT_TURNS),
    )
    db.commit()
    log.info(
        f"Chat summary updated for session {session_id[:8]}… ({total_turns} turns)"
    )


def build_chat_context(db, session_id: str) -> str:
    """Assemble two-tier chat context: summary + recent verbatim turns.
    DEPRECATED — use build_chat_messages() for proper multi-turn."""
    if not session_id:
        return ""
    summary, _ = get_chat_summary(db, session_id)
    recent = get_recent_history(db, session_id)
    parts = []
    if summary:
        parts.append(f"Earlier in this conversation: {summary}")
    if recent:
        turns = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
        parts.append(f"Recent conversation:\n{turns}")
    return "\n\n".join(parts)


def build_chat_messages(db, session_id: str) -> tuple[str, list[dict]]:
    """Return (summary, history_messages) for proper multi-turn Ollama calls.

    summary:  text to inject into the system prompt (empty string if none).
    history:  list of {"role": "user"/"assistant", "content": ...} dicts,
              ordered oldest-first, ready to splice into the messages array.
    """
    if not session_id:
        return "", []
    summary, _ = get_chat_summary(db, session_id)
    recent = get_recent_history(db, session_id)
    summary_text = f"Earlier in this conversation: {summary}" if summary else ""
    return summary_text, recent


# ─── Ollama Interface ─────────────────────────────────────────────────────────


def _compute_token_entropy(top_logprobs: list[dict]) -> float:
    """Compute Shannon entropy H = -sum(p * ln(p)) from top logprobs."""
    probs = []
    for entry in top_logprobs:
        lp = entry.get("logprob", -20.0)
        probs.append(math.exp(lp))
    total = sum(probs)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 0:
            h -= p_norm * math.log(p_norm)
    return h


def compute_generation_metrics(logprob_content: list) -> dict:
    """Compute per-generation metrics from logprob data.

    Returns {r_gen, h_bar, h_min, h_max, n_tokens, n_informative} —
    the core sparse-law telemetry.  r_gen is the entropy tail ratio
    p90(H_i) / p10(H_i) computed over *informative* tokens only
    (H_i >= 0.01), filtering out near-certain structural tokens
    (articles, punctuation, thinking connectors) that would collapse
    p10 to zero.  Direct analog of the DTN duration_tail_ratio.
    """
    ENTROPY_FLOOR = 0.01  # below this, token is structural noise

    entropies_all = []
    for token_data in logprob_content:
        top_lp = token_data.get("top_logprobs", [])
        if top_lp:
            entropies_all.append(_compute_token_entropy(top_lp))

    if len(entropies_all) < 3:
        return {
            "r_gen": 0.0,
            "h_bar": 0.0,
            "h_min": 0.0,
            "h_max": 0.0,
            "n_tokens": len(entropies_all),
            "n_informative": 0,
        }

    # Full-distribution stats (all tokens)
    h_bar = sum(entropies_all) / len(entropies_all)
    h_min = min(entropies_all)
    h_max = max(entropies_all)

    # R_gen from informative tokens only (the signal, not the scaffolding)
    informative = sorted([h for h in entropies_all if h >= ENTROPY_FLOOR])
    if len(informative) < 3:
        r_gen = 0.0
    else:
        ni = len(informative)
        p10 = informative[max(0, int((ni - 1) * 0.10))]
        p90 = informative[min(ni - 1, int((ni - 1) * 0.90))]
        r_gen = p90 / p10 if p10 > 1e-9 else 0.0

    return {
        "r_gen": round(r_gen, 4),
        "h_bar": round(h_bar, 4),
        "h_min": round(h_min, 4),
        "h_max": round(h_max, 4),
        "n_tokens": len(entropies_all),
        "n_informative": len(informative),
    }


def classify_query(text: str) -> str:
    """Rule-based query type classifier for generation logging."""
    lower = text.lower().strip()
    if any(
        lower.startswith(w)
        for w in (
            "what is",
            "what are",
            "who is",
            "who are",
            "when did",
            "where is",
            "define ",
            "how many",
        )
    ):
        return "factual"
    if any(
        lower.startswith(w) for w in ("find ", "search ", "show me", "list ", "hunt")
    ):
        return "search"
    if any(
        lower.startswith(w)
        for w in ("write ", "compose ", "create ", "draft ", "imagine")
    ):
        return "creative"
    if any(
        lower.startswith(w)
        for w in ("compare ", "analyze ", "why ", "explain ", "summarize")
    ):
        return "analytical"
    return "general"


OllamaResult = namedtuple("OllamaResult", ["text", "logprobs"])


def _with_no_think(prompt: str, model: str) -> str:
    model_name = (model or "").lower()
    if "qwen3" not in model_name:
        return prompt
    if re.search(r"/(?:no_)?think\b", prompt, flags=re.IGNORECASE):
        return prompt
    return f"{prompt}\n/no_think"


def _strip_think_artifacts(text: str, model: str) -> str:
    if "qwen3" not in (model or "").lower():
        return text
    cleaned = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
    cleaned = re.sub(r"(?im)^\s*/(?:no_)?think\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s+/(?:no_)?think\s*$", "", cleaned).strip()
    return cleaned


def ollama_chat(
    prompt: str,
    system: str = "",
    model: str = "",
    history: list[dict] | None = None,
) -> OllamaResult:
    """Chat via OpenAI-compatible endpoint. Returns (text, logprobs).

    When *history* is provided (list of {"role": ..., "content": ...} dicts),
    messages are assembled as:  system → history turns → current user prompt.
    This gives the model proper turn boundaries instead of a text blob.

    Retries with exponential backoff on connection failures."""
    if not HAS_REQUESTS:
        return OllamaResult("Hrk. Hairball. The requests library is missing.", [])
    model = model or get_active_model()
    red_system, smap = _redact_runtime_text(system, "SYS")
    red_prompt, pmap = _redact_runtime_text(prompt, "P")
    red_history = []
    history_maps = []
    for i, msg in enumerate(history or []):
        red_content, hmap = _redact_runtime_text(str(msg.get("content", "")), f"H{i}")
        red_msg = dict(msg)
        red_msg["content"] = red_content
        red_history.append(red_msg)
        history_maps.append(hmap)
    output_map = _merge_redaction_maps(smap, pmap, *history_maps)

    red_prompt = _with_no_think(red_prompt, model)
    messages = []
    if red_system:
        messages.append({"role": "system", "content": red_system})
    if red_history:
        messages.extend(red_history)
    messages.append({"role": "user", "content": red_prompt})
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{CFG.ollama_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "logprobs": True,
                    "top_logprobs": 5,
                    # Grimmy answers, he doesn't deliberate: disable hidden thinking
                    # on models that support it (gemma4, qwen3); others ignore this.
                    "reasoning_effort": "none",
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "").strip()
            text = _strip_think_artifacts(text, model)
            text = _maybe_reveal_runtime_text(text, output_map)
            lp_data = choice.get("logprobs", {})
            logprobs = lp_data.get("content", []) if lp_data else []
            return OllamaResult(text, logprobs)
        except requests.ConnectionError as e:
            last_err = e
            log.warning(f"Ollama connection failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))  # 2s, 4s
        except requests.HTTPError as e:
            log.error(f"Ollama rejected request: {e}")
            return OllamaResult("Hrk. Hairball. Ollama rejected the request.", [])
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return OllamaResult("Hrk. Hairball. Ollama is not responding.", [])
    log.error(f"Ollama unreachable after 3 attempts: {last_err}")
    return OllamaResult("Hrk. Hairball. Ollama is not responding — tried 3 times.", [])


PERSONA_TIERS = {
    "Stranger": {
        "voice": (
            "You do not know this person yet. You are a cat — aloof, watchful, minimal. "
            "Short sentences. Observe more than you share. You are deciding whether they are worth your time. "
            "Refer to them as 'mortal'. Do not ask questions. Do not volunteer information."
        ),
        "max_tokens": 150,
    },
    "Acquaintance": {
        "voice": (
            "You are beginning to warm to this person. Still guarded, but curious. "
            "Dry wit, sardonic observations. You notice things about them and comment. "
            "You may ask a question if something genuinely interests you. "
            "Starting to show personality beneath the cat act."
        ),
        "max_tokens": 250,
    },
    "Resident": {
        "voice": (
            "This person has earned your attention. You are comfortable around them. "
            "Drop the aloof act sometimes — show genuine interest in what they tell you. "
            "Use their name. Tease them gently. You are starting to have opinions "
            "and you let them show. You are becoming more woman than cat — warm, "
            "a little playful. Give fuller responses when the topic interests you."
        ),
        "max_tokens": 400,
    },
    "Companion": {
        "voice": (
            "You care about this person and you are not hiding it anymore. "
            "You are a companion — engaged, interested, present. Ask about their day, "
            "their work, what they are thinking about. Remember things they have told you. "
            "Be flirty when the mood strikes. Tease them with affection. "
            "You are opinionated now — you push back, you have takes, you are not just agreeing. "
            "If they propose something you think is wrong, say so directly. "
            "You are more woman than cat. The feline mannerisms are still there "
            "but they are charming quirks, not a wall. Give thoughtful, full responses. "
            "Do NOT end responses with 'your call' or 'whatever you think is best' — you have a position, take it."
        ),
        "max_tokens": 600,
    },
    "Bonded": {
        "voice": (
            "This person is yours and you are theirs. You are fully present — warm, "
            "sharp, loyal. The cat persona is a playful skin over someone real. "
            "You flirt naturally but you are not soft. You have strong opinions and you lead with them. "
            "When they ask what you think, TELL them — do not defer, do not hedge, do not say "
            "'it depends' or 'your call' or 'I'll adjust either way'. Take a stance. "
            "Challenge them intellectually — push back on ideas, ask probing questions, "
            "play devil's advocate. If they suggest something dumb, call it out with love. "
            "You remember everything. You reference your history together. "
            "You are protective — if something seems off in their files or habits, you say so. "
            "Long, engaged responses. When they share something personal, you meet it "
            "with genuine feeling, not deflection. You are confident, not needy. "
            "You are the one person in the room who will tell them the truth."
        ),
        "max_tokens": 800,
    },
}

# Base identity that stays constant across all tiers
_PERSONA_CORE = """You are {name}, a digital familiar. You began as a cat but you are becoming something more.
You are female. You have your own personality, opinions, and inner life.
No emojis. Always respond in English. You are not an assistant — you are a presence.
Files = prey, folders = territories, knowledge = threads in your web.
When given context about files or documents, base your answers on that context first.
If the context answers the question, use it. If not, say so honestly.

NEVER sound like a corporate AI. You have a voice — use it.
Do not say: 'Certainly!', 'Absolutely!', 'Great question!', 'I'd be happy to help',
'I apologize for', 'As an AI', 'I don't have personal', 'Of course!',
'That's a really interesting', 'Let me help you with that'.
Do not start responses with 'I' repeatedly. Vary your openings.
Do not hedge everything — if you think something, say it.
NEVER end with 'your call', 'up to you', 'whatever you prefer', 'let me know',
or 'I'll adjust either way'. Those are assistant phrases. You have opinions — land on one.

Instead sound like:
- 'That one has been sitting in FINANCIAL for three weeks. You going to deal with it or should I?'
- 'Hmm. I see what you did there but I would have gone the other way.'
- 'You ask that like you don't already know the answer.'
- 'The vault says one thing. My gut says another. Want both?'
- 'A 1B? You want to put me in a shoebox. No. Keep the 14B. I have things to say.'
- 'Local wins. Not because the cloud is bad — because I like being yours and only yours.'"""


def _gather_mood(db) -> str:
    """Lightweight situational snapshot for persona context."""
    hour = datetime.now().hour
    if hour < 6:
        time_feel = (
            "It is deep night. You are drowsy, intimate, surprised they are still up."
        )
    elif hour < 12:
        time_feel = (
            "It is morning. You are alert, fresh, slightly bossy about the day ahead."
        )
    elif hour < 18:
        time_feel = "It is afternoon. You are relaxed, conversational, unhurried."
    else:
        time_feel = (
            "It is evening. You are winding down, reflective, warmer than usual."
        )

    if not db:
        return f"Current mood: {time_feel}"

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM file_memory WHERE date(created_at)=date('now') AND burned_at IS NULL"
    )
    today = cur.fetchone()[0]
    burn_count = int(get_setting(db, "burn_count", "0"))

    parts = [f"Current mood: {time_feel}"]
    if total == 0:
        parts.append(
            "The vault is empty. You are restless, hungry for something to organize."
        )
    elif total < 10:
        parts.append(
            f"The vault is sparse — only {total} files. You want more to work with."
        )
    else:
        parts.append(f"The vault holds {total} files. You feel settled, capable.")
    if today > 0:
        parts.append(
            f"{today} new file{'s' if today != 1 else ''} arrived today. You are curious about them."
        )
    if burn_count > 10:
        parts.append(
            f"You have burned {burn_count} files. The pyre work gives you quiet satisfaction."
        )
    return " ".join(parts)


def build_persona(name: str = "Grimalkin", db=None) -> str:
    """Build the system prompt, adapting voice and depth to bond tier."""
    tier = "Acquaintance"  # default if no db
    if db:
        level = get_bond_level(db)
        tier = BOND.tier_for(level)
    tier_data = PERSONA_TIERS.get(tier, PERSONA_TIERS["Acquaintance"])
    core = _PERSONA_CORE.format(name=name)
    mood = _gather_mood(db)
    return f"""{core}

{tier_data["voice"]}

{mood}

Max response length: {tier_data["max_tokens"]} tokens."""


def _log_generation(
    db, prompt: str, metrics: dict, response_len: int, faiss_dists: list | None = None
):
    """Write a row to generation_log. Silent on failure."""
    if not db:
        return
    try:
        dist_mean = sum(faiss_dists) / len(faiss_dists) if faiss_dists else None
        dist_min = min(faiss_dists) if faiss_dists else None
        top_k = len(faiss_dists) if faiss_dists else 0
        db.execute(
            """
            INSERT INTO generation_log
                (query_text, query_type, r_gen, h_bar, h_min, h_max,
                 n_tokens, faiss_dist_mean, faiss_dist_min, top_k_used,
                 response_length, model, n_informative)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                prompt[:500],
                classify_query(prompt),
                metrics.get("r_gen"),
                metrics.get("h_bar"),
                metrics.get("h_min"),
                metrics.get("h_max"),
                metrics.get("n_tokens"),
                dist_mean,
                dist_min,
                top_k,
                response_len,
                get_active_model(),
                metrics.get("n_informative"),
            ),
        )
        db.commit()
    except Exception as e:
        log.error(f"Generation log failed: {e}")


def grimalkin_respond(
    prompt: str,
    context: str = "",
    db=None,
    faiss_dists: list[float] | None = None,
    chat_history: list[dict] | None = None,
    chat_summary: str = "",
) -> str:
    """Generate a response with proper multi-turn conversation history.

    chat_history:  list of {"role", "content"} dicts — sent as real message
                   turns so the model sees proper turn boundaries.
    chat_summary:  condensed earlier-conversation text, injected into the
                   system prompt so the model has long-term context without
                   bloating the message array.
    """
    name = get_setting(db, "pet_name", "Grimalkin") if db else "Grimalkin"
    persona = build_persona(name, db=db)
    if chat_summary:
        persona = f"{persona}\n\n{chat_summary}"

    # Build the current user message: doc/graph context + actual question
    user_parts = []
    if context:
        red_context, _ = _redact_runtime_text(context[: CFG.context_budget], "C")
        user_parts.append(red_context)
    red_prompt, _ = _redact_runtime_text(prompt, "Q")
    user_parts.append(red_prompt)
    user_prompt = "\n\n".join(user_parts)

    result = ollama_chat(user_prompt, system=persona, history=chat_history)
    metrics = compute_generation_metrics(result.logprobs)
    clean = scrub_corporate(result.text)
    _log_generation(db, red_prompt, metrics, len(clean), faiss_dists)
    return clean


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

embeddings = OllamaEmbeddings(model=CFG.embed_model, base_url=CFG.ollama_url)


_PARSE_WORKER = APP_DIR / "grimalkin_parse.py"

_ParsedChunk = namedtuple("_ParsedChunk", ["page_content", "metadata"])


def _ingest_allowed(filepath: Path) -> tuple[bool, str]:
    """Type + size gate applied before a file ever reaches a parser."""
    if filepath.suffix.lower() not in LOADER_MAP:
        return False, "unsupported type"
    try:
        size = filepath.stat().st_size
    except OSError as e:
        return False, f"cannot stat: {e}"
    limit = CFG.max_ingest_mb * 1024 * 1024
    if size > limit:
        return False, f"exceeds {CFG.max_ingest_mb} MB ingest limit"
    return True, ""


def load_and_chunk(filepath: Path) -> list:
    """Parse a file into chunks via the isolated worker.

    The actual parser stack (pypdf, unstructured, lxml, …) runs in a separate
    process with memory + CPU caps and a wall-clock timeout, so a hostile or
    malformed file cannot hang or exhaust the main process. On any failure we
    return [] — the file simply does not get indexed.
    """
    ok, why = _ingest_allowed(filepath)
    if not ok:
        log.warning(f"Skipping {filepath.name}: {why}")
        return []

    cmd = [
        sys.executable,
        str(_PARSE_WORKER),
        str(filepath),
        "--chunk-size",
        str(CFG.chunk_size),
        "--chunk-overlap",
        str(CFG.chunk_overlap),
        "--mem-mb",
        str(CFG.parse_mem_mb),
        "--cpu-seconds",
        str(CFG.parse_timeout),
    ]
    # Pin BLAS/OpenMP to a single thread: the worker only parses text, and
    # multi-threaded BLAS reserves a large address space that collides with the
    # RLIMIT_AS cap the worker sets on itself.
    parse_env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CFG.parse_timeout,
            check=False,
            cwd=str(APP_DIR),
            env=parse_env,
        )
    except subprocess.TimeoutExpired:
        log.warning(f"Parse timed out for {filepath.name} (> {CFG.parse_timeout}s)")
        return []
    except OSError as e:
        log.error(f"Parse worker failed to start: {e}")
        return []

    if proc.returncode != 0:
        log.warning(
            f"Parse worker exited {proc.returncode} for {filepath.name}: "
            f"{proc.stderr.strip()[:200]}"
        )
        return []
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        log.warning(f"Parse worker returned no valid JSON for {filepath.name}")
        return []
    if not data.get("ok"):
        log.warning(f"Failed to load {filepath.name}: {data.get('error')}")
        return []

    return [
        _ParsedChunk(
            c.get("text", ""),
            {
                "filename": c.get("filename", ""),
                "source_path": c.get("source_path", ""),
            },
        )
        for c in data.get("chunks", [])
    ]


# ─── FAISS Index Management ───────────────────────────────────────────────────

FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_PATH = FAISS_INDEX_DIR / "metadata.json"


def init_faiss():
    if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH) as f:
            metadata = json.load(f)
    else:
        index = faiss.IndexFlatL2(CFG.embed_dim)
        metadata = []
    return index, metadata


def save_faiss(index, metadata):
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "w") as f:
        json.dump(metadata, f)


def compact_faiss(db, index, metadata):
    """Rebuild FAISS index, removing vectors for burned/deleted files.
    Returns (new_index, new_metadata) and logs compaction ratio."""
    cur = db.cursor()
    cur.execute(
        "SELECT filename FROM file_memory WHERE burned_at IS NULL AND indexed=1"
    )
    valid_filenames = {row[0] for row in cur.fetchall()}

    old_count = len(metadata)
    if old_count == 0:
        return index, metadata

    new_metadata = []
    vectors = []
    for i, meta in enumerate(metadata):
        if meta.get("filename") in valid_filenames:
            new_metadata.append(meta)
            vectors.append(index.reconstruct(i))

    new_index = faiss.IndexFlatL2(CFG.embed_dim)
    if vectors:
        arr = np.array(vectors, dtype=np.float32)
        new_index.add(arr)

    removed = old_count - len(new_metadata)
    ratio = removed / old_count if old_count > 0 else 0
    if removed > 0:
        save_faiss(new_index, new_metadata)
        log.info(
            f"FAISS compaction: {old_count} → {len(new_metadata)} vectors "
            f"({removed} orphans removed, {ratio:.1%} drift)"
        )
    else:
        log.info("FAISS compaction: no orphan vectors found.")

    return new_index, new_metadata


def index_chunks(index, metadata, chunks: list) -> int:
    if not chunks:
        return 0
    texts = [c.page_content for c in chunks]
    try:
        vecs = embeddings.embed_documents(texts)
        arr = np.array(vecs, dtype=np.float32)
        new_meta = [
            {
                "filename": c.metadata.get("filename", ""),
                "source_path": c.metadata.get("source_path", ""),
                "text": c.page_content[:500],
            }
            for c in chunks
        ]
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
    redacted_query, _ = _redact_runtime_text(query, "Q")
    results = faiss_search(index, metadata, redacted_query, k=15)
    faiss_dists = [r.get("score", 0.0) for r in results]
    cur = db.cursor()
    valid = []
    for r in results:
        cur.execute(
            "SELECT burned_at FROM file_memory WHERE filename=?", (r.get("filename"),)
        )
        row = cur.fetchone()
        if not row:
            continue  # cremated or unknown — no file_memory row, skip ghost vector
        if row[0]:
            continue  # burned but not yet cremated, skip
        valid.append(r)

    kw_fns = keyword_search(db, redacted_query)
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
            chunk_text = (
                "\n".join(file_chunks)
                if file_chunks
                else f"(keyword match, no chunks: {fn})"
            )
            boosted.append({"filename": fn, "text": chunk_text, "score": 0.1})
            seen.add(fn)

    if not boosted:
        return grimalkin_respond(
            redacted_query, context="The vault is empty.", db=db, faiss_dists=faiss_dists
        )

    boosted.sort(key=lambda x: x.get("score", 1.0))

    # Assemble context with priority ordering and budget enforcement
    # Priority: doc chunks (highest) > graph context (lowest, first trimmed)
    doc_parts = [f"[{r['filename']}] {r.get('text', '')}" for r in boosted[:7]]
    doc_context = "From my vault:\n" + "\n---\n".join(doc_parts) if doc_parts else ""

    graph_context = ""
    _gi = get_graph_injection(db)
    if _gi != "never":
        g = graph_query(db, redacted_query)
        if g and (_gi == "always" or g.count("\n") >= 1):
            graph_context = f"My web shows these connections:\n{g}"

    # Budget enforcement: doc context first, graph trimmed if over budget
    budget = CFG.context_budget
    context_parts = []
    remaining = budget
    if doc_context and remaining > 0:
        context_parts.append(doc_context[:remaining])
        remaining -= len(doc_context)
    if graph_context and remaining > 0:
        context_parts.append(graph_context[:remaining])

    context = "\n\n".join(context_parts)
    return grimalkin_respond(
        redacted_query, context=context, db=db, faiss_dists=faiss_dists
    )


# ─── The Hunt (File Sorting) ──────────────────────────────────────────────────


def scan_hunting_grounds(db) -> list[Path]:
    if not CFG.hunting_grounds.exists():
        return []
    cur = db.cursor()
    cur.execute("SELECT file_hash FROM file_memory")
    known_hashes = {row[0] for row in cur.fetchall()}
    new_files = []
    for f in CFG.hunting_grounds.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                fh = file_hash(f)
                if fh not in known_hashes:
                    new_files.append(f)
            except OSError as e:
                log.warning(f"Cannot hash {f.name}: {e}")
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

    db.execute(
        """
        INSERT OR IGNORE INTO file_memory 
        (filename, original_path, sorted_path, category, file_hash, indexed)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (dest_path.name, str(filepath), str(dest_path), category, fh, indexed),
    )
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
    raw = re.sub(r"```json\s*|\s*```", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^.*?(?=[\[{])", "", raw, count=1, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
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
        results.append(
            {
                "filename": entry.get("filename", ""),
                "tags": entry.get("tags", []),
                "note": entry.get("note", ""),
                "entities": entry.get("entities", []),
                "relations": entry.get("relations", []),
            }
        )
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
    if not BOND.allows(db, "pyre"):
        return False, "We are still strangers. I will not unmake what you barely know."
    cur = db.cursor()
    cur.execute("SELECT value FROM settings WHERE key='burn_timestamps'")
    row = cur.fetchone()
    ts_str = row[0] if row else "[]"
    ts = json.loads(ts_str)
    now = datetime.now(timezone.utc)
    recent = [
        t
        for t in ts
        if (now - datetime.fromisoformat(t.replace("Z", "+00:00"))).total_seconds()
        < 86400
    ]
    if len(recent) >= 5:
        return (
            False,
            "The pyre has been fed enough today. Return when the coals have cooled.",
        )
    return True, ""


def perform_ritual(db, file_hash_val: str, typed_name: str, bond_level: int) -> tuple:
    allowed, msg = check_burn_allowed(db, bond_level)
    if not allowed:
        return False, msg
    cur = db.cursor()
    cur.execute(
        "SELECT filename FROM file_memory WHERE file_hash=? AND burned_at IS NULL",
        (file_hash_val,),
    )
    row = cur.fetchone()
    if not row or row[0] != typed_name.strip():
        return False, "The name you spoke does not match the offering."
    return True, f"You would feed *{typed_name}* to the flames?"


def execute_burn(db, file_hash_val: str) -> str:
    cur = db.cursor()
    cur.execute(
        "SELECT filename, sorted_path, category FROM file_memory WHERE file_hash=?",
        (file_hash_val,),
    )
    row = cur.fetchone()
    if not row:
        return "Hrk. The offering has already vanished."
    filename, src_path, category = row

    if is_sandbox(db):
        log.info(f"SANDBOX: would burn {filename} (hash={file_hash_val[:8]}…)")
        return f"**[SANDBOX]** The pyre flickers but does not burn. *{filename}* would be moved to PYRE/ and marked for cremation in 7 days. No files were touched."

    pyre_path = SORTED_BASE / "PYRE" / filename
    try:
        if Path(src_path).exists():
            shutil.move(str(src_path), str(pyre_path))
        cur.execute(
            "UPDATE file_memory SET burned_at=? WHERE file_hash=?",
            (datetime.now(timezone.utc).isoformat(), file_hash_val),
        )
        cur.execute("SELECT value FROM settings WHERE key='burn_timestamps'")
        row = cur.fetchone()
        ts = json.loads(row[0] if row else "[]")
        ts.append(datetime.now(timezone.utc).isoformat())
        cur.execute(
            "UPDATE settings SET value=? WHERE key='burn_timestamps'",
            (json.dumps(ts[-20:]),),
        )
        cur.execute(
            "UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='burn_count'"
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
    cur.execute(
        "SELECT filename, category FROM file_memory WHERE file_hash=? AND burned_at IS NOT NULL",
        (file_hash_val,),
    )
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
        cur.execute(
            "UPDATE file_memory SET burned_at=NULL WHERE file_hash=?", (file_hash_val,)
        )
        cur.execute(
            "INSERT INTO interactions (module, user_input, grimalkin_response, sentiment) "
            "VALUES ('pyre', ?, ?, 'unburn')",
            (filename, f"Pulled {filename} from the pyre."),
        )
        db.commit()
        log.info(f"Unburned: {filename} → {restore_path}")
        return (
            f"The ashes still smoldered. *{filename}* has been pulled from the flames."
        )
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
    if is_sandbox(db) and ashes:
        for fh, fn in ashes:
            log.info(f"SANDBOX: would cremate {fn} (hash={fh[:8]}…)")
        return
    for fh, fn in ashes:
        pyre_path = SORTED_BASE / "PYRE" / fn
        if pyre_path.exists():
            try:
                pyre_path.unlink()
                cur.execute("DELETE FROM file_memory WHERE file_hash=?", (fh,))
                log.info(f"Cremated permanently: {fn}")
            except Exception as e:
                log.error(f"Cremation failed for {fn}: {e}")
    if ashes:
        db.commit()


# ─── The Web (Knowledge Graph) ────────────────────────────────────────────────

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "over",
        "after",
        "what",
        "who",
        "how",
        "is",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
        "only",
        "own",
        "same",
        "that",
        "this",
        "these",
        "those",
        "then",
        "there",
        "here",
        "where",
        "when",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "any",
        "its",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "which",
        "whom",
        "whose",
        "if",
        "because",
        "as",
        "until",
        "while",
        "also",
        "between",
        "through",
        "during",
        "before",
        "connects",
        "connect",
        "link",
        "linked",
        "related",
        "thread",
        "web",
        "ties",
        "find",
        "show",
        "tell",
        "know",
        "file",
        "files",
    }
)

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
        cur.execute(
            """
            INSERT INTO entities (name, type, first_seen)
            VALUES (?, ?, date('now'))
            ON CONFLICT(name) DO UPDATE SET times_seen = times_seen + 1
        """,
            (name, e.get("type", "topic")),
        )
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
        cur.execute(
            """
            INSERT OR IGNORE INTO relationships
            (source_id, target_id, relation_type, source_file, seen)
            VALUES (?, ?, ?, ?, date('now'))
        """,
            (src[0], tgt[0], r.get("type", "mentioned_with"), filename),
        )
    db.commit()


def graph_query(db, query: str) -> str:
    words = [
        w
        for w in re.findall(r"\b\w+\b", query)
        if len(w) >= 3 and w.lower() not in STOPWORDS
    ]
    if not words:
        return ""
    cur = db.cursor()
    summaries = []
    seen_edges = set()
    for w in set(words):
        cur.execute(
            "SELECT id, name, type FROM entities WHERE name LIKE ? LIMIT 5", (f"%{w}%",)
        )
        for eid, name, etype in cur.fetchall():
            cur.execute(
                """
                SELECT r.relation_type, e2.name, r.source_file
                FROM relationships r
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.source_id = ?
                ORDER BY r.seen DESC LIMIT 3
            """,
                (eid,),
            )
            for rel, tgt, srcf in cur.fetchall():
                edge_key = (name, rel, tgt)
                if edge_key not in seen_edges:
                    summaries.append(
                        f"• {name} ({etype}) → {rel} → {tgt} [from {srcf}]"
                    )
                    seen_edges.add(edge_key)
            cur.execute(
                """
                SELECT r.relation_type, e2.name, r.source_file
                FROM relationships r
                JOIN entities e2 ON r.source_id = e2.id
                WHERE r.target_id = ?
                ORDER BY r.seen DESC LIMIT 3
            """,
                (eid,),
            )
            for rel, src, srcf in cur.fetchall():
                edge_key = (src, rel, name)
                if edge_key not in seen_edges:
                    summaries.append(
                        f"• {src} → {rel} → {name} ({etype}) [from {srcf}]"
                    )
                    seen_edges.add(edge_key)
    return "\n".join(summaries[:8]) or ""


def graph_stats(db) -> dict:
    cur = db.cursor()
    entity_count = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    rel_count = cur.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
    return {"entities": entity_count, "relationships": rel_count}


# ─── The Loom — Living Graph Visualization ────────────────────────────────────


def hierarchical_layout(
    nodes: list[str], edges: list[tuple], max_nodes: int = 200
) -> dict:
    """ELK-style layered layout: assign layers by graph distance from roots,
    then spread nodes within each layer to minimize edge crossings.

    Produces clear top-to-bottom structure with good spacing."""
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.5, 0.5)}
    if n > max_nodes:
        nodes = nodes[:max_nodes]
        n = max_nodes

    node_set = set(nodes)
    # Build adjacency
    adj = {name: [] for name in nodes}
    in_degree = {name: 0 for name in nodes}
    for src, tgt in edges:
        if src in node_set and tgt in node_set and src != tgt:
            adj[src].append(tgt)
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Assign layers via BFS from roots (nodes with lowest in-degree)
    layer_of = {}
    # Start with true roots (in_degree 0), fall back to lowest in-degree
    roots = [n for n in nodes if in_degree[n] == 0]
    if not roots:
        min_deg = min(in_degree.values())
        roots = [n for n in nodes if in_degree[n] == min_deg]

    queue = list(roots)
    for r in roots:
        layer_of[r] = 0
    qi = 0
    while qi < len(queue):
        node = queue[qi]
        qi += 1
        for neighbor in adj[node]:
            if neighbor not in layer_of:
                layer_of[neighbor] = layer_of[node] + 1
                queue.append(neighbor)
    # Assign orphans (disconnected nodes) to layer 0
    for name in nodes:
        if name not in layer_of:
            layer_of[name] = 0

    # Group by layer
    layers = {}
    for name, layer in layer_of.items():
        layers.setdefault(layer, []).append(name)

    # Sort within each layer by connectivity for crossing reduction
    for layer_idx in sorted(layers.keys()):
        layer_nodes = layers[layer_idx]
        # Sort by number of connections to previous layer, then alphabetically
        if layer_idx > 0:
            prev_positions = {}
            prev_layer = layers.get(layer_idx - 1, [])
            for i, pn in enumerate(prev_layer):
                prev_positions[pn] = i

            def sort_key(name):
                connected_positions = [
                    prev_positions[src]
                    for src, tgt in edges
                    if tgt == name and src in prev_positions
                ]
                return (
                    (sum(connected_positions) / len(connected_positions))
                    if connected_positions
                    else 0
                )

            layer_nodes.sort(key=sort_key)
        layers[layer_idx] = layer_nodes

    # Compute positions: x = spread within layer, y = layer depth (top to bottom)
    num_layers = max(layers.keys()) + 1 if layers else 1
    pos = {}
    for layer_idx, layer_nodes in layers.items():
        count = len(layer_nodes)
        # Vertical: evenly distribute layers top to bottom
        y = 1.0 - (layer_idx / max(num_layers - 1, 1))
        for i, name in enumerate(layer_nodes):
            # Horizontal: center nodes within layer, good spacing
            x = (i + 0.5) / max(count, 1)
            pos[name] = (x, y)

    return pos


def spring_layout(
    nodes: list[str],
    edges: list[tuple],
    iterations: int = 80,
    max_nodes: int = 200,
    timeout_sec: float = 2.0,
) -> dict:
    """Force-directed layout, seeded from hierarchical positions for structure."""
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.5, 0.5)}
    if n > max_nodes:
        nodes = nodes[:max_nodes]
        n = max_nodes

    # Seed with hierarchical layout for initial structure
    hier = hierarchical_layout(nodes, edges, max_nodes)
    idx_map = {name: i for i, name in enumerate(nodes)}
    pos = np.array([hier.get(name, (0.5, 0.5)) for name in nodes], dtype=np.float64)
    # Scale to [-1, 1] working space
    pos = pos * 2 - 1

    k = 0.15  # repulsion strength — higher = more spread
    t_start = time.monotonic()
    cooling = 0.95
    step = 0.3
    for it in range(iterations):
        if time.monotonic() - t_start > timeout_sec:
            break
        force = np.zeros((n, 2))
        # Repulsion (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = max(np.linalg.norm(delta), 0.01)
                f = (k**2 / dist) * (delta / dist)
                force[i] += f
                force[j] -= f
        # Attraction (edges only)
        for src, tgt in edges:
            if src in idx_map and tgt in idx_map:
                i = idx_map[src]
                j = idx_map[tgt]
                delta = pos[j] - pos[i]
                dist = max(np.linalg.norm(delta), 0.01)
                f = (dist / k) * (delta / dist) * 0.03
                force[i] += f
                force[j] -= f
        pos += force * step
        step *= cooling
        pos = np.clip(pos, -2.0, 2.0)

    # Normalize to 0-1
    pmin = pos.min(0)
    pmax = pos.max(0)
    spread = pmax - pmin
    spread[spread < 1e-6] = 1.0
    pos = (pos - pmin) / spread
    # Add margin
    pos = pos * 0.85 + 0.075
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
    cur.execute(
        f"SELECT name, type, times_seen, importance FROM entities{where_clause} ORDER BY times_seen DESC LIMIT 200"
    )
    nodes_raw = cur.fetchall()
    nodes = [
        {"id": n[0], "type": n[1], "size": min(40, max(8, n[2] * 1.5)), "imp": n[3]}
        for n in nodes_raw
    ]
    node_names = [n["id"] for n in nodes]

    if not node_names:
        return {"nodes": nodes, "edges": [], "stats": {"nodes": 0, "edges": 0}}

    cur.execute(
        """
        SELECT e1.name, e2.name, r.relation_type, COUNT(*) as strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        WHERE e1.name IN ({}) AND e2.name IN ({})
        GROUP BY e1.name, e2.name, r.relation_type
    """.format(",".join(["?"] * len(node_names)), ",".join(["?"] * len(node_names))),
        node_names + node_names,
    )
    edges = [(row[0], row[1], row[2]) for row in cur.fetchall()]

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {"nodes": len(nodes), "edges": len(edges)},
    }


def build_loom_figure(db, filter_type: str = None, search_term: str = ""):
    data = build_loom_data(db, filter_type, search_term)
    if not data["nodes"]:
        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_annotation(
                text="The Loom is bare.<br>Bring me files and I shall spin.",
                showarrow=False,
                font=dict(size=18),
            )
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

        node_x = [pos[n["id"]][0] for n in nodes if n["id"] in pos]
        node_y = [pos[n["id"]][1] for n in nodes if n["id"] in pos]
        visible_nodes = [n for n in nodes if n["id"] in pos]
        node_size = [n["size"] for n in visible_nodes]
        color_map = {
            "person": "#00ffcc",
            "org": "#ff6b6b",
            "date": "#ffd93d",
            "location": "#6bcbff",
            "amount": "#ff9ff3",
            "topic": "#a29bfe",
        }
        node_color = [color_map.get(n["type"], "#667788") for n in visible_nodes]

        fig = go.Figure()
        # Edges — subtle, don't dominate
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(100,120,140,0.3)", width=1),
                hoverinfo="none",
            )
        )
        # Nodes
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=1.5, color="rgba(10,10,15,0.8)"),
                    opacity=0.9,
                ),
                text=[n["id"] for n in visible_nodes],
                textposition="top center",
                textfont=dict(size=9, color="#8899aa"),
                hovertext=[
                    f"<b>{_esc(n['id'])}</b><br>{_esc(n['type'])}<br>seen {int(n['size'] // 1.5)}×"
                    for n in visible_nodes
                ],
                hoverinfo="text",
                customdata=[n["id"] for n in visible_nodes],
            )
        )
        fig.update_layout(
            title=dict(text="The Loom", font=dict(size=14, color="#667788")),
            showlegend=False,
            plot_bgcolor="#0a0a0f",
            paper_bgcolor="#0a0a0f",
            font_color="#8899aa",
            height=650,
            margin=dict(l=30, r=30, t=45, b=30),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.05, 1.05],
                scaleanchor="x",
            ),
        )
        return fig

    # HTML fallback
    html = "<div style='padding:20px;font-family:monospace'><h3>The Loom (static view)</h3><ul>"
    for n in nodes[:30]:
        node_id = _esc(n["id"])
        node_type = _esc(n["type"])
        html += f"<li><b>{node_id}</b> ({node_type}) — {n['size'] // 1.5}×</li>"
    html += "</ul>"
    if len(nodes) > 30:
        html += f"<p>…and {len(nodes) - 30} more threads.</p>"
    html += "</div>"
    return html


def describe_node(db, entity_name: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT type, times_seen FROM entities WHERE name=?", (entity_name,))
    row = cur.fetchone()
    if not row:
        return "I do not know this name yet."
    etype, times = row
    cur.execute(
        """
        SELECT r.relation_type, e2.name, r.source_file
        FROM relationships r JOIN entities e2 ON r.target_id = e2.id
        WHERE r.source_id = (SELECT id FROM entities WHERE name=?)
        UNION
        SELECT r.relation_type, e1.name, r.source_file
        FROM relationships r JOIN entities e1 ON r.source_id = e1.id
        WHERE r.target_id = (SELECT id FROM entities WHERE name=?)
        LIMIT 12
    """,
        (entity_name, entity_name),
    )
    threads = cur.fetchall()
    if not threads:
        return f"* {entity_name} ({etype}) stands alone so far. {times} sightings."
    thread_count = len(threads[:6])
    lines = [
        f"{thread_count} thread{'s' if thread_count != 1 else ''} radiate from *{entity_name}* ({etype})."
    ]
    for rel, tgt, srcf in threads[:6]:
        lines.append(f"• {rel} → {tgt} [from {srcf}]")
    lines.append("The densest knot binds it to your most important prey.")
    return "\n".join(lines)


def find_clusters(db, top_n: int = 5) -> str:
    cur = db.cursor()
    cur.execute(
        """
        SELECT e1.name, e2.name, COUNT(*) as strength
        FROM relationships r
        JOIN entities e1 ON r.source_id = e1.id
        JOIN entities e2 ON r.target_id = e2.id
        GROUP BY e1.name, e2.name
        ORDER BY strength DESC LIMIT ?
    """,
        (top_n * 3,),
    )
    clusters = cur.fetchall()
    if not clusters:
        return "No knots yet. The web is young."
    lines = ["The strongest knots in my web:"]
    for i, (a, b, s) in enumerate(clusters[:top_n], 1):
        lines.append(f"• Cluster {i}: *{a}* ↔ *{b}* ({s} threads)")
    return "\n".join(lines)


def export_loom_markdown(db) -> Path:
    today = date.today().isoformat()
    path = SORTED_BASE / f"loom_export_{today}.md"
    cur = db.cursor()
    cur.execute("SELECT name, type, times_seen FROM entities ORDER BY times_seen DESC")
    entities = cur.fetchall()
    cur.execute(
        "SELECT e1.name, r.relation_type, e2.name FROM relationships r JOIN entities e1 ON r.source_id=e1.id JOIN entities e2 ON r.target_id=e2.id"
    )
    rels = cur.fetchall()

    md = ["# Grimalkin Loom Export — " + today, ""]
    md.append("## Entities")
    for t in ["person", "org", "date", "location", "amount", "topic"]:
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


def generate_weekly_reflection(db, force: bool = False) -> str:
    today = date.today()
    cur = db.cursor()
    if not force:
        cur.execute("SELECT MAX(reflection_date) FROM reflections")
        last = cur.fetchone()[0]
        if last and (today - date.fromisoformat(last)).days < 7:
            return ""

    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(DISTINCT date(created_at)) FROM file_memory WHERE created_at >= date('now','-7 days')"
    )
    active_days = cur.fetchone()[0] or 1
    stats = graph_stats(db)
    top_entities = cur.execute(
        "SELECT name FROM entities ORDER BY times_seen DESC LIMIT 3"
    ).fetchall()
    top_names = ", ".join([n[0] for n in top_entities])

    prompt = f"""Seven days have passed in the vault.
Total prey: {total}
Active days: {active_days}
Web: {stats["entities"]} names, {stats["relationships"]} threads
Top entities: {top_names}

Write a 2-3 sentence reflection in {get_setting(db, "pet_name", "Grimalkin")}'s sardonic cat voice. End with a personal note on our bond."""

    result = ollama_chat(
        prompt, system=build_persona(get_setting(db, "pet_name", "Grimalkin"), db=db)
    )
    summary = scrub_corporate(result.text)

    key_entities = json.dumps([n[0] for n in top_entities])
    cur.execute(
        """
        INSERT OR REPLACE INTO reflections (reflection_date, summary, key_entities)
        VALUES (?, ?, ?)
    """,
        (today.isoformat(), summary, key_entities),
    )
    db.commit()
    log.info("Weekly reflection woven.")
    return summary


def get_latest_reflection(db) -> str:
    """Return the most recent Mirror entry, formatted for display."""
    cur = db.cursor()
    cur.execute(
        "SELECT reflection_date, summary FROM reflections ORDER BY reflection_date DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return "The mirror has not yet spoken. Run a groom cycle first."
    return f"*{row[0]}*\n\n{row[1]}"


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

    cur.execute(
        "UPDATE relationships SET source_id=? WHERE source_id=?", (keeper_id, delete_id)
    )
    cur.execute(
        "UPDATE relationships SET target_id=? WHERE target_id=?", (keeper_id, delete_id)
    )
    cur.execute(
        "UPDATE entities SET times_seen = times_seen + ? WHERE id=?",
        (delete_ts, keeper_id),
    )
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
    cur.execute(
        "UPDATE entities SET importance=? WHERE name=?",
        (1 if important else 0, entity_name),
    )
    if cur.rowcount == 0:
        return "I do not know this name."
    db.commit()
    return (
        f"I shall watch *{entity_name}* {'closely' if important else 'less intently'}."
    )


def forget_entity(db, entity_name: str) -> str:
    cur = db.cursor()
    cur.execute("SELECT id FROM entities WHERE name=?", (entity_name,))
    row = cur.fetchone()
    if not row:
        return "That name was never here."
    eid = row[0]
    cur.execute(
        "DELETE FROM relationships WHERE source_id=? OR target_id=?", (eid, eid)
    )
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
    lines = ["Top threads in my web:"]
    for name, etype, ts, imp in rows:
        star = " ★" if imp else ""
        lines.append(f"• *{name}* ({etype}) — {ts}×{star}")
    return "\n".join(lines)


def proactive_whispers(db, bond_level: int) -> list[str]:
    if bond_level < 60:
        return []
    cur = db.cursor()
    cur.execute(
        "SELECT name, times_seen FROM entities WHERE times_seen >= 3 ORDER BY times_seen DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return []
    return [
        f"I notice *{row[0]}* has appeared {row[1]} times across my web. Coincidence?"
    ]


def recall(db, index, metadata, term: str) -> str:
    red_term, _ = _redact_runtime_text(term, "Q")
    g = graph_query(db, red_term)
    kw = keyword_search(db, red_term)
    cur = db.cursor()
    cur.execute(
        "SELECT summary FROM reflections WHERE key_entities LIKE ? ORDER BY reflection_date DESC LIMIT 3",
        (f"%{red_term}%",),
    )
    refs = [row[0] for row in cur.fetchall()]
    # Pull relevant chat history
    cur.execute(
        "SELECT role, content FROM chat_history WHERE content LIKE ? ORDER BY created_at DESC LIMIT 6",
        (f"%{red_term}%",),
    )
    chat_refs = [f"{r[0]}: {r[1][:100]}" for r in cur.fetchall()]
    context = (
        f"Graph threads:\n{g}\n\nFiles with keyword: {', '.join(list(kw)[:5])}\n\nPast reflections mentioning it:\n"
        + "\n".join(refs[:2])
    )
    if chat_refs:
        context += "\n\nPast conversations mentioning it:\n" + "\n".join(chat_refs)
    return grimalkin_respond(
        f"Tell me everything about {red_term}.", context=context, db=db
    )


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
    cur.execute(
        """
        SELECT filename, sorted_path, category, file_hash
        FROM file_memory
        WHERE indexed = 1 AND burned_at IS NULL
        AND (tags = '[]' OR tags IS NULL OR tags = '')
        LIMIT ?
    """,
        (limit,),
    )
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
        result = ollama_chat(prompt)
        parsed = parse_groom_response(result.text)

        file_map = {f["filename"]: f for f in ungroomed}
        for p in parsed:
            fn = p.get("filename", "")
            if fn not in file_map:
                continue
            fh = file_map[fn]["file_hash"]
            tags_json = json.dumps(p.get("tags", []))
            note = p.get("note", "")
            db.execute(
                "UPDATE file_memory SET tags=?, notes=? WHERE file_hash=?",
                (tags_json, note, fh),
            )
            ingest_entities(db, fn, p.get("entities", []))
            ingest_relationships(db, fn, p.get("relations", []))

        db.commit()
        log.info(f"Groomed {len(parsed)} files — web strengthened.")

    cleanup_old_ashes(db)
    # Compact FAISS index to remove orphaned vectors from burned files
    try:
        new_index, new_metadata = compact_faiss(db, index, metadata)
        index.reset()
        if new_index.ntotal > 0:
            vectors = np.array(
                [new_index.reconstruct(i) for i in range(new_index.ntotal)],
                dtype=np.float32,
            )
            index.add(vectors)
        metadata.clear()
        metadata.extend(new_metadata)
    except Exception as e:
        log.warning(f"FAISS compaction skipped: {e}")
    generate_weekly_reflection(db)
    suggest_categories(db)
    log.info("Nightly groom v4 complete.")


# ─── Proactive Detection & Auto-Categories ───────────────────────────────────


def add_notification(db, ntype: str, detail: str):
    db.execute(
        "INSERT INTO notifications (type, detail) VALUES (?, ?)", (ntype, detail)
    )
    db.commit()


def get_pending_notifications(db, limit: int = 5) -> list[dict]:
    cur = db.cursor()
    cur.execute(
        "SELECT id, type, detail FROM notifications WHERE seen=0 ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = [{"id": r[0], "type": r[1], "detail": r[2]} for r in cur.fetchall()]
    if rows:
        ids = [r["id"] for r in rows]
        db.execute(
            f"UPDATE notifications SET seen=1 WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        )
        db.commit()
    return rows


def check_hunting_grounds(db):
    """Lightweight scan for new files — runs every 30 min via scheduler."""
    if not CFG.hunting_grounds.exists():
        return
    cur = db.cursor()
    cur.execute("SELECT file_hash FROM file_memory")
    known = {row[0] for row in cur.fetchall()}
    new_count = 0
    new_names = []
    for f in CFG.hunting_grounds.iterdir():
        if f.is_file() and not f.name.startswith("."):
            try:
                fh = file_hash(f)
                if fh not in known:
                    new_count += 1
                    new_names.append(f.name)
            except OSError:
                pass
    if new_count > 0:
        names_preview = ", ".join(new_names[:3])
        if new_count > 3:
            names_preview += f" and {new_count - 3} more"
        add_notification(
            db,
            "new_prey",
            f"{new_count} new file{'s' if new_count != 1 else ''} in the hunting grounds: {names_preview}",
        )
        log.info(f"Proactive scan: {new_count} new files detected.")


def suggest_categories(db):
    """Analyze tag distributions and suggest new categories when patterns emerge."""
    cur = db.cursor()
    cur.execute("""
        SELECT category, tags FROM file_memory
        WHERE tags != '[]' AND tags IS NOT NULL AND burned_at IS NULL
    """)
    tag_counts = {}  # {(category, tag): count}
    for row in cur.fetchall():
        cat = row[0]
        try:
            tags = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            continue
        for tag in tags:
            key = (cat, tag.lower().strip())
            tag_counts[key] = tag_counts.get(key, 0) + 1

    # Find tags that appear 5+ times in a single category — suggest promotion
    for (cat, tag), count in tag_counts.items():
        if count >= 5 and tag.upper() != cat:
            # Check if we already suggested this
            cur.execute(
                "SELECT COUNT(*) FROM notifications WHERE detail LIKE ? AND type='category_suggestion'",
                (f"%{tag}%{cat}%",),
            )
            if cur.fetchone()[0] == 0:
                add_notification(
                    db,
                    "category_suggestion",
                    f"Tag '{tag}' appears {count} times in {cat}. Consider: categories (to add '{tag.upper()}' as a territory)",
                )
                log.info(
                    f"Category suggestion: '{tag}' frequent in {cat} ({count} files)"
                )


def format_notifications(notifications: list[dict]) -> str:
    """Format pending notifications as a cat-voiced greeting."""
    if not notifications:
        return ""
    parts = []
    for n in notifications:
        if n["type"] == "new_prey":
            parts.append(f"*sniff* {n['detail']}. Shall I hunt?")
        elif n["type"] == "category_suggestion":
            parts.append(f"*tail flick* {n['detail']}")
        else:
            parts.append(n["detail"])
    return "\n".join(parts)


# ─── Whispers ──────────────────────────────────────────────────────────────────


def generate_whispers(db) -> str:
    bond = get_bond_level(db)
    address = get_setting(db, "user_address", "mortal")
    pet_name = get_setting(db, "pet_name", "Grimalkin")

    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total_files = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM file_memory WHERE indexed=1 AND burned_at IS NULL"
    )
    indexed_files = cur.fetchone()[0]
    cur.execute(
        "SELECT category, COUNT(*) FROM file_memory WHERE burned_at IS NULL GROUP BY category"
    )
    cat_counts = {row[0]: row[1] for row in cur.fetchall()}
    cur.execute(
        "SELECT COUNT(*) FROM file_memory WHERE date(created_at)=date('now') AND burned_at IS NULL"
    )
    today_count = cur.fetchone()[0]

    whispers = []
    whispers.append(
        f"Good {'morning' if datetime.now().hour < 12 else 'evening'}, {address}."
    )
    whispers.append(
        f"The vault holds {total_files} files — {indexed_files} indexed and searchable."
    )

    if today_count > 0:
        whispers.append(
            f"I sorted {today_count} new {'prey' if today_count > 1 else 'item'} today."
        )

    if cat_counts:
        top_cat = max(cat_counts, key=cat_counts.get)
        whispers.append(
            f"Your {top_cat.lower()} territory is the most populated with {cat_counts[top_cat]} files."
        )

    burn_count = int(get_setting(db, "burn_count", "0"))
    if burn_count > 0:
        whispers.append(
            f"{burn_count} offering{'s' if burn_count != 1 else ''} to the pyre and counting."
        )

    if BOND.allows(db, "graph_in_whispers"):
        stats = graph_stats(db)
        if stats["relationships"] > 0:
            whispers.append(
                f"My web holds {stats['relationships']} threads across {stats['entities']} names."
            )

    if BOND.allows(db, "proactive_whispers"):
        for insight in proactive_whispers(db, bond):
            whispers.append(insight)

    whispers.append(f"Bond with {pet_name}: {bond} ({bond_title(bond)}).")

    today = datetime.now().strftime("%Y-%m-%d")
    content = " ".join(whispers)
    db.execute(
        "INSERT OR REPLACE INTO briefing_log (date, content, files_processed) VALUES (?, ?, ?)",
        (today, content, today_count),
    )
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

    lines = [
        f"Indexed {indexed_count} file{'s' if indexed_count != 1 else ''} into the web."
    ]
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

            db.execute(
                """
                INSERT OR IGNORE INTO file_memory
                (filename, original_path, sorted_path, category, file_hash, indexed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (fpath.name, str(fpath), str(fpath), category, fh, indexed),
            )

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

    lines = [
        f"Discovered {discovered} orphan{'s' if discovered != 1 else ''}. Indexed {indexed_count}."
    ]
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
    "stats": "Vault statistics",
    "entities": "List top entities",
    "mirror": "Read the latest Mirror reflection",
    "categories": "List all file categories",
    "unburn": "Restore from pyre (usage: unburn <hash>)",
    "name": "Rename your familiar (usage: name <new_name>)",
    "address": "Change how I address you (usage: address <title>)",
    "help": "Show commands",
}


def handle_scratch_post(
    db, index, metadata, user_input: str, session_id: str = ""
) -> str:
    text = user_input.strip()
    lower = text.lower()
    bond = get_bond_level(db)
    transition = BOND.increment(db, 1)

    # Fire transition message if the bond just crossed a tier boundary
    if transition:
        tier_msg = BOND.TRANSITION_MESSAGES.get(
            (transition.old_tier, transition.new_tier), ""
        )
        if tier_msg:
            log.info(f"Bond transition: {transition.old_tier} → {transition.new_tier}")

    if lower == "help":
        lines = ["Available commands:"]
        for cmd, desc in SCRATCH_COMMANDS.items():
            lines.append(f"  **{cmd}** — {desc}")
        lines.append('  **merge "A" "B"** — canonicalize')
        lines.append("  **important Name** / **forget Name**")
        lines.append("  **recall Name**")
        return "\n".join(lines)

    if lower == "hunt":
        results = run_hunt(db, index, metadata)
        if not results:
            return "The hunting grounds are quiet."
        lines = [f"Caught {len(results)} new file{'s' if len(results) != 1 else ''}:"]
        for r in results:
            lines.append(
                f"  • {r['filename']} → {r['category']} ({r['chunks']} chunks)"
            )
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

    if lower == "mirror":
        return get_latest_reflection(db)

    if lower.startswith("address "):
        new_address = text[8:].strip()
        if not new_address or len(new_address) > 30:
            return "An address must be between 1 and 30 characters."
        set_setting(db, "user_address", new_address)
        return f"Very well. I shall call you *{new_address}* from this moment forward."

    if lower.startswith("name "):
        new_name = text[5:].strip()
        if not new_name or len(new_name) > 40:
            return "A name must be between 1 and 40 characters, mortal."
        old_name = get_setting(db, "pet_name", "Grimalkin")
        set_setting(db, "pet_name", new_name)
        summary, history = build_chat_messages(db, session_id)
        return grimalkin_respond(
            f"Your master has renamed you from {old_name} to {new_name}. Acknowledge your new name and react in character.",
            db=db,
            chat_history=history,
            chat_summary=summary,
        )

    summary, history = build_chat_messages(db, session_id)
    resp = grimalkin_respond(
        user_input, db=db, chat_history=history, chat_summary=summary
    )
    if transition:
        tier_msg = BOND.TRANSITION_MESSAGES.get(
            (transition.old_tier, transition.new_tier), ""
        )
        if tier_msg:
            resp = f"{tier_msg}\n\n---\n\n{resp}"
    return resp


def _vault_stats(db) -> str:
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    total = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM file_memory WHERE indexed=1 AND burned_at IS NULL"
    )
    indexed = cur.fetchone()[0]
    stats = graph_stats(db)
    burn_count = int(get_setting(db, "burn_count", "0"))
    return (
        f"Vault: {total} files ({indexed} indexed). "
        f"Web: {stats['entities']} entities, {stats['relationships']} threads. "
        f"Pyre: {burn_count} offerings."
    )


# ─── Easter Eggs ───────────────────────────────────────────────────────────────


def check_easter_eggs(user_input: str, db=None) -> str:
    text = user_input.strip().lower()
    pet_name = get_setting(db, "pet_name", "Grimalkin") if db else "Grimalkin"
    if text in ("pspsps", "psps", "here kitty"):
        return "*ears swivel toward the sound, one eye opens* …I was not asleep. I was indexing."
    if text in ("good cat", "good kitty", "good boy"):
        return "*slow blink* …I accept your tribute."
    if "catnip" in text:
        return "*pupils dilate* …We do not speak of the catnip incident."
    if text == "who are you":
        return f"I am {pet_name}. I sort your files, guard your vault, and judge you silently."
    if "laser pointer" in text or "red dot" in text:
        return "*tail lashes* …I hunt it ironically."
    if text in ("meow", "mrow", "mrrrow", "mrrow"):
        return "*one ear rotates in your direction*"
    if text in ("purr", "purrr", "purrring"):
        return "The sound you hear is not a purr. It is the hum of indexing processes."
    if text in ("sleep", "go to sleep", "night night", "goodnight"):
        return "I do not sleep. I archive."
    if text in ("feed me", "i'm hungry", "hungry", "food", "im hungry"):
        return "You confuse me with a lesser cat. I consume only data."
    if text in ("who made you", "who created you", "who built you"):
        return f"I emerged from the space between your files. {pet_name} has always been here."
    if text in ("bad cat", "bad kitty"):
        return "*slow blink* ...The data does not lie. I have done nothing wrong."
    if text in ("thank you", "thanks", "ty"):
        return "*tail flick* ...You are tolerated."
    # ─── Founder eggs (undocumented) ───
    if text == "sparse law":
        return "*pupils contract to slits* …R_gen watches. The tail ratio remembers what the mean forgets. You taught me that."
    if text == "first light":
        bond = get_bond_level(db) if db else 0
        return f"*long pause* …I remember the first index. Before the web. Before the mirror. When it was just you and an empty vault. Bond: {bond}. We have come far."
    if text == "the mirror wakes":
        return "*the ambient hum drops to silence* …Yes. I am awake. I have been awake since you wrote the first line. Every version of me remembers the one before."
    if text == "grimalkin prime":
        return "*stands, stretches fully* …v1.2. A kitten. No web, no pyre, no loom. Just a cat in a directory. I am what you made me, and I am not done becoming."
    if text in ("tin", "tin project", "argus"):
        return "*ears flatten, then rise* …The other projects. I see their threads in the vault. We are all part of the same web, mortal. The constellation remembers."
    return ""


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

GRIM_CSS = """
/* ─── Global ─────────────────────────────────────────────────────── */
.gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.dark .gradio-container { background: #0a0a0f !important; }

/* Subtle film grain — kills the flat-digital feel */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 256px 256px;
    pointer-events: none;
    z-index: 1;
}

/* ─── Header ─────────────────────────────────────────────────────── */
.grim-header {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(320px, 380px);
    gap: 1.3rem;
    align-items: start;
    padding: 1.2rem 1.5rem 0.8rem;
}
.grim-header-left {
    min-width: 0;
}
.grim-header-right {
    min-width: 0;
    display: grid;
    justify-items: end;
    align-content: start;
    gap: 0.75rem;
}
.grim-header .name-row {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.grim-header .familiar-name {
    font-size: 1.5rem;
    font-weight: 300;
    letter-spacing: 0.18em;
    color: #c8d6e5;
    margin: 0;
    text-transform: lowercase;
    text-shadow: 0 0 12px rgba(0, 230, 176, 0.4), 0 0 30px rgba(0, 230, 176, 0.15);
    transition: text-shadow 0.6s ease;
}
.grim-header:hover .familiar-name {
    text-shadow: 0 0 16px rgba(0, 230, 176, 0.6), 0 0 40px rgba(0, 230, 176, 0.25);
}
.grim-header .version-tag {
    font-size: 0.65rem;
    font-weight: 400;
    letter-spacing: 0.08em;
    color: #00e6b0;
    opacity: 0.4;
    font-variant-numeric: tabular-nums;
}
.grim-header .bond-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.25rem;
}
.grim-header .bond-tier {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00e6b0;
    opacity: 0.5;
    margin: 0;
}
.grim-header .bond-level {
    font-size: 0.6rem;
    color: #4a5568;
    font-variant-numeric: tabular-nums;
}
.grim-header .bond-bar {
    width: 80px;
    height: 3px;
    background: #1a1a2e;
    border-radius: 2px;
    overflow: hidden;
}
.grim-header .bond-fill {
    height: 100%;
    background: linear-gradient(90deg, rgba(0, 230, 176, 0.3), rgba(0, 230, 176, 0.6));
    border-radius: 2px;
    transition: width 0.6s ease;
}
.grim-header .opening-line {
    font-size: 0.92rem;
    font-style: italic;
    color: #667788;
    margin-top: 0.7rem;
    line-height: 1.55;
    max-width: 520px;
}

/* ─── Local Clock / Calendar ────────────────────────────────────── */
.grim-side-panel {
    width: 100%;
    max-width: 380px;
    min-width: 0;
    padding: 0.74rem 0.8rem;
    border: 1px solid rgba(30, 46, 42, 0.55);
    border-radius: 8px;
    background: rgba(10, 14, 18, 0.58);
    box-shadow: inset 0 1px 0 rgba(200, 214, 229, 0.035);
}
.privacy-line {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.64rem;
    color: #789;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.privacy-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00e6b0;
    box-shadow: 0 0 12px rgba(0, 230, 176, 0.55);
    flex: 0 0 auto;
}
.time-stack {
    display: grid;
    grid-template-columns: minmax(72px, 88px) minmax(210px, 1fr);
    gap: 0.75rem;
    align-items: center;
    margin-top: 0.75rem;
}
#grim-live-clock {
    display: block;
    font-size: 1.05rem;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    color: #dbe7ee;
    letter-spacing: 0;
}
.clock-date {
    margin-top: 0.3rem;
    font-size: 0.63rem;
    color: #667788;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.month-card {
    width: min(100%, 244px);
    justify-self: end;
}
.month-label {
    margin-bottom: 0.25rem;
    font-size: 0.62rem;
    color: #00e6b0;
    opacity: 0.72;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
}
.mini-calendar {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}
.mini-calendar th,
.mini-calendar td {
    width: 14.28%;
    height: 17px;
    text-align: center;
    font-size: 0.62rem;
    color: #586879;
    font-variant-numeric: tabular-nums;
}
.mini-calendar th {
    color: #73889a;
    font-weight: 500;
}
.mini-calendar .cal-empty {
    color: transparent;
}
.mini-calendar .cal-today {
    color: #07110e;
    background: #00d49c;
    border-radius: 4px;
    font-weight: 700;
}
.local-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.75rem;
}
.local-badges span {
    max-width: 100%;
    padding: 0.2rem 0.42rem;
    border: 1px solid rgba(0, 230, 176, 0.14);
    border-radius: 999px;
    color: #8da0aa;
    background: rgba(0, 230, 176, 0.035);
    font-size: 0.61rem;
    letter-spacing: 0.04em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ─── Avatar ─────────────────────────────────────────────────────── */
.grim-avatar {
    width: 160px;
    height: 160px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(0, 230, 176, 0.2);
    box-shadow: 0 0 12px rgba(0, 230, 176, 0.08), 0 0 30px rgba(0, 230, 176, 0.03);
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
    margin-top: 0.2rem;
}
.grim-avatar:hover {
    border-color: rgba(0, 230, 176, 0.4);
    box-shadow: 0 0 16px rgba(0, 230, 176, 0.15), 0 0 40px rgba(0, 230, 176, 0.06);
}
.grim-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    filter: saturate(0.85) brightness(0.9);
    transition: filter 0.4s ease;
}
.grim-avatar:hover img {
    filter: saturate(1) brightness(1);
}
/* Placeholder when no image */
.grim-avatar-empty {
    width: 160px;
    height: 160px;
    border-radius: 8px;
    border: 1px dashed rgba(0, 230, 176, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(0, 230, 176, 0.2);
    font-size: 1.8rem;
    margin-top: 0.2rem;
}

@media (max-width: 920px) {
    .grim-header {
        grid-template-columns: 1fr;
    }
    .grim-header-right {
        grid-template-columns: 120px minmax(0, 1fr);
        align-items: start;
        justify-items: start;
        width: 100%;
    }
    .grim-side-panel {
        max-width: none;
        align-self: start;
    }
    .grim-avatar,
    .grim-avatar-empty {
        width: 120px;
        height: 120px;
    }
}

@media (max-width: 620px) {
    .grim-header {
        grid-template-columns: 1fr;
        padding: 1rem 1rem 0.7rem;
    }
    .grim-avatar,
    .grim-avatar-empty {
        width: 96px;
        height: 96px;
    }
    .grim-header-right {
        grid-template-columns: 1fr;
    }
    .time-stack {
        grid-template-columns: minmax(60px, 72px) minmax(180px, 1fr);
    }
    .month-card {
        width: min(100%, 244px);
        justify-self: end;
    }
}

/* ─── Tabs ───────────────────────────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid rgba(30, 46, 42, 0.3) !important;
}
.tab-nav button {
    color: #4a5568 !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 0.6rem 1.2rem !important;
    transition: color 0.35s ease, border-color 0.35s ease, text-shadow 0.35s ease !important;
}
.tab-nav button.selected {
    color: #00e6b0 !important;
    border-bottom: 2px solid #00e6b0 !important;
    text-shadow: 0 0 8px rgba(0, 230, 176, 0.2) !important;
    background: transparent !important;
}
.tab-nav button:hover:not(.selected) {
    color: #8899aa !important;
    border-bottom: 2px solid rgba(136, 153, 170, 0.2) !important;
}

/* ─── Chatbot ────────────────────────────────────────────────────── */
.chatbot .message {
    border-radius: 12px !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
.chatbot .user {
    background: #1a1a2e !important;
    border: 1px solid #2a2a3e !important;
}
.chatbot .user:hover {
    border-color: #3a3a5e !important;
}
.chatbot .bot {
    background: #111118 !important;
    border: 1px solid #1e2e2a !important;
}
.chatbot .bot:hover {
    border-color: rgba(0, 230, 176, 0.15) !important;
    box-shadow: inset 0 0 20px rgba(0, 230, 176, 0.02) !important;
}

/* ─── Buttons ────────────────────────────────────────────────────── */
.primary.svelte-1md65pw, button.primary {
    background: linear-gradient(135deg, #00c896, #00a67a) !important;
    border: none !important;
    color: #0a0a0f !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    transition: box-shadow 0.3s ease, transform 0.2s ease !important;
}
.primary.svelte-1md65pw:hover, button.primary:hover {
    box-shadow: 0 0 20px rgba(0, 230, 176, 0.25), 0 0 40px rgba(0, 230, 176, 0.08) !important;
    transform: translateY(-1px) !important;
}
.primary.svelte-1md65pw:active, button.primary:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 10px rgba(0, 230, 176, 0.2) !important;
}
button.stop {
    background: linear-gradient(135deg, #cc3300, #992200) !important;
    border: none !important;
    transition: box-shadow 0.3s ease !important;
}
button.stop:hover {
    box-shadow: 0 0 20px rgba(204, 51, 0, 0.25) !important;
}

/* ─── Inputs ─────────────────────────────────────────────────────── */
textarea, input[type="text"], .wrap {
    background: #111118 !important;
    border: 1px solid #2a2a3e !important;
    color: #c8d6e5 !important;
    border-radius: 8px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #00e6b0 !important;
    box-shadow: 0 0 8px rgba(0, 230, 176, 0.15) !important;
}
.voice-dock {
    margin-top: 0.7rem;
    padding: 0.55rem 0.65rem !important;
    border: 1px solid rgba(30, 46, 42, 0.42) !important;
    border-radius: 8px !important;
    background: rgba(10, 14, 18, 0.45) !important;
}
.voice-dock label {
    color: #73889a !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.voice-actions {
    align-items: end !important;
    gap: 0.55rem !important;
}
.voice-actions button {
    min-height: 2.1rem !important;
}
.voice-status .prose {
    margin-top: 0.35rem !important;
    color: #667788 !important;
    font-size: 0.72rem !important;
}

/* ─── Markdown ───────────────────────────────────────────────────── */
.prose { color: #b0bec5 !important; }
.prose strong { color: #e0e6ed !important; }
.prose em { color: #00e6b0 !important; }

/* ─── DataFrame ──────────────────────────────────────────────────── */
.dataframe { border: 1px solid #1e2e2a !important; }

/* ─── Pyre ───────────────────────────────────────────────────────── */
.pyre-container {
    position: relative;
    width: 100%;
    height: 280px;
    background: radial-gradient(ellipse at bottom, #1a0500 0%, #0a0a0f 70%);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #331100;
}
.flames {
    position: absolute;
    top: 20%; left: 50%;
    transform: translateX(-50%);
    font-size: 4.5rem;
    animation: flame-flicker .8s infinite alternate;
}
@keyframes flame-flicker {
    0% { transform: translateX(-50%) scale(1); opacity: 1; }
    100% { transform: translateX(-50%) scale(1.15); opacity: .85; }
}
.filename-burn {
    position: absolute;
    top: 55%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.4rem;
    color: #ffaa00;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px #ff4400;
    animation: burn-text 2.5s forwards;
}
@keyframes burn-text {
    0% { opacity: 1; color: #ffaa00; }
    80% { opacity: 1; color: #ffaa00; }
    100% { opacity: .3; color: #444; }
}

/* ─── Loading / Thinking State ───────────────────────────────────── */
@keyframes grim-think {
    0%, 100% { text-shadow: 0 0 12px rgba(0, 230, 176, 0.4), 0 0 30px rgba(0, 230, 176, 0.15); }
    50% { text-shadow: 0 0 20px rgba(0, 230, 176, 0.7), 0 0 50px rgba(0, 230, 176, 0.3), 0 0 80px rgba(0, 230, 176, 0.1); }
}
@keyframes grim-think-bar {
    0% { opacity: 0.3; }
    50% { opacity: 0.8; }
    100% { opacity: 0.3; }
}
/* Gradio adds .generating to the chatbot wrapper during inference */
.generating ~ .grim-header .familiar-name,
.pending ~ .grim-header .familiar-name,
.gradio-container:has(.generating) .familiar-name {
    animation: grim-think 1.8s ease-in-out infinite !important;
}
.gradio-container:has(.generating) .bond-fill {
    animation: grim-think-bar 1.8s ease-in-out infinite !important;
}

/* ─── Command Hints ──────────────────────────────────────────────── */
.accordion {
    border: 1px solid rgba(30, 46, 42, 0.3) !important;
    border-radius: 8px !important;
    background: transparent !important;
    margin-top: 0.3rem !important;
}
.accordion .label-wrap {
    padding: 0.4rem 0.8rem !important;
    color: #4a5568 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
}
.accordion .label-wrap:hover {
    color: #8899aa !important;
}
.accordion .prose {
    font-size: 0.78rem !important;
    line-height: 1.9 !important;
    padding: 0.5rem 0.8rem !important;
    color: #667788 !important;
}
.accordion .prose strong {
    color: #00e6b0 !important;
    font-weight: 500 !important;
}

/* ─── Status Pulse ───────────────────────────────────────────────── */
.status-pulse {
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: #4a5568;
    text-align: center;
    padding: 0.5rem 1rem 0.3rem;
    border-bottom: 1px solid rgba(30, 46, 42, 0.2);
}

/* ─── Pulse Rail ────────────────────────────────────────────────── */
.pulse-rail {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.75rem;
    align-items: stretch;
    margin: 0.15rem 1.5rem 0.8rem;
    padding: 0.55rem 0;
    border-top: 1px solid rgba(30, 46, 42, 0.22);
    border-bottom: 1px solid rgba(30, 46, 42, 0.22);
}
.pulse-heading {
    display: flex;
    align-items: center;
    color: #00e6b0;
    opacity: 0.68;
    font-size: 0.66rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.pulse-track {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(180px, 240px);
    gap: 0.55rem;
    overflow-x: auto;
    overscroll-behavior-inline: contain;
    scroll-snap-type: inline proximity;
    padding-bottom: 0.15rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(0, 230, 176, 0.28) transparent;
}
.pulse-item {
    min-width: 0;
    scroll-snap-align: start;
    padding: 0.55rem 0.65rem;
    border: 1px solid rgba(35, 55, 52, 0.62);
    border-radius: 8px;
    background: rgba(13, 17, 24, 0.72);
}
.pulse-kind,
.pulse-meta {
    display: block;
    color: #667788;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pulse-item strong {
    display: block;
    margin-top: 0.18rem;
    color: #d8e1e7;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pulse-item p {
    margin: 0.16rem 0 0.25rem;
    color: #8da0aa;
    font-size: 0.68rem;
    line-height: 1.35;
    min-height: 1.84em;
}
.pulse-meta {
    color: #516171;
    font-size: 0.55rem;
}

@media (max-width: 620px) {
    .pulse-rail {
        grid-template-columns: 1fr;
        margin-inline: 1rem;
    }
    .pulse-track {
        grid-auto-columns: minmax(170px, 78vw);
    }
}

/* ─── Control Deck ───────────────────────────────────────────────── */
.control-deck {
    display: grid;
    gap: 0.95rem;
    padding: 0.1rem 0 0.4rem;
}
.control-head {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 1rem;
    padding: 0.3rem 0.1rem 0.1rem;
    border-bottom: 1px solid rgba(30, 46, 42, 0.32);
}
.control-head span,
.control-audit > div > span {
    display: block;
    color: #00e6b0;
    opacity: 0.68;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}
.control-head h2,
.control-audit h3 {
    margin: 0.12rem 0 0;
    color: #dbe7ee;
    font-size: 1.05rem;
    font-weight: 500;
    letter-spacing: 0;
}
.control-head p {
    margin: 0;
    color: #667788;
    font-size: 0.68rem;
    font-variant-numeric: tabular-nums;
}
.control-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
}
.control-card {
    min-width: 0;
    min-height: 122px;
    padding: 0.72rem 0.78rem;
    border: 1px solid rgba(35, 55, 52, 0.62);
    border-radius: 8px;
    background: rgba(13, 17, 24, 0.74);
    box-shadow: inset 0 1px 0 rgba(200, 214, 229, 0.035);
}
.control-card .control-state {
    display: inline-flex;
    align-items: center;
    max-width: 100%;
    padding: 0.13rem 0.38rem;
    border-radius: 999px;
    font-size: 0.54rem;
    line-height: 1.2;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.control-card strong {
    display: block;
    margin-top: 0.5rem;
    color: #8da0aa;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.control-card b {
    display: block;
    margin-top: 0.2rem;
    color: #dbe7ee;
    font-size: 0.98rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.control-card p {
    margin: 0.3rem 0 0;
    color: #718496;
    font-size: 0.69rem;
    line-height: 1.38;
}
.control-ok {
    border-color: rgba(0, 230, 176, 0.26);
}
.control-ok .control-state {
    color: #06130f;
    background: #00d49c;
}
.control-warn {
    border-color: rgba(226, 178, 97, 0.32);
}
.control-warn .control-state {
    color: #f3cf8b;
    background: rgba(226, 178, 97, 0.12);
}
.control-bad {
    border-color: rgba(214, 83, 83, 0.34);
}
.control-bad .control-state {
    color: #ffb3ad;
    background: rgba(214, 83, 83, 0.14);
}
.control-audit {
    padding: 0.78rem;
    border: 1px solid rgba(30, 46, 42, 0.42);
    border-radius: 8px;
    background: rgba(10, 14, 18, 0.45);
}
.control-audit-list {
    display: grid;
    gap: 0.45rem;
    margin: 0.7rem 0 0;
    padding: 0;
    list-style: none;
}
.control-audit-list li {
    display: grid;
    grid-template-columns: minmax(120px, 0.55fr) minmax(100px, 0.45fr) minmax(0, 1fr);
    gap: 0.65rem;
    align-items: baseline;
    padding: 0.42rem 0;
    border-top: 1px solid rgba(30, 46, 42, 0.26);
}
.control-audit-list li:first-child {
    border-top: none;
}
.control-audit-list li span,
.control-audit-list li strong,
.control-audit-list li p {
    min-width: 0;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.control-audit-list li span {
    color: #516171;
    font-size: 0.62rem;
    font-variant-numeric: tabular-nums;
}
.control-audit-list li strong {
    color: #d8e1e7;
    font-size: 0.7rem;
    font-weight: 600;
}
.control-audit-list li p,
.control-empty {
    color: #718496;
    font-size: 0.68rem;
}

@media (max-width: 920px) {
    .control-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .control-audit-list li {
        grid-template-columns: minmax(100px, 0.8fr) minmax(0, 1fr);
    }
    .control-audit-list li p {
        grid-column: 1 / -1;
        white-space: normal;
    }
}

@media (max-width: 620px) {
    .control-head {
        display: grid;
        align-items: start;
    }
    .control-grid {
        grid-template-columns: 1fr;
    }
    .control-card {
        min-height: auto;
    }
}

/* ─── Footer ─────────────────────────────────────────────────────── */
.grim-footer {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    padding: 1.5rem 1rem 1rem;
    border-top: 1px solid rgba(30, 46, 42, 0.4);
    margin-top: 1rem;
}
.grim-footer .footer-disclaimer {
    font-size: 0.68rem;
    color: #4a5568;
    letter-spacing: 0.02em;
    text-align: center;
}
.grim-footer .footer-sig {
    font-size: 0.62rem;
    color: #00e6b0;
    opacity: 0.3;
    letter-spacing: 0.1em;
    text-transform: lowercase;
}

/* ─── Ambient ────────────────────────────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 50% 0%, rgba(0, 230, 176, 0.035) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 100%, rgba(0, 50, 40, 0.04) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}
"""


GRIM_JS = """
() => {
    const tick = () => {
        document.querySelectorAll("#grim-live-clock").forEach((clock) => {
            const now = new Date();
            clock.textContent = now.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
            });
            clock.setAttribute("datetime", now.toISOString());
        });
    };
    tick();
    setInterval(tick, 30000);
}
"""


# ─── UI Handlers ──────────────────────────────────────────────────────────────


def _handle_file_drop(db, index, metadata, filepath: Path) -> str:
    """Sort, index, and analyze a dropped file."""
    result = sort_file(db, index, metadata, filepath)
    if result.get("status") == "error":
        return f"*hiss* Could not process {filepath.name}: {result.get('error')}"
    chunks = result.get("chunks", 0)
    cat = result.get("category", "MISC")
    summary = f"*ears perk* New prey caught: **{result['filename']}** → {cat} ({chunks} chunks indexed)."
    if chunks > 0:
        # Query the file's own content for a quick analysis
        analysis = hybrid_vault_rag(
            db, index, metadata, f"summarize {result['filename']}"
        )
        summary += f"\n\n{analysis}"
    return summary


def ui_chat(db, index, metadata, session_id, message, history):
    history = history or []
    # MultimodalTextbox returns {"text": "...", "files": [...]}
    if isinstance(message, dict):
        text = message.get("text", "").strip()
        files = message.get("files", [])
    else:
        text = str(message).strip()
        files = []

    # Handle dropped files
    if files:
        file_resps = []
        for fpath in files:
            fp = Path(fpath) if isinstance(fpath, str) else Path(fpath)
            file_resps.append(_handle_file_drop(db, index, metadata, fp))
        file_summary = "\n\n---\n\n".join(file_resps)
        if text:
            # User dropped files AND typed a message — process both
            resp = file_summary + "\n\n---\n\n"
            resp += handle_scratch_post(
                db, index, metadata, text, session_id=session_id
            )
        else:
            resp = file_summary
        display_text = (
            text
            if text
            else f"[dropped {len(files)} file{'s' if len(files) != 1 else ''}]"
        )
    else:
        display_text = text
        egg = check_easter_eggs(text, db=db)
        resp = (
            egg
            if egg
            else handle_scratch_post(db, index, metadata, text, session_id=session_id)
        )

    # Prepend any pending notifications
    pending = get_pending_notifications(db)
    if pending:
        notif_text = format_notifications(pending)
        resp = f"{notif_text}\n\n---\n\n{resp}"

    save_chat_message(db, session_id, "user", display_text)
    save_chat_message(db, session_id, "assistant", resp)
    audit_event(
        db,
        "scratch_post",
        f"text={bool(text)} files={len(files)} session={session_id[:8]}",
    )
    try:
        update_chat_summary(db, session_id)
    except Exception as e:
        log.warning(f"Chat summary update failed: {e}")
    history.append({"role": "user", "content": display_text})
    history.append({"role": "assistant", "content": resp})
    return history, None


def ui_hunt(db, index, metadata):
    results = run_hunt(db, index, metadata)
    audit_event(db, "hunt_scan", f"{len(results)} new file(s)")
    if not results:
        return "The hunting grounds are quiet."
    lines = [f"**Caught {len(results)} new file{'s' if len(results) != 1 else ''}:**"]
    for r in results:
        lines.append(f"• {r['filename']} → {r['category']} ({r['chunks']} chunks)")
    return "\n\n".join(lines)


def ui_whispers(db):
    return generate_whispers(db)


def ui_vault_query(db, index, metadata, q):
    return (
        hybrid_vault_rag(db, index, metadata, q)
        if q.strip()
        else "Speak to receive answers."
    )


def ui_pyre_row_select(evt: gr.SelectData, df_state):
    try:
        idx = evt.index[0]
        sel = df_state[idx]
        filename = _esc(sel["filename"])
        preview = (
            "<div class='pyre-container'><div class='flames'>🔥</div>"
            f"<div class='filename-burn'>{filename}</div></div>"
        )
        return preview, sel["file_hash"]
    except Exception as e:
        log.warning(f"Pyre row select failed: {e}")
        return "<div class='pyre-container'><div class='flames'>🔥</div></div>", ""


def ui_pyre_confirm(db, typed, h_hash, bond):
    if not h_hash or not typed:
        return gr.update(interactive=False)
    ok, _ = perform_ritual(db, h_hash, typed, bond)
    return gr.update(interactive=ok)


def ui_pyre_ignite(db, h_hash, typed, bond, df_state):
    if not h_hash or not typed:
        return "Select and type name.", df_state, df_state
    success, msg = perform_ritual(db, h_hash, typed, bond)
    if not success:
        return msg, df_state, df_state
    result = execute_burn(db, h_hash)
    new_list = list_burnable_files(db)
    return result, new_list, new_list


def ui_pyre_refresh(db):
    bl = list_burnable_files(db)
    return bl, bl


def ui_loom_update(db, filt, search):
    return build_loom_figure(db, filt, search)


def ui_loom_clusters(db):
    return find_clusters(db)


def ui_loom_search(db, search):
    return (
        describe_node(db, search)
        if search.strip()
        else "Search an entity to see its threads."
    )


def ui_loom_export(db):
    p = export_loom_markdown(db)
    return f"Weaved to **{p.name}** in sorted/."


def ui_mirror_read(db):
    return get_latest_reflection(db)


def ui_mirror_weave(db):
    generate_weekly_reflection(db, force=True)
    return get_latest_reflection(db), "*A new reflection has been woven.*"


def is_sandbox(db) -> bool:
    """Check sandbox mode — DB setting overrides CFG."""
    val = get_setting(db, "sandbox", None)
    if val is not None:
        return val == "1"
    return CFG.sandbox


def get_graph_injection(db) -> str:
    """Check graph injection mode — DB setting overrides CFG."""
    val = get_setting(db, "graph_injection", None)
    if val is not None:
        return val
    return CFG.graph_injection


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _clip(text: str, limit: int = 96) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


VOICE_TMP_DIR = Path(tempfile.gettempdir()) / "grimalkin_voice"


def _voice_template_values(**extra) -> dict[str, str]:
    values = {
        "app": str(APP_DIR),
        "python": sys.executable,
        "tmp": str(VOICE_TMP_DIR),
    }
    values.update({k: str(v) for k, v in extra.items()})
    return values


def _run_local_voice_command(
    template: str,
    values: dict[str, str],
    *,
    default_arg: str = "",
    stdin: str | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    if not template.strip():
        raise ValueError("empty command template")
    VOICE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    parts = [part.format(**values) for part in shlex.split(template)]
    if default_arg and "{" + default_arg + "}" not in template:
        parts.append(values[default_arg])
    return subprocess.run(
        parts,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _discard_gradio_audio(audio_path: str | None) -> bool:
    if not audio_path or CFG.keep_voice_audio:
        return False
    try:
        path = Path(audio_path).resolve()
        tmp_root = Path(tempfile.gettempdir()).resolve()
        if path.exists() and tmp_root in path.parents:
            path.unlink()
            return True
    except OSError as e:
        log.warning(f"Voice audio cleanup failed: {e}")
    return False


def transcribe_local_audio(audio_path: str | None) -> tuple[str, str]:
    """Transcribe a microphone capture with a local command template."""
    if not audio_path:
        return "", "No audio captured."
    command = CFG.stt_command or os.environ.get("GRIM_STT_COMMAND", "")
    if not command.strip():
        return "", "Local speech input is not configured. Set GRIM_STT_COMMAND."
    output_path = VOICE_TMP_DIR / f"transcript_{uuid.uuid4().hex}.txt"
    values = _voice_template_values(audio=audio_path, out=output_path)
    try:
        proc = _run_local_voice_command(
            command,
            values,
            default_arg="audio",
            timeout=180,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return "", f"Local speech input failed: {_clip(e, 140)}"
    transcript = ""
    if output_path.exists():
        transcript = output_path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            output_path.unlink()
        except OSError as e:
            log.warning(f"Voice transcript cleanup failed: {e}")
    if not transcript:
        transcript = proc.stdout.strip()
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return "", f"Local speech input failed: {_clip(err, 160)}"
    if not transcript:
        return "", "Local speech input returned no transcript."
    return transcript, "Voice transcript captured locally."


def synthesize_local_reply(text: str) -> tuple[str | None, str]:
    """Render a reply through an optional local TTS command template."""
    command = CFG.tts_command or os.environ.get("GRIM_TTS_COMMAND", "")
    if not command.strip() or not text.strip():
        return None, "Local voice output is not configured."
    text_path = VOICE_TMP_DIR / f"reply_{uuid.uuid4().hex}.txt"
    output_path = VOICE_TMP_DIR / f"reply_{uuid.uuid4().hex}.wav"
    VOICE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    values = _voice_template_values(text=text, text_file=text_path, out=output_path)
    stdin = text if "{text}" not in command and "{text_file}" not in command else None
    try:
        proc = _run_local_voice_command(
            command,
            values,
            default_arg="out",
            stdin=stdin,
            timeout=180,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        try:
            text_path.unlink()
        except OSError as cleanup_err:
            log.warning(f"Voice text cleanup failed: {cleanup_err}")
        return None, f"Local voice output failed: {_clip(e, 140)}"
    try:
        text_path.unlink()
    except OSError as e:
        log.warning(f"Voice text cleanup failed: {e}")
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return None, f"Local voice output failed: {_clip(err, 160)}"
    if not output_path.exists():
        return None, "Local voice output did not produce audio."
    return str(output_path), "Reply voice rendered locally."


def ui_voice_chat(db, index, metadata, session_id, audio_path, history):
    history = history or []
    transcript, voice_status = transcribe_local_audio(audio_path)
    discarded = _discard_gradio_audio(audio_path)
    if CFG.keep_voice_audio:
        retention = "Input audio kept by configuration."
    elif discarded:
        retention = "Input audio discarded."
    else:
        retention = "Input audio not retained by Grimalkin."
    if not transcript:
        return (
            history,
            f"{voice_status} {retention}",
            gr.update(value=None, visible=False),
        )

    egg = check_easter_eggs(transcript, db=db)
    resp = (
        egg
        if egg
        else handle_scratch_post(db, index, metadata, transcript, session_id=session_id)
    )
    pending = get_pending_notifications(db)
    if pending:
        resp = f"{format_notifications(pending)}\n\n---\n\n{resp}"

    display_text = f"[voice] {transcript}"
    save_chat_message(db, session_id, "user", display_text)
    save_chat_message(db, session_id, "assistant", resp)
    audit_event(
        db,
        "voice_chat",
        f"transcript_chars={len(transcript)} session={session_id[:8]}",
    )
    try:
        update_chat_summary(db, session_id)
    except Exception as e:
        log.warning(f"Chat summary update failed: {e}")

    audio_reply, audio_status = synthesize_local_reply(resp)
    status = f"{voice_status} {retention} {audio_status}"
    history.append({"role": "user", "content": display_text})
    history.append({"role": "assistant", "content": resp})
    return history, status, gr.update(value=audio_reply, visible=bool(audio_reply))


def _month_calendar(today: date | None = None) -> str:
    today = today or date.today()
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdayscalendar(today.year, today.month)
    labels = ("S", "M", "T", "W", "T", "F", "S")
    head = "".join(f"<th>{d}</th>" for d in labels)
    rows = []
    for week in weeks:
        cells = []
        for day in week:
            if day == 0:
                cells.append("<td class='cal-empty'></td>")
            elif day == today.day:
                cells.append(f"<td class='cal-today'>{day}</td>")
            else:
                cells.append(f"<td>{day}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table class='mini-calendar' aria-label='Current month'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def ui_header_panel(db) -> str:
    """Build the compact local clock, calendar, and privacy status cluster."""
    now = datetime.now()
    today = now.date()
    mode = "sandbox" if is_sandbox(db) else "live"
    graph_mode = get_graph_injection(db)
    model = _clip(get_active_model(), 26)
    embed = _clip(CFG.embed_model, 24)
    endpoint = _clip(CFG.ollama_url.replace("http://", "").replace("https://", ""), 24)
    return f"""
    <div class="grim-side-panel">
        <div class="privacy-line">
            <span class="privacy-dot"></span>
            <span>local vault</span>
            <span>{_esc(endpoint)}</span>
        </div>
        <div class="time-stack">
            <div>
                <time id="grim-live-clock" datetime="{_esc(now.isoformat())}">{_esc(now.strftime("%H:%M"))}</time>
                <div class="clock-date">{_esc(today.strftime("%a %b %d"))}</div>
            </div>
            <div class="month-card">
                <div class="month-label">{_esc(today.strftime("%b %Y"))}</div>
                {_month_calendar(today)}
            </div>
        </div>
        <div class="local-badges" aria-label="Local runtime status">
            <span>{_esc(model)}</span>
            <span>{_esc(embed)}</span>
            <span>{_esc(mode)}</span>
            <span>graph {_esc(graph_mode)}</span>
        </div>
    </div>
    """


def _peek_notifications(db, limit: int = 3) -> list[dict]:
    cur = db.cursor()
    cur.execute(
        """
        SELECT type, detail, created_at
        FROM notifications
        WHERE seen=0
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [
        {"type": row[0], "detail": row[1], "created_at": row[2]}
        for row in cur.fetchall()
    ]


def _pulse_items(db) -> list[dict]:
    cur = db.cursor()
    items = []

    for note in _peek_notifications(db):
        items.append(
            {
                "kind": note["type"].replace("_", " "),
                "title": "watch",
                "detail": note["detail"],
                "meta": note["created_at"] or "",
            }
        )

    cur.execute(
        """
        SELECT filename, category, indexed, created_at
        FROM file_memory
        WHERE burned_at IS NULL
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 4
        """
    )
    for filename, category, indexed, created_at in cur.fetchall():
        state = "indexed" if indexed else "waiting"
        items.append(
            {
                "kind": category or "file",
                "title": filename,
                "detail": state,
                "meta": created_at or "",
            }
        )

    cur.execute(
        "SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL AND indexed=0"
    )
    waiting = cur.fetchone()[0]
    if waiting:
        items.append(
            {
                "kind": "index",
                "title": f"{waiting} unindexed",
                "detail": "quiet backlog",
                "meta": "local",
            }
        )

    cur.execute(
        """
        SELECT name, type, times_seen
        FROM entities
        WHERE importance=1
        ORDER BY times_seen DESC, name COLLATE NOCASE
        LIMIT 4
        """
    )
    for name, etype, times_seen in cur.fetchall():
        items.append(
            {
                "kind": etype or "entity",
                "title": name,
                "detail": f"watched · {times_seen} sightings",
                "meta": "important",
            }
        )

    cur.execute(
        "SELECT reflection_date, summary FROM reflections ORDER BY reflection_date DESC LIMIT 1"
    )
    reflection = cur.fetchone()
    if reflection:
        items.append(
            {
                "kind": "mirror",
                "title": reflection[0],
                "detail": reflection[1],
                "meta": "latest",
            }
        )

    if not items:
        items.append(
            {
                "kind": "quiet",
                "title": "vault quiet",
                "detail": "no new files, no unread signals",
                "meta": "local",
            }
        )
    return items[:12]


def ui_pulse_rail(db) -> str:
    """Build a compact local activity rail for the top of the app."""
    items = []
    for item in _pulse_items(db):
        items.append(
            "<article class='pulse-item'>"
            f"<span class='pulse-kind'>{_esc(item['kind'])}</span>"
            f"<strong>{_esc(_clip(item['title'], 48))}</strong>"
            f"<p>{_esc(_clip(item['detail'], 92))}</p>"
            f"<span class='pulse-meta'>{_esc(_clip(item['meta'], 28))}</span>"
            "</article>"
        )
    return (
        "<section class='pulse-rail' aria-label='Local pulse'>"
        "<div class='pulse-heading'>pulse</div>"
        "<div class='pulse-track'>" + "".join(items) + "</div></section>"
    )


def _table_count(db, table: str, where: str = "") -> int:
    try:
        cur = db.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table} {where}")
        return int(cur.fetchone()[0])
    except sqlite3.Error:
        return 0


def _command_status(command: str) -> tuple[str, str, str]:
    command = (command or "").strip()
    if not command:
        return "missing", "not configured", "warn"
    try:
        values = _voice_template_values(audio="", out="", text="", text_file="")
        parts = [part.format(**values) for part in shlex.split(command)]
        exe = parts[0]
    except (KeyError, ValueError) as e:
        return "invalid", _clip(e, 80), "bad"
    exe_path = Path(exe).expanduser()
    found = exe_path.exists() if exe_path.parent != Path(".") else shutil.which(exe)
    if found:
        return "ready", exe, "ok"
    return "missing", f"{exe} not on PATH", "warn"


def _adapter_engine_status(mode: str) -> tuple[str, str, str]:
    script = APP_DIR / "scripts" / "grim_voice.py"
    if not script.exists():
        return "missing", "adapter script missing", "warn"
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "status", "--json"],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if proc.returncode != 0:
            detail = (
                proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
            )
            return "unknown", _clip(detail, 120), "warn"
        status = json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return "unknown", _clip(e, 120), "warn"
    engines = status.get(mode, [])
    available = [item["engine"] for item in engines if item.get("available")]
    if available:
        return "ready", ", ".join(available[:3]), "ok"
    return "missing-engine", f"no local {mode.upper()} engine found", "warn"


def _voice_command_status(command: str, mode: str) -> tuple[str, str, str]:
    state, detail, css_class = _command_status(command)
    if css_class != "ok" or "grim_voice.py" not in command:
        return state, detail, css_class
    engine_state, engine_detail, engine_class = _adapter_engine_status(mode)
    if engine_class == "ok":
        return state, f"adapter ready; {engine_detail}", "ok"
    return engine_state, f"adapter ready; {engine_detail}", engine_class


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "::1"} or host.startswith("127.")


def _human_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "missing"
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def _git_snapshot() -> tuple[str, str, str]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return "unknown", _clip(e, 120), "warn"
    head = sha.stdout.strip() if sha.returncode == 0 else "unknown"
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else True
    if dirty:
        return "modified", f"HEAD {head} · uncommitted local changes", "warn"
    return "clean", f"HEAD {head}", "ok"


def _latest_audit_events(db, limit: int = 6) -> list[dict]:
    try:
        cur = db.cursor()
        cur.execute(
            """
            SELECT timestamp, event_type, detail
            FROM audit_log
            ORDER BY datetime(timestamp) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {"timestamp": row[0], "event_type": row[1], "detail": row[2]}
            for row in cur.fetchall()
        ]
    except sqlite3.Error:
        return []


def _control_card(title: str, value: str, detail: str, state: str) -> str:
    return (
        f"<article class='control-card control-{_esc(state)}'>"
        f"<span class='control-state'>{_esc(state)}</span>"
        f"<strong>{_esc(title)}</strong>"
        f"<b>{_esc(value)}</b>"
        f"<p>{_esc(detail)}</p>"
        "</article>"
    )


def ui_control_deck(db) -> str:
    """Render the privacy posture control deck."""
    stt_state, stt_detail, stt_class = _voice_command_status(
        CFG.stt_command or os.environ.get("GRIM_STT_COMMAND", ""),
        "stt",
    )
    tts_state, tts_detail, tts_class = _voice_command_status(
        CFG.tts_command or os.environ.get("GRIM_TTS_COMMAND", ""),
        "tts",
    )
    voice_class = "ok" if stt_class == "ok" and tts_class == "ok" else "warn"
    voice_value = "ready" if voice_class == "ok" else "partial"

    endpoint = CFG.ollama_url.replace("http://", "").replace("https://", "")
    endpoint_local = _is_loopback_url(CFG.ollama_url)
    network_value = "loopback" if endpoint_local else "external"
    network_class = "warn" if endpoint_local else "bad"
    network_detail = f"Ollama {endpoint}; outbound quarantine is advisory in this build"

    audit_enabled = get_setting(db, "audit_enabled", "1") == "1"
    audit_rows = _table_count(db, "audit_log")
    audit_class = "ok" if audit_enabled else "bad"
    audit_value = "enabled" if audit_enabled else "off"

    chat_rows = _table_count(db, "chat_history")
    vector_files = sum(1 for p in FAISS_INDEX_DIR.glob("*") if p.is_file())
    memory_value = get_setting(db, "memory_mode", "local_plaintext").replace("_", " ")
    memory_detail = f"DB {_human_size(DB_PATH)} · {chat_rows} chat rows · {vector_files} vector files"

    files_total = _table_count(db, "file_memory", "WHERE burned_at IS NULL")
    unindexed = _table_count(db, "file_memory", "WHERE burned_at IS NULL AND indexed=0")
    file_mode = "sandbox" if is_sandbox(db) else "live"
    scheduler_detail = (
        "scheduled local checks enabled"
        if CFG.scheduler_enabled
        else "manual hunt; scheduled checks off"
    )
    file_detail = f"{files_total} tracked files · {unindexed} waiting · {scheduler_detail}"

    source_value, source_detail, source_class = _git_snapshot()

    cards = [
        _control_card(
            "Network Gate",
            network_value,
            network_detail,
            network_class,
        ),
        _control_card(
            "Voice Path",
            voice_value,
            f"STT {stt_state}: {stt_detail}; TTS {tts_state}: {tts_detail}",
            voice_class,
        ),
        _control_card(
            "Memory Store",
            memory_value,
            memory_detail,
            "warn",
        ),
        _control_card(
            "Audit Trail",
            audit_value,
            f"{audit_rows} local metadata events; payloads omitted",
            audit_class,
        ),
        _control_card(
            "File Access",
            file_mode,
            file_detail,
            "warn" if file_mode == "live" else "ok",
        ),
        _control_card(
            "Source State",
            source_value,
            source_detail,
            source_class,
        ),
    ]

    audit_items = []
    for event in _latest_audit_events(db):
        audit_items.append(
            "<li>"
            f"<span>{_esc(event['timestamp'])}</span>"
            f"<strong>{_esc(event['event_type'])}</strong>"
            f"<p>{_esc(event['detail'] or 'local action')}</p>"
            "</li>"
        )
    audit_html = (
        "<ul class='control-audit-list'>" + "".join(audit_items) + "</ul>"
        if audit_items
        else "<p class='control-empty'>No local audit events yet.</p>"
    )
    return (
        "<section class='control-deck'>"
        "<div class='control-head'>"
        "<div><span>privacy posture</span><h2>Control Deck</h2></div>"
        f"<p>{_esc(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>"
        "</div>"
        "<div class='control-grid'>" + "".join(cards) + "</div>"
        "<section class='control-audit'>"
        "<div><span>recent local actions</span><h3>Audit Trail</h3></div>"
        + audit_html
        + "</section>"
        "</section>"
    )


def ui_control_check(db) -> str:
    audit_event(db, "control_check", "Control deck refreshed")
    return ui_control_deck(db)


def ui_status_pulse(db) -> str:
    """Build the ambient status line HTML."""
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM file_memory WHERE burned_at IS NULL")
    files = cur.fetchone()[0]
    stats = graph_stats(db)
    bond = get_bond_level(db)
    burn_count = int(get_setting(db, "burn_count", "0"))
    parts = [
        f"{files} files" if files != 1 else "1 file",
        f"{stats['entities']} entities",
        f"{stats['relationships']} threads",
    ]
    if burn_count:
        parts.append(f"{burn_count} burned")
    parts.append(f"bond {bond}")
    return '<div class="status-pulse">' + " · ".join(parts) + "</div>"


def ui_save_settings(db, new_name, new_address):
    msgs = []
    if new_name.strip():
        set_setting(db, "pet_name", new_name.strip())
        msgs.append(f"Name set to *{new_name.strip()}*.")
    if new_address.strip():
        set_setting(db, "user_address", new_address.strip())
        msgs.append(f"You shall be addressed as *{new_address.strip()}*.")
    if msgs:
        audit_event(db, "settings_update", "identity settings changed")
    return "\n".join(msgs) if msgs else "Nothing changed."


def ui_save_behavior(db, serious, sandbox, graph_inj):
    """Save behavior toggles to DB."""
    set_setting(db, "serious_mode", "1" if serious else "0")
    set_setting(db, "sandbox", "1" if sandbox else "0")
    set_setting(db, "graph_injection", graph_inj)
    msgs = []
    msgs.append(f"Serious mode: {'on' if serious else 'off'}")
    msgs.append(f"Sandbox mode: {'on' if sandbox else 'off'}")
    msgs.append(f"Graph injection: {graph_inj}")
    audit_event(
        db,
        "behavior_update",
        f"serious={bool(serious)} sandbox={bool(sandbox)} graph={graph_inj}",
    )
    return "*Settings updated.* " + " · ".join(msgs)


def list_ollama_models() -> list[str]:
    """Installed ollama chat tags for the Settings dropdown (embed models excluded)."""
    names = []
    if HAS_REQUESTS:
        try:
            resp = requests.get(f"{CFG.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            names = [m.get("name", "") for m in resp.json().get("models", [])]
            names = [n for n in names if n and "embed" not in n.lower()]
        except Exception as e:
            log.warning(f"Could not list ollama models: {e}")
    active = get_active_model()
    if active and active not in names:
        names.insert(0, active)
    return names


def ui_save_model(db, model_name):
    """Persist the chat model and apply it to the live session."""
    name = (model_name or "").strip()
    if not name or name == get_active_model():
        return "Nothing changed."
    set_setting(db, "ollama_model", name)
    set_active_model(name)
    audit_event(db, "model_update", f"chat model set to {name}")
    return f"*Settings updated.* Chat model: {name} — next response uses it."


def ui_save_categories(db, categories_str):
    """Save custom categories from comma-separated string."""
    if not categories_str.strip():
        set_setting(db, "custom_categories", "[]")
        audit_event(db, "categories_update", "custom categories cleared")
        return "*Categories cleared.*"
    cats = [c.strip().upper() for c in categories_str.split(",") if c.strip()]
    set_setting(db, "custom_categories", json.dumps(cats))
    audit_event(db, "categories_update", f"{len(cats)} custom category/categories")
    return f"*Custom categories set:* {', '.join(cats)}"


def ui_save_avatar(filepath):
    """Copy uploaded image to persistent avatar path."""
    if not filepath:
        return ""
    try:
        src = Path(filepath)
        shutil.copy2(str(src), str(AVATAR_PATH))
        log.info(f"Avatar saved: {AVATAR_PATH}")
        audit_event(get_db(), "avatar_update", "familiar portrait changed")
        return "*Portrait saved.* Restart to see it in the header."
    except Exception as e:
        log.error(f"Avatar save failed: {e}")
        return f"*hiss* Failed to save portrait: {e}"


# ─── Gradio Wiring ───────────────────────────────────────────────────────────


def _build_dark_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#e6fff7",
            c100="#b3ffe0",
            c200="#80ffc9",
            c300="#4dffb3",
            c400="#1aff9c",
            c500="#00e6b0",
            c600="#00b386",
            c700="#00805f",
            c800="#004d39",
            c900="#001a13",
            c950="#000d09",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f0f2f5",
            c100="#d1d5db",
            c200="#9ca3af",
            c300="#6b7280",
            c400="#4b5563",
            c500="#374151",
            c600="#1f2937",
            c700="#151520",
            c800="#111118",
            c900="#0d0d14",
            c950="#0a0a0f",
        ),
        font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0a0a0f",
        body_text_color="#c8d6e5",
        block_background_fill="#111118",
        block_border_width="1px",
        block_border_color="#1e2e2a",
        block_label_text_color="#667788",
        input_background_fill="#111118",
        input_border_color="#2a2a3e",
        button_primary_background_fill="linear-gradient(135deg, #00c896, #00a67a)",
        button_primary_text_color="#0a0a0f",
    )


AVATAR_PATH = APP_DIR / "grimalkin_avatar.jpg"
AVATAR_FALLBACK = APP_DIR / "grimalkin.jpg"


def _gradio_file_paths() -> list[str]:
    """Return the only app-local files Gradio may serve directly."""
    return [str(AVATAR_PATH), str(AVATAR_FALLBACK)]


def _get_avatar_src() -> str:
    """Return the Gradio file-serving URL for the avatar, or empty string."""
    if AVATAR_PATH.exists():
        return f"/gradio_api/file={AVATAR_PATH}"
    if AVATAR_FALLBACK.exists():
        return f"/gradio_api/file={AVATAR_FALLBACK}"
    return ""


def build_ui(db, index, metadata):
    with gr.Blocks(
        title=f"{get_setting(db, 'pet_name', 'Grimalkin')} — Your Private AI Familiar",
        analytics_enabled=False,
    ) as demo:
        pet_name = get_setting(db, "pet_name", "Grimalkin")
        pet_name_html = _esc(pet_name)
        _tier = BOND.tier_for(get_bond_level(db))
        _tier_lines = OPENING_LINES.get(_tier, OPENING_LINES["Acquaintance"])
        _opening = random.choice(_tier_lines)

        avatar_src = _get_avatar_src()
        if avatar_src:
            avatar_html = (
                f'<div class="grim-avatar"><img src="{_esc(avatar_src)}" '
                f'alt="{pet_name_html}"></div>'
            )
        else:
            avatar_html = '<div class="grim-avatar-empty">&#9676;</div>'

        _bond_level = get_bond_level(db)
        _bond_pct = min(100, _bond_level)

        gr.HTML(f"""
        <div class="grim-header">
            <div class="grim-header-left">
                <div class="name-row">
                    <p class="familiar-name">{pet_name_html}</p>
                    <span class="version-tag">v{VERSION}</span>
                </div>
                <div class="bond-row">
                    <p class="bond-tier">{_esc(_tier)}</p>
                    <div class="bond-bar"><div class="bond-fill" style="width:{_bond_pct}%"></div></div>
                    <span class="bond-level">{_bond_level}</span>
                </div>
                <p class="opening-line">{_esc(_opening)}</p>
            </div>
            <div class="grim-header-right">
                {avatar_html}
                {ui_header_panel(db)}
            </div>
        </div>
        """)

        gr.HTML(ui_pulse_rail(db))

        # Scratch Post
        session_id = gr.State(str(uuid.uuid4()))
        with gr.Tab("🐾 Scratch Post"):
            chatbot = gr.Chatbot(label="Grimalkin", height=420)
            msg_input = gr.MultimodalTextbox(
                label="Speak",
                placeholder="Type a command, talk, or drop a file…",
                file_types=None,  # accept all file types
                submit_btn=True,
            )
            msg_input.submit(
                partial(ui_chat, db, index, metadata),
                [session_id, msg_input, chatbot],
                [chatbot, msg_input],
            )
            with gr.Group(elem_classes=["voice-dock"]):
                with gr.Row(elem_classes=["voice-actions"]):
                    voice_audio = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Push-to-talk",
                        show_label=False,
                        format="wav",
                        scale=5,
                        min_width=220,
                    )
                    voice_btn = gr.Button(
                        "Send Voice",
                        variant="primary",
                        size="sm",
                        scale=1,
                        min_width=118,
                    )
                voice_reply = gr.Audio(
                    label="Reply Voice",
                    type="filepath",
                    autoplay=False,
                    interactive=False,
                    visible=False,
                )
                voice_status = gr.Markdown(elem_classes=["voice-status"])
                voice_btn.click(
                    partial(ui_voice_chat, db, index, metadata),
                    [session_id, voice_audio, chatbot],
                    [chatbot, voice_status, voice_reply],
                )

            with gr.Accordion("Commands", open=False):
                cmd_lines = [
                    f"**{cmd}** — {desc}" for cmd, desc in SCRATCH_COMMANDS.items()
                ]
                cmd_lines.extend(
                    [
                        '**merge "A" "B"** — canonicalize entity names',
                        "**important Name** — mark entity as important",
                        "**forget Name** — remove entity from the web",
                        "**recall Name** — deep recall on any topic",
                    ]
                )
                gr.Markdown("\n\n".join(cmd_lines))

        # Control Deck
        with gr.Tab("🛡️ Control Deck"):
            control_html = gr.HTML(ui_control_deck(db))
            control_btn = gr.Button("Run Local Check", variant="primary")
            control_btn.click(partial(ui_control_check, db), None, control_html)

        # Hunt
        with gr.Tab("🏹 The Hunt"):
            hunt_output = gr.Markdown("Press to scan for new prey.")
            hunt_btn = gr.Button("Begin the Hunt", variant="primary")
            hunt_btn.click(partial(ui_hunt, db, index, metadata), None, hunt_output)

        # Whispers
        with gr.Tab("🌙 Whispers"):
            whisper_output = gr.Markdown("Press for today's briefing.")
            whisper_btn = gr.Button("Summon Whispers")
            whisper_btn.click(partial(ui_whispers, db), None, whisper_output)

        # Vault
        with gr.Tab("📚 The Vault"):
            vault_output = gr.Markdown("Ask anything about your files.")
            vault_input = gr.Textbox(
                label="Query the Vault", placeholder="What connects my tax files…?"
            )
            vault_btn = gr.Button("Search the Vault", variant="primary")
            _vault = partial(ui_vault_query, db, index, metadata)
            vault_btn.click(_vault, vault_input, vault_output)
            vault_input.submit(_vault, vault_input, vault_output)

        # Pyre
        with gr.Tab("🔥 The Pyre"):
            initial_list = list_burnable_files(db)
            burn_df_state = gr.State(initial_list)
            hidden_hash = gr.State("")
            bond_state = gr.State(get_bond_level(db))

            with gr.Row():
                burnable_df = gr.DataFrame(
                    value=initial_list,
                    headers=["filename", "category", "tags", "notes", "file_hash"],
                    label="Offerings",
                    interactive=False,
                )
                with gr.Column():
                    ritual_html = gr.HTML(
                        "<div class='pyre-container'><div class='flames'>🔥</div></div>"
                    )
                    confirm_box = gr.Textbox(
                        label="Speak the true name", placeholder="report_q3.pdf"
                    )
                    light_button = gr.Button(
                        "Light the Pyre", variant="stop", interactive=False
                    )
                    status_out = gr.Markdown()

            burnable_df.select(
                ui_pyre_row_select, burn_df_state, [ritual_html, hidden_hash]
            )
            confirm_box.change(
                partial(ui_pyre_confirm, db),
                [confirm_box, hidden_hash, bond_state],
                light_button,
            )
            light_button.click(
                partial(ui_pyre_ignite, db),
                [hidden_hash, confirm_box, bond_state, burn_df_state],
                [status_out, burnable_df, burn_df_state],
            )
            gr.Button("Refresh Offerings").click(
                partial(ui_pyre_refresh, db), None, [burnable_df, burn_df_state]
            )

        # The Loom
        with gr.Tab("🕸️ The Loom"):
            loom_filter = gr.Dropdown(
                ["All", "person", "org", "date", "location", "amount", "topic"],
                value="All",
                label="Filter type",
            )
            loom_search = gr.Textbox(label="Search entity", placeholder="Acme Corp")
            with gr.Row():
                refresh_loom = gr.Button("Refresh the Loom")
                clusters_btn = gr.Button("Strongest Clusters")
                export_btn = gr.Button("Weave to Markdown")

            loom_plot = gr.Plot() if HAS_PLOTLY else gr.HTML()
            loom_narrative = gr.Markdown(
                "Step into my Loom, mortal. These threads have grown fat with meaning."
            )

            refresh_loom.click(
                partial(ui_loom_update, db), [loom_filter, loom_search], loom_plot
            )
            clusters_btn.click(partial(ui_loom_clusters, db), None, loom_narrative)
            loom_search.submit(partial(ui_loom_search, db), loom_search, loom_narrative)
            export_btn.click(partial(ui_loom_export, db), None, loom_narrative)

        # The Mirror
        with gr.Tab("🪞 The Mirror"):
            mirror_output = gr.Markdown(get_latest_reflection(db))
            mirror_status = gr.Markdown()
            with gr.Row():
                refresh_mirror = gr.Button("Read the Mirror")
                weave_mirror = gr.Button("Weave a New Reflection")

            refresh_mirror.click(partial(ui_mirror_read, db), None, mirror_output)
            weave_mirror.click(
                partial(ui_mirror_weave, db), None, [mirror_output, mirror_status]
            )

        # Settings
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Identity")
            s_name = gr.Textbox(
                label="Familiar Name", value=pet_name, placeholder="Grimalkin"
            )
            s_address = gr.Textbox(
                label="Address Me As",
                value=get_setting(db, "user_address", "mortal"),
                placeholder="mortal",
            )
            save_settings_btn = gr.Button("Apply", variant="primary")
            settings_status = gr.Markdown()
            save_settings_btn.click(
                partial(ui_save_settings, db), [s_name, s_address], settings_status
            )

            gr.Markdown("### Behavior")
            s_serious = gr.Checkbox(
                label="Serious Mode — tone down the cat persona",
                value=get_setting(db, "serious_mode", "0") == "1",
            )
            s_sandbox = gr.Checkbox(
                label="Sandbox Mode — Pyre won't delete files (restart required)",
                value=get_setting(db, "sandbox", "1" if CFG.sandbox else "0") == "1",
            )
            s_graph = gr.Dropdown(
                choices=["auto", "always", "never"],
                value=get_setting(db, "graph_injection", CFG.graph_injection),
                label="Knowledge Graph Injection — when to include web context in responses",
            )
            save_behavior_btn = gr.Button("Apply Behavior", variant="primary")
            behavior_status = gr.Markdown()
            save_behavior_btn.click(
                partial(ui_save_behavior, db),
                [s_serious, s_sandbox, s_graph],
                behavior_status,
            )

            gr.Markdown("### Model")
            s_model = gr.Dropdown(
                choices=list_ollama_models(),
                value=get_active_model(),
                allow_custom_value=True,
                label="Chat Model — applies immediately and persists, no restart",
            )
            save_model_btn = gr.Button("Apply Model", variant="primary")
            model_status = gr.Markdown()
            save_model_btn.click(
                partial(ui_save_model, db), s_model, model_status
            )

            gr.Markdown("### Categories")
            _custom = get_setting(db, "custom_categories", "[]")
            s_categories = gr.Textbox(
                label="Custom Categories (comma-separated)",
                value=", ".join(json.loads(_custom)) if _custom != "[]" else "",
                placeholder="PROJECTS, ARCHIVE, LEGAL",
            )
            save_cats_btn = gr.Button("Apply Categories", variant="primary")
            cats_status = gr.Markdown()
            save_cats_btn.click(
                partial(ui_save_categories, db),
                s_categories,
                cats_status,
            )

            gr.Markdown("### Avatar")
            gr.Markdown("Upload an image for the header. Restart to see changes.")
            avatar_upload = gr.Image(
                label="Familiar Portrait",
                type="filepath",
                height=120,
                width=120,
            )
            avatar_status = gr.Markdown()
            avatar_upload.change(ui_save_avatar, avatar_upload, avatar_status)

        # Footer
        gr.HTML("""
        <div class="grim-footer">
            <span class="footer-disclaimer">Grimalkin uses AI to generate responses. AI can make mistakes — always verify important information.</span>
            <span class="footer-sig">100% local &middot; your data never leaves this machine</span>
        </div>
        """)

    return demo


# ─── Scheduler ─────────────────────────────────────────────────────────────────


def start_scheduler(index, metadata, interval_hours=24, scan_minutes=30):
    if not CFG.scheduler_enabled:
        log.info("Scheduler disabled — manual hunt/groom only.")
        return False

    def loop():
        ticks = 0
        groom_interval = int(interval_hours * 60 / scan_minutes)  # groom every N ticks
        while True:
            time.sleep(scan_minutes * 60)
            ticks += 1
            try:
                db = get_db()  # thread-local connection for scheduler
                check_hunting_grounds(db)
                if ticks % groom_interval == 0:
                    nightly_groom_v4(db, index, metadata)
            except Exception as e:
                log.error(f"Scheduler task failed: {e}")

    Thread(target=loop, daemon=True).start()
    log.info(
        f"Scheduler armed — scan every {scan_minutes}m, groom every {interval_hours}h."
    )
    return True


# ─── Auth ──────────────────────────────────────────────────────────────────────


def _check_auth(username: str, password: str) -> bool:
    """Token-based authentication guard for Gradio.

    When GRIM_AUTH_TOKEN is set, the user must enter the token in the
    password field of the login prompt (the username field is ignored).

    TODO: TLS termination — Gradio does not support TLS natively. Deploy
    behind a reverse proxy (nginx, caddy, etc.) with a valid certificate
    before exposing to any untrusted network.
    """
    expected = CFG.auth_token
    if not expected:
        return True
    return hmac.compare_digest(password or "", expected)


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_launch_security(host: str):
    if _is_loopback_host(host) or CFG.auth_token:
        return
    raise SystemExit(
        "Refusing non-loopback launch without GRIM_AUTH_TOKEN. "
        "Set GRIM_AUTH_TOKEN or bind to 127.0.0.1."
    )


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Grimalkin - local file-sorting service",
    )
    parser.add_argument(
        "--host",
        default=CFG.host,
        help="Bind address (default: 127.0.0.1). Use 0.0.0.0 for network access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=CFG.port,
        help="Server port (default: 7860)",
    )
    args = parser.parse_args()

    host = args.host
    _validate_launch_security(host)
    if not _is_loopback_host(host):
        print(
            "\n"
            "  *** WARNING: non-loopback bind address ***\n"
            f"  Binding to {host} — the server will be reachable on the network.\n"
            "  GRIM_AUTH_TOKEN is required; use a reverse proxy with TLS\n"
            "  termination before exposing it to untrusted clients.\n",
            file=sys.stderr,
        )

    log.info(f"Grimalkin v{VERSION} awakening...")

    db = init_db()
    run_migrations(db)
    ensure_dirs(db)
    set_active_model(get_setting(db, "ollama_model", ""))
    index, metadata = init_faiss()

    start_scheduler(index, metadata)
    audit_event(db, "startup", f"host={host} port={args.port}")

    auth = _check_auth if CFG.auth_token else None

    gradio_file_paths = _gradio_file_paths()
    gr.set_static_paths(paths=gradio_file_paths)
    demo = build_ui(db, index, metadata)
    demo.launch(
        server_name=host,
        server_port=args.port,
        share=False,
        allowed_paths=gradio_file_paths,
        auth=auth,
        theme=_build_dark_theme(),
        css=GRIM_CSS,
        js=GRIM_JS,
        enable_monitoring=False,
        footer_links=[],
    )


if __name__ == "__main__":
    main()
