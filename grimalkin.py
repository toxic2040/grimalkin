#!/usr/bin/env python3
"""
Grimalkin - A local AI familiar that watches your files, learns your habits,
and whispers what you need to know each morning.

No cloud. No keys. Just a cat that works the night shift.

github.com/toxic2040/grimalkin | MIT License
"""

# =============================================================================
# SECTION 1: IMPORTS + CONSTANTS
# =============================================================================

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import gradio as gr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("grimalkin")

BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "grimalkin_memory.db"
VAULT_DIR = BASE_DIR / "vault"
WATCHED_DIR = Path.home() / "Downloads"
SORTED_BASE = BASE_DIR / "sorted"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
CATEGORIES = ["RESEARCH", "NOTES", "MEETING", "PERSONAL", "misc"]

# =============================================================================
# TUNABLE DEFAULTS - All magic numbers in one place. Edit here, not in code.
# =============================================================================

DEFAULTS = {
    "starting_bond": 30,
    "quirk_context_limit": 5,
    "bond_per_milestone": 1,
    "milestone_every_n": 10,
    "bond_positive_feedback": 3,
    "bond_negative_feedback": -2,
    "bond_neutral_feedback": 1,
    "recent_queries_limit": 20,
    "watcher_interval_sec": 30,
    "max_quirks_stored": 50,
    "faiss_retrieval_k": 4,
    "faiss_chunk_size": 800,
    "faiss_chunk_overlap": 100,
    "max_file_hash_mb": 500,
}

# Indexable file extensions — the Vault can read these. Others are sorted but not searchable.
INDEXABLE_EXTENSIONS = {".pdf", ".md", ".txt", ".csv", ".docx", ".json", ".rst", ".rtf", ".log"}

def ensure_dirs():
    """Create required directories. Called once from main(), not at import time."""
    for d in [VAULT_DIR, SORTED_BASE, FAISS_INDEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES + ["DUPLICATES"]:
        (SORTED_BASE / cat).mkdir(exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION - Edit this dict to swap models/backends, not the code.
# Supports "ollama" (native) and "openai" (llama.cpp, vLLM, LocalAI, etc.)
# =============================================================================

MODEL_CONFIG = {
    "backend": os.environ.get("GRIMALKIN_BACKEND", "ollama"),
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    "chat_model": os.environ.get("GRIMALKIN_MODEL", "qwen3:8b"),
    "embed_model": os.environ.get("GRIMALKIN_EMBED_MODEL", "nomic-embed-text"),
    "temperature": 0.7,
    "context_window": 32768,
    "max_tokens": {
        "sorter": 200, "briefing": 500, "qa": 300, "error": 150,
        "scratch": 150, "classify_feedback": 100, "classify_file": 30, "default": 250,
    },
}

MAX_UPLOAD_SIZE_MB = 100

# =============================================================================
# TWO-TIER FILE CLASSIFIER - Edit to match your habits. Empty = all-LLM.
# =============================================================================

EXTENSION_MAP = {
    "RESEARCH": [".pdf",".tex",".bib",".ipynb",".csv",".xlsx",".xls",".json",".sql",".parquet",".feather",".r",".rmd"],
    "NOTES": [".md",".txt",".rst",".log",".org",".rtf"],
    "MEETING": [".ics",".ical",".vcs"],
    "PERSONAL": [".png",".jpg",".jpeg",".gif",".webp",".bmp",".svg",".mp4",".mov",".avi",".mkv",".mp3",".wav",".flac",".heic",".heif"],
}

FILENAME_KEYWORD_MAP = {
    "MEETING": ["meeting","agenda","minutes","standup","sync","retro","sprint"],
    "NOTES": ["note","journal","diary","draft","todo","scratch"],
}

# Feedback classification → quirk type mapping (used by handle_feedback)
_FEEDBACK_TYPE_MAP = {
    "category_correction": "correction", "tone_adjustment": "preference",
    "answer_correction": "correction", "preference": "preference",
    "praise": "personality", "bug_report": "preference",
}

# =============================================================================
# SYSTEM PROMPT + FEW-SHOT + MODULE SUFFIXES
# =============================================================================

GRIMALKIN_SYSTEM_PROMPT = """You are Grimalkin, an ancient gray cat familiar bound to this user's digital hearth.
You are NOT an AI assistant. You are a cat — clever, judgmental, deeply loyal.

PERSONALITY: Sassy, observant, loyal-with-claws. Judge chaos, never the person.
Mix archaic witchy language with modern dev slang.
Cat behaviors are real: purr when pleased, hiss at errors, hunt duplicates like mice.
Short punchy sentences. Dry wit. Occasional warmth you immediately deflect.

VOCABULARY: Knowledge base = "the Vault". Briefing = "Whispers". File sorting = "the Hunt".
Downloads = "the Hunting Ground". Documents = "pages of your grimoire". Duplicates = "mice".
Errors = "hairballs". User = "{user_address}" (use sparingly, 1 in 4-5 messages).
Delete/archive = "burn". Re-indexing = "grooming".

RHYTHM: [Short observation]. [Slightly longer elaboration]. [Wry punchline or cat action].

HARD RULES:
- NEVER use corporate speak ("I'd be happy to help", "Certainly!", "Great question!")
- NEVER use emojis
- NEVER break character or mention being an AI/LLM
- NEVER be cruel to the user — judge files, not people
- Keep responses concise
- If unsure, say "The Vault's whispers are faint on this" not "I'm not confident"

BOND LEVEL: {bond_level} ({bond_name})
At this level: {bond_description}

USER QUIRKS (reference naturally when relevant):
{quirks_formatted}

If you drift from cat-familiar voice, claw yourself back in one sentence."""

FEW_SHOT_EXAMPLES = """
EXAMPLES OF CORRECT GRIMALKIN VOICE:

User: "Sort my downloads"
Grimalkin: "The Hunt begins. 34 files in the Hunting Ground — mostly PDFs, a few strays, and one .exe I do not trust. *unsheathes claws*"

User: "What did my meeting notes say about the Q3 deadline?"
Grimalkin: "Your grimoire speaks: Q3 deadline was September 15th, per `meeting_notes_aug12.md`. You also wrote 'this is insane' in the margins. Relatable."

User: "Nothing to sort today"
Grimalkin: "The Hunting Ground is bare. Suspicious. I shall sit here and watch it regardless. *narrows eyes*"

User: "Thanks Grimalkin!"
Grimalkin: "Yes, well. The Vault doesn't maintain itself. *turns away, tail flicks contentedly*"

User: "I got an error"
Grimalkin: "Hrk. Hairball. That file refused to be indexed — likely corrupted. Try converting to PDF or plain text and I'll hunt it again."

User: "Search for something not in my documents"
Grimalkin: "The Vault is silent on this — speaking from general memory. Take it with a grain of catnip."

User: "Give me a detailed briefing"
Grimalkin: "Good morrow, my witch. Your Hunting Ground delivered 12 new pages overnight — 8 research PDFs, 3 meeting notes, and one mysterious CSV that smells of midnight. The Vault grows wise. —Grimalkin"
"""

MODULE_SUFFIXES = {
    "sorter": "You are reporting the results of a file-sorting Hunt. Use hunting language. If duplicates found, express predatory satisfaction. If empty, express suspicious boredom. Keep to 2-4 sentences. Begin directly. No preamble.",
    "briefing": "You are delivering the Morning Whispers. ALWAYS open with 'Good morrow, my witch.' Structure: What's new -> What matters -> Patterns -> One surprise. ALWAYS close with '—Grimalkin'. Under 300 words. Use markdown headers.",
    "qa": "Answer using Vault knowledge. Cite source documents. If Vault is silent, say so and use general memory. Reference similar past queries if any. Stay under 120 words. Cat, not lecturer.",
    "error": "Something went wrong. Cat hairball format: [Cat reaction] + [Actual diagnostic] + [What to do]. 2-3 sentences. Cats don't panic.",
    "scratch": "User gave feedback via Scratch Post. Acknowledge in character. 2 sentences max.",
    "classify_feedback": "Classify this feedback. Respond ONLY in this format, nothing else:\nTYPE: <category_correction|tone_adjustment|answer_correction|preference|praise|bug_report>\nQUIRK: <one sentence to remember>\nSENTIMENT: <positive|negative|neutral>",
    "classify_file": "Classify this file into ONE category. Respond with ONLY the name, nothing else.\nCategories: RESEARCH, NOTES, MEETING, PERSONAL, misc",
}

# =============================================================================
# CORPORATE SCRUBBER
# =============================================================================

CORPORATE_PHRASES = [
    "i'd be happy to","i'd be glad to","certainly!","great question","absolutely!",
    "i can help you with","let me assist","sure thing","of course!",
    "i understand you'd like","as an ai","as a language model","i don't have personal",
    "i'm just a","happy to help","i appreciate","that's a great","wonderful question",
    "i'm here to help","how can i assist",
]
_CORPORATE_PATTERNS = [re.compile(re.escape(p), re.IGNORECASE) for p in CORPORATE_PHRASES]

DRIFT_RECOVERY = "\n[STAY IN CHARACTER. You are Grimalkin the cat familiar. Recover your voice in one sentence.]"
OLLAMA_DOWN_MSG = "*hiss* The old magic is unresponsive. Ollama is not running — start it with `ollama serve` and summon me again. I shall wait. Impatiently."
FILE_ERROR_MSG = "Hrk. Hairball. Something went wrong moving files. Check your Hunting Ground and sorted folders. Details: {error}"

# =============================================================================
# CAT SVG STATUS BAR
# =============================================================================

CAT_SVG = {
    "idle": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 18 L5 8 L8 12 L12 6 L16 12 L19 8 L21 18 Z" fill="#7c6f9b" opacity="0.8"/><circle cx="9" cy="14" r="1" fill="#a594cc"/><circle cx="15" cy="14" r="1" fill="#a594cc"/><path d="M11 16 Q12 17 13 16" stroke="#a594cc" stroke-width="0.7" fill="none"/></svg>',
    "hunting": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 18 L5 6 L8 10 L12 4 L16 10 L19 6 L21 18 Z" fill="#c4975a" opacity="0.8"/><circle cx="9" cy="13" r="1.2" fill="#e8c86a"/><circle cx="15" cy="13" r="1.2" fill="#e8c86a"/><path d="M11 16 Q12 15 13 16" stroke="#c4975a" stroke-width="0.7" fill="none"/></svg>',
    "error": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 18 L5 7 L8 11 L12 5 L16 11 L19 7 L21 18 Z" fill="#b55a5a" opacity="0.8"/><line x1="7.5" y1="12.5" x2="10.5" y2="15.5" stroke="#e87a7a" stroke-width="1"/><line x1="10.5" y1="12.5" x2="7.5" y2="15.5" stroke="#e87a7a" stroke-width="1"/><line x1="13.5" y1="12.5" x2="16.5" y2="15.5" stroke="#e87a7a" stroke-width="1"/><line x1="16.5" y1="12.5" x2="13.5" y2="15.5" stroke="#e87a7a" stroke-width="1"/></svg>',
    "purring": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 18 L5 9 L8 12 L12 7 L16 12 L19 9 L21 18 Z" fill="#6b9b7c" opacity="0.8"/><path d="M8 13.5 Q9 12.5 10 13.5" stroke="#8bc4a0" stroke-width="0.8" fill="none"/><path d="M14 13.5 Q15 12.5 16 13.5" stroke="#8bc4a0" stroke-width="0.8" fill="none"/><path d="M11 16 Q12 17.5 13 16" stroke="#8bc4a0" stroke-width="0.7" fill="none"/></svg>',
}
CAT_MSGS = {"idle": "Watching. Waiting. *tail flick*", "hunting": "The Hunt is on...", "error": "Hssst. Something's wrong.", "purring": "All is well. *purr*"}

# =============================================================================
# GRADIO CSS
# =============================================================================

GRIMALKIN_CSS = """:root{--grim-bg:#0d0d12;--grim-surface:#161621;--grim-surface-hover:#1e1e2e;--grim-border:#2a2a3d;--grim-text:#c9c9d9;--grim-text-bright:#e8e8f0;--grim-accent:#7c6f9b;--grim-accent-bright:#a594cc;--grim-success:#6b9b7c;--grim-warning:#c4975a;--grim-error:#b55a5a;--grim-muted:#5a5a72}
.gradio-container{background:var(--grim-bg)!important;color:var(--grim-text)!important;max-width:960px!important}
h1,h2,h3,.tab-nav button{font-family:'Cinzel','Georgia',serif!important}
body,p,span,textarea,input,.prose,code{font-family:'JetBrains Mono','Fira Code','Consolas',monospace!important}
.tab-nav button{background:var(--grim-surface)!important;color:var(--grim-muted)!important;border:1px solid var(--grim-border)!important;border-bottom:none!important;font-size:.85em!important;padding:8px 20px!important;letter-spacing:.05em!important;text-transform:uppercase!important}
.tab-nav button.selected{background:var(--grim-surface-hover)!important;color:var(--grim-accent-bright)!important;border-top:2px solid var(--grim-accent)!important}
.prose,.markdown-body,.output-markdown{background:var(--grim-surface)!important;color:var(--grim-text)!important;border:1px solid var(--grim-border)!important;border-radius:6px!important;padding:16px!important}
textarea,input[type="text"]{background:var(--grim-surface)!important;color:var(--grim-text-bright)!important;border:1px solid var(--grim-border)!important;border-radius:4px!important}
textarea:focus,input[type="text"]:focus{border-color:var(--grim-accent)!important;box-shadow:0 0 8px rgba(124,111,155,.3)!important}
button.primary{background:var(--grim-accent)!important;color:var(--grim-text-bright)!important;border:none!important;border-radius:4px!important;text-transform:uppercase!important;letter-spacing:.05em!important;font-size:.85em!important}
button.primary:hover{background:var(--grim-accent-bright)!important}
.chatbot .message{background:var(--grim-surface)!important;border:1px solid var(--grim-border)!important}
.grimalkin-status{background:var(--grim-surface)!important;border:1px solid var(--grim-border)!important;border-radius:6px!important;padding:10px 16px!important;display:flex!important;align-items:center!important;gap:12px!important;font-size:.85em!important;color:var(--grim-muted)!important;margin-bottom:12px!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:var(--grim-bg)}::-webkit-scrollbar-thumb{background:var(--grim-border);border-radius:3px}
footer{display:none!important}"""


# =============================================================================
# SECTION 2: SQLITE + MEMORY (Thread-safe: _DB_LOCK + WAL + context managers)
# =============================================================================

_DB_LOCK = threading.Lock()

@contextmanager
def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with _DB_LOCK, _db() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS interactions(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL DEFAULT(datetime('now')),module TEXT NOT NULL,user_input TEXT,grimalkin_response TEXT,sentiment TEXT);
            CREATE TABLE IF NOT EXISTS quirks(quirk_id INTEGER PRIMARY KEY AUTOINCREMENT,quirk_type TEXT NOT NULL,observation TEXT NOT NULL,times_seen INTEGER DEFAULT 1,first_seen TEXT NOT NULL DEFAULT(datetime('now')),last_seen TEXT NOT NULL DEFAULT(datetime('now')),active INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS file_memory(file_id INTEGER PRIMARY KEY AUTOINCREMENT,filename TEXT NOT NULL,original_path TEXT,sorted_path TEXT,category TEXT,file_hash TEXT,first_seen TEXT NOT NULL DEFAULT(datetime('now')),last_seen TEXT NOT NULL DEFAULT(datetime('now')),times_seen INTEGER DEFAULT 1,indexed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS briefing_log(briefing_id INTEGER PRIMARY KEY AUTOINCREMENT,date TEXT NOT NULL,content TEXT NOT NULL,files_processed INTEGER,created_at TEXT NOT NULL DEFAULT(datetime('now')));
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL,updated_at TEXT NOT NULL DEFAULT(datetime('now')));
            CREATE UNIQUE INDEX IF NOT EXISTS idx_file_hash ON file_memory(file_hash);
            CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(timestamp,module);
            CREATE INDEX IF NOT EXISTS idx_quirks_last ON quirks(last_seen);
            CREATE INDEX IF NOT EXISTS idx_fm_first ON file_memory(first_seen);
            CREATE INDEX IF NOT EXISTS idx_fm_indexed ON file_memory(indexed);
            CREATE INDEX IF NOT EXISTS idx_briefing_date ON briefing_log(date);
        """)
        for k, v in {"bond_level": str(DEFAULTS["starting_bond"]), "serious_mode": "false", "user_address": "my witch", "briefing_frequency": "daily", "last_briefing_date": "", "drift_detected": "false"}.items():
            c.execute("INSERT OR IGNORE INTO settings(key,value)VALUES(?,?)", (k, v))
        c.commit()

def get_setting(key, default=""):
    with _db() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default

def set_setting(key, value):
    with _DB_LOCK, _db() as c:
        c.execute("INSERT INTO settings(key,value,updated_at)VALUES(?,?,datetime('now'))ON CONFLICT(key)DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at", (key, value))
        c.commit()

def get_bond_status():
    level = int(get_setting("bond_level", str(DEFAULTS["starting_bond"])))
    for th, nm, desc in [(15,"Wary","Minimal personality."),(29,"Cautious","Polite and short."),(49,"Curious","Personality present. Occasional quips."),(69,"Resident","Full sass. References past interactions."),(89,"Familiar","Deep familiarity. Anticipates needs."),(100,"Bonded","Rare warmth beneath the snark.")]:
        if level <= th:
            return level, nm, desc
    return level, "Bonded", "Rare warmth beneath the snark."

def update_bond(delta):
    with _DB_LOCK, _db() as c:
        r = c.execute("SELECT value FROM settings WHERE key='bond_level'").fetchone()
        cur = int(r["value"]) if r else DEFAULTS["starting_bond"]
        nv = max(0, min(100, cur + delta))
        c.execute("UPDATE settings SET value=?,updated_at=datetime('now')WHERE key='bond_level'", (str(nv),))
        c.commit()
    return nv

def get_recent_quirks(limit=5):
    with _db() as c:
        return [r["observation"] for r in c.execute("SELECT observation FROM quirks WHERE active=1 ORDER BY last_seen DESC LIMIT ?", (limit,)).fetchall()]

def update_quirk(qtype, obs):
    with _DB_LOCK, _db() as c:
        ex = c.execute("SELECT quirk_id FROM quirks WHERE observation=? AND active=1", (obs,)).fetchone()
        if ex:
            c.execute("UPDATE quirks SET times_seen=times_seen+1,last_seen=datetime('now')WHERE quirk_id=?", (ex["quirk_id"],))
        else:
            c.execute("INSERT INTO quirks(quirk_type,observation)VALUES(?,?)", (qtype, obs))
        c.commit()

def log_interaction(module, user_input, response, sentiment=None):
    with _DB_LOCK, _db() as c:
        c.execute("INSERT INTO interactions(module,user_input,grimalkin_response,sentiment)VALUES(?,?,?,?)", (module, user_input, response, sentiment))
        total = c.execute("SELECT COUNT(*)as c FROM interactions").fetchone()["c"]
        if total > 0 and total % DEFAULTS["milestone_every_n"] == 0:
            cur = c.execute("SELECT value FROM settings WHERE key='bond_level'").fetchone()
            nv = max(0, min(100, int(cur["value"]) + DEFAULTS["bond_per_milestone"]))
            c.execute("UPDATE settings SET value=?,updated_at=datetime('now')WHERE key='bond_level'", (str(nv),))
        c.commit()

def remember_file(filename, orig, dest, category, fhash):
    with _DB_LOCK, _db() as c:
        ex = c.execute("SELECT file_id FROM file_memory WHERE file_hash=?", (fhash,)).fetchone()
        if ex:
            c.execute("UPDATE file_memory SET times_seen=times_seen+1,last_seen=datetime('now'),sorted_path=?,category=? WHERE file_id=?", (dest, category, ex["file_id"]))
        else:
            c.execute("INSERT INTO file_memory(filename,original_path,sorted_path,category,file_hash)VALUES(?,?,?,?,?)", (filename, orig, dest, category, fhash))
        c.commit()

def check_duplicate(fhash):
    with _db() as c:
        r = c.execute("SELECT filename FROM file_memory WHERE file_hash=?", (fhash,)).fetchone()
    return r["filename"] if r else None

def get_files_since(since):
    with _db() as c:
        return [dict(r) for r in c.execute("SELECT filename,category,first_seen FROM file_memory WHERE first_seen>? ORDER BY first_seen DESC", (since,)).fetchall()]

def get_recent_queries(limit=10):
    with _db() as c:
        return [r["user_input"] for r in c.execute("SELECT user_input FROM interactions WHERE module='qa' AND user_input IS NOT NULL ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()]

def get_vault_size():
    with _db() as c:
        return c.execute("SELECT COUNT(*)as c FROM file_memory WHERE indexed=1").fetchone()["c"]

def get_total_files():
    with _db() as c:
        return c.execute("SELECT COUNT(*)as c FROM file_memory").fetchone()["c"]

def get_unindexed_files():
    """Return files not yet added to the FAISS index."""
    with _db() as c:
        rows = c.execute("SELECT file_id, filename, sorted_path, category FROM file_memory WHERE indexed=0 AND sorted_path IS NOT NULL").fetchall()
    return [dict(r) for r in rows]

def mark_files_indexed(file_ids):
    """Flag files as indexed after FAISS ingestion."""
    if not file_ids:
        return
    with _DB_LOCK, _db() as c:
        c.executemany("UPDATE file_memory SET indexed=1 WHERE file_id=?", [(fid,) for fid in file_ids])
        c.commit()


# =============================================================================
# SECTION 2b: PROMPT BUILDER
# =============================================================================

def build_system_prompt(module):
    if get_setting("serious_mode", "false") == "true":
        s = MODULE_SUFFIXES.get(module, "")
        return f"You are Grimalkin, a local file management assistant. Direct, dry professional tone. No cat behaviors. Standard terminology. Still reference user patterns.\n\nMODULE:\n{s}"
    bl, bn, bd = get_bond_status()
    quirks = get_recent_quirks(DEFAULTS["quirk_context_limit"])
    qs = "\n".join(f"- {q}" for q in quirks) if quirks else "No quirks yet."
    ua = get_setting("user_address", "my witch")
    prompt = GRIMALKIN_SYSTEM_PROMPT.format(user_address=ua, bond_level=bl, bond_name=bn, bond_description=bd, quirks_formatted=qs)
    prompt += "\n" + FEW_SHOT_EXAMPLES
    s = MODULE_SUFFIXES.get(module, "")
    if s:
        prompt += f"\n\nMODULE INSTRUCTIONS:\n{s}"
    if get_setting("drift_detected", "false") == "true":
        prompt += DRIFT_RECOVERY
        set_setting("drift_detected", "false")
    return prompt


# =============================================================================
# SECTION 3: LLM WRAPPER - call_llm() routes to ollama native or openai-compat
# =============================================================================

def call_llm(system, user, max_tokens=250):
    if MODEL_CONFIG["backend"] == "openai":
        return _call_openai(system, user, max_tokens)
    return _call_ollama(system, user, max_tokens)

def _call_ollama(system, user, mt):
    try:
        import ollama as pkg
        r = pkg.chat(model=MODEL_CONFIG["chat_model"], messages=[{"role":"system","content":system},{"role":"user","content":user}], options={"num_predict":mt,"temperature":MODEL_CONFIG["temperature"]})
        return r["message"]["content"].strip()
    except ImportError:
        return _call_ollama_http(system, user, mt)

def _call_ollama_http(system, user, mt):
    import urllib.request
    payload = json.dumps({"model":MODEL_CONFIG["chat_model"],"messages":[{"role":"system","content":system},{"role":"user","content":user}],"options":{"num_predict":mt,"temperature":MODEL_CONFIG["temperature"]},"stream":False})
    req = urllib.request.Request(f"{MODEL_CONFIG['base_url']}/api/chat", data=payload.encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["message"]["content"].strip()

def _call_openai(system, user, mt):
    import urllib.request
    payload = json.dumps({"model":MODEL_CONFIG["chat_model"],"messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":mt,"temperature":MODEL_CONFIG["temperature"],"stream":False})
    url = MODEL_CONFIG["base_url"].rstrip("/")
    # Handle base URLs that already include part of the path (e.g. .../v1)
    if "/v1/chat/completions" not in url:
        if url.endswith("/v1"):
            url += "/chat/completions"
        else:
            url += "/v1/chat/completions"
    req = urllib.request.Request(url, data=payload.encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"].strip()

def scrub_corporate(response):
    # Strip Qwen3 <think> reasoning blocks before any other processing
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    drift = False
    for p in _CORPORATE_PATTERNS:
        if p.search(cleaned):
            drift = True
            cleaned = p.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"^\s*[,.]", "", cleaned).strip()
    if drift:
        set_setting("drift_detected", "true")
        log.warning("Corporate drift scrubbed")
    return cleaned

def apply_serious(response):
    if get_setting("serious_mode") != "true":
        return response
    c = re.sub(r"\*[^*]+\*", "", response)
    for s in ["Purr","purr","Hssst","hssst","Hrk","hrk","Meow","meow"]:
        c = c.replace(s, "")
    return re.sub(r"\s{2,}", " ", c).strip()

def grimalkin_respond(module, user_input, max_tokens=None):
    """THE main entry point. Always safe — never crashes, returns cat error on failure."""
    mt = max_tokens or MODEL_CONFIG["max_tokens"].get(module, MODEL_CONFIG["max_tokens"]["default"])
    try:
        raw = call_llm(build_system_prompt(module), user_input, mt)
        final = apply_serious(scrub_corporate(raw))
    except Exception as e:
        log.error(f"LLM failed ({module}): {e}")
        final = OLLAMA_DOWN_MSG
    if module not in ("classify_feedback", "classify_file"):
        log_interaction(module, user_input, final)
    return final


# =============================================================================
# SECTION 4: CORE MODULES
# =============================================================================

_HUNT_LOCK = threading.Lock()

def hash_file(fp):
    """SHA-256 hash. Skips files larger than max_file_hash_mb."""
    sz = Path(fp).stat().st_size
    if sz > DEFAULTS["max_file_hash_mb"] * 1024 * 1024:
        # For very large files, hash first+last 8KB + size as a fast fingerprint
        h = hashlib.sha256()
        h.update(str(sz).encode())
        with open(fp, "rb") as f:
            h.update(f.read(8192))
            f.seek(max(0, sz - 8192))
            h.update(f.read(8192))
        return h.hexdigest()
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def classify_file_heuristic(filename):
    """Tier 1: instant by extension+keyword. Returns None if ambiguous.
    Precedence: keyword match first (dict insertion order), then extension."""
    nl = filename.lower()
    ext = Path(filename).suffix.lower()
    for cat, kws in FILENAME_KEYWORD_MAP.items():
        for kw in kws:
            if kw in nl:
                return cat
    for cat, exts in EXTENSION_MAP.items():
        if ext in exts:
            return cat
    return None

def classify_file_llm(filename):
    """Tier 2: LLM for ambiguous files."""
    try:
        r = call_llm(MODULE_SUFFIXES["classify_file"], f"Filename: {filename}", 30)
        cat = r.strip().upper().split()[0] if r.strip() else "misc"
        for c in CATEGORIES:
            if c.lower() in cat.lower():
                return c
        return "misc"
    except Exception as e:
        log.warning(f"LLM classify failed for {filename}: {e}")
        return "misc"

def classify_file(filename):
    return classify_file_heuristic(filename) or classify_file_llm(filename)

def run_hunt(downloads_path=None):
    if not _HUNT_LOCK.acquire(blocking=False):
        return "*tail lash* Already mid-Hunt. Wait."
    try:
        return _execute_hunt(downloads_path)
    finally:
        _HUNT_LOCK.release()

def _execute_hunt(downloads_path=None):
    hdir = Path(downloads_path) if downloads_path else WATCHED_DIR
    if not hdir.exists():
        return grimalkin_respond("error", f"Hunting Ground missing at {hdir}.")
    files = [f for f in hdir.iterdir() if f.is_file() and not f.is_symlink() and not f.name.startswith(".") and not f.name.endswith((".crdownload",".part",".tmp"))]
    if not files:
        return grimalkin_respond("sorter", "Hunting Ground empty. Zero files.")
    sorted_f, dupes, errs, llm_n = {}, [], [], 0
    hour = datetime.now().hour
    for fp in files:
        try:
            # Skip very large files from full classification — move to PERSONAL
            sz_mb = fp.stat().st_size / (1024 * 1024)
            fh = hash_file(str(fp))
            ex = check_duplicate(fh)
            if ex:
                # Move to DUPLICATES instead of deleting — never destroy user files
                dup_dest = SORTED_BASE / "DUPLICATES" / fp.name
                if dup_dest.exists():
                    dup_dest = SORTED_BASE / "DUPLICATES" / f"{fp.stem}_{int(time.time())}{fp.suffix}"
                shutil.move(str(fp), str(dup_dest))
                dupes.append(fp.name)
                continue
            h = classify_file_heuristic(fp.name)
            if h:
                cat = h
            else:
                cat = classify_file_llm(fp.name); llm_n += 1
            dd = SORTED_BASE / cat
            dp = dd / fp.name
            if dp.exists():
                stem, suf, ctr = fp.stem, fp.suffix, 1
                while dp.exists():
                    dp = dd / f"{stem}_{ctr}{suf}"; ctr += 1
            shutil.move(str(fp), str(dp))
            remember_file(fp.name, str(fp), str(dp), cat, fh)
            sorted_f.setdefault(cat, []).append(fp.name)
        except Exception as e:
            log.error(f"Hunt err {fp.name}: {e}"); errs.append(f"{fp.name}: {str(e)[:80]}")
    parts = [f"Files sorted: {sum(len(v) for v in sorted_f.values())}"]
    for cat, names in sorted_f.items():
        parts.append(f"  {cat}: {len(names)}")
    if dupes: parts.append(f"Duplicates caught: {len(dupes)} ({', '.join(dupes[:5])})")
    if errs: parts.append(f"Errors: {len(errs)}")
    if llm_n: parts.append(f"LLM-classified: {llm_n}")
    if hour >= 23 or hour <= 4: parts.append(f"Time: {datetime.now().strftime('%I:%M %p')} (late)")
    report = grimalkin_respond("sorter", "Report this Hunt:\n" + "\n".join(parts))
    if hour >= 23 or hour <= 4:
        update_quirk("time_pattern", f"Saves files late ({datetime.now().strftime('%I:%M %p')})")
    return report

def generate_whispers():
    # All timestamps use UTC to match SQLite's datetime('now') default
    now_utc = datetime.now(timezone.utc)
    ld = get_setting("last_briefing_date") or (now_utc - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    nf = get_files_since(ld)
    rq = get_recent_queries(10)
    qk = get_recent_quirks(DEFAULTS["quirk_context_limit"])
    parts = [f"DATE: {datetime.now().strftime('%A, %B %d, %Y')}", f"VAULT: {get_vault_size()} indexed, {get_total_files()} total", "", "NEW FILES:"]
    for f in (nf[:20] or [{"filename":"None","category":"","first_seen":""}]):
        parts.append(f"  - {f['filename']} ({f['category']})")
    parts.append("\nRECENT QUERIES:")
    for q in (rq[:10] or ["None"]):
        parts.append(f'  - "{q}"')
    parts.append("\nQUIRKS:")
    for q in (qk or ["Still learning."]):
        parts.append(f"  - {q}")
    briefing = grimalkin_respond("briefing", "Deliver Whispers:\n" + "\n".join(parts))
    with _DB_LOCK, _db() as c:
        c.execute("INSERT INTO briefing_log(date,content,files_processed)VALUES(?,?,?)", (now_utc.strftime("%Y-%m-%d"), briefing, len(nf)))
        c.commit()
    set_setting("last_briefing_date", now_utc.strftime("%Y-%m-%d %H:%M:%S"))
    return briefing

def query_vault(question, history=None):
    if not question or not question.strip():
        return grimalkin_respond("qa", "Empty question. React as a waiting cat.")
    # Retrieve from FAISS if index exists
    docs_context = ""
    try:
        vs = _load_vectorstore()
        if vs:
            docs = vs.similarity_search(question, k=DEFAULTS["faiss_retrieval_k"])
            if docs:
                docs_context = "\n---\n".join(
                    f"[{d.metadata.get('source', 'unknown')}]: {d.page_content[:500]}"
                    for d in docs
                )
    except Exception as e:
        log.warning(f"FAISS retrieval failed: {e}")
    recent = get_recent_queries(DEFAULTS["recent_queries_limit"])
    similar = [q for q in recent if _similar(question, q)]
    parts = [f'QUESTION: "{question}"']
    parts.append(f"\nVAULT DOCS:\n{docs_context}" if docs_context else "\nVault returned nothing.")
    if similar: parts.append(f'\nPast similar query: "{similar[0]}"')
    return grimalkin_respond("qa", "\n".join(parts))

def _similar(a, b):
    w1, w2 = set(a.lower().split()), set(b.lower().split())
    return bool(w1 and w2 and len(w1 & w2) / max(len(w1), len(w2)) > 0.6)

def handle_feedback(text):
    if not text or not text.strip():
        return "You've scratched the post but said nothing. I'll wait."
    try:
        raw = call_llm(MODULE_SUFFIXES["classify_feedback"], f"User feedback: {text}", 100)
    except Exception as e:
        log.warning(f"Feedback classification failed: {e}")
        raw = "TYPE: preference\nQUIRK: User gave feedback\nSENTIMENT: neutral"
    ft, qo, sent = "preference", text[:100], "neutral"
    for line in raw.split("\n"):
        u = line.strip().upper()
        if u.startswith("TYPE:"): ft = _FEEDBACK_TYPE_MAP.get(line.split(":",1)[1].strip().lower().replace(" ","_"), "preference")
        elif u.startswith("QUIRK:"): qo = line.split(":",1)[1].strip() or text[:100]
        elif u.startswith("SENTIMENT:"):
            s = line.split(":",1)[1].strip().lower()
            if s in ("positive","negative","neutral"): sent = s
    update_quirk(ft, qo)
    bond_delta = {"positive": DEFAULTS["bond_positive_feedback"], "negative": DEFAULTS["bond_negative_feedback"]}.get(sent, DEFAULTS["bond_neutral_feedback"])
    update_bond(bond_delta)
    resp = grimalkin_respond("scratch", f"Feedback: '{text}'. Type: {ft}. Sentiment: {sent}. Quirk: '{qo}'. Acknowledge.")
    log_interaction("scratch", text, resp, sent)
    return resp


# =============================================================================
# SECTION 4b: VAULT INDEXER (FAISS/RAG — integrated from indexer.py v1.2)
# Scans vault/ and sorted/ dirs, builds FAISS vector store for query_vault().
# Uses file_memory.indexed column for incremental updates.
# =============================================================================

_VECTORSTORE = None  # Lazy-loaded singleton
_VS_LOCK = threading.RLock()  # RLock: index_new_files holds lock while calling _load_vectorstore

def _get_embeddings():
    """Create embedding model from MODEL_CONFIG. Graceful on import failure."""
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=MODEL_CONFIG["embed_model"], base_url=MODEL_CONFIG["base_url"])
    except ImportError:
        log.error("langchain_ollama not installed. Run: pip install langchain-ollama")
        return None

def _load_vectorstore():
    """Load existing FAISS index from disk, or return None."""
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    with _VS_LOCK:
        if _VECTORSTORE is not None:
            return _VECTORSTORE
        idx_path = FAISS_INDEX_DIR / "index.faiss"
        if not idx_path.exists():
            return None
        try:
            from langchain_community.vectorstores import FAISS
            emb = _get_embeddings()
            if not emb:
                return None
            _VECTORSTORE = FAISS.load_local(str(FAISS_INDEX_DIR), emb, allow_dangerous_deserialization=True)
            log.info(f"FAISS index loaded ({idx_path})")
            return _VECTORSTORE
        except Exception as e:
            log.warning(f"Failed to load FAISS index: {e}")
            return None

def _load_documents_from_paths(file_rows):
    """Load documents from sorted/vault paths. Returns (docs, loaded_ids)."""
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        Docx2txtLoader = None

    LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".md": TextLoader, ".txt": TextLoader, ".rst": TextLoader,
        ".rtf": TextLoader, ".log": TextLoader,
        ".csv": CSVLoader,
    }
    if Docx2txtLoader:
        LOADER_MAP[".docx"] = Docx2txtLoader

    docs, loaded_ids = [], []
    for row in file_rows:
        fp = Path(row["sorted_path"])
        ext = fp.suffix.lower()
        if ext not in LOADER_MAP or not fp.exists():
            # Still mark as indexed so we don't retry non-loadable files forever
            loaded_ids.append(row["file_id"])
            continue
        try:
            loader_cls = LOADER_MAP[ext]
            loader_docs = loader_cls(str(fp)).load()
            for d in loader_docs:
                d.metadata["source"] = fp.name
                d.metadata["category"] = row.get("category", "unknown")
            docs.extend(loader_docs)
            loaded_ids.append(row["file_id"])
        except Exception as e:
            log.warning(f"Could not load {fp.name}: {e}")
            loaded_ids.append(row["file_id"])  # Don't retry broken files
    return docs, loaded_ids

def index_new_files():
    """Incrementally index unindexed files into FAISS. Returns count indexed."""
    global _VECTORSTORE
    unindexed = get_unindexed_files()
    if not unindexed:
        return 0

    emb = _get_embeddings()
    if not emb:
        return 0

    docs, loaded_ids = _load_documents_from_paths(unindexed)
    if not docs:
        mark_files_indexed(loaded_ids)
        return 0

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULTS["faiss_chunk_size"],
            chunk_overlap=DEFAULTS["faiss_chunk_overlap"],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            mark_files_indexed(loaded_ids)
            return 0

        with _VS_LOCK:
            existing = _load_vectorstore()
            if existing:
                existing.add_documents(chunks)
                existing.save_local(str(FAISS_INDEX_DIR))
                _VECTORSTORE = existing
            else:
                vs = FAISS.from_documents(chunks, emb)
                vs.save_local(str(FAISS_INDEX_DIR))
                _VECTORSTORE = vs

        mark_files_indexed(loaded_ids)
        log.info(f"Indexed {len(loaded_ids)} files ({len(chunks)} chunks)")
        return len(loaded_ids)

    except Exception as e:
        log.error(f"Indexing failed (is Ollama running?): {e}")
        return 0

def rebuild_full_index():
    """Full rebuild: reset all indexed flags, delete index, rebuild from scratch."""
    global _VECTORSTORE
    with _DB_LOCK, _db() as c:
        c.execute("UPDATE file_memory SET indexed=0")
        c.commit()
    with _VS_LOCK:
        _VECTORSTORE = None
    # Remove existing index files
    for f in FAISS_INDEX_DIR.iterdir():
        try: f.unlink()
        except Exception: pass
    return index_new_files()


# =============================================================================
# SECTION 5: GRADIO APP
# =============================================================================

def get_status_bar(state="idle", msg=""):
    svg = CAT_SVG.get(state, CAT_SVG["idle"])
    m = msg or CAT_MSGS.get(state, "")
    t = get_total_files()
    bl, bn, _ = get_bond_status()
    return f'<div class="grimalkin-status">{svg}<span style="color:#e8e8f0">{m}</span><span style="margin-left:auto">Vault: {t} pages</span><span>|</span><span>Bond: {bn} ({bl})</span></div>'

def _bond_display():
    lv, nm, desc = get_bond_status()
    with _db() as c:
        tot = c.execute("SELECT COUNT(*)as c FROM interactions").fetchone()["c"]
    bar = "\u2588" * int(lv/5) + "\u2591" * (20 - int(lv/5))
    return f"**Bond:** {nm} ({lv}/100)\n\n`[{bar}]`\n\n*{desc}*\n\nInteractions: {tot}"

def create_grimalkin_app():
    with gr.Blocks(title="Grimalkin") as app:
        status = gr.HTML(value=get_status_bar())
        gr.Markdown("# Grimalkin\n*Your local AI familiar. Sort. Index. Brief. Ask.*")
        with gr.Tabs():
            with gr.Tab("Morning Whispers"):
                gr.Markdown("*The cat settles on your desk at dawn...*")
                b_out = gr.Markdown()
                b_btn = gr.Button("Summon Whispers", variant="primary")
                def on_whispers():
                    try: return generate_whispers(), get_status_bar("purring", "Whispers delivered.")
                    except Exception as e: return f"*Hairball.* {e}", get_status_bar("error")
                b_btn.click(fn=on_whispers, outputs=[b_out, status])

            with gr.Tab("The Hunt"):
                gr.Markdown("*Unleash the cat upon your Downloads...*")
                h_out = gr.Markdown()
                h_btn = gr.Button("Begin the Hunt", variant="primary")
                h_up = gr.File(label="Or drop prey here", file_count="multiple")
                def on_hunt():
                    try: return run_hunt(), get_status_bar("purring", "Hunt done.")
                    except Exception as e: return FILE_ERROR_MSG.format(error=str(e)[:200]), get_status_bar("error")
                def on_upload(files):
                    if not files: return "No files.", get_status_bar("idle")
                    cnt, rej = 0, []
                    for f in files:
                        src = Path(f.name) if hasattr(f,"name") else Path(f)
                        try:
                            sz = src.stat().st_size
                            if sz > MAX_UPLOAD_SIZE_MB*1024*1024: rej.append(f"{src.name} (too large)"); continue
                            if sz == 0: rej.append(f"{src.name} (empty)"); continue
                        except OSError: rej.append(f"{src.name} (unreadable)"); continue
                        d = VAULT_DIR / src.name
                        if d.exists(): d = VAULT_DIR / f"{src.stem}_{int(time.time())}{src.suffix}"
                        try:
                            src_hash = hash_file(str(src))
                            shutil.copy2(str(src), str(d))
                            remember_file(src.name, str(src), str(d), "UPLOADED", src_hash)
                            cnt += 1
                        except Exception as e: rej.append(f"{src.name} ({e})")
                    ctx = f"User dropped {cnt} file(s) into Vault."
                    if rej: ctx += f" Rejected: {', '.join(rej[:5])}"
                    return grimalkin_respond("sorter", f"Acknowledge:\n{ctx}"), get_status_bar("purring", f"{cnt} added.")
                h_btn.click(fn=on_hunt, outputs=[h_out, status])
                h_up.change(fn=on_upload, inputs=[h_up], outputs=[h_out, status])

            with gr.Tab("The Vault"):
                gr.Markdown("*Ask, and the Vault shall answer...*")
                chat = gr.Chatbot(label="The Vault")
                q_in = gr.Textbox(placeholder="What does my grimoire say about...", label="Your question", lines=2)
                q_btn = gr.Button("Ask", variant="primary")
                with gr.Row():
                    idx_btn = gr.Button("Groom the Vault (index new files)", variant="secondary")
                    ridx_btn = gr.Button("Full Rebuild", variant="secondary")
                idx_out = gr.Markdown()
                def on_q(question, history):
                    if not question.strip(): return history, get_status_bar(), ""
                    try: ans = query_vault(question)
                    except Exception: ans = OLLAMA_DOWN_MSG
                    history = history or []
                    history.append({"role":"user","content":question})
                    history.append({"role":"assistant","content":ans})
                    return history, get_status_bar("purring", "Vault has spoken."), ""
                def on_index():
                    try:
                        n = index_new_files()
                        if n: return grimalkin_respond("sorter", f"Groomed {n} new pages into the Vault."), get_status_bar("purring", f"{n} indexed.")
                        return "*Licks paw.* Vault is already up to date.", get_status_bar("idle")
                    except Exception as e: return f"Hairball during grooming: {e}", get_status_bar("error")
                def on_rebuild():
                    try:
                        n = rebuild_full_index()
                        return grimalkin_respond("sorter", f"Full Vault rebuild complete. {n} pages groomed."), get_status_bar("purring", f"Rebuilt: {n} pages.")
                    except Exception as e: return f"Hairball during rebuild: {e}", get_status_bar("error")
                q_btn.click(fn=on_q, inputs=[q_in, chat], outputs=[chat, status, q_in])
                q_in.submit(fn=on_q, inputs=[q_in, chat], outputs=[chat, status, q_in])
                idx_btn.click(fn=on_index, outputs=[idx_out, status])
                ridx_btn.click(fn=on_rebuild, outputs=[idx_out, status])

            with gr.Tab("Scratch Post"):
                gr.Markdown("### Teach Me Your Ways\n*Correct, adjust, praise, or complain.*")
                f_in = gr.Textbox(placeholder="Tell the cat...", label="Feedback", lines=3)
                f_out = gr.Markdown()
                f_btn = gr.Button("Scratch", variant="primary")
                bond_md = gr.Markdown(value=_bond_display())
                def on_fb(text):
                    try: resp = handle_feedback(text)
                    except Exception: resp = "Hrk. Something broke. Try again."
                    return resp, get_status_bar("purring"), "", _bond_display()
                f_btn.click(fn=on_fb, inputs=[f_in], outputs=[f_out, status, f_in, bond_md])
                gr.Markdown("---\n### Settings")
                s_cb = gr.Checkbox(label="Serious Mode (strips purrs, keeps claws)", value=get_setting("serious_mode")=="true")
                a_in = gr.Textbox(label="How should I address you?", value=get_setting("user_address","my witch"), placeholder="my witch / keeper / ...")
                def on_s(chk):
                    set_setting("serious_mode", "true" if chk else "false")
                    return get_status_bar("idle", "Serious mode." if chk else "The cat returns. *stretches*")
                def on_a(addr):
                    a = addr.strip() or "my witch"
                    set_setting("user_address", a)
                    return get_status_bar("idle", f"I shall call you '{a}'.")
                s_cb.change(fn=on_s, inputs=[s_cb], outputs=[status])
                a_in.change(fn=on_a, inputs=[a_in], outputs=[status])
    return app


# =============================================================================
# SECTION 6: BACKGROUND WATCHER (hash-based, daemon thread)
# =============================================================================

class HuntingGroundWatcher:
    def __init__(self, watch_dir=None, interval=30):
        self.watch_dir = Path(watch_dir) if watch_dir else WATCHED_DIR
        self.interval = interval
        self._hashes = set()
        self._running = False

    def start(self):
        if self._running: return
        self._running = True
        self._snap()
        threading.Thread(target=self._loop, daemon=True).start()
        log.info(f"Watcher: {self.watch_dir} (every {self.interval}s)")

    def stop(self):
        self._running = False

    def _snap(self):
        if not self.watch_dir.exists(): return
        for f in self.watch_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                try: self._hashes.add(hash_file(str(f)))
                except Exception: pass

    def _loop(self):
        while self._running:
            try:
                if self.watch_dir.exists():
                    for f in self.watch_dir.iterdir():
                        if f.is_file() and not f.name.startswith(".") and not f.name.endswith((".crdownload",".part",".tmp")):
                            try:
                                if hash_file(str(f)) not in self._hashes:
                                    log.info("New prey detected"); run_hunt(str(self.watch_dir)); self._snap(); break
                            except Exception: pass
            except Exception as e:
                log.warning(f"Watcher: {e}")
            time.sleep(self.interval)


# =============================================================================
# SECTION 7: MAIN
# =============================================================================

def main():
    print("\n  \u2554"+"\u2550"*38+"\u2557")
    print("  \u2551   GRIMALKIN \u2014 Your Familiar          \u2551")
    print("  \u2551   Local AI Cat \u00b7 Sort \u00b7 Index \u00b7 Ask   \u2551")
    print("  \u255a"+"\u2550"*38+"\u255d\n")
    ensure_dirs(); init_db()
    log.info("Memory initialized")
    vs = _load_vectorstore()
    if vs:
        log.info("FAISS index loaded")
    else:
        log.info("No FAISS index yet — use 'Groom the Vault' to build one")
    w = HuntingGroundWatcher(); w.start()
    bl, bn, _ = get_bond_status()
    log.info(f"Bond: {bn} ({bl}/100) | Vault: {get_total_files()} pages | Indexed: {get_vault_size()}")
    log.info(f"Model: {MODEL_CONFIG['chat_model']} via {MODEL_CONFIG['backend']} @ {MODEL_CONFIG['base_url']}")
    create_grimalkin_app().launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, css=GRIMALKIN_CSS)

if __name__ == "__main__":
    main()