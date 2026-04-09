"""
Grimalkin test harness — pure-logic tests, no Ollama or Gradio required.
"""

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from grimalkin import (
    build_chat_context,
    classify_file,
    file_hash,
    get_chat_summary,
    get_recent_history,
    keyword_search,
    repair_json,
    save_chat_message,
    scrub_corporate,
    spring_layout,
    EXTENSION_MAP,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_test_db():
    """Create an in-memory DB with tables for keyword_search and chat memory tests."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE file_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            original_path TEXT,
            sorted_path TEXT,
            category TEXT,
            file_hash TEXT UNIQUE,
            indexed INTEGER DEFAULT 0,
            tags TEXT DEFAULT '[]',
            notes TEXT DEFAULT '',
            burned_at TIMESTAMP DEFAULT NULL
        )
    """)
    db.execute("""
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    db.execute("""
        CREATE TABLE chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE chat_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            summary TEXT NOT NULL,
            turn_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return db


def insert_file(db, filename, category="MISC", tags="[]", notes="", indexed=1):
    """Insert a test file into file_memory."""
    fh = hashlib.sha256(filename.encode()).hexdigest()
    db.execute(
        "INSERT INTO file_memory (filename, original_path, sorted_path, category, file_hash, indexed, tags, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (filename, f"/orig/{filename}", f"/sorted/{filename}", category, fh, indexed, tags, notes),
    )
    db.commit()


# ─── scrub_corporate ─────────────────────────────────────────────────────────

def test_scrub_removes_corporate_phrases():
    text = "Certainly! I'd be happy to help you with that."
    result = scrub_corporate(text)
    assert "Certainly" not in result
    assert "happy to help" not in result
    assert "you with that." in result


def test_scrub_preserves_clean_text():
    text = "The vault contains 42 files across three categories."
    assert scrub_corporate(text) == text


def test_scrub_case_insensitive():
    text = "ABSOLUTELY! as an ai, I cannot actually do that."
    result = scrub_corporate(text)
    assert "absolutely" not in result.lower()
    assert "as an ai" not in result.lower()


# ─── file_hash ────────────────────────────────────────────────────────────────

def test_file_hash_deterministic():
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
        f.write(b"grimalkin test data")
        path = Path(f.name)
    h1 = file_hash(path)
    h2 = file_hash(path)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest
    path.unlink()


def test_file_hash_differs_for_different_content():
    paths = []
    for content in [b"file one", b"file two"]:
        f = tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False)
        f.write(content)
        f.close()
        paths.append(Path(f.name))
    assert file_hash(paths[0]) != file_hash(paths[1])
    for p in paths:
        p.unlink()


# ─── classify_file ────────────────────────────────────────────────────────────

def test_classify_known_extensions():
    assert classify_file(Path("report.pdf")) == "FINANCIAL"
    assert classify_file(Path("data.csv")) == "FINANCIAL"
    assert classify_file(Path("photo.jpg")) == "MEDIA"
    assert classify_file(Path("song.mp3")) == "MEDIA"


def test_classify_unknown_extension():
    assert classify_file(Path("mystery.xyz")) == "MISC"


def test_classify_research_extensions():
    assert classify_file(Path("code.py")) == "RESEARCH"
    assert classify_file(Path("notes.md")) == "RESEARCH"


# ─── keyword_search ──────────────────────────────────────────────────────────

def test_keyword_search_by_filename():
    db = make_test_db()
    insert_file(db, "quarterly_report.pdf", category="FINANCIAL")
    insert_file(db, "vacation_photo.jpg", category="MEDIA")
    results = keyword_search(db, "quarterly")
    assert "quarterly_report.pdf" in results
    assert "vacation_photo.jpg" not in results


def test_keyword_search_by_tags():
    db = make_test_db()
    insert_file(db, "doc.pdf", tags='["invoice", "tax"]')
    results = keyword_search(db, "invoice")
    assert "doc.pdf" in results


def test_keyword_search_by_notes():
    db = make_test_db()
    insert_file(db, "file.txt", notes="Contains ARGUS whitepaper references")
    results = keyword_search(db, "ARGUS")
    assert "file.txt" in results


def test_keyword_search_multi_term():
    db = make_test_db()
    insert_file(db, "report.pdf", notes="quarterly earnings")
    insert_file(db, "photo.jpg", notes="vacation sunset")
    results = keyword_search(db, "quarterly sunset")
    assert "report.pdf" in results
    assert "photo.jpg" in results


def test_keyword_search_excludes_burned():
    db = make_test_db()
    insert_file(db, "alive.txt", notes="important data")
    insert_file(db, "dead.txt", notes="important data")
    db.execute("UPDATE file_memory SET burned_at = '2025-01-01' WHERE filename = 'dead.txt'")
    db.commit()
    results = keyword_search(db, "important")
    assert "alive.txt" in results
    assert "dead.txt" not in results


def test_keyword_search_excludes_unindexed():
    db = make_test_db()
    insert_file(db, "indexed.txt", notes="test data", indexed=1)
    insert_file(db, "unindexed.txt", notes="test data", indexed=0)
    results = keyword_search(db, "test")
    assert "indexed.txt" in results
    assert "unindexed.txt" not in results


def test_keyword_search_empty_query():
    db = make_test_db()
    insert_file(db, "file.txt")
    assert keyword_search(db, "") == set()
    assert keyword_search(db, " ") == set()


# ─── repair_json ──────────────────────────────────────────────────────────────

def test_repair_json_clean():
    raw = json.dumps({"files": [{"filename": "test.pdf", "tags": ["tax"]}]})
    result = repair_json(raw)
    assert result["files"][0]["filename"] == "test.pdf"


def test_repair_json_with_markdown_fence():
    raw = '```json\n{"files": [{"filename": "test.pdf"}]}\n```'
    result = repair_json(raw)
    assert result["files"][0]["filename"] == "test.pdf"


def test_repair_json_trailing_comma():
    raw = '{"files": [{"filename": "a.pdf",},]}'
    result = repair_json(raw)
    assert result["files"][0]["filename"] == "a.pdf"


def test_repair_json_bare_list():
    raw = '[{"filename": "a.pdf"}, {"filename": "b.pdf"}]'
    result = repair_json(raw)
    assert len(result["files"]) == 2


def test_repair_json_preamble_text():
    raw = 'Here are the results:\n\n{"files": [{"filename": "test.pdf"}]}'
    result = repair_json(raw)
    assert result["files"][0]["filename"] == "test.pdf"


def test_repair_json_total_garbage():
    result = repair_json("this is not json at all")
    assert result == {"files": []}


# ─── spring_layout ────────────────────────────────────────────────────────────

def test_spring_layout_empty():
    assert spring_layout([], []) == {}


def test_spring_layout_single_node():
    result = spring_layout(["A"], [])
    assert "A" in result
    assert result["A"] == (0.5, 0.5)


def test_spring_layout_returns_all_nodes():
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C")]
    np.random.seed(42)
    result = spring_layout(nodes, edges, iterations=20)
    assert set(result.keys()) == {"A", "B", "C"}


def test_spring_layout_positions_in_unit_square():
    nodes = ["A", "B", "C", "D"]
    edges = [("A", "B"), ("C", "D")]
    np.random.seed(42)
    result = spring_layout(nodes, edges, iterations=20)
    for name, (x, y) in result.items():
        assert 0.0 <= x <= 1.0, f"{name} x={x} out of [0,1]"
        assert 0.0 <= y <= 1.0, f"{name} y={y} out of [0,1]"


def test_spring_layout_deterministic_with_seed():
    nodes = ["A", "B", "C"]
    edges = [("A", "B")]
    np.random.seed(99)
    r1 = spring_layout(nodes, edges, iterations=30)
    np.random.seed(99)
    r2 = spring_layout(nodes, edges, iterations=30)
    for name in nodes:
        assert r1[name] == r2[name], f"{name} positions differ across runs"


# ─── chat memory ─────────────────────────────────────────────────────────────

def test_save_and_retrieve_chat():
    db = make_test_db()
    sid = "test-session-1"
    save_chat_message(db, sid, "user", "hello")
    save_chat_message(db, sid, "assistant", "greetings mortal")
    history = get_recent_history(db, sid, limit=4)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "greetings mortal"


def test_recent_history_ordering():
    db = make_test_db()
    sid = "test-session-2"
    for i in range(10):
        save_chat_message(db, sid, "user", f"msg {i}")
        save_chat_message(db, sid, "assistant", f"reply {i}")
    history = get_recent_history(db, sid, limit=3)
    assert len(history) == 6  # 3 pairs
    assert history[0]["content"] == "msg 7"  # 4th from last pair
    assert history[-1]["content"] == "reply 9"


def test_chat_summary_empty():
    db = make_test_db()
    summary, count = get_chat_summary(db, "nonexistent")
    assert summary == ""
    assert count == 0


def test_build_chat_context_empty():
    db = make_test_db()
    ctx = build_chat_context(db, "empty-session")
    assert ctx == ""


def test_build_chat_context_with_history():
    db = make_test_db()
    sid = "context-test"
    save_chat_message(db, sid, "user", "what is grimalkin?")
    save_chat_message(db, sid, "assistant", "I am your familiar.")
    ctx = build_chat_context(db, sid)
    assert "what is grimalkin?" in ctx
    assert "I am your familiar." in ctx


def test_session_isolation():
    db = make_test_db()
    save_chat_message(db, "session-a", "user", "hello from A")
    save_chat_message(db, "session-b", "user", "hello from B")
    history_a = get_recent_history(db, "session-a")
    history_b = get_recent_history(db, "session-b")
    assert len(history_a) == 1
    assert len(history_b) == 1
    assert history_a[0]["content"] == "hello from A"
    assert history_b[0]["content"] == "hello from B"


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    ok = run_all()
    sys.exit(0 if ok else 1)
