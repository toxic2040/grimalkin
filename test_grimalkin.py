"""
Grimalkin test harness — pure-logic tests, no Ollama or Gradio required.
"""

import hashlib
import json
import subprocess
import sqlite3
import sys
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import grimalkin
import grimalkin_loader
import numpy as np

from grimalkin_redact import redact, reveal, RedactPolicy, redact_hybrid

from grimalkin import (
    _check_auth,
    _command_status,
    _gradio_file_paths,
    _ingest_allowed,
    _is_loopback_host,
    _strip_think_artifacts,
    _validate_launch_security,
    _voice_template_values,
    _with_no_think,
    audit_event,
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
    ui_pyre_row_select,
    ui_control_deck,
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
    db.execute("""
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            detail TEXT
        )
    """)
    return db


def make_graph_db():
    """Create an in-memory DB with enough graph tables for Loom tests."""
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            type TEXT,
            times_seen INTEGER DEFAULT 1,
            importance INTEGER DEFAULT 0
        )
    """)
    db.execute("""
        CREATE TABLE relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            target_id INTEGER,
            relation_type TEXT,
            source_file TEXT
        )
    """)
    return db


def insert_file(db, filename, category="MISC", tags="[]", notes="", indexed=1):
    """Insert a test file into file_memory."""
    fh = hashlib.sha256(filename.encode()).hexdigest()
    db.execute(
        "INSERT INTO file_memory (filename, original_path, sorted_path, category, file_hash, indexed, tags, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            filename,
            f"/orig/{filename}",
            f"/sorted/{filename}",
            category,
            fh,
            indexed,
            tags,
            notes,
        ),
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


# ─── local control helpers ───────────────────────────────────────────────────


def test_qwen3_no_think_added_once():
    assert _with_no_think("hello", "qwen3:8b") == "hello\n/no_think"
    assert _with_no_think("hello\n/no_think", "qwen3:8b") == "hello\n/no_think"
    assert _with_no_think("hello", "llama3.2") == "hello"


def test_qwen3_think_artifacts_are_stripped():
    raw = "<think>hidden chain</think>\n\nVisible answer.\n/no_think"
    assert _strip_think_artifacts(raw, "qwen3:8b") == "Visible answer."
    assert _strip_think_artifacts(raw, "llama3.2") == raw


def test_command_status_handles_missing_and_ready_commands():
    missing_state, missing_detail, missing_class = _command_status("")
    assert (missing_state, missing_detail, missing_class) == (
        "missing",
        "not configured",
        "warn",
    )
    ready_state, ready_detail, ready_class = _command_status(f"{sys.executable} -V")
    assert ready_state == "ready"
    assert ready_detail == sys.executable
    assert ready_class == "ok"


def test_command_status_expands_voice_template_placeholders():
    values = _voice_template_values(audio="", out="", text_file="", text="")
    command = '"{python}" "{app}/scripts/grim_voice.py" status --json'
    state, detail, css_class = _command_status(command)
    assert state == "ready"
    assert detail == values["python"]
    assert css_class == "ok"


def test_audit_event_records_metadata_only():
    db = make_test_db()
    audit_event(db, "control_check", "deck refreshed")
    row = db.execute("SELECT event_type, detail FROM audit_log").fetchone()
    assert row == ("control_check", "deck refreshed")


def test_control_deck_escapes_audit_details():
    db = make_test_db()
    db.execute(
        "INSERT INTO audit_log (event_type, detail) VALUES (?, ?)",
        ("local", "<script>alert(1)</script>"),
    )
    db.commit()
    html = ui_control_deck(db)
    assert "Control Deck" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_voice_adapter_status_returns_json():
    script = Path(__file__).parent / "scripts" / "grim_voice.py"
    proc = subprocess.run(
        [sys.executable, str(script), "status", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    status = json.loads(proc.stdout)
    assert "stt" in status
    assert "tts" in status


def test_voice_adapter_marker_tts_writes_wav():
    script = Path(__file__).parent / "scripts" / "grim_voice.py"
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "reply.wav"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "tts",
                "--engine",
                "marker",
                "--text",
                "local test",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        with wave.open(str(out_path), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000


def test_gradio_file_paths_only_allow_avatar_files():
    paths = _gradio_file_paths()
    assert str(grimalkin.AVATAR_PATH) in paths
    assert str(grimalkin.AVATAR_FALLBACK) in paths
    assert str(grimalkin.APP_DIR) not in paths


def test_auth_uses_token_when_configured():
    old_cfg = grimalkin.CFG
    try:
        grimalkin.CFG = replace(old_cfg, auth_token="secret-token")
        assert _check_auth("ignored", "secret-token") is True
        assert _check_auth("ignored", "wrong-token") is False
        assert _check_auth("ignored", None) is False
    finally:
        grimalkin.CFG = old_cfg


def test_non_loopback_launch_requires_auth_token():
    old_cfg = grimalkin.CFG
    try:
        grimalkin.CFG = replace(old_cfg, auth_token="")
        assert _is_loopback_host("127.0.0.1") is True
        assert _is_loopback_host("::1") is True
        assert _is_loopback_host("0.0.0.0") is False
        try:
            _validate_launch_security("0.0.0.0")
        except SystemExit as exc:
            assert "GRIM_AUTH_TOKEN" in str(exc)
        else:
            raise AssertionError("non-loopback launch without auth token did not fail")

        grimalkin.CFG = replace(old_cfg, auth_token="secret-token")
        _validate_launch_security("0.0.0.0")
    finally:
        grimalkin.CFG = old_cfg


def test_pyre_row_select_escapes_filename():
    class Event:
        index = [0]

    rows = [{"filename": '<img src=x onerror="alert(1)">.pdf', "file_hash": "abc"}]
    preview, file_hash_value = ui_pyre_row_select(Event(), rows)
    assert file_hash_value == "abc"
    assert "<img src=x" not in preview
    assert "&lt;img src=x" in preview


def test_loom_html_fallback_escapes_entities():
    db = make_graph_db()
    db.execute(
        "INSERT INTO entities (name, type, times_seen, importance) VALUES (?, ?, ?, ?)",
        ('<img src=x onerror="alert(1)">', "topic<script>", 3, 0),
    )
    db.commit()
    old_plotly = grimalkin.HAS_PLOTLY
    try:
        grimalkin.HAS_PLOTLY = False
        html = grimalkin.build_loom_figure(db)
    finally:
        grimalkin.HAS_PLOTLY = old_plotly
    assert "<img src=x" not in html
    assert "topic<script>" not in html
    assert "&lt;img src=x" in html
    assert "topic&lt;script&gt;" in html


def test_core_loom_html_fallback_escapes_entities():
    db = make_graph_db()
    db.execute(
        "INSERT INTO entities (name, type, times_seen, importance) VALUES (?, ?, ?, ?)",
        ('<svg onload="alert(1)">', "person<script>", 2, 0),
    )
    db.commit()
    old_plotly = grimalkin.HAS_PLOTLY
    try:
        grimalkin.HAS_PLOTLY = False
        html = grimalkin.build_loom_figure(db)
    finally:
        grimalkin.HAS_PLOTLY = old_plotly
    assert "<svg" not in html
    assert "person<script>" not in html
    assert "&lt;svg" in html
    assert "person&lt;script&gt;" in html


# ─── ingestion gate + isolated parse worker ───────────────────────────────────


def test_ingest_gate_rejects_unsupported_type():
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
        path = Path(f.name)
    try:
        ok, why = _ingest_allowed(path)
        assert ok is False
        assert "unsupported" in why
    finally:
        path.unlink()


def test_ingest_gate_rejects_oversize():
    old_cfg = grimalkin.CFG
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"prey")
        path = Path(f.name)
    try:
        grimalkin.CFG = replace(old_cfg, max_ingest_mb=0)
        ok, why = _ingest_allowed(path)
        assert ok is False
        assert "ingest limit" in why
        # Gate short-circuits before the worker is ever spawned.
        assert grimalkin.load_and_chunk(path) == []
    finally:
        grimalkin.CFG = old_cfg
        path.unlink()


def test_parse_worker_returns_chunks_for_text():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Grimalkin guards the vault. The cat remembers every thread.")
        path = Path(f.name)
    try:
        chunks = grimalkin.load_and_chunk(path)
        assert len(chunks) >= 1
        assert "Grimalkin" in chunks[0].page_content
        assert chunks[0].metadata["filename"] == path.name
        assert chunks[0].metadata["source_path"] == str(path)
    finally:
        path.unlink()


def test_parse_worker_fails_closed_under_low_memory_cap():
    """A memory cap too small to even import the parser must degrade to [],
    never crash the caller — the containment guarantee. The cap tracks the
    worker's import floor (now leaner: grimalkin_loader, no numpy), so 24 MB is
    below the floor on this stack while 48 MB would import fine."""
    old_cfg = grimalkin.CFG
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("prey")
        path = Path(f.name)
    try:
        grimalkin.CFG = replace(old_cfg, parse_mem_mb=24)
        assert grimalkin.load_and_chunk(path) == []
    finally:
        grimalkin.CFG = old_cfg
        path.unlink()


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
    db.execute(
        "UPDATE file_memory SET burned_at = '2025-01-01' WHERE filename = 'dead.txt'"
    )
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


# ─── PII Redaction tests (options 1+2) ────────────────────────────────────────

def test_deterministic_redact_basic():
    policy = RedactPolicy()
    text = "SSN 123-45-6789, card 4111111111111111, email test@example.com, phone 555-123-4567, name John Smith"
    red, mapping = redact(text, policy)
    assert "123-45-6789" not in red
    assert "4111111111111111" not in red
    assert "test@example.com" not in red
    # non-PII context words preserved (fidelity)
    assert "SSN" in red
    assert "card" in red
    assert "email" in red
    assert "[SSN_1]" in red or "[CREDIT_CARD_1]" in red or "[EMAIL_1]" in red
    restored = reveal(red, mapping)
    assert "123-45-6789" in restored or "4111" in restored  # at least one restored
    print("deterministic basic ok")

def test_reveal_idempotent_and_no_leak():
    text = "My SSN is 078-05-1120"
    red, mp = redact(text)
    assert "078-05-1120" not in red
    restored = reveal(red, mp)
    assert "078-05-1120" in restored

def test_hybrid_falls_back_without_model():
    # Even without GLiNER, hybrid should not leak (deterministic layer protects)
    text = "John's SSN 123-45-6789"
    red, mp = redact_hybrid(text)
    assert "123-45-6789" not in red
    print("hybrid fallback executed without leak")

def test_redact_in_core_chunk_path():
    # Drives SHIPPED grimalkin_loader.load_and_chunk on .txt file (the parse-worker entry).
    # In envs without langchain_text_splitters it hits the documented txt fallback path
    # (still full redaction + metadata). Real splitter branch taken when the optional
    # dep is present; same entry point is exercised either way.
    import tempfile
    from pathlib import Path
    cfg = grimalkin_loader.GrimalkinConfig(pii_redaction="deterministic")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pii.txt"
        orig = "Owner: Alice Example, SSN: 111-22-3333 card 5555555555554444"
        p.write_text(orig)
        chunks = grimalkin_loader.load_and_chunk(p, cfg)
        assert len(chunks) > 0
        redacted_executed = False
        for c in chunks:
            if "111-22-3333" not in c.page_content and "5555555555554444" not in c.page_content:
                redacted_executed = True
            assert "SSN" in c.page_content or "[SSN" in c.page_content, "fidelity: SSN label must survive in redacted chunk"
            assert "card" in c.page_content.lower() or "[CREDIT" in c.page_content
        assert redacted_executed, "redaction must have executed on chunk content"
        print("chunk path redaction executed with fidelity")


def test_monolith_ollama_chat_redacts_prompt_history_and_system():
    if not grimalkin.HAS_REQUESTS:
        print("requests absent; skipping ollama redaction boundary test")
        return

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "model saw [P_SSN_1]"},
                        "logprobs": {"content": []},
                    }
                ]
            }

    def fake_post(url, json=None, timeout=120):
        captured["json"] = json
        return FakeResponse()

    with patch.object(grimalkin.requests, "post", fake_post):
        result = grimalkin.ollama_chat(
            "my SSN is 111-22-3333",
            system="operator email root@example.com",
            history=[{"role": "user", "content": "old card 4111111111111111"}],
        )

    sent = json.dumps(captured["json"]["messages"])
    assert "111-22-3333" not in sent
    assert "root@example.com" not in sent
    assert "4111111111111111" not in sent
    assert "[P_SSN_1]" in sent
    assert "111-22-3333" not in result.text


def test_active_model_override_reaches_ollama_chat():
    if not grimalkin.HAS_REQUESTS:
        print("requests absent; skipping active model override test")
        return

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "meow"}, "logprobs": {"content": []}}]}

    def fake_post(url, json=None, timeout=120):
        captured["model"] = json["model"]
        return FakeResponse()

    try:
        grimalkin.set_active_model("swapped-model:1b")
        with patch.object(grimalkin.requests, "post", fake_post):
            grimalkin.ollama_chat("hello")
        assert captured["model"] == "swapped-model:1b"
    finally:
        grimalkin.set_active_model("")
    assert grimalkin.get_active_model() == grimalkin.CFG.ollama_model


def test_ui_save_model_persists_and_applies():
    db = make_test_db()
    try:
        msg = grimalkin.ui_save_model(db, "  gemma-test:12b  ")
        assert "gemma-test:12b" in msg
        assert grimalkin.get_setting(db, "ollama_model") == "gemma-test:12b"
        assert grimalkin.get_active_model() == "gemma-test:12b"
        # applying the same model again is a no-op
        assert grimalkin.ui_save_model(db, "gemma-test:12b") == "Nothing changed."
        assert grimalkin.ui_save_model(db, "   ") == "Nothing changed."
        # the swap leaves an audit trail
        row = db.execute(
            "SELECT detail FROM audit_log WHERE event_type='model_update'"
        ).fetchone()
        assert row and "gemma-test:12b" in row[0]
    finally:
        grimalkin.set_active_model("")


def test_monolith_hybrid_vault_rag_redacts_query_before_retrieval():
    calls = {}

    class FakeDb:
        def cursor(self):
            return self

    def fake_faiss_search(index, metadata, query, k=15):
        calls["faiss"] = query
        return []

    def fake_keyword_search(db, query, limit=10):
        calls["keyword"] = query
        return set()

    def fake_graph_query(db, query):
        calls["graph"] = query
        return ""

    def fake_respond(prompt, context="", db=None, faiss_dists=None):
        calls["respond"] = prompt
        return "answer " + prompt

    with patch.object(grimalkin, "faiss_search", fake_faiss_search), \
        patch.object(grimalkin, "keyword_search", fake_keyword_search), \
        patch.object(grimalkin, "graph_query", fake_graph_query), \
        patch.object(grimalkin, "grimalkin_respond", fake_respond):
        result = grimalkin.hybrid_vault_rag(FakeDb(), object(), [], "find SSN 111-22-3333")

    assert "111-22-3333" not in calls["faiss"]
    assert "111-22-3333" not in calls["keyword"]
    assert "111-22-3333" not in calls["respond"]
    assert "111-22-3333" not in result

def test_redact_preserves_non_pii_context():
    """Fidelity fixture per strategy: non-PII labels/context words must stay literal in REDACTED text.
    Uses exact strings from demo and log. Asserts reveal roundtrips exactly.
    """
    from grimalkin_redact import redact, reveal
    cases = [
        "SSN 123-45-6789 card 4111111111111111 email a@b.com",
        "User with SSN 123-45-6789"
    ]
    non_pii_words = ["SSN", "card", "email", "User", "with"]
    for orig in cases:
        red, mp = redact(orig)
        for word in non_pii_words:
            if word in orig:
                assert word in red, f"non-PII context word '{word}' was mangled/removed in redacted for input: {orig!r} -> {red!r}"
        restored = reveal(red, mp)
        assert restored == orig, f"roundtrip failed: orig={orig!r} restored={restored!r}"
    print("test_redact_preserves_non_pii_context PASS (but expected to FAIL initially)")

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
