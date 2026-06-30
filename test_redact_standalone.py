"""
test_redact_standalone.py — pure redaction + seam tests, zero grimalkin monolith imports.

Imports only grimalkin_redact + the three extracted seams from grimalkin_core.
This allows pytest to run without faiss/langchain/gradio/SystemExit.

Run via: python -m pytest test_redact_standalone.py -q --tb=line
Or via scripts/verify_grimalkin_goal.py
"""

import sys
from pathlib import Path
from typing import List
import tempfile

# Standalone: only these
from grimalkin_redact import redact, reveal, RedactPolicy, redact_hybrid
from grimalkin_core import (
    GrimalkinConfig,
    prepare_query_for_llm,
    redact_chunk_pages,
    empty_vault_turn,
    merge_redaction_maps,
    maybe_reveal_llm_output,
)

# ─── tests ported/adapted ─────────────────────────────────────────────────────

def test_deterministic_redact_basic():
    policy = RedactPolicy()
    text = "SSN 123-45-6789, card 4111111111111111, email test@example.com, phone 555-123-4567, name is John Smith"
    red, mapping = redact(text, policy)
    assert "123-45-6789" not in red
    assert "4111111111111111" not in red
    assert "test@example.com" not in red
    assert "SSN" in red and "card" in red and "email" in red
    assert "[SSN_1]" in red or "[CREDIT_CARD_1]" in red or "[EMAIL_1]" in red
    restored = reveal(red, mapping)
    assert "123-45-6789" in restored
    print("deterministic basic ok")


def test_reveal_idempotent_and_no_leak():
    text = "My SSN is 078-05-1120"
    red, mp = redact(text)
    assert "078-05-1120" not in red
    restored = reveal(red, mp)
    assert "078-05-1120" in restored
    print("reveal ok")


def test_hybrid_falls_back_without_model():
    text = "John name is Test, SSN 123-45-6789"
    red, mp = redact_hybrid(text)
    assert "123-45-6789" not in red
    assert "SSN" in red
    print("hybrid fallback executed without leak")


def test_redact_in_core_chunk_path():
    """Drives the SHIPPED load_and_chunk (grimalkin_core) on a real .txt file.
    Exercises whatever branch the shipped code takes (fallback here because no langchain).
    Redaction via seam is asserted.
    """
    import grimalkin_core
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pii.txt"
        orig = "Owner: Alice Example, SSN: 111-22-3333 card 5555555555554444"
        p.write_text(orig)
        chunks = grimalkin_core.load_and_chunk(p, cfg)  # SHIPPED code path
        assert len(chunks) > 0
        redacted_executed = False
        for c in chunks:
            if "111-22-3333" not in c.page_content and "5555555555554444" not in c.page_content:
                redacted_executed = True
            assert "SSN" in c.page_content or "[SSN" in c.page_content, "fidelity: SSN label must survive"
            assert "card" in c.page_content.lower() or "[CREDIT" in c.page_content
        assert redacted_executed, "redaction must have executed on chunk content"
        print("chunk path (shipped load_and_chunk + seam) redaction executed with fidelity")


def test_redact_chunk_real_splitter_path_when_langchain_present():
    """When langchain_text_splitters is importable, drive the real Recursive splitter
    + redaction seam (as the shipped load_and_chunk would after its split step).
    Asserts placeholder produced. Skips cleanly when dep absent (current env).
    """
    import pytest
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pytest.skip("langchain_text_splitters absent; cannot drive/assert real splitter path in this env")
    from langchain_core.documents import Document
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    docs = [Document(page_content="My name is Bob. SSN 123-45-6789 lives here.")]
    split = splitter.split_documents(docs)
    # drive redaction on the real-split chunks (mirrors post-split code in shipped load_and_chunk)
    red = redact_chunk_pages([type("D", (), {"page_content": d.page_content, "metadata": {}})() for d in split], cfg)
    for c in red:
        assert "123-45-6789" not in c.page_content
        assert "[SSN" in c.page_content or "SSN" in c.page_content
    print("real Recursive splitter + redaction path driven and asserted")


def test_redact_before_respond_path():
    """Uses prepare seam + recording to show redaction-before-LLM."""
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    prompt = "Hi, my name is Test User and SSN is 222-33-4444"
    received = []

    def recording_respond(red_p):
        received.append(red_p)
        return "LLM-RECEIVED: " + red_p

    # Simulate the respond path: redact before LLM; default output stays redacted.
    red_prompt, pmap = prepare_query_for_llm(prompt, cfg, "P")
    # no extra context
    red_ctx, cmap = "", {}
    full_map = merge_redaction_maps(pmap, cmap)
    llm_out = recording_respond(red_prompt)
    final = maybe_reveal_llm_output(llm_out, full_map, cfg)

    assert len(received) == 1
    assert "222-33-4444" not in received[0], "raw PII leaked to LLM"
    assert "[P_SSN_1]" in received[0] or "[P_GIVEN_NAME" in received[0], "no scoped placeholder sent to LLM"
    assert "222-33-4444" not in final, "default response rehydrated PII"
    print("respond path via seam: redaction before LLM confirmed")


def test_redact_preserves_non_pii_context():
    """Fidelity: non-PII labels stay literal in REDACTED; exact roundtrip."""
    cases = [
        "SSN 123-45-6789 card 4111111111111111 email a@b.com",
        "User with SSN 123-45-6789"
    ]
    non_pii_words = ["SSN", "card", "email", "User", "with"]
    for orig in cases:
        red, mp = redact(orig)
        for word in non_pii_words:
            if word in orig:
                assert word in red, f"non-PII context word '{word}' mangled: {orig!r} -> {red!r}"
        restored = reveal(red, mp)
        assert restored == orig, f"roundtrip failed: {orig!r} != {restored!r}"
    print("fidelity ok")


def test_empty_vault_turn_redacts_before_llm_and_keeps_default_output_redacted():
    """Drives the empty_vault_turn seam with recording respond_fn.
    Asserts: fn saw redacted (placeholder), return value remains redacted by default.
    """
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    received = []

    def recording_respond(red_q: str) -> str:
        received.append(red_q)
        # simulate LLM echoing what it received (redacted)
        return "LLM-RECEIVED: " + red_q

    query = "Hi, my name is Test User and SSN is 222-33-4444"
    result = empty_vault_turn(query, recording_respond, cfg)

    assert len(received) == 1
    sent = received[0]
    assert "222-33-4444" not in sent, "raw PII reached respond_fn"
    assert "[Q_SSN_1]" in sent or "[Q_GIVEN_NAME" in sent
    assert "222-33-4444" not in result, "default empty-vault response rehydrated PII"
    print("empty_vault_turn seam: redacted to LLM, default output redacted PASS")


def test_scoped_placeholders_keep_query_and_context_maps_separate():
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    q_red, qmap = prepare_query_for_llm("query SSN 111-22-3333", cfg, "Q")
    c_red, cmap = prepare_query_for_llm("context SSN 222-33-4444", cfg, "C")
    merged = merge_redaction_maps(qmap, cmap)

    assert "[Q_SSN_1]" in q_red
    assert "[C_SSN_1]" in c_red
    assert reveal("[Q_SSN_1] [C_SSN_1]", merged) == "111-22-3333 222-33-4444"


def test_core_rag_redacts_before_retrieval_and_llm():
    import grimalkin_core
    from types import SimpleNamespace

    cfg = GrimalkinConfig(pii_redaction="deterministic")
    calls = {"memory": [], "llm": []}

    ctx = SimpleNamespace()
    ctx.config = cfg
    ctx.memory = SimpleNamespace(search=lambda q, k=15: calls["memory"].append(q) or [])
    ctx.db = SimpleNamespace(cursor=lambda: SimpleNamespace())
    ctx.llm = SimpleNamespace(
        respond=lambda prompt, context="", persona="": calls["llm"].append((prompt, context)) or f"LLM {prompt}"
    )

    original_keyword = grimalkin_core.keyword_search
    original_persona = grimalkin_core.build_enhanced_persona
    try:
        grimalkin_core.keyword_search = lambda db, q, limit=10: set()
        grimalkin_core.build_enhanced_persona = lambda ctx, task_type="general": "persona"
        result = grimalkin_core.hybrid_vault_rag(ctx, "find SSN 111-22-3333")
    finally:
        grimalkin_core.keyword_search = original_keyword
        grimalkin_core.build_enhanced_persona = original_persona

    assert calls["memory"] and "111-22-3333" not in calls["memory"][0]
    assert calls["llm"] and "111-22-3333" not in calls["llm"][0][0]
    assert "111-22-3333" not in result


# ─── runner (for direct or pytest) ────────────────────────────────────────────

def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        name = test.__name__
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
