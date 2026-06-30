"""test_redact_standalone.py — redaction engine + ingestion loader, no monolith.

Imports only grimalkin_redact (the shared engine) and grimalkin_loader (the live
file-ingestion path). Runs without faiss/langchain/gradio, so redaction has a fast
test surface independent of the UI.

Run via: python -m pytest test_redact_standalone.py -q
"""

import tempfile
from pathlib import Path

from grimalkin_redact import redact, reveal, RedactPolicy, redact_hybrid
from grimalkin_loader import load_and_chunk, redact_chunk_pages
from grimalkin_interfaces import GrimalkinConfig


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


def test_reveal_idempotent_and_no_leak():
    text = "My SSN is 078-05-1120"
    red, mp = redact(text)
    assert "078-05-1120" not in red
    assert "078-05-1120" in reveal(red, mp)


def test_hybrid_falls_back_without_model():
    text = "John name is Test, SSN 123-45-6789"
    red, mp = redact_hybrid(text)
    assert "123-45-6789" not in red
    assert "SSN" in red


def test_redact_preserves_non_pii_context():
    """Fidelity: non-PII labels stay literal in REDACTED; exact roundtrip."""
    cases = [
        "SSN 123-45-6789 card 4111111111111111 email a@b.com",
        "User with SSN 123-45-6789",
    ]
    non_pii_words = ["SSN", "card", "email", "User", "with"]
    for orig in cases:
        red, mp = redact(orig)
        for word in non_pii_words:
            if word in orig:
                assert word in red, f"non-PII context word '{word}' mangled: {orig!r} -> {red!r}"
        assert reveal(red, mp) == orig, f"roundtrip failed: {orig!r}"


def test_ingestion_chunk_path_redacts():
    """Drives the SHIPPED load_and_chunk (grimalkin_loader) on a real .txt file —
    the same entry the sandboxed parse worker uses. Redaction must run on chunk
    content with label fidelity intact. Hits the txt fallback when langchain is
    absent; the real splitter branch when present. Same entry point either way."""
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pii.txt"
        p.write_text("Owner: Alice Example, SSN: 111-22-3333 card 5555555555554444")
        chunks = load_and_chunk(p, cfg)
        assert len(chunks) > 0
        redacted_executed = False
        for c in chunks:
            if "111-22-3333" not in c.page_content and "5555555555554444" not in c.page_content:
                redacted_executed = True
            assert "SSN" in c.page_content or "[SSN" in c.page_content, "fidelity: SSN label must survive"
            assert "card" in c.page_content.lower() or "[CREDIT" in c.page_content
        assert redacted_executed, "redaction must have executed on chunk content"


def test_redact_chunk_real_splitter_path_when_langchain_present():
    """When langchain_text_splitters is importable, drive the real Recursive
    splitter + the redact_chunk_pages seam (as load_and_chunk does post-split).
    Skips cleanly when the optional dep is absent."""
    import pytest

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        pytest.skip("langchain_text_splitters absent; cannot drive the real splitter path here")
    cfg = GrimalkinConfig(pii_redaction="deterministic")
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    from langchain_core.documents import Document
    docs = [Document(page_content="My name is Bob. SSN 123-45-6789 lives here.")]
    split = splitter.split_documents(docs)
    red = redact_chunk_pages(
        [type("D", (), {"page_content": d.page_content, "metadata": {}})() for d in split], cfg
    )
    for c in red:
        assert "123-45-6789" not in c.page_content
        assert "[SSN" in c.page_content or "SSN" in c.page_content
