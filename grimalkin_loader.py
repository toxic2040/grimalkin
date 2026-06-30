"""
Grimalkin document loader
=========================

The one piece of the old `grimalkin_core` that the running app actually used:
load a file, split it into chunks, and redact PII before the text reaches the
FAISS index. The sandboxed parse worker (`grimalkin_parse.py`) spawns this; the
monolith's own `load_and_chunk` is the parent that spawns the worker.

Stdlib + the lazy langchain loader stack + the shared `grimalkin_redact` engine.
"""

import logging
from pathlib import Path
from typing import List, Optional

from grimalkin_interfaces import GrimalkinConfig
from grimalkin_redact import RedactPolicy, make_policy_from_config, redact, redact_hybrid

log = logging.getLogger("grimalkin")


# Lazy imports for loaders (avoid pulling langchain unless chunking files)
def _get_text_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def _get_loader_map():
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredWordDocumentLoader,
        CSVLoader,
    )
    return {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".html": TextLoader,
        ".htm": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".csv": CSVLoader,
    }


def _scoped_policy(cfg: Optional[GrimalkinConfig], scope: str = "") -> RedactPolicy:
    policy = make_policy_from_config(cfg)
    policy.placeholder_prefix = scope
    return policy


def redact_chunk_pages(chunks: List, cfg: Optional[GrimalkinConfig] = None) -> List:
    """Apply redaction (deterministic or hybrid per cfg) to chunk.page_content in
    place before indexing. Returns the same chunks. PII never reaches FAISS."""
    if not chunks:
        return chunks
    cfg = cfg or GrimalkinConfig()
    mode = getattr(cfg, "pii_redaction", "deterministic")
    if mode == "off":
        return chunks
    for chunk in chunks:
        if not hasattr(chunk, "metadata") or chunk.metadata is None:
            chunk.metadata = {}
        original = chunk.page_content
        if mode == "hybrid":
            redacted, mapping = redact_hybrid(original, _scoped_policy(cfg, "DOC"))
        else:
            redacted, mapping = redact(original, _scoped_policy(cfg, "DOC"))
        chunk.page_content = redacted
        if mapping:
            chunk.metadata["pii_redacted"] = "true"
    return chunks


LOADER_MAP = None  # populated lazily


def load_and_chunk(filepath: Path, config: GrimalkinConfig = None) -> list:
    """Load a file and split into redacted chunks with metadata."""
    global LOADER_MAP
    ext = filepath.suffix.lower()
    if LOADER_MAP is None:
        try:
            LOADER_MAP = _get_loader_map()
        except Exception:
            LOADER_MAP = {}
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        # Fallback simple loader for txt/md when langchain missing (for tests / env)
        if ext in (".txt", ".md"):
            try:
                with open(str(filepath), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                chunk_size = config.chunk_size if config else 800
                chunk_overlap = config.chunk_overlap if config else 100
                docs = []
                i = 0
                while i < len(content):
                    chunk_text = content[i:i + chunk_size]

                    class SimpleDoc:
                        def __init__(self, t, meta):
                            self.page_content = t
                            self.metadata = meta

                    docs.append(SimpleDoc(chunk_text, {}))
                    i += chunk_size - chunk_overlap if chunk_overlap < chunk_size else chunk_size
                chunks = docs
                cfg = config or GrimalkinConfig()
                for chunk in chunks:
                    chunk.metadata["filename"] = filepath.name
                    chunk.metadata["source_path"] = str(filepath)
                return redact_chunk_pages(chunks, cfg)
            except Exception as e:
                log.warning(f"Failed fallback load {filepath.name}: {e}")
                return []
        return []

    chunk_size = config.chunk_size if config else 800
    chunk_overlap = config.chunk_overlap if config else 100
    Splitter = _get_text_splitter()
    splitter = Splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    try:
        loader = loader_cls(str(filepath))
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        cfg = config or GrimalkinConfig()
        for chunk in chunks:
            chunk.metadata["filename"] = filepath.name
            chunk.metadata["source_path"] = str(filepath)
        return redact_chunk_pages(chunks, cfg)
    except Exception as e:
        log.warning(f"Failed to load {filepath.name}: {e}")
        return []
