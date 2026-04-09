"""
Grimalkin v5.0 — Interface Contracts
=====================================

Drop-in abstraction layer for the v4.0 monolith.
These protocols define the seams. Current implementations wrap existing code 1:1.
Future upgrades (ChromaDB, llama.cpp ROCm, council) swap the implementation,
not the callers.

Usage in main():
    config = GrimalkinConfig()
    llm = OllamaBackend(config)          # swap to: ROCmBackend(config)
    memory = FaissMemoryStore(config)     # swap to: ChromaMemoryStore(config)
    # everything else receives these interfaces, never raw index/metadata

—Grimalkin
"""

from __future__ import annotations

import threading
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared result type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OllamaResult = namedtuple("OllamaResult", ["text", "logprobs"])

log = logging.getLogger("grimalkin")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration — replaces module-level constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GrimalkinConfig:
    """Single source of truth for all tunables. Swap models at runtime."""

    # Models
    main_model: str = "qwen3:8b"           # GPU primary — swap to 14B Q5 when ready
    router_model: str = ""                  # CPU routing — empty = no router yet
    embed_model: str = "nomic-embed-text"   # CPU embeddings via Ollama

    # Inference backend
    ollama_url: str = "http://localhost:11434"
    inference_timeout: int = 120
    gpu_layers: int = 99                    # -ngl for llama.cpp / Ollama

    # Memory / FAISS
    faiss_dim: int = 768
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_vectors: int = 120_000              # prune threshold (future)
    max_persona_tokens: int = 512           # cap on persona system prompt

    # Paths
    app_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def db_path(self) -> Path:
        return self.app_dir / "grimalkin.db"

    @property
    def faiss_index_path(self) -> Path:
        return self.app_dir / "faiss_index" / "index.faiss"

    @property
    def faiss_meta_path(self) -> Path:
        return self.app_dir / "faiss_index" / "metadata.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Backend Protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMBackend(ABC):
    """
    Everything that talks to a language model goes through this interface.

    Current callers in v4.0:
        ollama_chat()        → infer()
        grimalkin_respond()  → respond() [convenience wrapper]
        embeddings.embed_*   → embed_texts() / embed_query()

    The GPU lock lives HERE, not in callers. Any implementation that
    touches the GPU must acquire self._gpu_lock before inference.
    """

    @abstractmethod
    def infer(self, prompt: str, system: str = "", model: str = "") -> str:
        """Single-turn LLM completion. Maps to current ollama_chat()."""
        ...

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed. Maps to current embeddings.embed_documents()."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Single query embed. Maps to current embeddings.embed_query()."""
        ...

    def respond(self, prompt: str, context: str = "", persona: str = "") -> str:
        """
        Convenience: context + prompt → infer with persona.
        Maps to current grimalkin_respond(). Shared across all backends.
        """
        full = f"{context}\n\n{prompt}" if context else prompt
        return self.infer(full, system=persona)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Backend: Ollama (wraps your current code exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OllamaBackend(LLMBackend):
    """
    1:1 wrap of existing ollama_chat + OllamaEmbeddings.
    No behavior change. Just behind the interface now.
    """

    def __init__(self, config: GrimalkinConfig):
        self.config = config
        self._gpu_lock = threading.Lock()

        # Lazy import to match current pattern
        from langchain_community.embeddings import OllamaEmbeddings
        self._embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_url,
        )

    def infer(self, prompt: str, system: str = "", model: str = "") -> str:
        import requests as req

        model = model or self.config.main_model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # GPU lock: only one inference at a time on single-GPU rig
        with self._gpu_lock:
            try:
                resp = req.post(
                    f"{self.config.ollama_url}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                    timeout=self.config.inference_timeout,
                )
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "").strip()
            except Exception as e:
                log.error(f"Ollama error: {e}")
                return "Hrk. Hairball. Ollama is not responding."

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Backend: ROCm llama.cpp (future — skeleton only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ROCmBackend(LLMBackend):
    """
    Future: llama.cpp server with HIP on RX 7900 GRE.
    Same interface, different engine. Callers don't change.

    Start server externally:
        ./llama-server -m qwen3-14b-q5_k_m.gguf -ngl 99 --host 0.0.0.0 -c 8192
    Then point config.ollama_url at its /v1/chat/completions endpoint.
    """

    def __init__(self, config: GrimalkinConfig):
        self.config = config
        self._gpu_lock = threading.Lock()
        # Embeddings still via Ollama on CPU — no reason to waste GPU
        from langchain_community.embeddings import OllamaEmbeddings
        self._embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_url,
        )

    def infer(self, prompt: str, system: str = "", model: str = "") -> str:
        import requests as req

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with self._gpu_lock:
            try:
                # llama.cpp server uses OpenAI-compatible endpoint
                resp = req.post(
                    f"{self.config.ollama_url}/v1/chat/completions",
                    json={"messages": messages, "max_tokens": 1024},
                    timeout=self.config.inference_timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.error(f"llama.cpp ROCm error: {e}")
                return "Hrk. Hairball. The GPU familiar is not responding."

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Store Protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ChunkRecord:
    """What goes into and comes out of the vector store."""
    text: str
    filename: str
    source_path: str
    file_hash: str
    score: float = 0.0
    vector_id: int = -1


class MemoryStore(ABC):
    """
    Everything that touches vectors/embeddings goes through this interface.

    Current callers in v4.0:
        init_faiss()              → __init__ / load
        index_chunks()            → add_chunks()
        faiss_search()            → search()
        save_faiss()              → save() [called internally]
        (missing in v4.0)         → remove_file()  ← the ghost vector fix

    The threading lock lives HERE. Callers never manage locks.
    """

    @abstractmethod
    def add_chunks(self, chunks: list, file_hash: str) -> int:
        """Embed and store chunks. Returns count added."""
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[ChunkRecord]:
        """Semantic search. Returns scored results."""
        ...

    @abstractmethod
    def remove_file(self, file_hash: str) -> int:
        """Remove all vectors for a file. Returns count removed. THE FIX."""
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist to disk."""
        ...

    @abstractmethod
    def total_vectors(self) -> int:
        """Expose index.ntotal for observability."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Store: Hardened FAISS (Path 1 — wraps current code + fixes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FaissMemoryStore(MemoryStore):
    """
    Wraps existing FAISS code with three fixes:
    1. IndexIDMap2 instead of IndexFlatL2 (enables remove_ids)
    2. threading.Lock on all mutations
    3. file_hash tracked in every metadata entry

    Drop-in replacement. Same data, same search quality.
    """

    def __init__(self, config: GrimalkinConfig, llm: LLMBackend):
        import faiss
        import json

        self.config = config
        self._llm = llm  # for embeddings
        self._lock = threading.Lock()
        self._next_id = 0

        idx_path = config.faiss_index_path
        meta_path = config.faiss_meta_path

        if idx_path.exists() and meta_path.exists():
            self._index = faiss.read_index(str(idx_path))
            with open(meta_path) as f:
                self._metadata = json.load(f)
            # Recover next_id from existing metadata
            existing_ids = [m.get("vector_id", 0) for m in self._metadata]
            self._next_id = max(existing_ids, default=-1) + 1
        else:
            base = faiss.IndexFlatL2(config.faiss_dim)
            self._index = faiss.IndexIDMap2(base)
            self._metadata = []
            self._next_id = 0

        log.info(f"MemoryStore loaded: {self._index.ntotal} vectors, "
                 f"{len(self._metadata)} metadata entries")

    def add_chunks(self, chunks: list, file_hash: str) -> int:
        """
        Maps to current index_chunks() but with:
        - file_hash tracked per entry
        - vector_id assigned via IndexIDMap2
        - lock held during mutation
        """
        import numpy as np

        if not chunks:
            return 0

        texts = [c.page_content for c in chunks]
        try:
            vecs = np.array(self._llm.embed_texts(texts), dtype=np.float32)
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            return 0

        with self._lock:
            ids = np.arange(
                self._next_id,
                self._next_id + len(chunks),
                dtype=np.int64,
            )
            self._index.add_with_ids(vecs, ids)

            for i, chunk in enumerate(chunks):
                self._metadata.append({
                    "vector_id": int(ids[i]),
                    "filename": chunk.metadata.get("filename", ""),
                    "source_path": chunk.metadata.get("source_path", ""),
                    "text": chunk.page_content[:500],
                    "file_hash": file_hash,
                })

            self._next_id += len(chunks)

        return len(chunks)

    def search(self, query: str, k: int = 5) -> list[ChunkRecord]:
        """Maps to current faiss_search(). Lock-safe read."""
        import numpy as np

        if self._index.ntotal == 0:
            return []

        try:
            vec = np.array(
                self._llm.embed_query(query), dtype=np.float32
            ).reshape(1, -1)
        except Exception as e:
            log.error(f"Query embedding failed: {e}")
            return []

        with self._lock:
            k_actual = min(k, self._index.ntotal)
            distances, indices = self._index.search(vec, k_actual)

            # Build ID → metadata lookup for safe access
            id_to_meta = {m["vector_id"]: m for m in self._metadata
                          if "vector_id" in m}

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                meta = id_to_meta.get(int(idx))
                if not meta:
                    continue
                results.append(ChunkRecord(
                    text=meta.get("text", ""),
                    filename=meta.get("filename", ""),
                    source_path=meta.get("source_path", ""),
                    file_hash=meta.get("file_hash", ""),
                    score=float(dist),
                    vector_id=int(idx),
                ))

        return results

    def remove_file(self, file_hash: str) -> int:
        """
        THE MISSING PIECE in v4.0.
        Called from execute_burn(). Removes ghost vectors immediately.
        """
        import numpy as np

        with self._lock:
            to_remove = [
                m["vector_id"] for m in self._metadata
                if m.get("file_hash") == file_hash and "vector_id" in m
            ]
            if not to_remove:
                return 0

            self._index.remove_ids(np.array(to_remove, dtype=np.int64))
            self._metadata = [
                m for m in self._metadata
                if m.get("file_hash") != file_hash
            ]

        self.save()
        log.info(f"Removed {len(to_remove)} vectors for file_hash={file_hash[:12]}…")
        return len(to_remove)

    def save(self) -> None:
        """Maps to current save_faiss(). Atomic-ish."""
        import faiss
        import json

        with self._lock:
            faiss.write_index(self._index, str(self.config.faiss_index_path))
            with open(self.config.faiss_meta_path, "w") as f:
                json.dump(self._metadata, f)

    def total_vectors(self) -> int:
        return self._index.ntotal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Memory Store: ChromaDB (future — skeleton)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ChromaMemoryStore(MemoryStore):
    """
    Future drop-in when vault exceeds ~10k chunks.
    Same interface. HNSW index, native metadata filtering, built-in persistence.
    Callers don't change at all.
    """

    def __init__(self, config: GrimalkinConfig, llm: LLMBackend):
        self.config = config
        self._llm = llm
        # pip install chromadb
        # from langchain_community.vectorstores import Chroma
        raise NotImplementedError("Enable when ready: pip install chromadb")

    def add_chunks(self, chunks: list, file_hash: str) -> int:
        ...  # chroma.add_documents(docs)

    def search(self, query: str, k: int = 5) -> list[ChunkRecord]:
        ...  # chroma.similarity_search(query, k=k, filter={"burned_at": None})

    def remove_file(self, file_hash: str) -> int:
        ...  # chroma.delete(filter={"file_hash": file_hash})

    def save(self) -> None:
        pass  # ChromaDB auto-persists

    def total_vectors(self) -> int:
        ...  # return chroma._collection.count()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feedback Store — the learning mechanism
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeedbackStore:
    """
    NOT in v4.0. This is the missing piece for "learn and adapt."

    When the user corrects Grimalkin, store the correction keyed by task type.
    At inference time, retrieve relevant past corrections and inject into
    the system prompt. This is how the familiar gets smarter over time.

    Schema addition to init_db():
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,         -- 'vault_query', 'groom', 'classify', 'general'
            original_query TEXT,
            bad_response TEXT,
            correction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_corrections_type ON corrections(task_type);
    """

    def __init__(self, db):
        self.db = db
        self._ensure_table()

    def _ensure_table(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                original_query TEXT,
                bad_response TEXT,
                correction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_corrections_type ON corrections(task_type);
        """)
        self.db.commit()

    def record(self, task_type: str, query: str, bad_response: str, correction: str):
        """User says 'wrong, it should be X' → store it."""
        self.db.execute(
            "INSERT INTO corrections (task_type, original_query, bad_response, correction) "
            "VALUES (?, ?, ?, ?)",
            (task_type, query, bad_response, correction),
        )
        self.db.commit()
        log.info(f"Correction recorded for task_type={task_type}")

    def get_relevant(self, task_type: str, query: str, limit: int = 3) -> list[dict]:
        """
        Retrieve past corrections relevant to current task.
        Inject these into the system prompt so the model doesn't repeat mistakes.

        Simple keyword + type match for now.
        Future: embed corrections and do semantic retrieval.
        """
        cur = self.db.cursor()

        # First: exact task type match, most recent
        cur.execute(
            "SELECT original_query, correction FROM corrections "
            "WHERE task_type = ? ORDER BY created_at DESC LIMIT ?",
            (task_type, limit),
        )
        results = [{"query": r[0], "correction": r[1]} for r in cur.fetchall()]
        return results

    def build_correction_context(self, task_type: str, query: str) -> str:
        """
        Returns a string to prepend to the system prompt.
        Empty string if no relevant corrections exist.
        """
        relevant = self.get_relevant(task_type, query)
        if not relevant:
            return ""

        lines = ["Past corrections to remember:"]
        for r in relevant:
            lines.append(f"- When asked about '{r['query']}', "
                         f"the correct answer was: {r['correction']}")
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application Context — the v5.0 startup bundle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AppContext:
    """
    Single object that carries all runtime state.

    Replaces the (db, index, metadata) triple that threads through every v4.0
    function call. In v5.0, callers receive ctx and access ctx.db, ctx.llm,
    ctx.memory, ctx.feedback instead of positional index/metadata arguments.

    Construct via make_context() in grimalkin_features.py:
        ctx = make_context()                       # default: Ollama + FAISS
        ctx = make_context(backend="rocm")         # future: ROCm backend
    """

    db: Any                 # sqlite3.Connection — shared across threads (WAL mode)
    config: GrimalkinConfig
    llm: LLMBackend
    memory: MemoryStore
    feedback: FeedbackStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# How it all wires together — updated main()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def example_main():
    """
    Shows how the refactored startup looks.
    Compare to current main() on line 1769 of grimalkin.py:

    CURRENT:
        db = init_db()
        index, metadata = init_faiss()
        start_scheduler(db, index, metadata)
        demo = build_ui(db, index, metadata)

    REFACTORED:
        db = init_db()
        config = GrimalkinConfig()
        llm = OllamaBackend(config)              # ← swap to ROCmBackend later
        memory = FaissMemoryStore(config, llm)    # ← swap to ChromaMemoryStore later
        feedback = FeedbackStore(db)              # ← NEW: learning mechanism
        start_scheduler(db, memory)
        demo = build_ui(db, llm, memory, feedback)

    Functions that currently take (db, index, metadata) now take (db, memory).
    Functions that currently call ollama_chat() now call llm.infer().
    Functions that currently call grimalkin_respond() now call llm.respond().

    The migration is mechanical — find/replace the signatures, swap the calls.
    Zero behavior change on day one.
    """
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Migration cheat sheet — what changes in existing functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MIGRATION_NOTES = """
EXISTING FUNCTION              → WHAT CHANGES
─────────────────────────────────────────────────────────
ollama_chat(prompt, system)    → llm.infer(prompt, system)
grimalkin_respond(p, ctx, db)  → llm.respond(p, ctx, persona)
embeddings.embed_documents(t)  → llm.embed_texts(t)      [via memory store]
embeddings.embed_query(t)      → llm.embed_query(t)      [via memory store]

init_faiss()                   → FaissMemoryStore(config, llm)
index_chunks(idx, meta, ch)    → memory.add_chunks(chunks, file_hash)
faiss_search(idx, meta, q, k)  → memory.search(query, k)
save_faiss(idx, meta)          → memory.save()
(missing)                      → memory.remove_file(fh)  ← called in execute_burn

sort_file(db, idx, meta, fp)   → sort_file(db, memory, fp)
run_hunt(db, idx, meta)        → run_hunt(db, memory)
hybrid_vault_rag(db,i,m, q)    → hybrid_vault_rag(db, memory, q)
nightly_groom_v4(db, i, m)     → nightly_groom_v4(db, llm, memory)
reindex_unindexed(db, i, m)    → reindex_unindexed(db, memory)
ingest_sorted(db, i, m)        → ingest_sorted(db, memory)
recall(db, i, m, term)         → recall(db, llm, memory, term)
build_ui(db, i, m)             → build_ui(db, llm, memory, feedback)
start_scheduler(db, i, m)      → start_scheduler(db, memory)

execute_burn(db, fh)           → execute_burn(db, memory, fh)
                                  ADD: memory.remove_file(fh)  ← ghost vector fix

handle_scratch_post(db,i,m,u)  → handle_scratch_post(db, llm, memory, feedback, u)
                                  ADD: feedback.build_correction_context() in prompt

NEW COMMANDS in Scratch Post:
    "correct <text>"           → feedback.record(task_type, last_query, last_resp, text)
    "corrections"              → list recent corrections
─────────────────────────────────────────────────────────
Total functions that change signature: ~15
Total new LOC for interfaces:          ~350
Total behavior change on day one:      ZERO (same Ollama, same FAISS, same results)
"""
