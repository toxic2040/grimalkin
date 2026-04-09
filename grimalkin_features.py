"""
Grimalkin v5.0 — Feature Bridge
=================================

Thin adapter between the v4.1 monolith (grimalkin.py) and the v5.0 component
stack (grimalkin_core.py + grimalkin_interfaces.py).

Dependency flow:
    grimalkin.py  →  grimalkin_features.py  →  grimalkin_core.py  →  grimalkin_interfaces.py

What this file does:
    1. make_context()     — builds a fully-wired AppContext (replaces the old
                            db / init_faiss() / embeddings startup sequence).
    2. ollama_chat()      — backward-compat shim that preserves the OllamaResult
                            (text, logprobs) signature that compute_generation_metrics
                            depends on, while routing through the v5.0 GPU lock.
    3. grimalkin_respond() — backward-compat shim: old (prompt, context, db) API
                             → new ctx.llm.respond() with live persona.
    4. Re-exports         — every grimalkin_core and grimalkin_interfaces symbol
                            available from a single import for grimalkin.py call sites.

Migration strategy (mechanical, one call site at a time):
    # Old pattern in grimalkin.py:
    result = ollama_chat(prompt, system)
    text, logprobs = result.text, result.logprobs

    # Migrated pattern:
    from grimalkin_features import make_context, ollama_chat
    ctx = make_context()                         # once at startup
    result = ollama_chat(prompt, system, ctx=ctx)  # passes GPU lock + config

Zero behavior change on day one. Same Ollama endpoint, same FAISS index, same results.

—Grimalkin
"""

import logging
import time
from pathlib import Path

from grimalkin_interfaces import (
    # Core types
    GrimalkinConfig,
    AppContext,
    OllamaResult,
    LLMBackend,
    MemoryStore,
    ChunkRecord,
    FeedbackStore,
    # Backends
    OllamaBackend,
    ROCmBackend,
    # Memory stores
    FaissMemoryStore,
    ChromaMemoryStore,
)
from grimalkin_core import (
    # DB lifecycle
    init_db,
    run_all_migrations,
    ensure_dirs,
    # Settings
    get_setting,
    set_setting,
    get_bond_level,
    increment_bond,
    bond_title,
    # Persona
    build_persona,
    build_enhanced_persona,
    # File operations
    file_hash,
    classify_file,
    load_and_chunk,
    scan_hunting_grounds,
    sort_file,
    run_hunt,
    reindex_unindexed,
    ingest_sorted,
    # Knowledge graph
    ingest_entities,
    ingest_relationships,
    graph_stats,
    semantic_graph_context,
    # Retrieval
    keyword_search,
    hybrid_vault_rag,
    recall,
    # Loom visualization
    build_loom_data,
    build_loom_figure,
    describe_node,
    find_clusters,
    list_top_entities,
    export_loom_markdown,
    # Entity management
    merge_entity,
    forget_entity,
    set_entity_importance,
    # Whispers
    proactive_whispers,
    # Migration
    migrate_faiss_v4_to_v5,
)

log = logging.getLogger("grimalkin")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Factory — the v5.0 startup pattern
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_context(
    config: GrimalkinConfig = None,
    backend: str = "ollama",
) -> AppContext:
    """
    Build a fully-wired AppContext — the v5.0 startup pattern.

    Replaces the current main() startup sequence:
        db = init_db()
        index, metadata = init_faiss()
        embeddings = OllamaEmbeddings(...)

    With a single call:
        ctx = make_context()
        # then pass ctx to everything instead of (db, index, metadata)

    Parameters
    ----------
    config : GrimalkinConfig or None
        Configuration object. None → default GrimalkinConfig() (reads from
        the directory containing this file).
    backend : str
        "ollama"  — OllamaBackend (default, current behavior)
        "rocm"    — ROCmBackend (future: llama.cpp server on RX 7900 GRE)

    Returns
    -------
    AppContext
        Fully initialized context. The memory store is loaded from disk if an
        existing FAISS index exists, or initialized fresh. All schema migrations
        are applied idempotently.
    """
    if config is None:
        config = GrimalkinConfig()

    db = init_db(config)
    run_all_migrations(db)
    ensure_dirs(config, db)

    if backend == "rocm":
        llm = ROCmBackend(config)
        log.info("LLM backend: ROCm (llama.cpp)")
    else:
        llm = OllamaBackend(config)
        log.info(f"LLM backend: Ollama ({config.main_model})")

    memory = FaissMemoryStore(config, llm)
    feedback = FeedbackStore(db)

    return AppContext(db=db, config=config, llm=llm, memory=memory, feedback=feedback)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Backward-compat shims — preserve v4.1 call-site signatures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ollama_chat(
    prompt: str,
    system: str = "",
    model: str = "",
    ctx: AppContext = None,
) -> OllamaResult:
    """
    Backward-compat shim: returns OllamaResult(text, logprobs).

    Preserves the signature that compute_generation_metrics depends on.
    Routes through the AppContext GPU lock when ctx is provided, which prevents
    concurrent inference on a single-GPU rig.

    Old call site (grimalkin.py):
        result = ollama_chat(prompt, system)
        text, logprobs = result.text, result.logprobs

    Migrated call site:
        result = ollama_chat(prompt, system, ctx=ctx)  # same return type
        text, logprobs = result.text, result.logprobs  # unchanged

    Parameters
    ----------
    prompt : str
    system : str
        System prompt (persona). Empty = no system turn.
    model : str
        Override model name. Empty = config.main_model.
    ctx : AppContext or None
        When provided, uses ctx.llm's GPU lock and config.
        When None, creates a one-shot backend (isolated calls / scripts).

    Returns
    -------
    OllamaResult(text: str, logprobs: list)
        logprobs is the raw content list from Ollama's logprobs field,
        suitable for passing directly to compute_generation_metrics().
    """
    try:
        import requests as req
    except ImportError:
        return OllamaResult("Hrk. Hairball. The requests library is missing.", [])

    if ctx is not None:
        cfg = ctx.config
        lock = ctx.llm._gpu_lock if hasattr(ctx.llm, "_gpu_lock") else None
    else:
        cfg = GrimalkinConfig()
        lock = None

    resolved_model = model or cfg.main_model
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": resolved_model,
        "messages": messages,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
    }

    last_err = None
    for attempt in range(3):
        try:
            _do_request = lambda: req.post(
                f"{cfg.ollama_url}/v1/chat/completions",
                json=payload,
                timeout=cfg.inference_timeout,
            )

            if lock is not None:
                with lock:
                    resp = _do_request()
            else:
                resp = _do_request()

            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "").strip()
            lp_data = choice.get("logprobs", {})
            logprobs = lp_data.get("content", []) if lp_data else []
            return OllamaResult(text, logprobs)

        except req.ConnectionError as e:
            last_err = e
            log.warning(f"Ollama connection failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
        except req.HTTPError as e:
            log.error(f"Ollama rejected request: {e}")
            return OllamaResult("Hrk. Hairball. Ollama rejected the request.", [])
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return OllamaResult("Hrk. Hairball. Ollama is not responding.", [])

    log.error(f"Ollama unreachable after 3 attempts: {last_err}")
    return OllamaResult("Hrk. Hairball. Ollama is not responding — tried 3 times.", [])


def grimalkin_respond(
    prompt: str,
    context: str = "",
    db=None,
    ctx: AppContext = None,
    task_type: str = "general",
) -> str:
    """
    Backward-compat shim: old (prompt, context, db) → ctx.llm.respond().

    Old call site (grimalkin.py):
        response = grimalkin_respond(prompt, context_str, db)

    Migrated call site (explicit persona + corrections):
        response = grimalkin_respond(prompt, context_str, ctx=ctx, task_type="vault_query")

    When ctx is None (un-migrated call sites), falls back to a one-shot
    OllamaBackend with no persona enrichment — same behavior as v4.0.
    """
    if ctx is not None:
        persona = build_enhanced_persona(ctx, task_type)
        return ctx.llm.respond(prompt, context=context, persona=persona)

    # Fallback: no ctx — one-shot backend, basic persona only
    cfg = GrimalkinConfig()
    llm = OllamaBackend(cfg)
    if db is not None:
        persona = build_persona(db, cfg)
    else:
        persona = ""
    return llm.respond(prompt, context=context, persona=persona)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API surface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    # ── Types ──────────────────────────────────────────────────────────────
    "GrimalkinConfig",
    "AppContext",
    "OllamaResult",
    "LLMBackend",
    "MemoryStore",
    "ChunkRecord",
    "FeedbackStore",
    "OllamaBackend",
    "ROCmBackend",
    "FaissMemoryStore",
    "ChromaMemoryStore",
    # ── Factory ────────────────────────────────────────────────────────────
    "make_context",
    # ── Backward-compat shims ──────────────────────────────────────────────
    "ollama_chat",
    "grimalkin_respond",
    # ── DB lifecycle ────────────────────────────────────────────────────────
    "init_db",
    "run_all_migrations",
    "ensure_dirs",
    # ── Settings ────────────────────────────────────────────────────────────
    "get_setting",
    "set_setting",
    "get_bond_level",
    "increment_bond",
    "bond_title",
    # ── Persona ─────────────────────────────────────────────────────────────
    "build_persona",
    "build_enhanced_persona",
    # ── File operations ─────────────────────────────────────────────────────
    "file_hash",
    "classify_file",
    "load_and_chunk",
    "scan_hunting_grounds",
    "sort_file",
    "run_hunt",
    "reindex_unindexed",
    "ingest_sorted",
    # ── Knowledge graph ──────────────────────────────────────────────────────
    "ingest_entities",
    "ingest_relationships",
    "graph_stats",
    "semantic_graph_context",
    # ── Retrieval ────────────────────────────────────────────────────────────
    "keyword_search",
    "hybrid_vault_rag",
    "recall",
    # ── Loom ─────────────────────────────────────────────────────────────────
    "build_loom_data",
    "build_loom_figure",
    "describe_node",
    "find_clusters",
    "list_top_entities",
    "export_loom_markdown",
    # ── Entity management ────────────────────────────────────────────────────
    "merge_entity",
    "forget_entity",
    "set_entity_importance",
    # ── Misc ──────────────────────────────────────────────────────────────────
    "proactive_whispers",
    "migrate_faiss_v4_to_v5",
]
