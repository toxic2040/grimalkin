# Grimalkin v2.0 — Integrated Build Changelog

**Source:** `grimalkin.py` (722 LOC) + `indexer.py` (67 LOC) → `grimalkin.py` (983 LOC)

---

## Bug Fixes (from review)

| # | Fix | Lines | Severity |
|---|-----|-------|----------|
| 1 | **UTC/local time mismatch** — `generate_whispers` now uses `datetime.now(timezone.utc)` to match SQLite's `datetime('now')` (UTC). Briefings no longer miss files for non-UTC users. | 583–600 | HIGH |
| 2 | **6 bare `except:` → `except Exception:`** — watcher, Vault tab, Scratch Post, feedback handler. No longer swallows `KeyboardInterrupt`/`SystemExit`. | scattered | HIGH |
| 3 | **Duplicate files moved to `sorted/DUPLICATES/` instead of `fp.unlink()`** — user files are never silently deleted. | 510–518 | HIGH |
| 4 | **Qwen3 `<think>` tag stripping** — `scrub_corporate()` now strips `<think>...</think>` blocks before other processing. | 424–427 | HIGH |
| 5 | **`_call_openai` URL double-`/v1` bug** — now handles `base_url` ending in `/v1` correctly. | 414–420 | MEDIUM |
| 6 | **`on_upload` hashes source before copy** — no more hashing partial files on interrupted writes. | 848–850 | MEDIUM |
| 7 | **`log_interaction` bond bump now atomic** — count + bond update happen inside the same `_DB_LOCK` context. No race between count and update. | 312–321 | MEDIUM |
| 8 | **WAL + `synchronous=NORMAL` set per-connection** in `_db()` context manager, not just in `init_db()`. Survives DB file deletion/recreation. | 244–250 | MEDIUM |
| 9 | **Symlinks skipped in `run_hunt`** — `not f.is_symlink()` added to file filter. | 503 | LOW |
| 10 | **`hash_file` size gate** — files >500 MB use fast fingerprint (first+last 8KB + size) instead of full SHA-256. | 470–482 | LOW |

## Indexer Integration

| Feature | What it does |
|---------|-------------|
| `_load_vectorstore()` | Lazy singleton — loads FAISS index from disk on first query, not at startup |
| `_get_embeddings()` | Creates `OllamaEmbeddings` from `MODEL_CONFIG` (single source of truth for model + URL) |
| `_load_documents_from_paths()` | Loads files by extension: PDF, MD, TXT, CSV, DOCX, RST, RTF, LOG, JSON |
| `index_new_files()` | **Incremental** — only indexes files where `file_memory.indexed=0`. Adds to existing FAISS index. |
| `rebuild_full_index()` | Resets all `indexed` flags, deletes index files, rebuilds from scratch |
| `get_unindexed_files()` | DB helper: `SELECT ... WHERE indexed=0` |
| `mark_files_indexed()` | DB helper: batch `UPDATE indexed=1` |

### UI additions
- **"Groom the Vault"** button (incremental index) in Vault tab
- **"Full Rebuild"** button in Vault tab
- Startup log now shows FAISS index status

### What was dropped from indexer.py
- Emoji print statements (replaced with `log.info`)
- Hardcoded `knowledge/` path (now uses `VAULT_DIR` + `SORTED_BASE` via `file_memory.sorted_path`)
- Hardcoded `EMBEDDING_MODEL` (now reads `MODEL_CONFIG["embed_model"]`)
- Hardcoded Ollama URL (now reads `MODEL_CONFIG["base_url"]`)
- `DirectoryLoader` glob scanning (replaced with DB-driven `get_unindexed_files` — only loads files we've actually sorted)
- Full rebuild as the only mode (now incremental-first, rebuild on demand)

## Structural Improvements

| Change | Why |
|--------|-----|
| `DEFAULTS` dict at top | All magic numbers in one place (bond values, intervals, FAISS chunk sizes, etc.) |
| `_FEEDBACK_TYPE_MAP` at module level | Was inline in `handle_feedback()` — now discoverable and editable |
| `INDEXABLE_EXTENSIONS` set | Documents which file types the Vault can actually read |
| `DUPLICATES` category folder | Created by `ensure_dirs()` |
| PEP 8 imports (one per line) | Clean `git blame` |
| `idx_fm_indexed` and `idx_briefing_date` indexes | Added to `init_db()` for query performance |
| `_VS_LOCK` is `RLock` | Prevents deadlock when `index_new_files` → `_load_vectorstore` (nested lock acquisition) |

## New Dependencies (pip install)

The FAISS/RAG features require these. The app runs fine without them — it just can't index or search:

```
pip install langchain-ollama langchain-community langchain-text-splitters faiss-cpu pypdf
```

Optional for `.docx` support:
```
pip install docx2txt
```

## LOC Budget

| Component | Before | After |
|-----------|--------|-------|
| grimalkin.py | 722 | 983 |
| indexer.py | 67 | 0 (folded in) |
| **Total** | **789** | **983** |
| Net new LOC | — | **+194** |

Still well under the 2000 LOC pain threshold for a single file.
