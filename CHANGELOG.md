# Grimalkin — Changelog

---

## v4.1 — The Mirror Wakes

**The Mirror now has a tab.** It was documented. Now it exists.

### New Features

| Feature | Description |
|---------|-------------|
| **🪞 Mirror tab** | View the latest reflection and weave new ones on demand — no longer scheduler-only |
| **⚙️ Settings tab** | Change familiar name and address title directly in the UI |
| `mirror` command | Read the latest Mirror reflection from Scratch Post |
| `address` command | Change how Grimalkin addresses you (`address captain`, `address old friend`, etc.) |
| Opening line | Random flavor quote shown on each session start |
| 7 new easter eggs | `meow`, `purr`, `sleep`, `feed me`, `who made you`, `bad cat`, `thank you` |

### Bug Fixes

| Fix | Detail |
|-----|--------|
| `describe_node` hardcoded "Seven" | Thread count now reflects actual relationships |
| `find_clusters` hardcoded intro | No longer says "Seventeen names dance" regardless of data |
| `generate_weekly_reflection` system prompt | Now uses current `pet_name` instead of hardcoded "Grimalkin" |
| `OllamaEmbeddings` import | Tries `langchain_ollama` first, falls back to `langchain_community` for forward compatibility |
| `PERSONA_SYSTEM` global removed | Was a stale module-level variable; replaced with inline `build_persona()` call at usage site |

---

## v4.0 — The Veil Lifts

The Loom came alive. The Mirror began forming. The cat got a name.

### New Features

| Feature | Description |
|---------|-------------|
| **🪞 The Mirror** | Weekly reflections generated from vault activity and stored in `reflections` table |
| **`pet_name` setting** | Rename your familiar with `name <new_name>` — persists across sessions |
| **`build_persona(name)`** | Dynamic system prompt builder using current familiar name |
| **🕸️ The Loom** | Force-directed knowledge graph visualization with Plotly + HTML fallback |
| `spring_layout()` | Pure NumPy force-directed layout — no networkx dependency |
| `describe_node()` | Entity deep-dive: type, sightings, all connected threads |
| `find_clusters()` | Surface the most densely connected entity pairs |
| `export_loom_markdown()` | Dump the full web to a markdown file in sorted/ |
| `merge_entity()` | Canonicalize duplicate entities, dedup relationships |
| `set_entity_importance()` | Flag entities as important (★ in listings) |
| `forget_entity()` | Remove an entity and all its relationships |
| `recall` command | Cross-source synthesis: files + graph + past reflections |
| `proactive_whispers()` | Bond ≥ 60 unlocks proactive entity insights in Whispers |
| `migrate_v4()` | Adds `reflections` table and `importance` column via safe ALTER TABLE |

---

## v3.0 — The Pyre and The Web

Fire and memory. Files could be destroyed. Entities could be known.

### New Features

| Feature | Description |
|---------|-------------|
| **🔥 The Pyre** | Ritual file deletion: 3-step ceremony, bond gate (≥ 30), name confirmation |
| 7-day ash cooling | Files move to `sorted/PYRE/` and sit for 7 days before permanent deletion |
| `unburn` command | Rescue files from the Pyre before cremation |
| **Knowledge Graph** | Entity + relationship extraction via LLM during nightly groom |
| `entities`, `stats` commands | Graph visibility from Scratch Post |
| Nightly groom | Automated tag/note/entity extraction on a 24h scheduler |
| `groom` command | Trigger nightly groom manually |
| Custom categories | User-defined sort categories stored in settings |
| `migrate_v3()` | Adds `entities`, `relationships` tables; `burned_at` column on `file_memory` |
| `PYRE` + `DUPLICATES` folders | Created automatically by `ensure_dirs()` |

---

## v2.1 — Hybrid Search

The vault got smarter. Keyword blindness was fixed.

### Changes

| Change | Description |
|--------|-------------|
| `keyword_search()` | Multi-term OR matching — "ARGUS whitepaper" now finds files with either word |
| `hybrid_vault_rag()` | Merged FAISS semantic + keyword results with score boosting |
| `ingest` command | Discover and index orphan files already present in sorted/ |
| `index` command | Re-index files that failed or were skipped on first pass |
| Removed `route_vault_query` | Was a passthrough wrapper; inlined into callers |

---

## v2.0 — The Full Rebuild

Single file. One database. One cat. Everything that came before was practice.

**Source:** `grimalkin.py` (722 LOC) + `indexer.py` (67 LOC) → `grimalkin.py` (983 LOC)

### Bug Fixes

| # | Fix | Severity |
|---|-----|----------|
| 1 | UTC/local time mismatch in `generate_whispers` | HIGH |
| 2 | 6 bare `except:` → `except Exception:` across all modules | HIGH |
| 3 | Duplicate files moved to `sorted/DUPLICATES/` instead of silently deleted | HIGH |
| 4 | Qwen3 `<think>` tag stripping in `scrub_corporate()` | HIGH |
| 5 | `_call_openai` URL double-`/v1` bug | MEDIUM |
| 6 | `on_upload` hashes source before copy | MEDIUM |
| 7 | `log_interaction` bond bump now atomic within `_DB_LOCK` | MEDIUM |
| 8 | WAL + `synchronous=NORMAL` set per-connection, not just at init | MEDIUM |
| 9 | Symlinks skipped in `run_hunt` | LOW |
| 10 | `hash_file` size gate: files >500 MB use fast fingerprint | LOW |

### Structural Changes

- Lazy FAISS index loading — loads on first query, not at startup
- `DEFAULTS` dict: all magic numbers in one place
- `INDEXABLE_EXTENSIONS` set: explicit declaration of what the vault can read
- Bond system: every interaction increments bond (0–100)
- Whispers: daily briefings with bond-gated detail levels
- `idx_fm_indexed` and `idx_briefing_date` indexes for query performance
- `_VS_LOCK` as `RLock` to prevent deadlock on nested lock acquisition
