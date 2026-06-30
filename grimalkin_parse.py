#!/usr/bin/env python3
"""
Grimalkin parse worker
======================

Runs a single document through the LangChain loaders in an isolated child
process so a hostile or malformed file cannot hang, exhaust memory, or take
down the main Grimalkin process. The parent (`load_and_chunk` in grimalkin.py)
spawns this with a wall-clock timeout; this process additionally caps its own
address space and CPU time before importing any parser.

Contract:
    python grimalkin_parse.py <path> --chunk-size N --chunk-overlap M \
        --mem-mb K --cpu-seconds S

    Emits one JSON object on stdout:
        {"ok": true,  "chunks": [{"text": ..., "filename": ..., "source_path": ...}, ...]}
        {"ok": false, "error": "<reason>"}

Only stdlib + the core loader stack are imported here. Resource limits are
applied first, so even the heavy parser imports run inside the cap.
"""

import argparse
import json
import sys


def _apply_limits(mem_mb: int, cpu_seconds: int) -> None:
    """Cap address space and CPU time. POSIX only; a no-op elsewhere."""
    try:
        import resource
    except ImportError:
        return
    if mem_mb > 0:
        cap = mem_mb * 1024 * 1024
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            ceiling = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
        except (ValueError, OSError):
            pass
    if cpu_seconds > 0:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ValueError, OSError):
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated document parse worker.")
    parser.add_argument("path")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--mem-mb", type=int, default=1024)
    parser.add_argument("--cpu-seconds", type=int, default=55)
    args = parser.parse_args()

    _apply_limits(args.mem_mb, args.cpu_seconds)

    # Heavy imports happen after the limits are in place, on purpose.
    from pathlib import Path

    from grimalkin_loader import load_and_chunk
    from grimalkin_interfaces import GrimalkinConfig

    config = GrimalkinConfig(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    try:
        chunks = load_and_chunk(Path(args.path), config)
        payload = {
            "ok": True,
            "chunks": [
                {
                    "text": c.page_content,
                    "filename": c.metadata.get("filename", ""),
                    "source_path": c.metadata.get("source_path", ""),
                }
                for c in chunks
            ],
        }
    except MemoryError:
        payload = {"ok": False, "error": "memory limit exceeded during parse"}
    except Exception as e:  # noqa: BLE001 — worker boundary; report and exit clean
        payload = {"ok": False, "error": str(e)[:200]}

    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
