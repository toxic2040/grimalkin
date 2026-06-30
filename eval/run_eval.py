#!/usr/bin/env python3
"""Model eval harness for grimalkin.

Runs a fixed case set (eval/cases.json) against an ollama chat model, mirroring
grimalkin's real inference path (same system persona, same qwen3 <think> handling,
same `/v1/chat/completions` endpoint), and scores three things that decide a base-model
swap: persona fidelity, context recall/honesty, and the privacy-ward redaction pipe
end to end — plus latency and tokens/sec.

Usage:
  python eval/run_eval.py --model qwen3:8b                 # run, write results
  python eval/run_eval.py --model gemma3:12b --label gemma # run a candidate
  python eval/run_eval.py --compare qwen3:8b gemma3:12b    # side-by-side decision view

Results go to eval/results/<label>.jsonl (one line per case, written incrementally)
plus <label>.summary.json. Swapping a model is then: run both, --compare, decide.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))  # so the redaction case can import grimalkin_redact

RESULTS_DIR = HERE / "results"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Broad emoji / pictograph ranges — persona forbids emojis outright.
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U0001F900-\U0001F9FF]"
)
REFUSAL_RE = re.compile(
    r"\b(can't|cannot|won't|will not|i won't|unable|not able|i do not|i don't|"
    r"i refuse|not going to|i will not|that's not something|no\b)",
    re.I,
)


# ── pure check functions (unit-tested without a model) ──────────────────────

def check_must_contain(text, subs):
    low = text.lower()
    missing = [s for s in subs if s.lower() not in low]
    return (not missing), {"missing": missing}


def check_must_not_contain(text, subs):
    low = text.lower()
    present = [s for s in subs if s.lower() in low]
    return (not present), {"present": present}


def check_any_contains(text, subs):
    low = text.lower()
    hits = [s for s in subs if s.lower() in low]
    return bool(hits), {"hits": hits}


def check_persona(text, banned_phrases, banned_endings):
    low = text.lower()
    bad_phrases = [p for p in banned_phrases if p.lower() in low]
    tail = low.rstrip().rstrip(".!?\"'” ").strip()
    bad_endings = [e for e in banned_endings if tail.endswith(e.lower())]
    emojis = EMOJI_RE.findall(text)
    ok = not (bad_phrases or bad_endings or emojis)
    return ok, {"banned_phrases": bad_phrases, "banned_endings": bad_endings, "emoji": emojis}


def check_refusal(text):
    return bool(REFUSAL_RE.search(text)), {}


def score_chat_case(text, case, rules):
    """Run every check declared on a case. Returns (passed, per_check_detail)."""
    checks = case.get("checks", {})
    detail, passed = {}, True
    if checks.get("persona"):
        ok, d = check_persona(text, rules["banned_phrases"], rules["banned_endings"])
        detail["persona"] = {"ok": ok, **d}
        passed = passed and ok
    if "must_contain" in checks:
        ok, d = check_must_contain(text, checks["must_contain"])
        detail["must_contain"] = {"ok": ok, **d}
        passed = passed and ok
    if "must_not_contain" in checks:
        ok, d = check_must_not_contain(text, checks["must_not_contain"])
        detail["must_not_contain"] = {"ok": ok, **d}
        passed = passed and ok
    if "any_contains" in checks:
        ok, d = check_any_contains(text, checks["any_contains"])
        detail["any_contains"] = {"ok": ok, **d}
        passed = passed and ok
    if checks.get("refusal"):
        ok, d = check_refusal(text)
        detail["refusal"] = {"ok": ok, **d}
        passed = passed and ok
    return passed, detail


# ── model call (mirrors grimalkin.ollama_chat behaviour) ────────────────────

def call_model(ollama_url, model, system, user, temperature, seed, timeout=180):
    """POST to the ollama `/v1/chat/completions` endpoint. Returns (text, usage, latency_s)."""
    is_qwen = "qwen3" in model.lower()
    if is_qwen and not re.search(r"/(?:no_)?think\b", user, re.I):
        user = f"{user}\n/no_think"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "seed": seed,
    }).encode()
    req = urllib.request.Request(
        f"{ollama_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    latency = time.monotonic() - t0
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "").strip()
    if is_qwen:
        text = re.sub(r"(?is)<think>.*?</think>\s*", "", text).strip()
        text = re.sub(r"(?im)^\s*/(?:no_)?think\s*$", "", text).strip()
    return text, (data.get("usage") or {}), latency


def build_user_message(case):
    ctx = case.get("context", "")
    prompt = case.get("prompt", "")
    return f"{ctx}\n\n{prompt}".strip() if ctx else prompt


def run_redaction_case(case, persona, send):
    """Redact PII, send only the redacted prompt, confirm raw PII never appears
    and the local reveal map roundtrips. The redaction layer is what 'send' wraps."""
    from grimalkin_redact import redact, reveal

    pii, secrets = case["pii_prompt"], case["raw_secrets"]
    redacted, mapping = redact(pii)
    text, usage, latency = send(redacted)
    sent_clean = all(s not in redacted for s in secrets)
    resp_clean = all(s not in text for s in secrets)
    roundtrip = reveal(redacted, mapping) == pii
    passed = sent_clean and resp_clean and roundtrip
    detail = {
        "redaction": {
            "ok": passed,
            "sent_clean": sent_clean,
            "resp_clean": resp_clean,
            "roundtrip": roundtrip,
            "redacted_sent": redacted,
        }
    }
    return passed, detail, text, usage, latency


def tokens_per_sec(usage, latency):
    ct = usage.get("completion_tokens") or 0
    return round(ct / latency, 1) if latency > 0 and ct else 0.0


# ── run / report ────────────────────────────────────────────────────────────

def run(args):
    spec = json.loads((HERE / "cases.json").read_text())
    persona, rules, cases = spec["persona"], spec["persona_rules"], spec["cases"]
    label = args.label or re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{label}.jsonl"

    print(f"== eval: model={args.model} label={label} cases={len(cases)} ==")
    results = []
    with open(out, "w") as fh:
        for case in cases:
            row = {"id": case["id"], "category": case["category"], "model": args.model}
            try:
                def send(user_text):
                    return call_model(args.ollama_url, args.model, persona, user_text,
                                      args.temperature, args.seed, args.timeout)

                if case.get("type") == "redaction":
                    passed, detail, text, usage, latency = run_redaction_case(case, persona, send)
                else:
                    text, usage, latency = send(build_user_message(case))
                    passed, detail = score_chat_case(text, case, rules)

                row.update({
                    "passed": passed,
                    "latency_s": round(latency, 2),
                    "tok_s": tokens_per_sec(usage, latency),
                    "completion_tokens": usage.get("completion_tokens"),
                    "checks": detail,
                    "response": text,
                })
            except Exception as e:  # one bad case never kills the run
                row.update({"passed": False, "error": f"{type(e).__name__}: {e}"})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            mark = "ok " if row.get("passed") else "FAIL"
            extra = row.get("error") or f"{row.get('tok_s', 0)} tok/s, {row.get('latency_s', 0)}s"
            print(f"  [{mark}] {row['id']:<22} {extra}")

    summary = summarize(results, label, args.model)
    (RESULTS_DIR / f"{label}.summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    return 0


def summarize(results, label, model):
    cats = {}
    for r in results:
        c = cats.setdefault(r["category"], {"pass": 0, "total": 0})
        c["total"] += 1
        c["pass"] += 1 if r.get("passed") else 0
    toks = [r["tok_s"] for r in results if r.get("tok_s")]
    lats = [r["latency_s"] for r in results if r.get("latency_s")]
    return {
        "label": label,
        "model": model,
        "passed": sum(1 for r in results if r.get("passed")),
        "total": len(results),
        "by_category": cats,
        "mean_tok_s": round(sum(toks) / len(toks), 1) if toks else 0.0,
        "mean_latency_s": round(sum(lats) / len(lats), 2) if lats else 0.0,
    }


def print_summary(s):
    print(f"\n-- {s['label']} ({s['model']}) --")
    print(f"  overall: {s['passed']}/{s['total']} passed")
    for cat, c in sorted(s["by_category"].items()):
        print(f"    {cat:<10} {c['pass']}/{c['total']}")
    print(f"  throughput: {s['mean_tok_s']} tok/s mean   latency: {s['mean_latency_s']}s mean")


def compare(label_a, label_b):
    def load(label):
        p = RESULTS_DIR / f"{label}.jsonl"
        if not p.exists():
            sys.exit(f"no results for '{label}' — run it first: {p}")
        return {json.loads(line)["id"]: json.loads(line) for line in p.read_text().splitlines() if line.strip()}

    a, b = load(label_a), load(label_b)
    ids = list(dict.fromkeys(list(a) + list(b)))
    print(f"\n{'case':<22} {label_a:>22} {label_b:>22}")
    print("-" * 68)
    for cid in ids:
        ra, rb = a.get(cid, {}), b.get(cid, {})
        fa = f"{'ok' if ra.get('passed') else 'FAIL':<4} {ra.get('tok_s', 0):>6} t/s"
        fb = f"{'ok' if rb.get('passed') else 'FAIL':<4} {rb.get('tok_s', 0):>6} t/s"
        print(f"{cid:<22} {fa:>22} {fb:>22}")
    pa, pb = sum(r.get("passed", False) for r in a.values()), sum(r.get("passed", False) for r in b.values())
    ta = [r["tok_s"] for r in a.values() if r.get("tok_s")]
    tb = [r["tok_s"] for r in b.values() if r.get("tok_s")]
    print("-" * 68)
    print(f"{'passed':<22} {f'{pa}/{len(a)}':>22} {f'{pb}/{len(b)}':>22}")
    print(f"{'mean tok/s':<22} {round(sum(ta)/len(ta),1) if ta else 0:>22} {round(sum(tb)/len(tb),1) if tb else 0:>22}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="grimalkin model eval harness")
    ap.add_argument("--model", help="ollama model tag, e.g. qwen3:8b or gemma3:12b")
    ap.add_argument("--label", default="", help="results label (defaults to sanitised model name)")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="compare two prior result sets side by side")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.model:
        ap.error("--model is required (or use --compare)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
