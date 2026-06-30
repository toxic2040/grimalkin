"""
grimalkin_redact.py — Local PII Redaction for Grimalkin

Deterministic layer (regex + validators) for structured PII:
- SSNs (with structure rules)
- Credit cards (Luhn)
- Emails
- Phones (basic US/international patterns)
- Routing / account numbers (basic)
- IP addresses
- URLs (optional)

Produces stable placeholders e.g. [SSN_1], [EMAIL_1], [NAME_1] (basic name heuristics for demo).

Provides:
- redact(text, policy=None) -> (redacted_text, mapping)
- reveal(text, mapping) -> restored_text

Mapping is {placeholder: original} for local rehydration only.
Never sent to LLM or index.

Configurable keep-set for coarse context (CITY, STATE, ZIP kept by default for RAG utility).

Pure functions, no side effects. Auditable < 400 LOC core.

For grimalkin: call on user prompts before LLM and on chunks before FAISS/memory.
"""

import re
from typing import Dict, Tuple, Set, Optional, List
from dataclasses import dataclass, field

log = __import__("logging").getLogger("grimalkin")

# ─── Policy ────────────────────────────────────────────────────────────────────

DEFAULT_KEEP: Set[str] = {"CITY", "STATE", "ZIP_CODE"}  # coarse geography useful for context

@dataclass
class RedactPolicy:
    keep: Set[str] = field(default_factory=lambda: set(DEFAULT_KEEP))
    redact_names: bool = True
    redact_structured: bool = True
    placeholder_prefix: str = ""

    def should_redact(self, label: str) -> bool:
        return label not in self.keep

# ─── Validators ────────────────────────────────────────────────────────────────

def luhn_valid(number: str) -> bool:
    """Luhn check for credit cards etc. Strips non-digits."""
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    checksum = 0
    odd = True
    for d in reversed(digits):
        if odd:
            checksum += d
        else:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        odd = not odd
    return checksum % 10 == 0

def ssn_struct_valid(ssn: str) -> bool:
    """US SSN structural validity (area 001-899 except 000/666/9xx, etc.)."""
    m = re.match(r"^(\d{3})-(\d{2})-(\d{4})$", ssn)
    if not m:
        return False
    area, group, serial = m.groups()
    if area in ("000", "666") or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    return True

# ─── Patterns (deterministic first pass) ───────────────────────────────────────

# Structured (high precision)
STRUCTURED_PATTERNS: List[Tuple[str, re.Pattern, callable]] = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ssn_struct_valid),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b"), luhn_valid),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), lambda x: True),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), lambda x: True),
    ("ROUTING_NUMBER", re.compile(r"\b\d{9}\b"), lambda x: True),  # loose; refine in practice
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), lambda x: True),
    ("URL", re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"), lambda x: True),
]

# Basic name heuristics (for deterministic demo; ML will improve)
NAME_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # GIVEN_NAME requires prefix; capture one or two capitalized words for full names
    ("GIVEN_NAME", re.compile(r"(?i)\b(name is |called |mr\. |mrs\. |ms\. |dr\. )([A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?(?:\s+[A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?)?)\b")),
    ("SURNAME", re.compile(r"(?i)\b([A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?)\b(?=\s+(?:st\.|street|ave\.|avenue|rd\.|road|lane|dr\.|drive|ct\.|court|blvd|way|,\s*[A-Z]))")),
]

# Rough address components (street line focus) - more anchored
ADDRESS_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("BUILDING_NUMBER", re.compile(r"\b(\d{1,5})\s+(?:[A-Z][a-z]+ )?(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.|Court|Ct\.|Way|Blvd\.)\b", re.I)),
    ("STREET_NAME", re.compile(r"\b(?:\d{1,5}\s+)?([A-Z][A-Za-z\s]+?(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.|Court|Ct\.|Way|Blvd\.))\b", re.I)),
]

# Tokens that should never be redacted as names (labels like "SSN", "card" etc. in context)
LABEL_DENYLIST = {"SSN", "CARD", "CREDIT_CARD", "EMAIL", "PHONE", "IP_ADDRESS", "URL", "ROUTING_NUMBER", "TAX_ID", "GOVERNMENT_ID", "PASSPORT", "DRIVERS_LICENSE", "USER", "WITH", "MY", "NAME", "IS", "LIVES", "AT", "HELLO"}

# ─── Core Redaction Engine ─────────────────────────────────────────────────────

def _stable_placeholder(label: str, counter: int, prefix: str = "") -> str:
    if prefix:
        clean = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_").upper()
        if clean:
            return f"[{clean}_{label}_{counter}]"
    return f"[{label}_{counter}]"

def _collect_deterministic_spans(text: str, policy: RedactPolicy) -> list[tuple[int, int, str, str]]:
    """Return list of (start, end, label, original_value) for PII spans to redact.
    Shared by redact() and redact_hybrid().
    """
    spans = []
    counters = {}  # per label for placeholder numbering, but numbering done later in rebuild? No, for val uniqueness we track here.

    # Track seen values to reuse placeholders
    value_to_ph = {}

    def should_add(label: str, val: str) -> bool:
        if not policy.should_redact(label):
            return False
        if val.upper() in LABEL_DENYLIST and label in ("GIVEN_NAME", "SURNAME"):
            return False
        return True

    # Structured first
    for label, pattern, validator in STRUCTURED_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            if validator and not validator(val):
                continue
            if should_add(label, val):
                if val not in value_to_ph:
                    counters[label] = counters.get(label, 0) + 1
                    ph = _stable_placeholder(label, counters[label], policy.placeholder_prefix)
                    value_to_ph[val] = ph
                spans.append((m.start(), m.end(), label, val))

    # Names (GIVEN requires prefix group)
    if policy.redact_names:
        for label, pattern in NAME_PATTERNS:
            for m in pattern.finditer(text):
                if label == "GIVEN_NAME":
                    if m.lastindex and m.lastindex >= 2:
                        val = m.group(2).strip()
                        start, end = m.start(2), m.end(2)
                    else:
                        continue  # no prefix, skip
                else:
                    val = m.group(1).strip() if m.lastindex else m.group(0).strip()
                    start, end = m.start(), m.end()
                if should_add(label, val):
                    if val not in value_to_ph:
                        counters[label] = counters.get(label, 0) + 1
                        ph = _stable_placeholder(label, counters[label], policy.placeholder_prefix)
                        value_to_ph[val] = ph
                    spans.append((start, end, label, val))

    # Addresses
    for label, pattern in ADDRESS_PATTERNS:
        for m in pattern.finditer(text):
            val = m.group(0)
            if should_add(label, val):
                if val not in value_to_ph:
                    counters[label] = counters.get(label, 0) + 1
                    ph = _stable_placeholder(label, counters[label], policy.placeholder_prefix)
                    value_to_ph[val] = ph
                spans.append((m.start(), m.end(), label, val))

    return spans

def _rebuild_from_spans(
    text: str,
    spans: list[tuple[int, int, str, str]],
    prefix: str = "",
) -> tuple[str, dict[str, str]]:
    """Rebuild text replacing spans with placeholders. Returns (redacted, mapping ph->orig)."""
    mapping = {}

    spans = sorted(spans, key=lambda x: x[0])
    non_overlap = []
    for s in spans:
        if not non_overlap or s[0] >= non_overlap[-1][1]:
            non_overlap.append(s)

    result_parts = []
    last = 0
    value_to_ph = {}
    counters = {}
    for start, end, label, val in non_overlap:
        if val not in value_to_ph:
            counters[label] = counters.get(label, 0) + 1
            ph = _stable_placeholder(label, counters[label], prefix)
            value_to_ph[val] = ph
            mapping[ph] = val
        else:
            ph = value_to_ph[val]
        result_parts.append(text[last:start])
        result_parts.append(ph)
        last = end
    result_parts.append(text[last:])
    redacted = ''.join(result_parts)
    return redacted, mapping

def redact(text: str, policy: Optional[RedactPolicy] = None) -> Tuple[str, Dict[str, str]]:
    """
    Redact PII using deterministic layer. Shared pipeline.
    """
    if policy is None:
        policy = RedactPolicy()
    spans = _collect_deterministic_spans(text, policy)
    return _rebuild_from_spans(text, spans, policy.placeholder_prefix)

def reveal(text: str, mapping: Dict[str, str]) -> str:
    """Restore originals from placeholders using the provided local mapping."""
    if not mapping:
        return text
    result = text
    # Replace longest first to avoid prefix issues
    for ph in sorted(mapping.keys(), key=len, reverse=True):
        orig = mapping[ph]
        result = result.replace(ph, orig)
    return result

# ─── Convenience for grimalkin usage ───────────────────────────────────────────

def redact_for_llm(text: str, policy: Optional[RedactPolicy] = None) -> Tuple[str, Dict[str, str]]:
    """Redact before sending to LLM or index."""
    return redact(text, policy)

def make_policy_from_config(cfg: Optional[object] = None) -> RedactPolicy:
    """Bridge for grimalkin config if present (supports GrimalkinConfig or dict)."""
    if cfg is None:
        return RedactPolicy()
    # dict or dataclass-like
    keep = getattr(cfg, "pii_keep", None) or (cfg.get("pii_keep") if isinstance(cfg, dict) else None) or DEFAULT_KEEP
    if isinstance(keep, str):
        keep = {x.strip() for x in re.split(r"[, ]+", keep) if x.strip()}
    elif isinstance(keep, (list, tuple)):
        keep = set(keep)
    mode = getattr(cfg, "pii_redaction", "deterministic") if not isinstance(cfg, dict) else cfg.get("pii_redaction", "deterministic")
    redact_names = mode != "off"
    return RedactPolicy(keep=keep if isinstance(keep, set) else set(keep), redact_names=redact_names)

# ─── Hybrid support (Option 2): deterministic + small open PII model ──────────

_HYBRID_MODEL = None
_HYBRID_AVAILABLE = None

def _try_load_gliner_pii():
    """Lazy load a local GLiNER PII model when explicitly configured."""
    global _HYBRID_MODEL, _HYBRID_AVAILABLE
    if _HYBRID_AVAILABLE is not None:
        return _HYBRID_MODEL
    model_ref = __import__("os").environ.get("GRIM_GLINER_PII_MODEL", "").strip()
    if not model_ref:
        _HYBRID_AVAILABLE = False
        log.info("Hybrid redactor: GRIM_GLINER_PII_MODEL unset; using deterministic only")
        return None
    allow_downloads = __import__("os").environ.get("GRIM_ALLOW_MODEL_DOWNLOADS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not __import__("pathlib").Path(model_ref).exists() and not allow_downloads:
        _HYBRID_AVAILABLE = False
        log.info("Hybrid redactor: remote model refs require GRIM_ALLOW_MODEL_DOWNLOADS=1")
        return None
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_ref)
        _HYBRID_MODEL = model
        _HYBRID_AVAILABLE = True
        log.info("Hybrid redactor: GLiNER-PII small loaded")
    except Exception as e:
        _HYBRID_AVAILABLE = False
        log.info(f"Hybrid redactor: GLiNER not available ({e}); using deterministic only")
    return _HYBRID_MODEL

def _ml_pii_spans(text: str, labels: List[str] = None) -> List[dict]:
    """Use small model to get PII spans if loaded."""
    model = _try_load_gliner_pii()
    if not model:
        return []
    if labels is None:
        labels = ["person", "name", "address", "ssn", "phone", "email", "id", "account", "government id", "passport", "credit card"]
    try:
        entities = model.predict_entities(text, labels)
        # Normalize to our format
        spans = []
        for e in entities:
            spans.append({
                "label": e.get("label", "PII").upper().replace(" ", "_"),
                "text": e.get("text", ""),
                "start": e.get("start", 0),
                "end": e.get("end", 0),
            })
        return spans
    except Exception:
        return []

def redact_hybrid(text: str, policy: Optional[RedactPolicy] = None) -> Tuple[str, Dict[str, str]]:
    """Hybrid: deterministic + ML spans. Uses the SINGLE shared collect + rebuild pipeline."""
    if policy is None:
        policy = RedactPolicy()
    det_spans = _collect_deterministic_spans(text, policy)
    # append ML spans (with their offsets from original text)
    ml = _ml_pii_spans(text)
    for s in ml:
        lbl = s["label"]
        orig = s.get("text", "").strip()
        if not orig or not policy.should_redact(lbl):
            continue
        start = s.get("start", 0)
        end = s.get("end", start + len(orig))
        det_spans.append((start, end, lbl, orig))
    return _rebuild_from_spans(text, det_spans, policy.placeholder_prefix)

# Simple self-test helpers (not tests)
if __name__ == "__main__":
    sample = "My name is John Doe, SSN 123-45-6789, card 4111 1111 1111 1111, email john@doe.com, lives at 123 Main St, phone (555) 123-4567"
    red, mp = redact(sample)
    print("REDACTED (det):", red)
    print("MAPPING:", mp)
    print("REVEALED:", reveal(red, mp))

    # Hybrid demo (may fall back)
    hred, hmp = redact_hybrid(sample)
    print("REDACTED (hybrid):", hred)
