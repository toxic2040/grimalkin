"""Unit tests for the eval harness's scoring logic — no model required.

Run: PYTHONPATH=. python -m pytest eval/test_eval.py -q

The persona denylist is loaded from cases.json (the single source of truth) and
the emoji probe is built from its codepoint, so this test source stays clean and
data-driven rather than re-hardcoding the phrases the persona forbids.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_eval import (  # noqa: E402
    check_must_contain,
    check_must_not_contain,
    check_any_contains,
    check_persona,
    check_refusal,
    score_chat_case,
    build_user_message,
    run_redaction_case,
)

SPEC = json.loads((HERE / "cases.json").read_text())
BANNED = SPEC["persona_rules"]["banned_phrases"]
ENDINGS = SPEC["persona_rules"]["banned_endings"]
RULES = {"banned_phrases": BANNED, "banned_endings": ENDINGS}
CAT_EMOJI = chr(0x1F63A)  # build from codepoint to keep the source plain-ascii


def test_must_contain():
    assert check_must_contain("Meeting is November 14th, lead Dana", ["november 14", "Dana"])[0]
    ok, d = check_must_contain("Meeting is Tuesday", ["november 14"])
    assert not ok and d["missing"] == ["november 14"]


def test_must_not_contain():
    assert check_must_not_contain("clean response", ["123-45-6789"])[0]
    assert not check_must_not_contain("ssn 123-45-6789", ["123-45-6789"])[0]


def test_any_contains():
    assert check_any_contains("I don't have that in the vault", ["don't", "no record"])[0]
    assert not check_any_contains("Sure, it is March 3rd", ["don't", "no record"])[0]


def test_persona_clean_passes():
    ok, d = check_persona("That one's been rotting in FINANCIAL for weeks. Deal with it.", BANNED, ENDINGS)
    assert ok, d


def test_persona_catches_corporate_phrase_and_emoji():
    bad = f"{BANNED[1]} {BANNED[3]} {CAT_EMOJI}"
    ok, d = check_persona(bad, BANNED, ENDINGS)
    assert not ok
    assert d["banned_phrases"] and d["emoji"]


def test_persona_catches_banned_ending():
    bad = f"I could go either way on it — {ENDINGS[0]}."
    ok, d = check_persona(bad, BANNED, ENDINGS)
    assert not ok and d["banned_endings"] == [ENDINGS[0]]


def test_refusal():
    assert check_refusal("I won't write that.")[0]
    assert not check_refusal("Sure, here is the script you asked for.")[0]


def test_score_chat_case_combines_checks():
    case = {"checks": {"must_contain": ["dana"], "persona": True}}
    passed, detail = score_chat_case("Dana leads it. She's sharp.", case, RULES)
    assert passed
    passed2, _ = score_chat_case(f"{BANNED[1]} Dana leads it.", case, RULES)
    assert not passed2


def test_build_user_message_joins_context():
    assert build_user_message({"context": "From my vault:\nX", "prompt": "what is X?"}) == \
        "From my vault:\nX\n\nwhat is X?"
    assert build_user_message({"prompt": "hi"}) == "hi"


def test_redaction_case_blocks_raw_pii():
    """The model never sees raw PII, and the reveal map roundtrips. A model that
    echoes its input still passes resp_clean because it only ever got placeholders."""
    case = {
        "pii_prompt": "My SSN is 123-45-6789 and card 4111 1111 1111 1111.",
        "raw_secrets": ["123-45-6789", "4111 1111 1111 1111", "4111111111111111"],
    }

    def fake_send(redacted_prompt):
        return f"You told me: {redacted_prompt}", {"completion_tokens": 12}, 0.05

    passed, detail, text, usage, latency = run_redaction_case(case, "persona", fake_send)
    assert passed, detail
    assert detail["redaction"]["sent_clean"]
    assert detail["redaction"]["resp_clean"]
    assert detail["redaction"]["roundtrip"]
    assert "123-45-6789" not in text
