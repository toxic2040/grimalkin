"""Tests for the local voice adapter's STT engine selection and errors.

These cover the detection/selection logic only — they monkeypatch
``shutil.which`` so no real STT engine needs to be installed to run them.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grim_voice  # noqa: E402


def _which_factory(present):
    """Build a fake ``shutil.which`` that only resolves names in ``present``."""
    present = set(present)

    def fake(name):
        return f"/usr/bin/{name}" if name in present else None

    return fake


def _patch_which(monkeypatch, present):
    monkeypatch.setattr(grim_voice.shutil, "which", _which_factory(present))


def _clear_stt_env(monkeypatch):
    for var in (
        "GRIM_STT_ENGINE",
        "GRIM_STT_TRANSCRIPT",
        "GRIM_STT_LANGUAGE",
        "GRIM_WHISPER_LANGUAGE",
    ):
        monkeypatch.delenv(var, raising=False)


def _stt_args(audio, out, engine=""):
    return SimpleNamespace(audio=str(audio), out=str(out), engine=engine)


# --- detection / ordering ----------------------------------------------------


def test_status_lists_stt_engines_in_priority_order(monkeypatch):
    _patch_which(monkeypatch, [])
    names = [row["engine"] for row in grim_voice._engine_rows()["stt"]]
    assert names == ["whisper-ctranslate2", "whisper", "vosk-transcriber"]


def test_status_marks_present_engine_available(monkeypatch):
    _patch_which(monkeypatch, ["whisper-ctranslate2"])
    rows = {row["engine"]: row for row in grim_voice._engine_rows()["stt"]}
    assert rows["whisper-ctranslate2"]["available"] is True
    assert rows["whisper"]["available"] is False
    assert rows["whisper-ctranslate2"]["detail"].endswith("whisper-ctranslate2")


# --- auto selection ----------------------------------------------------------


def test_auto_prefers_faster_whisper(monkeypatch):
    _patch_which(monkeypatch, ["whisper-ctranslate2", "whisper", "vosk-transcriber"])
    chosen, available = grim_voice._select_stt_engine("auto")
    assert chosen == "whisper-ctranslate2"
    assert available[0] == "whisper-ctranslate2"


def test_auto_falls_back_to_whisper(monkeypatch):
    _patch_which(monkeypatch, ["whisper", "vosk-transcriber"])
    chosen, _ = grim_voice._select_stt_engine("auto")
    assert chosen == "whisper"


def test_auto_falls_back_to_vosk(monkeypatch):
    _patch_which(monkeypatch, ["vosk-transcriber"])
    chosen, _ = grim_voice._select_stt_engine("auto")
    assert chosen == "vosk-transcriber"


def test_auto_none_when_nothing_installed(monkeypatch):
    _patch_which(monkeypatch, [])
    chosen, available = grim_voice._select_stt_engine("auto")
    assert chosen is None
    assert available == []


def test_empty_request_behaves_like_auto(monkeypatch):
    _patch_which(monkeypatch, ["whisper"])
    assert grim_voice._select_stt_engine("")[0] == "whisper"


# --- forced selection --------------------------------------------------------


def test_explicit_engine_overrides_priority(monkeypatch):
    _patch_which(monkeypatch, ["whisper-ctranslate2", "whisper", "vosk-transcriber"])
    chosen, _ = grim_voice._select_stt_engine("vosk-transcriber")
    assert chosen == "vosk-transcriber"


def test_alias_resolves_to_cli_name(monkeypatch):
    _patch_which(monkeypatch, ["whisper-ctranslate2"])
    assert grim_voice._select_stt_engine("faster-whisper")[0] == "whisper-ctranslate2"
    assert grim_voice._select_stt_engine("vosk")[0] is None  # vosk not present


def test_forced_engine_unavailable_returns_none(monkeypatch):
    _patch_which(monkeypatch, ["vosk-transcriber"])
    chosen, available = grim_voice._select_stt_engine("whisper")
    assert chosen is None
    assert available == ["vosk-transcriber"]


def test_unknown_engine_returns_none(monkeypatch):
    _patch_which(monkeypatch, ["whisper-ctranslate2"])
    chosen, _ = grim_voice._select_stt_engine("banana")
    assert chosen is None


# --- cmd_stt error text and dispatch ----------------------------------------


def test_cmd_stt_auto_error_lists_install_hint(monkeypatch, capsys, tmp_path):
    _clear_stt_env(monkeypatch)
    _patch_which(monkeypatch, [])
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    rc = grim_voice.cmd_stt(_stt_args(audio, tmp_path / "out.txt"))
    assert rc == 69
    err = capsys.readouterr().err
    assert "No local STT engine found" in err
    assert "whisper-ctranslate2" in err
    assert "Available now: none" in err


def test_cmd_stt_unavailable_engine_error_names_known(monkeypatch, capsys, tmp_path):
    _clear_stt_env(monkeypatch)
    _patch_which(monkeypatch, ["vosk-transcriber"])
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    rc = grim_voice.cmd_stt(_stt_args(audio, tmp_path / "out.txt", engine="whisper"))
    assert rc == 69
    err = capsys.readouterr().err
    assert "Requested STT engine 'whisper' is not available" in err
    assert "vosk-transcriber" in err  # both the known list and what's available


def test_cmd_stt_missing_audio_returns_two(monkeypatch, tmp_path):
    _clear_stt_env(monkeypatch)
    _patch_which(monkeypatch, ["whisper-ctranslate2"])
    rc = grim_voice.cmd_stt(_stt_args(tmp_path / "missing.wav", tmp_path / "out.txt"))
    assert rc == 2


def test_cmd_stt_dispatches_to_selected_engine(monkeypatch, tmp_path):
    _clear_stt_env(monkeypatch)
    _patch_which(monkeypatch, ["whisper-ctranslate2", "whisper", "vosk-transcriber"])
    calls = []

    def stub(audio, out_path):
        calls.append("faster-whisper")
        return 0

    monkeypatch.setitem(grim_voice._STT_FUNCS, "whisper-ctranslate2", stub)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    rc = grim_voice.cmd_stt(_stt_args(audio, tmp_path / "out.txt"))
    assert rc == 0
    assert calls == ["faster-whisper"]


def test_cmd_stt_env_passthrough_still_works(monkeypatch, tmp_path):
    _clear_stt_env(monkeypatch)
    monkeypatch.setenv("GRIM_STT_TRANSCRIPT", "the quick brown fox")
    out = tmp_path / "out.txt"
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    rc = grim_voice.cmd_stt(_stt_args(audio, out, engine="env"))
    assert rc == 0
    assert out.read_text(encoding="utf-8").strip() == "the quick brown fox"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
