#!/usr/bin/env python3
"""Local STT/TTS adapter commands for Grimalkin's voice dock."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def _which(name: str) -> str | None:
    return shutil.which(name)


def _engine_rows() -> dict[str, list[dict[str, object]]]:
    piper_model = os.environ.get("GRIM_PIPER_MODEL", "")
    return {
        "stt": [
            {
                "engine": "whisper",
                "available": bool(_which("whisper")),
                "detail": _which("whisper") or "not on PATH",
            },
            {
                "engine": "vosk-transcriber",
                "available": bool(_which("vosk-transcriber")),
                "detail": _which("vosk-transcriber") or "not on PATH",
            },
        ],
        "tts": [
            {
                "engine": "piper",
                "available": bool(_which("piper") and piper_model and Path(piper_model).exists()),
                "detail": piper_model or "set GRIM_PIPER_MODEL",
            },
            {
                "engine": "espeak-ng",
                "available": bool(_which("espeak-ng")),
                "detail": _which("espeak-ng") or "not on PATH",
            },
            {
                "engine": "espeak",
                "available": bool(_which("espeak")),
                "detail": _which("espeak") or "not on PATH",
            },
            {
                "engine": "pico2wave",
                "available": bool(_which("pico2wave")),
                "detail": _which("pico2wave") or "not on PATH",
            },
            {
                "engine": "flite",
                "available": bool(_which("flite")),
                "detail": _which("flite") or "not on PATH",
            },
            {
                "engine": "spd-say",
                "available": bool(_which("spd-say")),
                "detail": _which("spd-say") or "not on PATH",
            },
        ],
    }


def _available(mode: str) -> list[str]:
    return [row["engine"] for row in _engine_rows()[mode] if row["available"]]


def _run(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _write_marker_wav(path: Path, seconds: float = 0.12, sample_rate: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def _read_text(args) -> str:
    if args.text:
        return args.text
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8", errors="replace")
    return sys.stdin.read()


def _write_transcript(out_path: Path, text: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.strip() + "\n", encoding="utf-8")


def _stt_whisper(audio: Path, out_path: Path) -> int:
    model = os.environ.get("GRIM_WHISPER_MODEL", "base")
    language = os.environ.get("GRIM_WHISPER_LANGUAGE", "")
    with tempfile.TemporaryDirectory(prefix="grimalkin_whisper_") as tmp:
        cmd = [
            "whisper",
            str(audio),
            "--model",
            model,
            "--output_format",
            "txt",
            "--output_dir",
            tmp,
        ]
        if language:
            cmd.extend(["--language", language])
        proc = _run(cmd)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or proc.stdout)
            return proc.returncode
        transcript = Path(tmp, audio.stem + ".txt")
        if not transcript.exists():
            sys.stderr.write("whisper did not produce a transcript\n")
            return 1
        _write_transcript(out_path, transcript.read_text(encoding="utf-8", errors="replace"))
    print(f"transcript written via whisper: {out_path}")
    return 0


def _stt_vosk(audio: Path, out_path: Path) -> int:
    cmd = ["vosk-transcriber", "-i", str(audio), "-o", str(out_path)]
    proc = _run(cmd)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"transcript written via vosk-transcriber: {out_path}")
    return 0


def cmd_stt(args) -> int:
    audio = Path(args.audio)
    out_path = Path(args.out)
    if not audio.exists():
        sys.stderr.write(f"audio file not found: {audio}\n")
        return 2

    engine = args.engine or os.environ.get("GRIM_STT_ENGINE", "auto")
    if engine == "env":
        text = os.environ.get("GRIM_STT_TRANSCRIPT", "")
        if not text.strip():
            sys.stderr.write("GRIM_STT_TRANSCRIPT is empty\n")
            return 1
        _write_transcript(out_path, text)
        print(f"transcript written via env: {out_path}")
        return 0

    if engine in ("auto", "whisper") and _which("whisper"):
        return _stt_whisper(audio, out_path)
    if engine in ("auto", "vosk-transcriber") and _which("vosk-transcriber"):
        return _stt_vosk(audio, out_path)

    available = ", ".join(_available("stt")) or "none"
    sys.stderr.write(
        "No local STT engine found. Install whisper or vosk-transcriber, "
        f"or set GRIM_STT_ENGINE explicitly. Available now: {available}\n"
    )
    return 69


def _tts_piper(text: str, out_path: Path) -> int:
    model = os.environ.get("GRIM_PIPER_MODEL", "")
    if not model:
        sys.stderr.write("set GRIM_PIPER_MODEL to a local Piper .onnx voice\n")
        return 2
    cmd = ["piper", "--model", model, "--output_file", str(out_path)]
    speaker = os.environ.get("GRIM_PIPER_SPEAKER", "")
    if speaker:
        cmd.extend(["--speaker", speaker])
    proc = _run(cmd, input_text=text)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"speech rendered via piper: {out_path}")
    return 0


def _tts_espeak(binary: str, text: str, out_path: Path) -> int:
    voice = os.environ.get("GRIM_ESPEAK_VOICE", "")
    speed = os.environ.get("GRIM_ESPEAK_SPEED", "")
    cmd = [binary]
    if voice:
        cmd.extend(["-v", voice])
    if speed:
        cmd.extend(["-s", speed])
    cmd.extend(["-w", str(out_path), text])
    proc = _run(cmd)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"speech rendered via {binary}: {out_path}")
    return 0


def _tts_pico2wave(text: str, out_path: Path) -> int:
    proc = _run(["pico2wave", "-w", str(out_path), text])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"speech rendered via pico2wave: {out_path}")
    return 0


def _tts_flite(text: str, out_path: Path) -> int:
    proc = _run(["flite", "-t", text, "-o", str(out_path)])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    print(f"speech rendered via flite: {out_path}")
    return 0


def _tts_spd_say(text: str, out_path: Path) -> int:
    proc = _run(["spd-say", text])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        return proc.returncode
    _write_marker_wav(out_path)
    print(f"speech spoken via spd-say; marker audio written: {out_path}")
    return 0


def cmd_tts(args) -> int:
    out_path = Path(args.out)
    text = _read_text(args).strip()
    if not text:
        sys.stderr.write("no text provided\n")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = args.engine or os.environ.get("GRIM_TTS_ENGINE", "auto")
    if engine == "marker":
        _write_marker_wav(out_path)
        print(f"marker audio written: {out_path}")
        return 0
    if engine in ("auto", "piper") and _which("piper") and os.environ.get("GRIM_PIPER_MODEL"):
        return _tts_piper(text, out_path)
    if engine in ("auto", "espeak-ng") and _which("espeak-ng"):
        return _tts_espeak("espeak-ng", text, out_path)
    if engine in ("auto", "espeak") and _which("espeak"):
        return _tts_espeak("espeak", text, out_path)
    if engine in ("auto", "pico2wave") and _which("pico2wave"):
        return _tts_pico2wave(text, out_path)
    if engine in ("auto", "flite") and _which("flite"):
        return _tts_flite(text, out_path)
    if engine in ("auto", "spd-say") and _which("spd-say"):
        return _tts_spd_say(text, out_path)

    available = ", ".join(_available("tts")) or "none"
    sys.stderr.write(
        "No local TTS engine found. Install Piper/espeak/flite, or set "
        f"GRIM_TTS_ENGINE explicitly. Available now: {available}\n"
    )
    return 69


def cmd_status(args) -> int:
    rows = _engine_rows()
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    for mode, engines in rows.items():
        print(mode.upper())
        for row in engines:
            marker = "yes" if row["available"] else "no"
            print(f"  {row['engine']}: {marker} ({row['detail']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grimalkin local voice adapters")
    sub = parser.add_subparsers(dest="cmd", required=True)

    stt = sub.add_parser("stt", help="transcribe a local audio file")
    stt.add_argument("--audio", required=True)
    stt.add_argument("--out", required=True)
    stt.add_argument("--engine", default="")
    stt.set_defaults(func=cmd_stt)

    tts = sub.add_parser("tts", help="render or speak local text")
    tts.add_argument("--out", required=True)
    tts.add_argument("--text", default="")
    tts.add_argument("--text-file", default="")
    tts.add_argument("--engine", default="")
    tts.set_defaults(func=cmd_tts)

    status = sub.add_parser("status", help="show local engine availability")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
