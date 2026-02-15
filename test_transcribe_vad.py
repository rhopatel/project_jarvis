#!/usr/bin/env python3
"""
Transcribe all vad_test/*.wav files using the same whisper setup as Jarvis.
Run after test_vad_silence.py to verify transcription quality across silence limits.
"""

import os
import re
import sys
import time
import subprocess
import wave

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAD_TEST_DIR = os.path.join(SCRIPT_DIR, "vad_test")
WHISPER_DIR = os.path.join(SCRIPT_DIR, "../whisper.cpp")
WHISPER_BIN = os.path.abspath(os.path.join(WHISPER_DIR, "build/bin/whisper-cli"))
WHISPER_MODEL = os.path.abspath(os.path.join(WHISPER_DIR, "models/ggml-tiny.en-q4_0.bin"))
if not os.path.exists(WHISPER_MODEL):
    WHISPER_MODEL = os.path.abspath(os.path.join(WHISPER_DIR, "models/ggml-tiny.en.bin"))
WHISPER_PROMPT = "Jarvis Jarvis light on lights off turn on turn off switch"

_whisper_env = os.environ.copy()
for _d in ["../whisper.cpp/build/src", "../whisper.cpp/build/ggml/src"]:
    _abs = os.path.abspath(os.path.join(SCRIPT_DIR, _d))
    if os.path.isdir(_abs):
        _whisper_env["LD_LIBRARY_PATH"] = _abs + ":" + _whisper_env.get("LD_LIBRARY_PATH", "")


def fix_jarvis_mishearing(text: str) -> str:
    if not text or len(text) < 5:
        return text
    lowered = text.lower()
    for mis in (r"^driver'?s?\s+", r"^drivers\s+", r"^driver\s+is\s+"):
        if re.search(mis, lowered):
            return "Jarvis " + re.sub(mis, "", text, flags=re.I).strip()
    return text


def transcribe(path: str):
    """Returns (text, duration_sec)."""
    t0 = time.time()
    nthreads = min(4, (os.cpu_count() or 4))
    cmd = [
        WHISPER_BIN, "-m", WHISPER_MODEL, "-f", path,
        "-t", str(nthreads), "-bs", "1", "-bo", "1", "-ml", "32",
        "--no-timestamps", "-l", "en",
    ]
    if WHISPER_PROMPT:
        cmd.extend(["--prompt", WHISPER_PROMPT])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=_whisper_env, cwd=SCRIPT_DIR)
    except subprocess.TimeoutExpired:
        return ("[timeout]", 0)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return (f"[error: {r.stderr[:100]}]", elapsed)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", r.stdout).strip()
    text = fix_jarvis_mishearing(text)
    return (text, elapsed)


def main():
    if not os.path.isdir(VAD_TEST_DIR):
        print("Run test_vad_silence.py first to create vad_test/")
        return 1
    wavs = sorted(f for f in os.listdir(VAD_TEST_DIR) if f.endswith(".wav"))
    if not wavs:
        print("No .wav files in vad_test/")
        return 1

    print("Transcribing vad_test/*.wav (same setup as Jarvis):\n")
    for f in wavs:
        path = os.path.join(VAD_TEST_DIR, f)
        with wave.open(path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        text, elapsed = transcribe(path)
        print(f"  {f:25} ({dur:.1f}s audio) → \"{text}\"  [{elapsed:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
