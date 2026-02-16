#!/usr/bin/env python3
"""
Jarvis Voice Assistant
Wake word activated assistant using whisper.cpp, Ollama, and Piper TTS.
"""

import subprocess
import os
import struct
import math
import time
import sys
import threading
import wave
import collections
import requests
import json
import signal
import atexit
import re
import tempfile
from typing import Optional


# ── Configuration ─────────────────────────────────────────────

DEBUG = 0  # 0 = clean (YOU/JARVIS only), 1 = verbose. Override: --debug or JARVIS_DEBUG=1
VAD_DEBUG_SAVE     = 0            # 1 = save captured WAV to vad_captures/ (verify full command), 0 = off
WAKE_WORD          = "Jarvis"
SIMILARITY_THRESH = 0.65  # Wake word match; raise to cut false positives (georgis/jorgas fixed in _JARVIS_LIKE)
THRESHOLD          = 700         # RMS threshold for voice activity detection (calibrated)
SILENCE_LIMIT = 1.1         # seconds of silence to end utterance
SILENCE_LIMIT_CONV = 1.0    # Shorter in conversation so "light on" etc. stop sooner → use tiny model
PRE_AUDIO          = 0.4         # seconds of pre-roll (catch start of "Jarvis")
CONVERSATION_TIMEOUT = 30        # seconds of no speech to exit conversation
MIN_UTTERANCE_DURATION = 0.25     # seconds: ignore captures shorter than this (reduces false triggers like "you")
# LLM: False = smart-home only (lights, desk, floor), no chat. True = full assistant.
LLM_ENABLED        = False
OLLAMA_IP          = "10.0.0.224"
OLLAMA_MODEL       = "gemma3:12b"      # Best overall assistant
RATE               = 16000

# TP-Link Kasa - direct local control (no cloud)
# Newer bulbs may need TP-Link account credentials - leave empty for older devices
KASA_USERNAME      = ""  # e.g. "you@email.com" for Tapo/newer Kasa
KASA_PASSWORD      = ""  # TP-Link cloud password

# Device aliases - must match names in Kasa app.
KASA_DEVICES = ("floor", "desk")  # Floor (bedside), desk (desk lamp)
CHUNK              = 1024

# System prompt: load from system_prompt.txt (gitignored) or use default
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SYSTEM_PROMPT_PATH = os.path.join(_SCRIPT_DIR, "system_prompt.txt")
_DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful voice assistant. "
    "Keep responses short and concise unless asked to elaborate. "
    "You speak out loud: reply in plain conversational language only. "
    "Never use markdown (no asterisks, bullets, bold, code blocks). "
    "Respond the way a human would when talking—no special formatting."
)


def _load_system_prompt() -> str:
    """Load system prompt from system_prompt.txt if it exists, else use default."""
    try:
        if os.path.isfile(_SYSTEM_PROMPT_PATH):
            with open(_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    return text
    except Exception:
        pass
    return _DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = _load_system_prompt()


# ── Paths (relative to script, works regardless of cwd) ───────

_WHISPER_BASE = os.path.join(_SCRIPT_DIR, "..", "whisper.cpp")
WHISPER_BIN = os.path.join(_WHISPER_BASE, "build", "bin", "whisper-cli")
WHISPER_MODEL_FAST = os.path.join(_WHISPER_BASE, "models", "ggml-tiny.en-q4_0.bin")      # <2s: quick commands
WHISPER_MODEL_ACCURATE = os.path.join(_WHISPER_BASE, "models", "ggml-base.en-q4_0.bin")  # >=2s: conversation
WHISPER_SHORT_THRESH = 3.0  # seconds: use tiny below this (quicker for "light on"), base above
# Short clips: bias toward quick commands (lights, floor, desk, on, off, colors)
_dev_phrase = " ".join(f"turn {d} on turn {d} off {d} on {d} off" for d in KASA_DEVICES)
WHISPER_PROMPT_QUICK = (
    "Jarvis Jarvis Jarvis light on lights off turn on turn off "
    f"{_dev_phrase} {' '.join(KASA_DEVICES)} "
    "red orange yellow green blue indigo violet purple"
)
WHISPER_PROMPT_CONV = ""  # No biasing for long clips


def _detect_mic_device() -> str:
    """Auto-detect first available ALSA capture device."""
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return "default"
        for line in result.stdout.splitlines():
            match = re.search(r"card (\d+):.*device (\d+):", line)
            if match:
                return f"plughw:{match.group(1)},{match.group(2)}"
        return "default"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "default"


MIC_DEVICE = _detect_mic_device()
RAM_WAV = "/dev/shm/jarvis_vad.wav"

PIPER_VOICE     = "en_US-ryan-high"  # Higher quality; fallback to medium if high missing
PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")


# ── Whisper subprocess environment (built once) ──────────────

_whisper_env = os.environ.copy()
for _d in ["build/src", "build/ggml/src"]:
    _abs = os.path.join(_WHISPER_BASE, _d)
    if os.path.isdir(_abs):
        _whisper_env["LD_LIBRARY_PATH"] = (
            _abs + ":" + _whisper_env.get("LD_LIBRARY_PATH", "")
        )


# ── Colours ───────────────────────────────────────────────────

BLUE    = "\033[1;34m"
GREEN   = "\033[1;32m"
YELLOW  = "\033[1;33m"
CYAN    = "\033[1;36m"
MAGENTA = "\033[1;35m"
RESET   = "\033[0m"


# ── Globals ───────────────────────────────────────────────────

_tts_voice: object = None
_mic: subprocess.Popen = None


def dbg(*args, **kwargs):
    """Print only when DEBUG is on."""
    if DEBUG:
        print(*args, **kwargs)


# ── Audio helpers ─────────────────────────────────────────────

def get_rms(data: bytes) -> float:
    """RMS of a buffer of signed 16-bit LE samples."""
    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"<{n}h", data)
    return math.sqrt(sum(s * s for s in shorts) / n)


def save_wav(chunks: list, path: str):
    """Write raw PCM chunks to a 16 kHz mono WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(chunks))


def play_wav(path: str):
    """Play a WAV via paplay (PipeWire/Pulse); fall back to aplay (ALSA) on failure."""
    r = subprocess.run(["paplay", path], capture_output=True, timeout=120)
    if r.returncode != 0:
        subprocess.run(["aplay", "-q", path], stderr=subprocess.DEVNULL, timeout=120)


# ── Whisper transcription ────────────────────────────────────

def _trim_trailing_silence(path: str, thresh: float = 1500) -> str:
    """Trim trailing silence from WAV. Higher thresh = trim more (works with ambient noise)."""
    try:
        with wave.open(path, "rb") as wf:
            nch, sw, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        n = len(frames) // (sw * nch)
        if n < 2:
            return path
        chunk_sz = int(rate * 0.02)  # 20ms chunks
        trim_end = n
        quiet_chunks = 0
        need_quiet = int(0.15 * rate / chunk_sz)  # 150ms quiet = cut (trim faster)
        for i in range(n - chunk_sz, 0, -chunk_sz):
            chunk = frames[i * sw * nch : (i + chunk_sz) * sw * nch]
            if len(chunk) < chunk_sz * sw * nch:
                break
            rms = get_rms(chunk)
            if rms < thresh:
                quiet_chunks += 1
                if quiet_chunks >= need_quiet:
                    trim_end = i + chunk_sz
                    break
            else:
                quiet_chunks = 0
        min_samples = int(rate * 0.5)  # Keep at least 0.5s
        if trim_end < n * 0.9 and trim_end >= min_samples:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(nch)
                wf.setsampwidth(sw)
                wf.setframerate(rate)
                wf.writeframes(frames[: trim_end * sw * nch])
    except Exception:
        pass
    return path


def _get_wav_duration(path: str) -> float:
    """Return duration in seconds."""
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def _warm_whisper():
    """Preload both Whisper models so first transcription is fast."""
    for secs in (0.5, 4.0):  # warm tiny (short) and base (long)
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(b"\x00\x00" * int(RATE * secs))
            transcribe(path)
        except Exception:
            pass
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


def transcribe(path: str) -> tuple[str, float]:
    """Run whisper.cpp on a WAV file. Tiny (<3s) for speed, base for longer conversation."""
    path = _trim_trailing_silence(path)
    dur = _get_wav_duration(path)
    model = WHISPER_MODEL_FAST if dur < WHISPER_SHORT_THRESH else WHISPER_MODEL_ACCURATE
    t0 = time.time()
    nthreads = min(4, (os.cpu_count() or 4))
    prompt = WHISPER_PROMPT_QUICK if dur < WHISPER_SHORT_THRESH else WHISPER_PROMPT_CONV
    cmd = [
        WHISPER_BIN, "-m", model, "-f", path,
        "-t", str(nthreads), "-bs", "1", "-bo", "1",
        "-ml", "32" if dur < WHISPER_SHORT_THRESH else "64",
        "--no-timestamps", "-l", "en",
    ]
    if prompt:
        cmd.extend(["--prompt", prompt])
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30, env=_whisper_env,
        )
    except subprocess.TimeoutExpired:
        return ("", 0.0)

    dbg(f"{BLUE}[PERF]{RESET} Transcription ({dur:.1f}s, {'tiny' if dur < WHISPER_SHORT_THRESH else 'base'}): {time.time() - t0:.2f}s")

    if r.returncode != 0:
        dbg(f"{BLUE}[DEBUG]{RESET} Whisper stderr: {r.stderr[:200]}")
        return ("", dur)

    # Strip whisper artefacts: [blank_audio], (silence), etc.
    text = re.sub(r"\[.*?\]|\(.*?\)", "", r.stdout).strip()
    # Fix wake-word mishearings so we detect "Jarvis" (driver's, computer, etc.)
    text = _fix_jarvis_mishearing(text)
    text = _fix_command_mishearings(text)
    return (text, dur)


# Words that sound like "Jarvis" - map to Jarvis for wake-word detection
_JARVIS_LIKE = (
    "travis", "darvis", "charvis", "charvus", "charis", "garvis", "yarvis", "jarvin", "jarvus",
    "jarvis", "driver's", "drivers", "computer", "you",
    "draw", "drop", "drive", "rose", "garbage", "georgis", "jorgas",  # Whisper mishearings
)

# Whisper mishearings of command phrases (apply after wake-word fix)
_CMD_MISHEARINGS = {
    "sternal light": "turn off light",
    "sternal lights": "turn off lights",
    "strong flight": "turn on light",
    "strong flights": "turn on lights",
    "desk loop": "desk off",
    "floor loop": "floor off",
    "does red": "desk red",
    "does red end": "desk red",
    "des red": "desk red",
    "disc red": "desk red",
    "des red": "desk red",
    "a slight line": "lights on",
    "a slight lines": "lights on",
    "just for all": "lights on",
    "just for all lights": "lights on",
}


def _fix_command_mishearings(text: str) -> str:
    """Fix Whisper mishearings of command phrases (e.g. sternal light -> turn off light)."""
    t = text.lower()
    for bad, good in _CMD_MISHEARINGS.items():
        if bad in t:
            t = t.replace(bad, good)
    return t


def _fix_jarvis_mishearing(text: str) -> str:
    """Fix wake-word mishearings: travis, charvis, you, draw, etc. -> Jarvis."""
    if not text or len(text) < 3:  # allow "you" (3 chars)
        return text
    words = text.split()
    if not words:
        return text
    first = words[0].lower().rstrip(".,!?;:")
    if first in _JARVIS_LIKE or similarity(first, WAKE_WORD) >= 0.6:
        return "Jarvis " + " ".join(words[1:]).strip()
    return text


# ── Text helpers ──────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """Levenshtein-based similarity in [0, 1]."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return 1.0 - prev[-1] / max(len(a), len(b))


def extract_wake_command(text: str) -> Optional[str]:
    """If text contains the wake word, return the command (wake word removed).
    Returns None when wake word is absent."""
    words = text.lower().split()
    for i, w in enumerate(words):
        if similarity(w.rstrip(".,!?;:"), WAKE_WORD) >= SIMILARITY_THRESH:
            cmd = " ".join(words[:i] + words[i + 1:]).strip()
            dbg(f"{BLUE}[DEBUG]{RESET} Heard: '{text}' -> command: '{cmd}'")
            return cmd if len(cmd) >= 2 else ""
    return None


def strip_wake_word(text: str) -> str:
    """Remove wake word from text (for conversation mode, in case user says it
    out of habit)."""
    words = text.lower().split()
    return " ".join(
        w for w in words
        if similarity(w.rstrip(".,!?;:"), WAKE_WORD) < SIMILARITY_THRESH
    ).strip()


# ── TTS ───────────────────────────────────────────────────────

def load_tts():
    """Load the Piper voice model once at startup."""
    global _tts_voice
    import warnings

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

    from pathlib import Path

    voice_dir = Path(PIPER_VOICE_DIR)
    voice_dir.mkdir(parents=True, exist_ok=True)
    model_path  = voice_dir / f"{PIPER_VOICE}.onnx"
    config_path = voice_dir / f"{PIPER_VOICE}.onnx.json"

    # Suppress C++ ONNX Runtime warnings during import
    saved_fd = os.dup(2)
    devnull  = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        from piper.voice import PiperVoice
        from piper.download_voices import download_voice
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)

    if not model_path.exists() or not config_path.exists():
        print(f"Downloading Piper voice ({PIPER_VOICE})...")
        download_voice(PIPER_VOICE, voice_dir)

    _tts_voice = PiperVoice.load(str(model_path), use_cuda=False)
    dbg(f"{GREEN}[OK]{RESET} TTS loaded")


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown and problematic Unicode before TTS. Piper can't speak asterisks or IPA."""
    if not text:
        return ""
    # Remove markdown bold/italic (**, *), bullets (-, *, •), leading " * "
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)   # **bold** -> bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)      # *italic* -> italic
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.M)  # bullet lines
    text = re.sub(r"\n\s*[-*•]\s*", ". ", text)    # convert bullets to sentence breaks
    text = re.sub(r"\s*\*\s*", " ", text)          # lone asterisks
    # Remove combining/IPA chars Piper can't pronounce (e.g. U+0329)
    text = "".join(c for c in text if not (0x0300 <= ord(c) <= 0x036F))
    return text.strip()


def _split_sentences(text: str) -> list:
    """Split text into sentences. Play each as soon as synthesized for lower latency."""
    text = _sanitize_for_tts(text.strip())
    if not text:
        return []
    # Split on sentence boundaries (. ! ?) - keep delimiters
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return [text]
    return sentences


def speak(text: str):
    """Synthesise text with Piper and play via PulseAudio.
    Only speaks when LLM is enabled and Ollama/GPU is reachable.
    """
    if not text or not text.strip():
        return
    if not LLM_ENABLED or not _ollama_connected:
        return

    from piper.config import SynthesisConfig

    cfg = SynthesisConfig(speaker_id=None, volume=0.25)
    sentences = _split_sentences(text)

    for sent in sentences:
        if not sent.strip():
            continue
        raw_path = tempfile.mktemp(suffix=".wav", prefix="jarvis_tts_")
        out_path = tempfile.mktemp(suffix=".wav", prefix="jarvis_play_")
        try:
            with wave.open(raw_path, "wb") as wf:
                _tts_voice.synthesize_wav(sent, wf, syn_config=cfg)
            with wave.open(raw_path, "rb") as rf:
                params = rf.getparams()
                audio = rf.readframes(rf.getnframes())
            dup_ms = 150  # USB speaker cuts start; prepend dup so each sentence is audible
            if dup_ms > 0:
                dup_bytes = int(params.framerate * dup_ms / 1000) * params.sampwidth * params.nchannels
                dup = audio[:min(dup_bytes, len(audio))]
                with wave.open(out_path, "wb") as wf:
                    wf.setparams(params)
                    wf.writeframes(dup + audio)
            else:
                with wave.open(out_path, "wb") as wf:
                    wf.setparams(params)
                    wf.writeframes(audio)
            play_wav(out_path)
        except OSError:
            pass
        finally:
            for p in (raw_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ── Microphone / VAD ─────────────────────────────────────────

def _save_vad_debug(wav_path: str, text: str):
    """Save captured WAV for verification - listen to check we got the full command."""
    try:
        import shutil
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vad_captures")
        os.makedirs(d, exist_ok=True)
        with wave.open(wav_path, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        safe = re.sub(r"[^\w\s]", "", text)[:40].replace(" ", "_") or "empty"
        dst = os.path.join(d, f"{int(time.time())}_{dur:.1f}s_{safe}.wav")
        shutil.copy2(wav_path, dst)
        print(f"  [VAD] saved {dur:.1f}s → {dst}")
    except Exception:
        pass


def wait_for_silence(settle: float = 0.5):
    """Read and discard mic data until the room is quiet for *settle* seconds."""
    quiet_since = None
    while True:
        chunk = _mic.stdout.read(CHUNK * 2)
        if not chunk:
            return
        if get_rms(chunk) < THRESHOLD:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= settle:
                return
        else:
            quiet_since = None


def wait_for_speech(timeout: float = None, silence_limit: float = SILENCE_LIMIT) -> list:
    """Block until speech is detected, record until silence returns.

    Returns the list of audio chunks, or [] on timeout / stream end.
    silence_limit: seconds of silence to end utterance (shorter = faster cut-off).
    """
    pre = collections.deque(maxlen=int(RATE / CHUNK * PRE_AUDIO))
    chunks = []
    speaking = False
    silence_start = None
    wait_start = time.time()

    while True:
        chunk = _mic.stdout.read(CHUNK * 2)
        if not chunk:
            return []

        rms = get_rms(chunk)
        pre.append(chunk)

        if rms >= THRESHOLD:
            if not speaking:
                speaking = True
                chunks = list(pre)
            else:
                chunks.append(chunk)
            silence_start = None
        elif speaking:
            chunks.append(chunk)
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_limit:
                return chunks
        else:
            # Still waiting for speech
            if timeout and time.time() - wait_start >= timeout:
                return []

    return chunks


# ── Ollama ────────────────────────────────────────────────────

# ── TP-Link Kasa (local) ───────────────────────────────────────

_kasa_devices: list = []  # [(ip, alias, model), ...]
_kasa_creds = None


def _kasa_discover():
    """Discover and cache Kasa device IPs. Uses fresh connection per device when controlling."""
    global _kasa_devices, _kasa_creds
    import asyncio
    from kasa import Discover, Credentials

    def _load_creds():
        global _kasa_creds
        user = KASA_USERNAME or os.environ.get("KASA_USERNAME", "").strip()
        pw = KASA_PASSWORD or os.environ.get("KASA_PASSWORD", "").strip()
        if not user or not pw:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line.strip())
                        if m:
                            k, v = m.group(1), m.group(2).strip().strip("'\"")
                            if k == "KASA_USERNAME":
                                user = v
                            elif k == "KASA_PASSWORD":
                                pw = v
        _kasa_creds = Credentials(user, pw) if user and pw else None

    _load_creds()

    async def _discover():
        try:
            found = await Discover.discover(credentials=_kasa_creds)
            devices = []
            for ip, dev in found.items():
                await dev.update()
                devices.append((str(ip), dev.alias or str(ip), dev.model))
            devices.sort(key=lambda x: (0 if "130" in str(x[2]) else 1, x[1]))
            return devices
        except Exception as e:
            dbg(f"{BLUE}[KASA]{RESET} Discover error: {e}")
            return []

    _kasa_devices = asyncio.run(_discover())


def _kasa_get_device_by_alias(alias: str) -> Optional[tuple]:
    """Return (ip, alias, model) for device whose alias matches (case-insensitive), or None."""
    global _kasa_devices
    if not _kasa_devices:
        _kasa_discover()
    want = alias.lower()
    for ip, a, model in _kasa_devices:
        if a and a.lower() == want:
            return (ip, a, model)
    return None


# ROYGBIV: hue 0-360 (HSV)
_KASA_COLORS = {
    "red": 0, "orange": 30, "yellow": 60, "green": 120,
    "blue": 240, "indigo": 260, "violet": 280, "purple": 280,
}

# Percent words -> 0-100
_KASA_BRIGHTNESS_WORDS = {
    "full": 100, "max": 100, "bright": 100, "all": 100,
    "three quarters": 75, "three quarter": 75,
    "half": 50, "mid": 50, "medium": 50,
    "quarter": 25, "low": 25, "dim": 25,
}


def control_kasa_device(alias: str, brightness: Optional[int] = None,
                       color: Optional[str] = None, on: Optional[bool] = None) -> bool:
    """Control a single Kasa device by alias. Main=brightness only, Desk=color or brightness."""
    global _kasa_devices, _kasa_creds
    entry = _kasa_get_device_by_alias(alias)
    if not entry:
        return False
    ip, _, model = entry

    import asyncio
    from kasa import Discover, Module

    async def _control():
        try:
            dev = await Discover.discover_single(ip, credentials=_kasa_creds)
            await dev.update()
            if Module.Light not in dev.modules:
                return False
            light = dev.modules[Module.Light]
            await dev.turn_on()
            if on is False:
                await dev.turn_off()
                return True
            if color is not None and light.has_feature("hsv"):
                hue = _KASA_COLORS.get(color.lower(), 0)
                bri = brightness if brightness is not None else 100
                await light.set_hsv(hue, 100, bri)
            elif brightness is not None:
                if light.has_feature("brightness"):
                    await light.set_brightness(brightness)
                elif light.has_feature("hsv"):
                    await light.set_hsv(0, 0, brightness)
            await dev.update()
            return True
        except Exception as e:
            dbg(f"{BLUE}[KASA]{RESET} {alias}: {e}")
            return False

    return asyncio.run(_control())


def control_kasa_lights(on: bool) -> bool:
    """Control each device with a fresh connection. LB130 uses set_hsv for reliability."""
    global _kasa_devices, _kasa_creds
    if not _kasa_devices:
        _kasa_discover()
    if not _kasa_devices:
        return False
    import asyncio
    from kasa import Discover, Module

    async def _turn_on_dev(dev):
        await dev.turn_on()
        try:
            if Module.Light in dev.modules:
                light = dev.modules[Module.Light]
                if light.has_feature("hsv"):
                    await light.set_hsv(0, 0, 100)
                elif light.has_feature("brightness"):
                    await light.set_brightness(100)
        except Exception:
            pass
        await dev.update()

    async def _control_one(ip: str, alias: str):
        try:
            dev = await Discover.discover_single(ip, credentials=_kasa_creds)
            await dev.update()
            if on:
                await _turn_on_dev(dev)
            else:
                await dev.turn_off()
            return True
        except Exception as e:
            dbg(f"{BLUE}[KASA]{RESET} {alias}: {e}")
            return False

    async def _control():
        results = await asyncio.gather(
            *(_control_one(ip, alias) for ip, alias, _ in _kasa_devices)
        )
        return any(results)

    return asyncio.run(_control())


# ── Command routing (layered) ─────────────────────────────────────────
# Layer 3: Local actions (Kasa, etc.) - runs on Pi, no PC needed
# Layer 4: Ollama LLM - only if Layer 3 does not match


# Exclude negations: "don't turn off", "don't turn on"
_LIGHT_NEGATION = re.compile(r"\b(?:don't|dont|do not|never)\s+(?:turn|switch)", re.I)


def match_quick_command(text: str) -> Optional[tuple]:
    """Keyword match: both lights ON/OFF, floor ON/OFF, desk ON/OFF, desk color.
    Returns (name, reaction, params) or None."""
    cmd = strip_wake_word(text).lower().strip() or text.lower().strip()
    if _LIGHT_NEGATION.search(cmd) or len(cmd) < 2:
        return None
    cmd_words = set(re.findall(r"\w+", cmd))

    # Desk color (desk only); "desk" alone -> desk on
    if "desk" in cmd_words:
        for color in _KASA_COLORS:
            if color in cmd_words or color in cmd:
                return ("KASA_COLOR", "kasa_device", {"device": "desk", "color": color})
        if "on" in cmd_words:
            return ("KASA_DEVICE_ON", "kasa_device", {"device": "desk", "on": True})
        if "off" in cmd_words:
            return ("KASA_DEVICE_OFF", "kasa_device", {"device": "desk", "on": False})
        # "desk" alone -> desk on (common shorthand)
        return ("KASA_DEVICE_ON", "kasa_device", {"device": "desk", "on": True})

    # Floor
    if "floor" in cmd_words:
        if "on" in cmd_words:
            return ("KASA_DEVICE_ON", "kasa_device", {"device": "floor", "on": True})
        if "off" in cmd_words:
            return ("KASA_DEVICE_OFF", "kasa_device", {"device": "floor", "on": False})

    # Both lights (light/lights or bare turn on/off)
    has_light = "light" in cmd_words or "lights" in cmd_words
    has_turn = "turn" in cmd_words
    if has_light or has_turn:
        if "on" in cmd_words:
            return ("LIGHT_ON", "kasa_on", {})
        if "off" in cmd_words:
            return ("LIGHT_OFF", "kasa_off", {})

    return None


def _parse_brightness(value: str) -> Optional[int]:
    """Parse '100%', '50 percent', 'full', 'half', etc. -> 0-100 or None."""
    value = value.strip().lower().replace(" percent", "%")
    if value in _KASA_BRIGHTNESS_WORDS:
        return _KASA_BRIGHTNESS_WORDS[value]
    m = re.match(r"(\d+)\s*%?", value)
    if m:
        n = int(m.group(1))
        return max(0, min(100, n))
    return None


def _run_quick_reaction(reaction: str, params: Optional[dict] = None) -> bool:
    """Dispatch quick command reaction. Returns True on success."""
    params = params or {}
    if reaction == "kasa_on":
        return control_kasa_lights(on=True)
    if reaction == "kasa_off":
        return control_kasa_lights(on=False)
    if reaction == "kasa_device":
        device = params.get("device")
        brightness = params.get("brightness")
        color = params.get("color")
        on = params.get("on")
        if device:
            if on is not None:
                return control_kasa_device(device, on=on)
            return control_kasa_device(device, brightness=brightness, color=color)
    return False


# Conversation history for multi-turn context (max 10 turns)
_chat_history: list = []
_CHAT_HISTORY_MAX = 10

# Ollama connection: kept updated by background keepalive; quick commands work without it
_ollama_connected = False
_ollama_stop_event = threading.Event()
OLLAMA_KEEPALIVE_INTERVAL = 15


def _check_ollama() -> bool:
    """Verify Ollama is reachable and our model exists. Returns True on success."""
    try:
        r = requests.get(f"http://{OLLAMA_IP}:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        want = OLLAMA_MODEL.split(":")[0]
        return any(m.get("name", "").startswith(want) for m in models)
    except requests.exceptions.RequestException:
        return False


def _warm_ollama_model() -> None:
    """Send a minimal request to load the model and prime it with the system prompt."""
    try:
        url = f"http://{OLLAMA_IP}:11434/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Hi"},
            ],
            "stream": True,
        }
        with requests.post(url, json=payload, stream=True, timeout=30) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    if json.loads(line).get("done"):
                        break
                except json.JSONDecodeError:
                    pass
    except requests.exceptions.RequestException:
        pass


def _ollama_keepalive_loop() -> None:
    """Background thread: periodically check Ollama, warm model when first connected."""
    global _ollama_connected
    last_connected = False
    while not _ollama_stop_event.wait(OLLAMA_KEEPALIVE_INTERVAL):
        connected = _check_ollama()
        _ollama_connected = connected
        if connected and not last_connected:
            dbg(f"{GREEN}[OLLAMA]{RESET} Connected, warming model...")
            _warm_ollama_model()
        last_connected = connected


def query_ollama(prompt: str, history: Optional[list] = None) -> str:
    """Send prompt to Ollama chat API with conversation history. Returns assistant response or '' on failure."""
    global _ollama_connected
    url = f"http://{OLLAMA_IP}:11434/api/chat"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": True}
    text = ""
    timeout = 60 if _ollama_connected else 5
    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            _ollama_connected = True
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if isinstance(chunk, str):
                        text += chunk
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException as e:
        _ollama_connected = False
        print(f"{BLUE}[ERROR]{RESET} Ollama: {e}")
    return text


def ask_jarvis(text: str, dur: float = 0) -> bool:
    """Run quick command if matched (lights, floor, desk), else query LLM.
    Returns True if we gave a verbal LLM reply (enter conversation mode)."""
    global _chat_history
    m = match_quick_command(text)
    if m is not None:
        name, reaction, params = m[0], m[1], m[2] if len(m) > 2 else {}
        print(f"{CYAN}COMMAND:{RESET} {name}")
        ok = _run_quick_reaction(reaction, params)
        if not ok:
            err = "No Kasa devices found. Check they're on your network."
            print(f"{BLUE}[ERROR]{RESET} {err}\n")
            speak(err)
        return False

    # No quick match → LLM if enabled, else silence
    if not LLM_ENABLED:
        return False
    print(f"{CYAN}YOU:{RESET} {text}")
    dbg(f"{YELLOW}[THINKING]{RESET}...")
    response = query_ollama(text, history=_chat_history)
    if response:
        _chat_history.append({"role": "user", "content": text})
        _chat_history.append({"role": "assistant", "content": response})
        if len(_chat_history) > _CHAT_HISTORY_MAX * 2:
            _chat_history = _chat_history[-_CHAT_HISTORY_MAX * 2 :]
        print(f"{GREEN}JARVIS:{RESET} {response}\n")
        speak(response)
        return True  # Verbal reply → enter conversation mode
    # No quick command, Ollama offline → silence
    return False


# ── Conversation mode ─────────────────────────────────────────

def conversation():
    """After wake word, keep talking without repeating 'Jarvis'.
    Exits after CONVERSATION_TIMEOUT seconds of silence.
    """
    while True:
        wait_for_silence()
        pi_led_set_ready(True)   # Green = listening for next utterance

        dbg(f"{YELLOW}[CONVERSATION]{RESET} Listening...")

        chunks = wait_for_speech(timeout=CONVERSATION_TIMEOUT, silence_limit=SILENCE_LIMIT_CONV)
        if not chunks:
            dbg(f"{BLUE}[INFO]{RESET} Conversation timeout.")
            _chat_history.clear()  # Fresh context for next session
            return
        pi_led_set_ready(False)   # Red = thinking/speaking

        save_wav(chunks, RAM_WAV)
        text, dur = transcribe(RAM_WAV)
        if VAD_DEBUG_SAVE:
            _save_vad_debug(RAM_WAV, text)
        try:
            os.remove(RAM_WAV)
        except OSError:
            pass
        if not text or len(text) < 2:
            continue
        # Skip very short captures unless they match a quick command (reduces false triggers)
        short_match = match_quick_command(text)
        if dur < MIN_UTTERANCE_DURATION and short_match is None:
            continue

        ask_jarvis(text, dur=dur)


# ── Raspberry Pi LED status ─────────────────────────────────────
# Green = ready to listen. Red = thinking/speaking. Requires root.

_PI_LED_GREEN = None   # ACT/led0 - ready state
_PI_LED_RED = None     # PWR/led1 - busy state
_PI_LED_SAVED = {}     # {led_path: original_trigger}
_PI_LED_OK = False     # True if both LEDs found and we have permission


def _pi_led_init() -> bool:
    """Discover green (ACT) and red (PWR) LEDs. Save triggers. Returns True if both found.
    Only runs when effective UID is 0 (root) - i.e. when started via systemd service."""
    global _PI_LED_GREEN, _PI_LED_RED, _PI_LED_SAVED
    if os.geteuid() != 0:
        return False
    base = "/sys/class/leds"
    if not os.path.isdir(base):
        return False
    green_candidates = ["act", "led0", "green"]
    red_candidates = ["pwr", "power", "led1", "red"]
    for name in os.listdir(base):
        led = os.path.join(base, name)
        trigger_path = os.path.join(led, "trigger")
        brightness_path = os.path.join(led, "brightness")
        if not os.path.isfile(trigger_path) or not os.path.isfile(brightness_path):
            continue
        lower = name.lower()
        try:
            with open(trigger_path) as f:
                raw = f.read()
            m = re.search(r"\[(\w+)\]", raw)
            saved = m.group(1) if m else "input"
            if any(c in lower for c in green_candidates) and _PI_LED_GREEN is None:
                _PI_LED_GREEN = led
                _PI_LED_SAVED[led] = saved
            elif any(c in lower for c in red_candidates) and _PI_LED_RED is None:
                _PI_LED_RED = led
                _PI_LED_SAVED[led] = saved
        except (OSError, PermissionError):
            pass
    global _PI_LED_OK
    _PI_LED_OK = _PI_LED_GREEN is not None and _PI_LED_RED is not None
    return _PI_LED_OK


def _pi_led_set(led_path: str, on: bool):
    """Set one LED solid on or off."""
    if not led_path:
        return
    trigger_path = os.path.join(led_path, "trigger")
    brightness_path = os.path.join(led_path, "brightness")
    try:
        with open(trigger_path, "w") as f:
            f.write("none\n")
        with open(brightness_path, "w") as f:
            f.write("1\n" if on else "0\n")
    except (OSError, PermissionError):
        pass


def _pi_led_restore():
    """Restore both LEDs to their original triggers (on exit)."""
    for led_path, trigger in _PI_LED_SAVED.items():
        try:
            with open(os.path.join(led_path, "trigger"), "w") as f:
                f.write(trigger + "\n")
        except (OSError, PermissionError):
            pass


def pi_led_set_ready(ready: bool):
    """Set LED state: True = green (ready to talk), False = red (thinking/speaking)."""
    if not _PI_LED_OK:
        return
    if ready:
        _pi_led_set(_PI_LED_GREEN, True)
        _pi_led_set(_PI_LED_RED, False)
    else:
        _pi_led_set(_PI_LED_GREEN, False)
        _pi_led_set(_PI_LED_RED, True)


# ── Cleanup ───────────────────────────────────────────────────

def cleanup():
    """Terminate the microphone subprocess, stop Ollama keepalive, and restore Pi LEDs."""
    _ollama_stop_event.set()
    _pi_led_restore()
    if _mic and _mic.poll() is None:
        _mic.terminate()
        try:
            _mic.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _mic.kill()


def on_signal(signum, frame):
    print()
    cleanup()
    sys.exit(0)


# ── Main ──────────────────────────────────────────────────────

def _run_wake_word_loop():
    """Main loop: wait for speech, check for wake word, route to LLM or quick commands."""
    while True:
        pi_led_set_ready(True)   # Green = listening
        chunks = wait_for_speech()
        if not chunks:
            break
        pi_led_set_ready(False)   # Red = thinking/speaking

        save_wav(chunks, RAM_WAV)
        path = _trim_trailing_silence(RAM_WAV)
        dur = _get_wav_duration(path)

        text, dur = transcribe(path)  # tiny <3s (fast), base >=3s (accurate for conversation)
        if VAD_DEBUG_SAVE:
            _save_vad_debug(RAM_WAV, text)
        try:
            os.remove(RAM_WAV)
        except OSError:
            pass
        if not text:
            dbg(f"{BLUE}[DEBUG]{RESET} Transcription empty, skipping")
            pi_led_set_ready(True)
            continue
        short_match = match_quick_command(text)
        if dur < MIN_UTTERANCE_DURATION and short_match is None:
            dbg(f"{BLUE}[DEBUG]{RESET} Very short capture, no command match, skipping")
            pi_led_set_ready(True)
            continue

        command = extract_wake_command(text)
        if command is None:
            dbg(f"{BLUE}[DEBUG]{RESET} No wake word in '{text}' — say 'Jarvis' first, skipping")
            pi_led_set_ready(True)
            continue

        if len(command) >= 2:
            entered_conversation = ask_jarvis(text, dur=dur)
        else:
            # "Jarvis" alone: only enter conversation if LLM enabled
            speak("Yes?")
            entered_conversation = LLM_ENABLED

        if entered_conversation:
            dbg(f"{GREEN}[CONVERSATION]{RESET} Active")
            conversation()
            wait_for_silence()
            dbg(f"{YELLOW}[LISTENING]{RESET} Wake word mode")


def main():
    global _mic, DEBUG
    if "--debug" in sys.argv or os.environ.get("JARVIS_DEBUG"):
        DEBUG = 1
        if "--debug" in sys.argv:
            sys.argv.remove("--debug")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    atexit.register(cleanup)

    if DEBUG:
        print(f"{BLUE}=== JARVIS ==={RESET}")
        print(f"  Model: {OLLAMA_MODEL}  Whisper: tiny<{WHISPER_SHORT_THRESH}s else base")
        print(f"  Mic: {MIC_DEVICE}  Threshold: {THRESHOLD}\n")

    # Validate paths
    for path, label in [
        (WHISPER_BIN, "Whisper binary"),
        (WHISPER_MODEL_FAST, "Whisper tiny model"),
        (WHISPER_MODEL_ACCURATE, "Whisper base model"),
    ]:
        if not os.path.exists(path):
            print(f"{BLUE}[ERROR]{RESET} {label} not found: {path}")
            return

    # Load TTS voice
    load_tts()

    # Start Ollama keepalive (quick commands work without it; LLM needs connection)
    global _ollama_connected
    _ollama_connected = _check_ollama()
    _ollama_stop_event.clear()
    keepalive = threading.Thread(target=_ollama_keepalive_loop, daemon=True)
    keepalive.start()

    # Start microphone
    _mic = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
         "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    time.sleep(0.2)
    if _mic.poll() is not None:
        err = _mic.stderr.read().decode(errors="ignore").strip()
        print(f"{BLUE}[ERROR]{RESET} Microphone failed: {err}")
        return

    # Discard first second of audio (USB mic initialisation spike)
    for _ in range(int(RATE / CHUNK)):
        _mic.stdout.read(CHUNK * 2)

    # Initialize everything before "ready" so first command is fast
    dbg(f"{BLUE}[STARTUP]{RESET} Warming Whisper...")
    _warm_whisper()
    dbg(f"{BLUE}[STARTUP]{RESET} Discovering Kasa devices...")
    _kasa_discover()

    print(f"{GREEN}Jarvis ready.{RESET}\n")

    if _pi_led_init():
        pi_led_set_ready(True)  # Green = ready to listen (Pi only, needs root)

    _run_wake_word_loop()


if __name__ == "__main__":
    main()
