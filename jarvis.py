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
import wave
import collections
import requests
import json
import signal
import atexit
import re
import select
import tempfile
from typing import Optional


# ── Configuration ─────────────────────────────────────────────

DEBUG              = 0            # 0 = clean (YOU/JARVIS only), 1 = verbose
VAD_DEBUG_SAVE     = 0            # 1 = save captured WAV to vad_captures/ (verify full command), 0 = off
WAKE_WORD          = "Jarvis"
SIMILARITY_THRESH  = 0.6
THRESHOLD          = 1000        # RMS threshold for voice activity detection
SILENCE_LIMIT      = 0.7         # seconds of silence (0.5 was cutting off "light on")
SILENCE_LIMIT_CONV = 1.2        # seconds of silence in conversation mode (longer pauses ok)
PRE_AUDIO          = 0.25        # seconds of pre-roll before speech (minimal)
CONVERSATION_TIMEOUT = 30        # seconds of no speech to exit conversation
SILENCE_PAD_MS     = 150         # ms of silence prepended to TTS for USB sink wake
#BARGEIN_THRESHOLD  = 1800        # RMS to trigger barge-in (above speaker bleed)
#BARGEIN_CHUNKS     = 3           # consecutive loud chunks needed (~200ms, prevents false triggers)
OLLAMA_IP          = "10.0.0.224"
OLLAMA_MODEL       = "gemma3:12b"
RATE               = 16000

# TP-Link Kasa - direct local control (no cloud)
# Newer bulbs may need TP-Link account credentials - leave empty for older devices
KASA_USERNAME      = ""  # e.g. "you@email.com" for Tapo/newer Kasa
KASA_PASSWORD      = ""  # TP-Link cloud password

# Curated quick commands: (NAME, voice_phrases, reaction)
# NAME: ALL_CAPS identifier shown when triggered
# reaction: "kasa_on" | "kasa_off" | ... (dispatched below)
QUICK_COMMANDS     = [
    ("LIGHT_ON",  ["turn on lights", "lights on", "switch on lights", "light on"], "kasa_on"),
    ("LIGHT_OFF", ["turn off lights", "lights off", "switch off lights", "light off"], "kasa_off"),
]
# "light" alone = LIGHT_ON (for cut-off) but not when they said "light off"
QUICK_FALLBACK    = ("LIGHT_ON", "light", "kasa_on")  # (name, phrase, reaction) if phrase in cmd and "light off" not in cmd
CHUNK              = 1024

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful AI assistant. "
    "You are running in 'Voice Mode'. "
    "Your responses should be concise, natural, and conversational. "
    "Speak as if you're having a conversation. "
    "Do not use markdown formatting or special characters."
)


# ── Paths (fixed for this machine) ───────────────────────────

WHISPER_BIN   = os.path.abspath("../whisper.cpp/build/bin/whisper-cli")
# Prefer quantized model (faster, smaller) - fallback to full if missing
WHISPER_MODEL = os.path.abspath("../whisper.cpp/models/ggml-tiny.en-q4_0.bin")
if not os.path.exists(WHISPER_MODEL):
    WHISPER_MODEL = os.path.abspath("../whisper.cpp/models/ggml-tiny.en.bin")
# Initial prompt biases transcription toward these phrases (reduces "lay on him" -> "light on" etc.)
WHISPER_PROMPT = "Jarvis Jarvis light on lights off turn on turn off switch"


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
RAM_WAV       = "/dev/shm/jarvis_vad.wav"
TTS_WAV       = "/dev/shm/jarvis_tts.wav"
TTS_RAW_WAV   = "/dev/shm/jarvis_tts_raw.wav"

PIPER_VOICE     = "en_US-ryan-medium"
PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")


# ── Whisper subprocess environment (built once) ──────────────

_whisper_env = os.environ.copy()
for _d in ["../whisper.cpp/build/src", "../whisper.cpp/build/ggml/src"]:
    _abs = os.path.abspath(_d)
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
    """Play a WAV through PipeWire/PulseAudio while monitoring the mic.

    """
    subprocess.run(["paplay", path], stderr=subprocess.DEVNULL, timeout=120)


# ── Whisper transcription ────────────────────────────────────

def _trim_trailing_silence(path: str, thresh: float = 800) -> str:
    """Trim trailing silence from WAV to reduce whisper processing time. Returns path."""
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
        need_quiet = int(0.25 * rate / chunk_sz)  # 250ms silence = cut
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


def _warm_whisper():
    """Preload Whisper model so first real transcription is fast."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"\x00\x00" * int(RATE * 0.5))  # 0.5s silence
        transcribe(path)
    except Exception:
        pass
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


def transcribe(path: str) -> str:
    """Run whisper.cpp on a WAV file and return cleaned text."""
    path = _trim_trailing_silence(path)
    t0 = time.time()
    nthreads = min(4, (os.cpu_count() or 4))
    cmd = [
        WHISPER_BIN, "-m", WHISPER_MODEL, "-f", path,
        "-t", str(nthreads), "-bs", "1", "-bo", "1",
        "-ml", "32",  # Short commands only
        "--no-timestamps", "-l", "en",
    ]
    if WHISPER_PROMPT:
        cmd.extend(["--prompt", WHISPER_PROMPT])
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30, env=_whisper_env,
        )
    except subprocess.TimeoutExpired:
        return ""

    dbg(f"{BLUE}[PERF]{RESET} Transcription: {time.time() - t0:.2f}s")

    if r.returncode != 0:
        dbg(f"{BLUE}[DEBUG]{RESET} Whisper stderr: {r.stderr[:200]}")
        return ""

    # Strip whisper artefacts: [blank_audio], (silence), etc.
    text = re.sub(r"\[.*?\]|\(.*?\)", "", r.stdout).strip()
    # Fix common mishearings of "Jarvis" at start (e.g. "driver's light off")
    text = _fix_jarvis_mishearing(text)
    return text


def _fix_jarvis_mishearing(text: str) -> str:
    """Replace common mishearings of 'Jarvis' at sentence start."""
    if not text or len(text) < 5:
        return text
    lowered = text.lower()
    # "driver's", "drivers", "driver is" etc. at start -> "Jarvis"
    for mis in (r"^driver'?s?\s+", r"^drivers\s+", r"^driver\s+is\s+"):
        if re.search(mis, lowered):
            return "Jarvis " + re.sub(mis, "", text, flags=re.I).strip()
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


def speak(text: str):
    """Synthesise text with Piper, pad with silence, and play via PulseAudio.
    """
    if not text or not text.strip():
        return

    from piper.config import SynthesisConfig

    cfg = SynthesisConfig(speaker_id=None, volume=0.25)

    # Generate raw speech
    with wave.open(TTS_RAW_WAV, "wb") as wf:
        _tts_voice.synthesize_wav(text, wf, syn_config=cfg)

    # Read it back, prepend silence for USB sink wake-up, write final WAV
    with wave.open(TTS_RAW_WAV, "rb") as rf:
        params = rf.getparams()
        audio  = rf.readframes(rf.getnframes())

    pad_bytes = int(params.framerate * SILENCE_PAD_MS / 1000) * params.sampwidth * params.nchannels
    pad = b"\x00" * pad_bytes

    with wave.open(TTS_WAV, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(pad + audio)

    # Debug: keep a copy
    if DEBUG:
        import shutil
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_debug")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, f"tts_{int(time.time())}.wav")
        shutil.copy2(TTS_WAV, dst)
        dbg(f"{BLUE}[DEBUG]{RESET} WAV saved: {dst}")

    play_wav(TTS_WAV)
    for p in (TTS_WAV, TTS_RAW_WAV):
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


def match_quick_command(text: str) -> Optional[tuple]:
    """If text matches a quick command phrase, return (NAME, reaction). Else None."""
    cmd = text.lower().strip()
    for name, phrases, reaction in QUICK_COMMANDS:
        for phrase in phrases:
            if phrase in cmd:
                dbg(f"{BLUE}[QUICK]{RESET} Matched '{phrase}' -> {name}")
                return (name, reaction)
    # Fallback: "light" alone = on (handles cut-off) but not "light off"
    if QUICK_FALLBACK and QUICK_FALLBACK[1] in cmd and "light off" not in cmd:
        dbg(f"{BLUE}[QUICK]{RESET} Fallback '{QUICK_FALLBACK[1]}' -> {QUICK_FALLBACK[0]}")
        return (QUICK_FALLBACK[0], QUICK_FALLBACK[2])
    return None


def _run_quick_reaction(reaction: str) -> bool:
    """Dispatch quick command reaction. Returns True on success."""
    if reaction == "kasa_on":
        return control_kasa_lights(on=True)
    if reaction == "kasa_off":
        return control_kasa_lights(on=False)
    return False


def query_ollama(prompt: str) -> str:
    """Send a prompt to Ollama and return the full response."""
    url = f"http://{OLLAMA_IP}:11434/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
    }
    text = ""
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text += data.get("response", "")
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.RequestException as e:
        print(f"{BLUE}[ERROR]{RESET} Ollama: {e}")
    return text


def ask_jarvis(text: str) -> bool:
    """Layer 3 first (local actions), Layer 4 only if no match.
    text: full utterance (include 'Jarvis' for LLM). COMMAND vs YOU based on Layer 3 match.
    Returns True if we gave a verbal LLM reply (enter conversation mode)."""
    m = match_quick_command(text)
    if m is not None:
        name, reaction = m
        print(f"{CYAN}COMMAND:{RESET} {name}")
        ok = _run_quick_reaction(reaction)
        if not ok:
            err = "No Kasa devices found. Check they're on your network."
            print(f"{BLUE}[ERROR]{RESET} {err}\n")
            speak(err)
        return False

    print(f"{CYAN}YOU:{RESET} {text}")
    dbg(f"{YELLOW}[THINKING]{RESET}...")
    response = query_ollama(text)
    if response:
        print(f"{GREEN}JARVIS:{RESET} {response}\n")
        speak(response)
        return True  # Verbal reply → enter conversation mode
    return False


# ── Conversation mode ─────────────────────────────────────────

def conversation():
    """After wake word, keep talking without repeating 'Jarvis'.
    Exits after CONVERSATION_TIMEOUT seconds of silence.

    """
    while True:
        wait_for_silence()

        dbg(f"{YELLOW}[CONVERSATION]{RESET} Listening...")

        chunks = wait_for_speech(timeout=CONVERSATION_TIMEOUT, silence_limit=SILENCE_LIMIT_CONV)
        if not chunks:
            dbg(f"{BLUE}[INFO]{RESET} Conversation timeout.")
            return

        save_wav(chunks, RAM_WAV)
        text = transcribe(RAM_WAV)
        if VAD_DEBUG_SAVE:
            _save_vad_debug(RAM_WAV, text)
        try:
            os.remove(RAM_WAV)
        except OSError:
            pass
        if not text or len(text) < 2:
            continue

        # Don't edit: pass full text including "Jarvis" to LLM
        ask_jarvis(text)


# ── Raspberry Pi LED status ─────────────────────────────────────

_PI_LED_PWR = None
_PI_LED_SAVED_TRIGGER = None


def _pi_led_set_running(running: bool):
    """Set the Pi's PWR (red) LED: solid on = Jarvis running, restore default on exit.
    Fails silently if not on a Pi or without permission.
    """
    global _PI_LED_PWR, _PI_LED_SAVED_TRIGGER
    base = "/sys/class/leds"
    if not os.path.isdir(base):
        return
    for name in os.listdir(base):
        if "pwr" in name.lower() or "power" in name.lower() or name == "led1":
            led = os.path.join(base, name)
            trigger_path = os.path.join(led, "trigger")
            brightness_path = os.path.join(led, "brightness")
            if not os.path.isfile(trigger_path):
                continue
            try:
                if running:
                    with open(trigger_path) as f:
                        raw = f.read()
                    m = re.search(r"\[(\w+)\]", raw)
                    _PI_LED_SAVED_TRIGGER = m.group(1) if m else "input"
                    _PI_LED_PWR = led
                    with open(trigger_path, "w") as f:
                        f.write("none\n")
                    if os.path.isfile(brightness_path):
                        with open(brightness_path, "w") as f:
                            f.write("1\n")
                elif _PI_LED_PWR == led and _PI_LED_SAVED_TRIGGER:
                    with open(trigger_path, "w") as f:
                        f.write(_PI_LED_SAVED_TRIGGER)
            except (OSError, PermissionError):
                pass
            break


# ── Cleanup ───────────────────────────────────────────────────

def cleanup():
    """Terminate the microphone subprocess and restore Pi LED."""
    _pi_led_set_running(False)
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

def main():
    global _mic

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    atexit.register(cleanup)

    if DEBUG:
        print(f"{BLUE}=== JARVIS ==={RESET}")
        print(f"  Model: {OLLAMA_MODEL}  Whisper: {os.path.basename(WHISPER_MODEL)}")
        print(f"  Mic: {MIC_DEVICE}  Threshold: {THRESHOLD}\n")

    # Validate paths
    for path, label in [(WHISPER_BIN, "Whisper binary"), (WHISPER_MODEL, "Whisper model")]:
        if not os.path.exists(path):
            print(f"{BLUE}[ERROR]{RESET} {label} not found: {path}")
            return

    # Load TTS voice
    load_tts()

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

    _pi_led_set_running(True)  # Solid red PWR LED = Jarvis running (Pi only)

    # ── Wake-word loop ────────────────────────────────────────
    while True:
        chunks = wait_for_speech()
        if not chunks:
            break

        save_wav(chunks, RAM_WAV)
        text = transcribe(RAM_WAV)
        if VAD_DEBUG_SAVE:
            _save_vad_debug(RAM_WAV, text)
        try:
            os.remove(RAM_WAV)
        except OSError:
            pass
        if not text:
            continue

        command = extract_wake_command(text)
        if command is None:
            continue

        if len(command) >= 2:
            entered_conversation = ask_jarvis(text)
        else:
            speak("Yes?")
            entered_conversation = True

        # Conversation mode only when Jarvis gave a verbal reply (LLM or "Yes?")
        if entered_conversation:
            dbg(f"{GREEN}[CONVERSATION]{RESET} Active")
            conversation()
            wait_for_silence()
            dbg(f"{YELLOW}[LISTENING]{RESET} Wake word mode")


if __name__ == "__main__":
    main()
