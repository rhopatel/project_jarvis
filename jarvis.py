#!/usr/bin/env python3
"""
Jarvis Voice Assistant
Wake word activated personal assistant using whisper.cpp, Ollama, and Piper TTS
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
import threading
import signal
import atexit
import re
from typing import Tuple, Optional

# --- CONFIGURATION ---
DEBUG_MODE = 0                # 0 = clean output (only YOU/JARVIS), 1 = verbose debug
WAKE_WORD = "Jarvis"
SIMILARITY_THRESHOLD = 0.6  # Lowered from 0.7 for better detection
WHISPER_THREADS = "4"
THRESHOLD = 1000              # RMS threshold for VAD (may need adjustment based on mic)
SILENCE_LIMIT = 2             # seconds of silence to stop recording
PREV_AUDIO = 0.5              #kusal seconds of pre-audio buffer
CONVERSATION_TIMEOUT = 30     # seconds of silence before exiting conversation mode
PC_IP = "10.0.0.224"
MODEL = "gemma3:12b"

# --- PROMPT ---
SYSTEM_PROMPT = (
    "You are Jarvis, a helpful AI assistant. "
    "You are running in 'Voice Mode'. "
    "Your responses should be concise, natural, and conversational. "
    "Speak as if you're having a conversation. "
    "Do not use markdown formatting or special characters."
)

# --- PATHS ---
WHISPER_BIN = "./main"
if not os.path.exists(WHISPER_BIN):
    WHISPER_BIN = "./build/bin/whisper-cli"
    if not os.path.exists(WHISPER_BIN):
        # Try relative to whisper.cpp directory
        WHISPER_BIN = "../whisper.cpp/build/bin/whisper-cli"

WHISPER_MODEL = "models/ggml-tiny.en.bin"
if not os.path.exists(WHISPER_MODEL):
    WHISPER_MODEL = "../whisper.cpp/models/ggml-tiny.en.bin"

# Set LD_LIBRARY_PATH for whisper shared libraries
WHISPER_LIB_DIR = os.path.abspath("../whisper.cpp/build/src")
GGML_LIB_DIR = os.path.abspath("../whisper.cpp/build/ggml/src")

# Build LD_LIBRARY_PATH with both directories
lib_paths = []
if os.path.exists(WHISPER_LIB_DIR):
    lib_paths.append(WHISPER_LIB_DIR)
if os.path.exists(GGML_LIB_DIR):
    lib_paths.append(GGML_LIB_DIR)

if lib_paths:
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_path = ":".join(lib_paths)
    if current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{current_ld_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

RAM_WAV = "/dev/shm/vad_capture.wav"

# --- CONSTANTS ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1

# --- COLORS ---
BLUE = "\033[1;34m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
RESET = "\033[0m"

def dbg(*args, **kwargs):
    """Print only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)

# --- TTS CONFIGURATION ---
# Piper TTS voice model - will be downloaded on first use if not present
PIPER_VOICE_NAME = "en_US-ryan-medium"
PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")
PIPER_SPEAKER_ID = None  # Set for multi-speaker models, None for single-speaker

# Global: pre-loaded TTS voice (loaded once at startup)
_tts_voice = None

def play_wav(wav_path: str) -> bool:
    """Play a WAV file through PipeWire/PulseAudio (prevents USB speaker first-word cutoff)."""
    # paplay goes through the sound server, which keeps the USB device managed
    # and buffers audio during SUSPENDED->RUNNING transitions.
    # Falls back to aplay direct ALSA only if paplay is unavailable.
    commands = [
        ["paplay", wav_path],
        ["aplay", "-q", "-D", "plughw:2,0", wav_path],
        ["aplay", "-q", wav_path],
    ]
    for cmd in commands:
        try:
            proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
            active_processes.append(proc)
            proc.wait(timeout=60)
            if proc in active_processes:
                active_processes.remove(proc)
            if proc.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc in active_processes:
                active_processes.remove(proc)
        except FileNotFoundError:
            # Command not installed, try next
            continue
        except Exception:
            if proc in active_processes:
                active_processes.remove(proc)
            continue
    return False


def similarity(s0: str, s1: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein distance.
    Returns a float between 0.0 (completely different) and 1.0 (identical).
    """
    s0 = s0.lower().strip()
    s1 = s1.lower().strip()
    
    if not s0 and not s1:
        return 1.0
    if not s0 or not s1:
        return 0.0
    
    len0 = len(s0) + 1
    len1 = len(s1) + 1
    
    # Initialize previous column
    prev_col = list(range(len1))
    
    for i in range(len0):
        col = [i]
        for j in range(1, len1):
            cost = 0 if (i > 0 and j > 0 and s0[i-1] == s1[j-1]) else 1
            col.append(min(
                1 + col[j-1],      # insertion
                1 + prev_col[j],    # deletion
                prev_col[j-1] + cost # substitution
            ))
        prev_col = col
    
    dist = prev_col[len1 - 1]
    max_len = max(len(s0), len(s1))
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def get_words(text: str) -> list:
    """Split text into words."""
    return text.lower().strip().split()


def get_rms(data: bytes) -> float:
    """Calculate RMS (Root Mean Square) of audio data."""
    count = int(len(data) / 2)
    if count == 0:
        return 0.0
    format_str = "<%dh" % count
    shorts = struct.unpack(format_str, data)
    sum_squares = 0.0
    for sample in shorts:
        sum_squares += sample * sample
    return math.sqrt(sum_squares / count) if count > 0 else 0.0


def save_wav(audio_data: list, filename: str) -> bool:
    """Save audio data to WAV file."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(audio_data))
        return True
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Failed to save WAV: {e}")
        return False


def transcribe(wav_file: str) -> str:
    """Transcribe audio file using whisper.cpp."""
    if not os.path.exists(wav_file):
        return ""
    
    cmd = [
        WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_file,
        "-t", WHISPER_THREADS, "-bs", "1", "-bo", "1",
        "--no-timestamps", "-l", "en"
    ]
    
    # Ensure LD_LIBRARY_PATH is set for subprocess
    env = os.environ.copy()
    lib_paths = []
    if WHISPER_LIB_DIR and os.path.exists(WHISPER_LIB_DIR):
        lib_paths.append(WHISPER_LIB_DIR)
    if GGML_LIB_DIR and os.path.exists(GGML_LIB_DIR):
        lib_paths.append(GGML_LIB_DIR)
    
    if lib_paths:
        current_ld_path = env.get("LD_LIBRARY_PATH", "")
        new_ld_path = ":".join(lib_paths)
        if current_ld_path:
            env["LD_LIBRARY_PATH"] = f"{new_ld_path}:{current_ld_path}"
        else:
            env["LD_LIBRARY_PATH"] = new_ld_path
    
    try:
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            dbg(f"{BLUE}[DEBUG] Whisper error: {result.stderr[:200]}{RESET}")
            return ""
        
        text = result.stdout.strip()
        dbg(f"{BLUE}[PERF]{RESET} Transcription: {elapsed:.2f}s")
        return text
    except subprocess.TimeoutExpired:
        print(f"{BLUE}[ERROR]{RESET} Whisper transcription timed out")
        return ""
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Transcription failed: {e}")
        return ""


def clean_transcription(text: str) -> str:
    """
    Remove whisper artifacts like [blank_audio], (silence), [MUSIC], etc.
    Returns the cleaned text, or empty string if nothing real remains.
    """
    if not text:
        return ""
    # Strip bracketed and parenthesized whisper tags
    cleaned = re.sub(r'\[.*?\]', '', text)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    cleaned = cleaned.strip()
    return cleaned


def extract_command(text: str) -> Optional[str]:
    """
    Check if text contains the wake word 'Jarvis' and extract the command.
    Returns the command text (with 'Jarvis' removed), or None if wake word not found.
    """
    if not text:
        return None
    
    words = get_words(text)
    if not words:
        return None
    
    # Check if any word matches "Jarvis"
    jarvis_index = None
    for i, word in enumerate(words):
        clean_word = word.rstrip('.,!?;:')
        if similarity(clean_word, WAKE_WORD) >= SIMILARITY_THRESHOLD:
            jarvis_index = i
            break
    
    if jarvis_index is None:
        return None
    
    # Remove "Jarvis" and return the rest as the command
    command_words = words[:jarvis_index] + words[jarvis_index + 1:]
    command = " ".join(command_words).strip()
    
    dbg(f"{BLUE}[DEBUG]{RESET} Heard: '{text}' -> Wake word found, command: '{command}'")
    return command if len(command) >= 2 else ""


def query_ollama_stream(prompt: str):
    """
    Query Ollama API and yield response tokens as they arrive.
    Returns full response text.
    """
    url = f"http://{PC_IP}:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True
    }
    
    full_response = ""
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            full_response += token
                            yield token
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.RequestException as e:
        print(f"\n{BLUE}[ERROR]{RESET} Ollama connection failed: {e}")
        yield None
    
    return full_response


def load_tts_voice():
    """
    Pre-load the Piper TTS voice model.
    Suppresses ONNX Runtime GPU discovery warnings by temporarily redirecting stderr.
    """
    global _tts_voice
    
    try:
        import warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        warnings.filterwarnings('ignore', category=UserWarning, module='onnxruntime')
        
        from pathlib import Path
        
        voice_dir = Path(PIPER_VOICE_DIR)
        voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx"
        config_path = voice_dir / f"{PIPER_VOICE_NAME}.onnx.json"
        
        # Suppress C++ ONNX Runtime warnings by redirecting fd 2 (stderr)
        # Must wrap ALL piper imports since onnxruntime initializes at import time
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        try:
            from piper.voice import PiperVoice
            from piper.download_voices import download_voice
        finally:
            os.dup2(stderr_fd, 2)
            os.close(devnull)
            os.close(stderr_fd)
        
        if not model_path.exists() or not config_path.exists():
            print(f"{BLUE}[INFO]{RESET} Downloading Piper voice model (first time only)...")
            download_voice(PIPER_VOICE_NAME, voice_dir)
        
        _tts_voice = PiperVoice.load(str(model_path), use_cuda=False)
        
        dbg(f"{GREEN}[OK]{RESET} TTS voice loaded: {PIPER_VOICE_NAME}")
        return True
    except Exception as e:
        print(f"{BLUE}[WARNING]{RESET} Failed to load TTS voice: {e}")  # always show - TTS is critical
        return False


def speak_text(text: str) -> bool:
    """
    Convert text to speech using pre-loaded Piper TTS voice and play it.
    """
    global _tts_voice
    
    if not text or not text.strip():
        return False
    
    wav_path = "/dev/shm/jarvis_tts.wav"
    
    try:
        if _tts_voice is None:
            load_tts_voice()
        
        if _tts_voice is None:
            raise RuntimeError("TTS voice not loaded")
        
        from piper.config import SynthesisConfig
        syn_config = SynthesisConfig(speaker_id=PIPER_SPEAKER_ID, volume=0.25)
        
        raw_wav_path = "/dev/shm/jarvis_tts_raw.wav"
        with wave.open(raw_wav_path, 'wb') as wav_file:
            _tts_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        
        # Prepend a small silence so the sound server has time to fully
        # wake the USB sink before speech begins.
        SILENCE_MS = 150
        with wave.open(raw_wav_path, 'rb') as rf:
            params = rf.getparams()
            audio_data = rf.readframes(rf.getnframes())
        silence_samples = int(params.framerate * SILENCE_MS / 1000)
        silence_bytes = b'\x00' * (silence_samples * params.sampwidth * params.nchannels)
        with wave.open(wav_path, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(silence_bytes + audio_data)
        
        # Save persistent copy when debugging
        if DEBUG_MODE:
            import shutil
            tts_debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_debug")
            os.makedirs(tts_debug_dir, exist_ok=True)
            debug_name = f"tts_{int(time.time())}.wav"
            debug_path = os.path.join(tts_debug_dir, debug_name)
            shutil.copy2(wav_path, debug_path)
            dbg(f"{BLUE}[DEBUG]{RESET} TTS WAV saved: {debug_path}")
        
        if play_wav(wav_path):
            return True
        
        print(f"{BLUE}[ERROR]{RESET} Audio playback failed")
        return False
    
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} TTS failed: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
    
    # Fallback: use espeak
    try:
        dbg(f"{BLUE}[INFO]{RESET} Using espeak fallback")
        result = subprocess.run(
            ["espeak", "-s", "150", "-v", "en", text],
            stderr=subprocess.DEVNULL, timeout=30
        )
        return result.returncode == 0
    except FileNotFoundError:
        print(f"{BLUE}[ERROR]{RESET} No TTS engine available")
        return False
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} espeak failed: {e}")
        return False


# Global process tracking for cleanup
active_processes = []

def cleanup_processes():
    """Kill all active subprocesses."""
    for proc in active_processes[:]:  # Copy list to avoid modification during iteration
        try:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except Exception:
            pass
        finally:
            if proc in active_processes:
                active_processes.remove(proc)

def signal_handler(signum, frame):
    """Handle termination signals."""
    dbg(f"\n{BLUE}[SHUTDOWN]{RESET} Received signal {signum}, cleaning up...")
    print()
    cleanup_processes()
    sys.exit(0)




def record_until_silence(process, audio_buffer: collections.deque, silence_limit: float = None) -> list:
    """
    Record audio until silence is detected.
    Returns list of audio chunks (mono).
    """
    if silence_limit is None:
        silence_limit = SILENCE_LIMIT
    
    recording = []
    silence_start = None
    got_speech = False
    
    # Add pre-audio buffer
    recording.extend(audio_buffer)
    
    dbg(f"{BLUE}[RECORDING]{RESET} Listening...", end="", flush=True)
    
    while True:
        chunk = process.stdout.read(CHUNK * 2)
        if not chunk:
            break
        
        recording.append(chunk)
        rms = get_rms(chunk)
        
        if rms >= THRESHOLD:
            got_speech = True
            silence_start = None
        else:
            if silence_start is None:
                silence_start = time.time()
            elif got_speech and time.time() - silence_start > silence_limit:
                # Only stop after we've heard speech + silence
                break
            elif not got_speech and time.time() - silence_start > silence_limit + 2:
                # If we never heard speech, wait a bit longer then give up
                break
        
        # Visual feedback
        if DEBUG_MODE:
            print(".", end="", flush=True)
    
    if DEBUG_MODE:
        print()  # New line after dots
    return recording


def drain_mic(process):
    """
    Drain any buffered audio from the microphone.
    Call this after TTS playback to discard audio of Jarvis talking.
    """
    import select
    try:
        while select.select([process.stdout], [], [], 0)[0]:
            discarded = process.stdout.read(CHUNK * 2)
            if not discarded:
                break
    except Exception:
        pass


def wait_for_silence(process, settle_time: float = 0.5):
    """
    After Jarvis finishes speaking, read and discard audio until the room
    is quiet (RMS < THRESHOLD) for at least `settle_time` seconds.
    This prevents the VAD from immediately triggering on speaker echo.
    """
    quiet_since = None
    while True:
        chunk = process.stdout.read(CHUNK * 2)
        if not chunk:
            return
        rms = get_rms(chunk)
        if rms < THRESHOLD:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= settle_time:
                return  # Room is quiet, safe to start listening
        else:
            quiet_since = None  # Still noisy, keep discarding


def send_to_ollama(command: str, process) -> Optional[str]:
    """
    Send a command to Ollama, speak the response.
    Returns the response text, or None on failure.
    """
    print(f"{CYAN}YOU:{RESET} {command}")
    
    # Query Ollama
    dbg(f"{YELLOW}[THINKING]{RESET}...")
    response_text = ""
    
    try:
        for token in query_ollama_stream(command):
            if token is None:
                break
            response_text += token
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Failed to get response: {e}")
        return None
    
    if response_text:
        print(f"{GREEN}JARVIS:{RESET} {response_text}")
        print()
        
        # Speak response
        dbg(f"{MAGENTA}[SPEAKING]{RESET}...")
        speak_text(response_text)
    
    return response_text


def conversation_loop(process):
    """
    Conversation mode: keep listening and responding until extended silence.
    No wake word needed after the first activation.
    Uses the same VAD logic as the main loop -- waits patiently for RMS to
    exceed the threshold before recording, then waits for silence to end the
    utterance.  Exits only after CONVERSATION_TIMEOUT seconds of total silence.
    """
    pre_audio = collections.deque(maxlen=int(RATE / CHUNK * PREV_AUDIO))
    
    while True:
        dbg(f"{YELLOW}[CONVERSATION]{RESET} Listening... (say nothing to exit)")
        
        # --- Wait for speech (identical to main loop VAD) ---
        speech_chunks = []
        is_speaking = False
        silence_start = None
        wait_start = time.time()
        
        while True:
            chunk = process.stdout.read(CHUNK * 2)
            if not chunk:
                return
            
            rms = get_rms(chunk)
            pre_audio.append(chunk)
            
            if rms >= THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    speech_chunks = list(pre_audio)
                else:
                    speech_chunks.append(chunk)
                silence_start = None
            elif is_speaking:
                speech_chunks.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_LIMIT:
                    # Speech ended
                    break
            else:
                # No speech yet -- wait patiently up to CONVERSATION_TIMEOUT
                if time.time() - wait_start >= CONVERSATION_TIMEOUT:
                    dbg(f"{BLUE}[INFO]{RESET} No input detected, exiting conversation mode.")
                    return
        
        if not speech_chunks:
            continue
        
        if not save_wav(speech_chunks, RAM_WAV):
            continue
        
        # --- Transcribe and filter ---
        dbg(f"{YELLOW}[TRANSCRIBING]{RESET}...")
        raw_text = transcribe(RAM_WAV)
        full_text = clean_transcription(raw_text)
        
        if not full_text or len(full_text.strip()) < 2:
            # Whisper returned silence/artifact -- NOT real speech.
            # Don't count this against the user; just go back to waiting.
            dbg(f"{BLUE}[...]{RESET} No speech detected, still listening...")
            pre_audio.clear()
            continue
        
        # Remove wake word if user said it out of habit
        words = get_words(full_text)
        command_words = []
        for word in words:
            clean = word.rstrip('.,!?;:')
            if similarity(clean, WAKE_WORD) < SIMILARITY_THRESHOLD:
                command_words.append(word)
        
        full_command = " ".join(command_words).strip()
        
        if len(full_command) < 2:
            continue
        
        send_to_ollama(full_command, process)
        pre_audio.clear()
        wait_for_silence(process)


def main():
    """Main event loop."""
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_processes)
    
    if DEBUG_MODE:
        print(f"{BLUE}=== JARVIS VOICE ASSISTANT ==={RESET}")
        print(f"{BLUE}   * Wake Word:     {WAKE_WORD}{RESET}")
        print(f"{BLUE}   * Model:         {MODEL}{RESET}")
        print(f"{BLUE}   * Mic Input:     plughw:4,0 (AIRHUG 21){RESET}")
        print(f"{BLUE}   * Threshold:    {THRESHOLD}{RESET}")
        print(f"{BLUE}   * Whisper Model: {WHISPER_MODEL}{RESET}")
        print()
    
    # Check dependencies
    if not os.path.exists(WHISPER_BIN):
        print(f"{BLUE}[ERROR]{RESET} Whisper binary not found at {WHISPER_BIN}")
        print(f"{BLUE}[INFO]{RESET} Please ensure whisper.cpp is built and binary is accessible")
        return
    
    if not os.path.exists(WHISPER_MODEL):
        print(f"{BLUE}[ERROR]{RESET} Whisper model not found at {WHISPER_MODEL}")
        print(f"{BLUE}[INFO]{RESET} Please download a whisper model")
        return
    
    # Pre-load TTS voice (suppresses ONNX warning here at startup)
    dbg(f"{BLUE}[INIT]{RESET} Loading TTS voice...")
    load_tts_voice()
    
    # Start audio capture
    # Use plughw:4,0 for AIRHUG 21 microphone (card 4) - handles sample rate conversion
    record_cmd = [
        "arecord", "-D", "plughw:4,0", "-f", "S16_LE",
        "-c", "1", "-r", str(RATE), "-t", "raw", "-q"
    ]
    
    try:
        process = subprocess.Popen(
            record_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        # Give it a moment to start, then check if it's still running
        time.sleep(0.2)
        if process.poll() is not None:
            _, stderr = process.communicate()
            error_msg = stderr.decode('utf-8', errors='ignore') if stderr else "Unknown error"
            print(f"{BLUE}[ERROR]{RESET} Failed to start microphone: {error_msg.strip()}")
            print(f"{BLUE}[INFO]{RESET} Please check that AIRHUG 21 microphone (card 4) is connected")
            return
        else:
            dbg(f"{GREEN}[OK]{RESET} Using microphone: plughw:4,0 (AIRHUG 21)")
            # Discard first ~1 second of audio (USB mic initialization spike)
            discard_chunks = int(RATE / CHUNK * 1.0)
            for _ in range(discard_chunks):
                process.stdout.read(CHUNK * 2)
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Failed to start microphone: {e}")
        cleanup_processes()
        return
    
    # Audio buffer for pre-speech capture
    pre_audio = collections.deque(maxlen=int(RATE / CHUNK * PREV_AUDIO))
    
    if not DEBUG_MODE:
        print(f"{GREEN}Jarvis ready.{RESET}")
    else:
        print(f"{YELLOW}[LISTENING]{RESET} Say '{WAKE_WORD}' to activate...")
    print()
    
    try:
        speech_chunks = []
        is_speaking = False
        silence_start = None
        meter_counter = 0
        METER_INTERVAL = int(RATE / CHUNK * 0.1)  # Update meter every ~100ms
        METER_WIDTH = 40  # Character width of the bar
        
        while True:
            chunk = process.stdout.read(CHUNK * 2)
            if not chunk:
                dbg(f"{BLUE}[DEBUG]{RESET} No audio data received, exiting...")
                break
            
            rms = get_rms(chunk)
            pre_audio.append(chunk)
            
            # Live RMS meter (updates ~10x per second, only when not speaking/processing)
            if DEBUG_MODE:
                meter_counter += 1
                if not is_speaking and meter_counter >= METER_INTERVAL:
                    meter_counter = 0
                    bar_level = min(int(rms / 200), METER_WIDTH)  # Scale: 200 RMS per char
                    thresh_pos = min(int(THRESHOLD / 200), METER_WIDTH)
                    bar = ""
                    for i in range(METER_WIDTH):
                        if i < bar_level:
                            bar += "\033[1;32m#\033[0m" if i < thresh_pos else "\033[1;31m#\033[0m"
                        elif i == thresh_pos:
                            bar += "\033[1;33m|\033[0m"
                        else:
                            bar += " "
                    print(f"\r  RMS [{bar}] {rms:5.0f}/{THRESHOLD}", end="", flush=True)
            
            if rms >= THRESHOLD:
                # Voice detected
                if not is_speaking:
                    if DEBUG_MODE:
                        print()  # Clear meter line
                    is_speaking = True
                    speech_chunks = list(pre_audio)
                else:
                    speech_chunks.append(chunk)
                silence_start = None
            elif is_speaking:
                # Below threshold but we were speaking - track silence
                speech_chunks.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_LIMIT:
                    # Speech ended - transcribe it
                    is_speaking = False
                    silence_start = None
                    
                    if save_wav(speech_chunks, RAM_WAV):
                        dbg(f"{YELLOW}[TRANSCRIBING]{RESET}...")
                        text = clean_transcription(transcribe(RAM_WAV))
                        
                        if text:
                            command = extract_command(text)
                            
                            if command is not None:
                                # "Jarvis" was in the sentence
                                dbg(f"{GREEN}[WAKE WORD DETECTED]{RESET}")
                                
                                if len(command) >= 2:
                                    send_to_ollama(command, process)
                                else:
                                    # Just said "Jarvis" with no command
                                    speak_text("Yes?")
                                
                                # Let the room settle before listening again
                                wait_for_silence(process)
                                
                                # Enter conversation mode
                                dbg(f"{GREEN}[CONVERSATION MODE]{RESET} Keep talking, no need to say '{WAKE_WORD}'")
                                conversation_loop(process)
                                
                                # Back to wake word mode
                                pre_audio.clear()
                                wait_for_silence(process)
                                
                                dbg(f"{YELLOW}[LISTENING]{RESET} Say '{WAKE_WORD}' to activate...")
                    
                    speech_chunks = []
    
    except KeyboardInterrupt:
        dbg(f"\n{BLUE}[SHUTDOWN]{RESET} Stopping...")
        print()
    except Exception as e:
        print(f"\n{BLUE}[ERROR]{RESET} Unexpected error: {e}")  # always show
    finally:
        cleanup_processes()


if __name__ == "__main__":
    main()
