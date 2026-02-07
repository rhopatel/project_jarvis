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
from typing import Tuple, Optional

# --- CONFIGURATION ---
WAKE_WORD = "Jarvis"
SIMILARITY_THRESHOLD = 0.6  # Lowered from 0.7 for better detection
WAKE_CHUNK_DURATION = 2.5  # seconds for wake word detection chunks
WHISPER_THREADS = "1"
THRESHOLD = 2000              # RMS threshold for VAD (may need adjustment based on mic)
SILENCE_LIMIT = 3             # seconds of silence to stop recording
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

WHISPER_MODEL = "models/ggml-base.en.bin"
if not os.path.exists(WHISPER_MODEL):
    WHISPER_MODEL = "../whisper.cpp/models/ggml-base.en.bin"

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
WAKE_WAV = "/dev/shm/wake_chunk.wav"

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

# --- TTS CONFIGURATION ---
# Piper TTS voice model - will be downloaded on first use if not present
PIPER_VOICE_NAME = "en_US-kusal-medium"
PIPER_VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")
PIPER_SPEAKER_ID = None  # Set for multi-speaker models, None for single-speaker

# Global: pre-loaded TTS voice (loaded once at startup)
_tts_voice = None


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
        "-ac", "512", "-t", WHISPER_THREADS, "--no-timestamps"
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        
        if result.returncode != 0:
            print(f"{BLUE}[DEBUG] Whisper error: {result.stderr[:200]}{RESET}")
            return ""
        
        text = result.stdout.strip()
        return text
    except subprocess.TimeoutExpired:
        print(f"{BLUE}[ERROR]{RESET} Whisper transcription timed out")
        return ""
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Transcription failed: {e}")
        return ""


def detect_wake_word(audio_chunk: list) -> Tuple[bool, str]:
    """
    Detect wake word in audio chunk.
    Returns (matched: bool, command_text: str)
    """
    if not save_wav(audio_chunk, WAKE_WAV):
        return False, ""
    
    text = transcribe(WAKE_WAV)
    if not text:
        return False, ""
    
    words = get_words(text)
    if not words:
        return False, ""
    
    # Check if first word matches wake word
    first_word = words[0] if words else ""
    # Remove punctuation for better matching
    first_word_clean = first_word.rstrip('.,!?;:')
    sim = similarity(first_word_clean, WAKE_WORD)
    
    # Only print debug if similarity is close (to reduce noise)
    if sim >= SIMILARITY_THRESHOLD - 0.2:
        print(f"{BLUE}[DEBUG]{RESET} Heard: '{text}' -> First word: '{first_word_clean}', similarity: {sim:.2f}")
    
    if sim >= SIMILARITY_THRESHOLD:
        # Extract command (everything after wake word)
        command_words = words[1:] if len(words) > 1 else []
        command_text = " ".join(command_words)
        return True, command_text
    
    return False, ""


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
        
        print(f"{GREEN}[OK]{RESET} TTS voice loaded: {PIPER_VOICE_NAME}")
        return True
    except Exception as e:
        print(f"{BLUE}[WARNING]{RESET} Failed to load TTS voice: {e}")
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
        syn_config = SynthesisConfig(speaker_id=PIPER_SPEAKER_ID, volume=0.5)
        
        with wave.open(wav_path, 'wb') as wav_file:
            _tts_voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        
        # Play the WAV file with aplay
        for aplay_device in ["plughw:2,0", "default"]:
            try:
                aplay_proc = subprocess.Popen(
                    ["aplay", "-q", "-D", aplay_device, wav_path],
                    stderr=subprocess.PIPE
                )
                active_processes.append(aplay_proc)
                _, stderr = aplay_proc.communicate(timeout=30)
                if aplay_proc in active_processes:
                    active_processes.remove(aplay_proc)
                if aplay_proc.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                if aplay_proc.poll() is None:
                    aplay_proc.kill()
                    aplay_proc.wait()
                if aplay_proc in active_processes:
                    active_processes.remove(aplay_proc)
            except Exception:
                if aplay_proc in active_processes:
                    active_processes.remove(aplay_proc)
                continue
        
        print(f"{BLUE}[ERROR]{RESET} All audio devices failed for TTS playback")
        return False
    
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} TTS failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback: use espeak
    try:
        print(f"{BLUE}[INFO]{RESET} Using espeak fallback")
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
    print(f"\n{BLUE}[SHUTDOWN]{RESET} Received signal {signum}, cleaning up...")
    cleanup_processes()
    sys.exit(0)


def play_ping_sound():
    """Play a simple ping/beep sound to indicate listening."""
    try:
        # Generate a two-tone ping (880Hz + 1320Hz, 0.25 seconds)
        sample_rate = 22050
        duration = 0.25
        num_samples = int(sample_rate * duration)
        
        audio_data = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Two harmonics for a pleasant chime
            tone = (math.sin(2 * math.pi * 880 * t) * 0.6 +
                    math.sin(2 * math.pi * 1320 * t) * 0.4)
            # Quick attack, smooth decay
            envelope = min(1.0, t * 50) * math.exp(-t * 8)
            sample = int(tone * envelope * 28000)
            # Clamp to 16-bit range
            sample = max(-32768, min(32767, sample))
            # Pack as signed 16-bit little-endian
            audio_data.extend(struct.pack('<h', sample))
        
        # Play using aplay - try USB speaker first, then default
        for aplay_device in ["plughw:2,0", "default"]:
            try:
                proc = subprocess.Popen(
                    ["aplay", "-q", "-D", aplay_device, "-f", "S16_LE",
                     "-r", str(sample_rate), "-c", "1"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                proc.communicate(input=bytes(audio_data), timeout=3)
                if proc.returncode == 0:
                    return
            except Exception:
                continue
    except Exception:
        pass  # Ping is nice-to-have, not critical


def record_until_silence(process, audio_buffer: collections.deque, silence_limit: float = None, play_ping: bool = True) -> list:
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
    
    # Play ping sound to indicate listening
    if play_ping:
        play_ping_sound()
    
    print(f"{BLUE}[RECORDING]{RESET} Listening...", end="", flush=True)
    
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
        print(".", end="", flush=True)
    
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


def process_command(process, audio_buffer):
    """
    Record, transcribe, query Ollama, and speak a single command.
    Returns the response text, or None if no valid command was captured.
    """
    recording = record_until_silence(process, audio_buffer)
    
    if not save_wav(recording, RAM_WAV):
        print(f"{BLUE}[ERROR]{RESET} Failed to save recording")
        return None
    
    # Transcribe
    print(f"{YELLOW}[TRANSCRIBING]{RESET}...")
    full_text = transcribe(RAM_WAV)
    
    if not full_text:
        print(f"{BLUE}[IGNORED]{RESET} No speech detected")
        return None
    
    # Remove wake word from transcription if present
    words = get_words(full_text)
    if words and similarity(words[0], WAKE_WORD) >= SIMILARITY_THRESHOLD:
        words = words[1:]
    
    full_command = " ".join(words) if words else full_text
    
    if len(full_command.strip()) < 2:
        print(f"{BLUE}[IGNORED]{RESET} Too short")
        return None
    
    print(f"{CYAN}YOU:{RESET} {full_command}")
    
    # Query Ollama
    print(f"{YELLOW}[THINKING]{RESET}...")
    response_text = ""
    
    for token in query_ollama_stream(full_command):
        if token is None:
            break
        response_text += token
    
    if response_text:
        print(f"{GREEN}JARVIS:{RESET} {response_text}")
        print()
        
        # Speak response
        print(f"{MAGENTA}[SPEAKING]{RESET}...")
        speak_text(response_text)
        # Drain mic buffer so we don't hear ourselves
        drain_mic(process)
        print()
    
    return response_text


def conversation_loop(process, audio_buffer):
    """
    Conversation mode: keep listening and responding until extended silence.
    No wake word needed after the first activation.
    """
    empty_count = 0
    max_empty = 2  # Exit conversation after 2 consecutive empty inputs
    
    while True:
        print(f"{YELLOW}[CONVERSATION]{RESET} Listening... (say nothing for {CONVERSATION_TIMEOUT}s to exit)")
        
        # Record with a longer silence limit to be patient
        recording = record_until_silence(process, audio_buffer, play_ping=False)
        
        if not save_wav(recording, RAM_WAV):
            continue
        
        print(f"{YELLOW}[TRANSCRIBING]{RESET}...")
        full_text = transcribe(RAM_WAV)
        
        if not full_text or len(full_text.strip()) < 2:
            empty_count += 1
            if empty_count >= max_empty:
                print(f"{BLUE}[INFO]{RESET} No more input detected, exiting conversation mode.")
                return
            print(f"{BLUE}[...]{RESET} Didn't catch that, still listening...")
            continue
        
        # Got speech - reset empty counter
        empty_count = 0
        
        # Remove wake word if user said it again out of habit
        words = get_words(full_text)
        if words and similarity(words[0], WAKE_WORD) >= SIMILARITY_THRESHOLD:
            words = words[1:]
        
        full_command = " ".join(words) if words else full_text
        
        if len(full_command.strip()) < 2:
            continue
        
        print(f"{CYAN}YOU:{RESET} {full_command}")
        
        # Query Ollama
        print(f"{YELLOW}[THINKING]{RESET}...")
        response_text = ""
        
        try:
            for token in query_ollama_stream(full_command):
                if token is None:
                    break
                response_text += token
        except Exception as e:
            print(f"{BLUE}[ERROR]{RESET} Failed to get response: {e}")
            continue
        
        if response_text:
            print(f"{GREEN}JARVIS:{RESET} {response_text}")
            print()
            print(f"{MAGENTA}[SPEAKING]{RESET}...")
            speak_text(response_text)
            # Drain mic buffer so we don't hear ourselves
            drain_mic(process)
            print()
        
        audio_buffer.clear()


def main():
    """Main event loop."""
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_processes)
    
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
    print(f"{BLUE}[INIT]{RESET} Loading TTS voice...")
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
            print(f"{GREEN}[OK]{RESET} Using microphone: plughw:4,0 (AIRHUG 21)")
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Failed to start microphone: {e}")
        cleanup_processes()
        return
    
    # Audio buffers
    wake_buffer = collections.deque(maxlen=int(RATE / CHUNK * WAKE_CHUNK_DURATION))
    audio_buffer = collections.deque(maxlen=int(RATE / CHUNK * PREV_AUDIO))
    
    wake_check_counter = 0
    wake_check_interval = int(RATE / CHUNK * 0.5)  # Check every 0.5 seconds
    
    print(f"{YELLOW}[LISTENING]{RESET} Say '{WAKE_WORD}' to activate...")
    print(f"{BLUE}[DEBUG]{RESET} Monitoring audio levels (threshold: {THRESHOLD})")
    print()
    
    try:
        while True:
            # Read chunk (mono, 16kHz from plughw device)
            chunk = process.stdout.read(CHUNK * 2)
            if not chunk:
                print(f"{BLUE}[DEBUG]{RESET} No audio data received, exiting...")
                break
            
            rms = get_rms(chunk)
            
            # Add to wake word detection buffer
            wake_buffer.append(chunk)
            audio_buffer.append(chunk)
            wake_check_counter += 1
            
            # Check for wake word periodically (every 0.5 seconds) once buffer is full
            if (len(wake_buffer) >= int(RATE / CHUNK * WAKE_CHUNK_DURATION) and 
                wake_check_counter >= wake_check_interval):
                wake_check_counter = 0
                
                # Process wake word detection chunk
                wake_chunk = list(wake_buffer)
                matched, command_text = detect_wake_word(wake_chunk)
                
                if matched:
                    print(f"{GREEN}[WAKE WORD DETECTED]{RESET}")
                    
                    try:
                        # Process the first command (with ping sound)
                        result = process_command(process, audio_buffer)
                        
                        if result:
                            # Enter conversation mode - keep talking without wake word
                            print(f"{GREEN}[CONVERSATION MODE]{RESET} Keep talking, no need to say '{WAKE_WORD}'")
                            conversation_loop(process, audio_buffer)
                        
                    except Exception as e:
                        print(f"{BLUE}[ERROR]{RESET} Error processing command: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Back to wake word mode
                    wake_buffer.clear()
                    audio_buffer.clear()
                    wake_check_counter = 0
                    
                    print(f"{YELLOW}[LISTENING]{RESET} Say '{WAKE_WORD}' to activate...")
                    print()
    
    except KeyboardInterrupt:
        print(f"\n{BLUE}[SHUTDOWN]{RESET} Stopping...")
    except Exception as e:
        print(f"\n{BLUE}[ERROR]{RESET} Unexpected error: {e}")
    finally:
        cleanup_processes()


if __name__ == "__main__":
    main()
