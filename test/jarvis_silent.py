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

# --- CONFIGURATION ---
WHISPER_THREADS = "1"
THRESHOLD = 2000              # Adjusted for your cleaner audio
SILENCE_LIMIT = 2
PREV_AUDIO = 0.5
PC_IP = "10.0.0.224"
MODEL = "gemma3:12b"

# --- PROMPT ---
SYSTEM_PROMPT = (
    "You are Jarvis. You are running in 'Console Mode'. "
    "Your responses should be concise, helpful, and text-based. "
    "Do not use markdown formatting."
)

# --- PATHS ---
WHISPER_BIN = "./main"
if not os.path.exists(WHISPER_BIN): WHISPER_BIN = "./build/bin/whisper-cli"

WHISPER_MODEL = "models/ggml-base.en.bin"
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
RESET = "\033[0m"

def get_rms(data):
    count = int(len(data) / 2)
    format = "<%dh" % (count)
    shorts = struct.unpack(format, data)
    sum_squares = 0.0
    for sample in shorts:
        sum_squares += sample * sample
    return math.sqrt(sum_squares / count)

def save_wav(audio_data):
    with wave.open(RAM_WAV, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))

def transcribe():
    if not os.path.exists(RAM_WAV): return ""
    
    # DEBUG MODE: removed --no-prints so we can see errors
    cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", RAM_WAV, "-ac", "512", "-t", WHISPER_THREADS, "--no-timestamps"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # DEBUG PRINTING
    if result.returncode != 0:
        print(f"\n{BLUE}[DEBUG] CRASH: {result.stderr}{RESET}")
        return ""
    
    text = result.stdout.strip()
    if not text:
        print(f"\n{BLUE}[DEBUG] HEARD SILENCE (Check Mic Volume){RESET}")
        
    return text

def ask_ollama(prompt):
    url = f"http://{PC_IP}:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True 
    }
    
    print(f"{GREEN}JARVIS:{RESET} ", end="", flush=True)
    
    try:
        with requests.post(url, json=payload, stream=True, timeout=30) as response:
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    word = chunk.get("response", "")
                    print(word, end="", flush=True)
            print("\n" + "-"*50)
            
    except Exception as e:
        print(f"\n{BLUE}[ERROR]{RESET} Connection Failed: {e}")

def main():
    print(f"{BLUE}=== JARVIS CONSOLE MODE (DEBUG) ==={RESET}")
    print(f"{BLUE}   * Model:        {MODEL}{RESET}")
    print(f"{BLUE}   * Mic Input:    BOOSTED (Card 2){RESET}")
    print(f"{BLUE}   * Threshold:    {THRESHOLD}{RESET}")

    record_cmd = ["arecord", "-D", "plug:boosted", "-f", "S16_LE", "-c", "1", "-r", "16000", "-t", "raw", "-q"]
    try:
        process = subprocess.Popen(record_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"{BLUE}[ERROR]{RESET} Mic fail: {e}")
        return

    audio_buffer = collections.deque(maxlen=int(RATE / CHUNK * PREV_AUDIO))
    recording = []
    silence_start = None
    is_recording = False
    
    print(f"{YELLOW}[LISTENING]...{RESET}")
    
    try:
        while True:
            chunk = process.stdout.read(CHUNK * 2)
            if not chunk: break
            rms = get_rms(chunk)
            
            if is_recording:
                recording.append(chunk)
                if rms < THRESHOLD:
                    if silence_start is None: silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_LIMIT:
                        is_recording = False
                        print(f"\n{YELLOW}[THINKING]...{RESET}")
                        
                        save_wav(recording)
                        text = transcribe()
                        
                        if len(text) > 2:
                            print(f"{CYAN}YOU:{RESET} {text}")
                            ask_ollama(text)
                        else:
                            print(f"{BLUE}[IGNORED] (No text found){RESET}")
                            
                        print(f"{YELLOW}[LISTENING]...{RESET}")
                        recording = []
                        audio_buffer.clear()
                else:
                    silence_start = None
            else:
                audio_buffer.append(chunk)
                if rms > THRESHOLD:
                    print(f"\r{BLUE}[RECORDING]{RESET} Voice detected!   ", end="")
                    is_recording = True
                    silence_start = None
                    recording.extend(audio_buffer)
                    recording.append(chunk)

    except KeyboardInterrupt:
        process.kill()

if __name__ == "__main__":
    main()
