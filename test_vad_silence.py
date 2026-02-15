#!/usr/bin/env python3
"""
Test different SILENCE_LIMIT values for VAD.
Say the SAME command each time (e.g. "Jarvis light on").
Saves each clip with the silence limit in the filename.
Play them back to find which value captures your full command without cutting off.
"""

import os
import sys
import struct
import math
import time
import wave
import collections
import subprocess
import re

# Same config as jarvis
RATE = 16000
CHUNK = 1024
THRESHOLD = 1000
PRE_AUDIO = 0.25

# Test range: 0 to 1.5 seconds (0 = cuts on first quiet chunk)
SILENCE_LIMITS = [0, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "vad_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_mic():
    try:
        r = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return "default"
        for line in r.stdout.splitlines():
            m = re.search(r"card (\d+):.*device (\d+):", line)
            if m:
                return f"plughw:{m.group(1)},{m.group(2)}"
    except Exception:
        pass
    return "default"


def get_rms(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"<{n}h", data)
    return math.sqrt(sum(s * s for s in shorts) / n)


def save_wav(chunks: list, path: str):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(chunks))


def wait_for_speech(mic, silence_limit: float, timeout: float = 15) -> list:
    pre = collections.deque(maxlen=int(RATE / CHUNK * PRE_AUDIO))
    chunks = []
    speaking = False
    silence_start = None
    wait_start = time.time()
    while True:
        chunk = mic.stdout.read(CHUNK * 2)
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
            if timeout and time.time() - wait_start >= timeout:
                return []
    return chunks


def main():
    mic_device = detect_mic()
    print(f"Mic: {mic_device}\n")
    print(f"Testing {len(SILENCE_LIMITS)} silence limits: {SILENCE_LIMITS}")
    print("Say the SAME command each time (e.g. 'Jarvis light on')\n")

    mic = subprocess.Popen(
        ["arecord", "-D", mic_device, "-f", "S16_LE", "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    time.sleep(0.3)
    if mic.poll() is not None:
        print("Mic failed:", mic.stderr.read().decode())
        return 1

    # Discard USB init spike
    for _ in range(int(RATE / CHUNK)):
        mic.stdout.read(CHUNK * 2)

    for i, limit in enumerate(SILENCE_LIMITS):
        print(f"\n[{i+1}/{len(SILENCE_LIMITS)}] Silence limit = {limit}s — say your command NOW...")
        chunks = wait_for_speech(mic, silence_limit=limit)
        if not chunks:
            print(f"  (no speech / timeout)")
            continue
        dur = len(chunks) * CHUNK / RATE
        fname = f"silence_{limit:.2f}s.wav"
        path = os.path.join(OUTPUT_DIR, fname)
        save_wav(chunks, path)
        print(f"  Saved {dur:.1f}s → {path}")

    mic.terminate()
    try:
        mic.wait(timeout=2)
    except subprocess.TimeoutExpired:
        mic.kill()

    print(f"\nDone. Clips saved to: {OUTPUT_DIR}/")
    print("Play them back (e.g. paplay vad_test/silence_0.50s.wav) to compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
