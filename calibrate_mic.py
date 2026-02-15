#!/usr/bin/env python3
"""
Microphone calibration helper for Jarvis.
Records 5 seconds, analyzes levels, suggests adjustments.

Run: python calibrate_mic.py
Uses same mic device as jarvis.py.
"""
import os
import sys
import struct
import math
import subprocess
import wave
import tempfile

# Reuse jarvis config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jarvis import MIC_DEVICE, RATE, CHUNK, THRESHOLD


def get_rms(data: bytes) -> float:
    n = len(data) // 2
    if n == 0:
        return 0.0
    shorts = struct.unpack(f"<{n}h", data)
    return math.sqrt(sum(s * s for s in shorts) / n)


def main():
    print("Jarvis Mic Calibration")
    print(f"  Device: {MIC_DEVICE}")
    print(f"  Rate: {RATE} Hz")
    print()
    print("Speak at normal volume for 5 seconds...")
    print("(Say a few sentences, then stay silent)")
    print()

    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
    )
    chunks = []
    n_chunks = int(5 * RATE / CHUNK)
    for i in range(n_chunks):
        data = proc.stdout.read(CHUNK * 2)
        if not data:
            break
        chunks.append(data)
    proc.terminate()
    proc.wait()

    if not chunks:
        print("No audio captured. Check mic connection and device.")
        return 1

    # Analyze
    all_data = b"".join(chunks)
    rms_overall = get_rms(all_data)
    # Per-chunk RMS to see dynamics
    rms_list = [get_rms(c) for c in chunks]
    rms_max = max(rms_list)
    rms_avg_speech = sum(r for r in rms_list if r > 500) / max(1, sum(1 for r in rms_list if r > 500))
    rms_min_silence = min(r for r in rms_list if r < 2000) if any(r < 2000 for r in rms_list) else 0

    print("Results:")
    print(f"  Overall RMS:     {rms_overall:.0f}")
    print(f"  Peak (loudest):  {rms_max:.0f}")
    print(f"  Avg when speech: {rms_avg_speech:.0f}" if rms_avg_speech else "  (no clear speech detected)")
    print(f"  Silence floor:   {rms_min_silence:.0f}")
    print()
    print(f"  Current THRESHOLD in jarvis.py: {THRESHOLD}")
    print()
    suggested = int(rms_avg_speech * 0.3) if rms_avg_speech else 800
    if suggested < 500:
        suggested = 500
    if suggested > 3000:
        suggested = 3000
    print(f"  Suggested THRESHOLD: {suggested} (edit jarvis.py THRESHOLD = {suggested})")
    print()
    print("  If speech is weak, raise mic volume: amixer -c N set Capture 90%")
    print("  List capture devices: arecord -l")
    return 0


if __name__ == "__main__":
    sys.exit(main())
