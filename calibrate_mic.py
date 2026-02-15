#!/usr/bin/env python3
"""
Microphone calibration helper for Jarvis.
Records ambient noise, then speech, and suggests a THRESHOLD that works in your environment.

IMPORTANT: Run this with your typical background noise (heater, AC, etc.) so the
threshold is set above ambient. Otherwise Jarvis may not hear you over the noise.
"""
import os
import sys
import struct
import math
import subprocess

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
    print("Run this with your heater/AC/etc. ON so the threshold works in noisy conditions.")
    print()
    print("Phase 1: Stay SILENT for 3 seconds (heater/AC on - measures ambient noise).")
    input("  Press Enter to start... ")
    print("  Recording ambient...")

    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
    )
    ambient_chunks = []
    for _ in range(int(3 * RATE / CHUNK)):
        data = proc.stdout.read(CHUNK * 2)
        if not data:
            break
        ambient_chunks.append(data)
    proc.terminate()
    proc.wait()

    ambient_rms = [get_rms(c) for c in ambient_chunks]
    rms_ambient = sum(ambient_rms) / len(ambient_rms) if ambient_rms else 0
    rms_ambient_peak = max(ambient_rms) if ambient_rms else 0

    print()
    print("Phase 2: SPEAK at normal volume for 4 seconds.")
    input("  Press Enter to start... ")
    print("  Recording speech...")

    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
    )
    speech_chunks = []
    for _ in range(int(4 * RATE / CHUNK)):
        data = proc.stdout.read(CHUNK * 2)
        if not data:
            break
        speech_chunks.append(data)
    proc.terminate()
    proc.wait()

    if not speech_chunks:
        print("No audio captured. Check mic connection and device.")
        return 1

    speech_rms = [get_rms(c) for c in speech_chunks]
    rms_speech = sum(r for r in speech_rms if r > rms_ambient * 1.2) / max(
        1, sum(1 for r in speech_rms if r > rms_ambient * 1.2)
    ) if any(r > rms_ambient * 1.2 for r in speech_rms) else max(speech_rms)

    print()
    print("Results:")
    print(f"  Ambient (noise) avg: {rms_ambient:.0f}")
    print(f"  Ambient peak:        {rms_ambient_peak:.0f}")
    print(f"  Speech avg:          {rms_speech:.0f}")
    print()
    print(f"  Current THRESHOLD in jarvis.py: {THRESHOLD}")
    print()

    # Threshold must be BELOW speech or we never trigger. Above ambient_avg reduces false triggers.
    if rms_speech > rms_ambient_peak and (rms_speech - rms_ambient_peak) > 100:
        # Good separation: put threshold 25% above ambient peak
        suggested = int(rms_ambient_peak + (rms_speech - rms_ambient_peak) * 0.25)
    else:
        # Heater/noise louder than speech: threshold must be below speech to ever trigger.
        # Use 80% of speech avg to catch quieter syllables. May get false triggers from noise spikes.
        suggested = int(rms_speech * 0.8)

    suggested = max(400, min(suggested, 4000))
    # Never suggest above speech - would make you inaudible
    suggested = min(suggested, int(rms_speech * 0.95))

    print(f"  Suggested THRESHOLD: {suggested} (edit jarvis.py THRESHOLD = {suggested})")
    print()
    if rms_speech < rms_ambient_peak:
        print("  NOTE: Heater peaks are louder than your speech. You may get occasional false")
        print("  triggers from noise spikes, but threshold is set so your voice can activate Jarvis.")
    elif rms_speech < rms_ambient * 1.5:
        print("  WARNING: Speech barely above ambient. Speak louder or move mic closer.")
    print("  If speech is weak: amixer -c N set Capture 90%")
    print("  List devices: arecord -l")
    return 0


if __name__ == "__main__":
    sys.exit(main())
