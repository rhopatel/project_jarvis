#!/usr/bin/env python3
"""
Test wake-word similarity scores. Run this to see what scores you actually get
when saying "Jarvis" vs when moving around (false positives).

Usage: ./test_wake_scores.sh

Say "Jarvis" several times, then move around the room (don't speak). Compare
the max scores: you want threshold above the false-positive scores but below
your real "Jarvis" scores. Then set SIMILARITY_THRESH in jarvis.py.
"""
import os
import subprocess
import sys
import time

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvis import (
    WAKE_WORD,
    SIMILARITY_THRESH,
    MIC_DEVICE,
    RATE,
    CHUNK,
    RAM_WAV,
    similarity,
    save_wav,
    transcribe,
    wait_for_speech,
    _trim_trailing_silence,
    _get_wav_duration,
)
import jarvis

# Disable verbose debug
jarvis.DEBUG = 0


def main():
    print("Wake-word score test")
    print(f"  Threshold in jarvis.py: {SIMILARITY_THRESH}")
    print(f"  Mic: {MIC_DEVICE}")
    print()
    print("Say 'Jarvis' several times, then move around (no speech) to capture false positives.")
    print("Ctrl+C when done.")
    print()

    # Init mic (same as jarvis main)
    jarvis._mic = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-c", "1", "-r", str(RATE), "-t", "raw", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    time.sleep(0.2)
    if jarvis._mic.poll() is not None:
        print("ERROR: Microphone failed")
        return
    for _ in range(int(RATE / CHUNK)):
        jarvis._mic.stdout.read(CHUNK * 2)

    n = 0
    try:
        while True:
            n += 1
            print(f"[{n}] Listening... ", end="", flush=True)
            chunks = wait_for_speech()
            if not chunks:
                print("(stream end)")
                break
            save_wav(chunks, RAM_WAV)
            path = _trim_trailing_silence(RAM_WAV)
            dur = _get_wav_duration(path)
            text, _ = transcribe(path)
            try:
                os.remove(RAM_WAV)
            except OSError:
                pass

            if not text:
                print(f"empty (duration {dur:.1f}s)")
                continue

            words = text.lower().split()
            scores = []
            for w in words:
                clean = w.rstrip(".,!?;:")
                if clean:
                    s = similarity(clean, WAKE_WORD)
                    scores.append((clean, s))

            max_score = max((s for _, s in scores), default=0)
            would_trigger = max_score >= SIMILARITY_THRESH

            print(f"{dur:.1f}s  \"{text}\"")
            for word, sc in scores:
                bar = "***" if sc >= SIMILARITY_THRESH else "   "
                print(f"      {bar} '{word}' -> {sc:.3f}")
            print(f"      max={max_score:.3f}  threshold={SIMILARITY_THRESH}  TRIGGER={would_trigger}")
            print()
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        if jarvis._mic and jarvis._mic.poll() is None:
            jarvis._mic.terminate()
            jarvis._mic.wait()


if __name__ == "__main__":
    main()
