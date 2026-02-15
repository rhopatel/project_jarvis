#!/bin/bash
# Run wake-word similarity score test. See what scores you get when saying
# "Jarvis" vs when moving around (false positives). Use results to tune
# SIMILARITY_THRESH in jarvis.py.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

[ ! -d "venv" ] && echo "Error: venv not found" && exit 1
source venv/bin/activate

WHISPER_LIB_DIR="$SCRIPT_DIR/../whisper.cpp/build/src"
GGML_LIB_DIR="$SCRIPT_DIR/../whisper.cpp/build/ggml/src"
[ -d "$WHISPER_LIB_DIR" ] && [ -d "$GGML_LIB_DIR" ] && \
  export LD_LIBRARY_PATH="$WHISPER_LIB_DIR:$GGML_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

for c in 0 1 2 3 4 5; do
  amixer -c $c set Capture 100% 2>/dev/null >/dev/null && break
  amixer -c $c set Mic 100% 2>/dev/null >/dev/null && break
done

python test/test_wake_word_scores.py
