#!/bin/bash
# Wrapper script to run Jarvis Voice Assistant
# Automatically activates virtual environment and runs jarvis.py

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create it with: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set LD_LIBRARY_PATH for whisper shared libraries
WHISPER_LIB_DIR="$SCRIPT_DIR/../whisper.cpp/build/src"
GGML_LIB_DIR="$SCRIPT_DIR/../whisper.cpp/build/ggml/src"

if [ -d "$WHISPER_LIB_DIR" ] && [ -d "$GGML_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$WHISPER_LIB_DIR:$GGML_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Force CPU-only execution for Piper TTS (no GPU)
export CUDA_VISIBLE_DEVICES=""

# Ensure Python output is not buffered (important for non-TTY environments)
export PYTHONUNBUFFERED=1

# CPU performance mode (skip if no sudo)
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$gov" ] && echo performance 2>/dev/null | sudo tee "$gov" >/dev/null
done

# Boost mic capture volume (tries first few ALSA cards)
for c in 0 1 2 3 4 5; do
    amixer -c $c set Capture 100% 2>/dev/null && break
    amixer -c $c set Mic 100% 2>/dev/null && break
done

# Check if jarvis.py exists
if [ ! -f "jarvis.py" ]; then
    echo "Error: jarvis.py not found in $SCRIPT_DIR"
    exit 1
fi

# Run jarvis.py with all arguments passed to this script
python jarvis.py "$@"
