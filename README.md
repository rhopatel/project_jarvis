# Jarvis Voice Assistant

A wake-word activated personal assistant running on Raspberry Pi, powered by:
- **whisper.cpp** for speech-to-text transcription
- **Ollama** (on PC) for LLM processing
- **Piper TTS** for text-to-speech

## Features

- Wake word detection ("Jarvis")
- Continuous listening with VAD (Voice Activity Detection)
- Real-time transcription using whisper.cpp
- LLM processing via Ollama API
- Natural voice output using Piper TTS

## Prerequisites

### Hardware
- Raspberry Pi with microphone (using `plug:boosted` ALSA device)
- PC running Ollama server accessible on network

### Software
- Python 3.8+
- whisper.cpp built and accessible
- Ollama running on PC at `10.0.0.224:11434`
- ALSA audio system configured

## Setup

### 1. Clone and Setup Environment

```bash
cd /home/rohan/Documents/project_jarvis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Whisper.cpp

Ensure whisper.cpp is built and the binary is accessible:
- Binary: `./build/bin/whisper-cli` or `./main`
- Model: `models/ggml-base.en.bin`

Jarvis uses `base.en` (better accuracy than tiny). In whisper.cpp:
```bash
./models/download-ggml-model.sh base.en
# Optional: quantize for speed (smaller, faster). Build first: cmake --build build
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q4_0.bin q4_0
```
Also try pre-built quantized: `./models/download-ggml-model.sh base.en-q5_1`. Falls back to tiny if base missing.

### 3. Configure Ollama

Default: `gemma3:12b`. Update in `jarvis.py` if needed:
```python
OLLAMA_IP = "10.0.0.224"   # Your PC's IP
OLLAMA_MODEL = "gemma3:12b"
```

### 4. Setup Piper TTS

Piper TTS downloads voices on first use. Default: `en_US-ryan-high` (best quality). Change in `jarvis.py`:
```python
PIPER_VOICE = "en_US-ryan-high"
```

### 5. (Optional) Mic Calibration

If transcription is off, calibrate your mic levels:
```bash
python calibrate_mic.py
```
Speak for 5 seconds; it suggests a THRESHOLD value for `jarvis.py`.

### 6. (Optional) Set CPU Performance Mode

For better responsiveness, set CPU governor to performance mode:
```bash
sudo ./set_performance_mode.sh
```

Or manually:
```bash
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee "$cpu"
done
```

## Usage

### Quick Start (Recommended)
Use the wrapper script which automatically activates the virtual environment:
```bash
./run_jarvis.sh
```

### Manual Start
If you prefer to activate the environment manually:
```bash
source venv/bin/activate
python jarvis.py
```

Or run directly (if venv is already activated):
```bash
./jarvis.py
```

### How It Works

1. **Listening**: Jarvis continuously listens for the wake word "Jarvis"
2. **Wake Word Detection**: When detected, it starts recording your command
3. **Recording**: Records until 2 seconds of silence
4. **Transcription**: Uses whisper.cpp to transcribe your speech
5. **Processing**: Sends text to Ollama on your PC for LLM processing
6. **Response**: Speaks the response using Piper TTS

### Example Interaction

```
[LISTENING] Say 'Jarvis' to activate...

[WAKE WORD DETECTED]
[RECORDING] Listening for command....
YOU: what's the weather like today

[THINKING]...
JARVIS: I don't have access to real-time weather data...

[SPEAKING]...
```

## Deployment (Auto-start on Boot)

To run Jarvis automatically when the Raspberry Pi boots:

```bash
sudo ./install_service.sh
sudo systemctl enable jarvis
sudo systemctl start jarvis
```

- **Status**: `sudo systemctl status jarvis`
- **Logs**: `journalctl -u jarvis -f`
- **Stop**: `sudo systemctl stop jarvis`
- **Disable**: `sudo systemctl disable jarvis` (stops auto-start)

The service waits for network (Kasa, Ollama) and audio before starting, and restarts if it exits.

### Raspberry Pi LED Indicator

- **Green** = ready to listen (say "Jarvis" to activate)
- **Red** = thinking or speaking (wait until green to talk again)
- On exit, both LEDs revert to normal

**LED control requires root.** Run `sudo ./run_jarvis.sh` for LED status. The systemd service runs as your user for audio, so the LED won't change unless the service is configured to run as root.

### Systemd: Disable, Restart, Reinstall

| Goal | Commands |
|------|----------|
| **Stop and disable** (no auto-start) | `sudo systemctl stop jarvis` then `sudo systemctl disable jarvis` |
| **Apply code changes** (no reinstall) | `sudo systemctl restart jarvis` |
| **Reinstall** (new paths/user, updated service file) | `sudo ./install_service.sh` then `sudo systemctl daemon-reload` and `sudo systemctl restart jarvis` |

## Configuration

Key settings in `jarvis.py`:

```python
# Wake word
WAKE_WORD = "Jarvis"
SIMILARITY_THRESH = 0.6    # Wake word match threshold (0.0-1.0)

# Audio
THRESHOLD = 500            # RMS threshold for VAD (calibrate with calibrate_mic.py)
SILENCE_LIMIT = 1.1        # Seconds of silence to stop recording
SILENCE_LIMIT_CONV = 1.5   # Silence limit in conversation mode

# Ollama
OLLAMA_IP = "10.0.0.224"
OLLAMA_MODEL = "gemma3:12b"

# TTS
PIPER_VOICE = "en_US-ryan-high"
```

Whisper uses auto thread count (up to 4). Use `calibrate_mic.py` to find the right THRESHOLD for your mic.

## Troubleshooting

### Microphone Not Working
- Check ALSA device: `arecord -l`
- Verify `plug:boosted` device exists
- Test recording: `arecord -D plug:boosted -f S16_LE -r 16000 test.wav`

### Whisper Not Found
- Ensure whisper.cpp is built: `cd ../whisper.cpp && make`
- Update `WHISPER_BIN` path in `jarvis.py`

### Ollama Connection Failed
- Verify Ollama is running on PC: `curl http://10.0.0.224:11434/api/tags`
- Check firewall settings
- Update `OLLAMA_IP` in jarvis.py if different

### Service Runs but No Audio
The service runs as your user with `XDG_RUNTIME_DIR` for PipeWire. Ensure you're logged in (desktop session) when the service starts. If audio still fails, check `journalctl -u jarvis -f` for errors.

### TTS Not Working
- Install espeak as fallback: `sudo apt-get install espeak espeak-data`
- Check Piper TTS installation: `python -m piper_tts --help`
- Voice models download automatically on first use

### High Latency
- Set CPU to performance mode (see above)
- Reduce `WAKE_CHUNK_DURATION` for faster wake detection
- Use smaller whisper model (base.en instead of medium.en)

## Files

- `jarvis.py` - Main voice assistant script
- `run_jarvis.sh` - Wrapper script (auto-activates venv and runs jarvis.py)
- `jarvis.service` - systemd unit for auto-start on boot
- `install_service.sh` - Installs the systemd service
- `test/jarvis_silent.py` - Console-only reference version (legacy)
- `requirements.txt` - Python dependencies
- `set_performance_mode.sh` - CPU performance mode script
- `README.md` - This file

## Notes

- Audio files are stored in `/dev/shm/` (RAM disk) for speed
- Wake word detection uses similarity matching (not exact match)
- System prompt is optimized for voice responses (concise, natural)
- TTS falls back to espeak if Piper fails

## License

See individual component licenses:
- whisper.cpp: MIT
- Ollama: MIT
- Piper TTS: MIT
