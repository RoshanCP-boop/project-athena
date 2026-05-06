# Sage — Handheld Local-First AI Assistant

A privacy-focused voice assistant running on a Raspberry Pi 5 with optional
cloud "deep-think" mode using groq.

## What it does

- **Wake-word detection** ("hey jarvis") via [openWakeWord](https://github.com/dscripka/openWakeWord)
- **Speech-to-text** via [whisper.cpp](https://github.com/ggerganov/whisper.cpp) running as a local HTTP server, or Groq's `whisper-large-v3-turbo` in online mode
- **LLM inference** via local [Ollama](https://ollama.ai) (`qwen2.5:1.5b` by default), or Groq's `llama-3.3-70b-versatile` in online mode
- **Text-to-speech** via local [Piper](https://github.com/rhasspy/piper) (`en_US-ryan-medium`), or Groq's Orpheus in online mode
- **Tool calling** for time, calculator, timers, long-term memory, and physical hardware (servo gestures, LED ring once wired)
- **Long-term memory** in a local SQLite database, with auto-summarized conversation history
- **Barge-in interruption** — say "hey jarvis" or tap the screen mid-reply to cut Sage off
- **pygame UI** that runs on the attached touchscreen

## Hardware

- Raspberry Pi 5 (8 GB) with NVMe SSD
- Adafruit INA219 power monitor (I2C) for future battery / power-adaptive model switching
- SG90 micro servo on hardware PWM (GPIO 18) for embodied "head" gestures
- WS2812 12-LED NeoPixel ring (planned, on SPI MOSI / GPIO 10)
- DS-213K buttons (planned, GPIO inputs for the deep-think toggle)
- USB microphone (C-Media PCM2902, USB 1.1)
- 5" DSI touchscreen
- AB13X USB audio dongle + PAM8403 amp + 40 mm speaker (planned, replaces HDMI audio for portable use)

## Project layout

```
app.py                 # main pygame UI + voice pipeline
listen.py              # CLI version of the same pipeline (no GUI)
hardware.py            # Servo + LedRing singletons with cooperative-cancel actions
tools.py               # Local tool implementations + Ollama function-call schemas
memory.py              # SQLite-backed long-term memory
cloud.py               # Groq adapter for online LLM/STT/TTS
wake_calibrate.py      # Standalone wake-word threshold calibration helper
power_test.py          # INA219 sanity-check script
servo_test.py          # Servo sweep test
vad_test.py            # Voice-activity-detection sanity test
```

## Setup (high-level)

1. Flash Raspberry Pi OS (Bookworm), enable I2C and SPI via `raspi-config`.
2. Install Ollama, pull `qwen2.5:1.5b` and `qwen2.5:0.5b`.
3. Build whisper.cpp into `~/sage/vendor/whisper.cpp/` and start `whisper-server` as a systemd user service.
4. `pip install -r requirements.txt` (TODO: add this — for now see imports in `app.py`).
5. Download a Piper voice into `~/sage/vendor/piper-voices/` (default: `en_US-ryan-medium`).
6. Copy `.env.example` → `.env` and fill in `GROQ_API_KEY` if you want online mode.
7. `DISPLAY=:0 python app.py`

## License

Personal academic project. No license declared.
