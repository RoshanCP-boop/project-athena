"""
Live VAD test. Captures mic audio continuously, detects speech vs silence,
and prints an indicator so you can see the VAD reacting to your voice.

Tuning knobs:
  VAD_AGGRESSIVENESS -- 0 (most permissive) to 3 (most strict).
  FRAME_MS           -- 10, 20, or 30 ms. Bigger = smoother, more latency.
"""

import sys
import numpy as np
import sounddevice as sd
import webrtcvad

MIC_INDEX = None
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
VAD_AGGRESSIVENESS = 2

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

print(f"Listening on device {MIC_INDEX}. Speak to see activity. Ctrl-C to exit.\n")

def callback(indata, frames, time_info, status):
    if status:
        print(f"[audio status] {status}", file=sys.stderr)
    pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
    is_speech = vad.is_speech(pcm16, SAMPLE_RATE)
    rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
    bar = "#" * min(40, int(rms * 400))
    tag = "SPEECH " if is_speech else "silence"
    print(f"\r{tag} | {bar:<40} | rms={rms:.4f}", end="", flush=True)

try:
    with sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=callback,
    ):
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("\nStopped.")
