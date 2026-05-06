"""
Wake-word threshold calibration helper.

Standalone tool that loads the same openWakeWord model app.py uses,
listens to the mic, and prints the peak score every 500ms. Use it to
figure out the right value for SAGE_WAKE_THRESHOLD in your .env.

Procedure:
  1. Run the app: `python wake_calibrate.py`
     (uses SAGE_WAKE_MODEL from .env, or pass --model /path/to.onnx)
  2. Stay silent for 5 seconds. Note the peak — call it `noise_floor`.
     If it's above ~0.05 your room is noisy / mic is hot.
  3. Say "hey sage" 8-10 times, varying volume and distance the way
     you'd actually use it. Note the *lowest* score that "hey sage"
     produces — call it `min_hit`.
  4. Set SAGE_WAKE_THRESHOLD halfway between noise_floor and min_hit
     (and at least 0.05 below min_hit). Restart app.py.

Examples (mine):
  noise_floor=0.04, min_hit=0.55  ->  threshold=0.45 (good headroom)
  noise_floor=0.10, min_hit=0.40  ->  threshold=0.30 (custom model
    needs lower threshold; consider retraining with more samples)

Tips for better wake detection:
  - Distance: 30-60 cm is the sweet spot; closer can clip, farther
    drops too much signal.
  - Cadence: say "hey sage" as one phrase ("HEY-sage"), not two
    separate words with a pause. Short, crisp.
  - Custom-model trick: most colab-trained models match best at the
    cadence the training audio was recorded at. If yours sounds clipped
    or rushed, mimic it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("SAGE_WAKE_MODEL", "hey_jarvis"),
        help="openWakeWord built-in name OR absolute path to a .onnx/.tflite file.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Sounddevice input index (default 0)."
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.05,
        help="Suppress scores below this in the live log (default 0.05).",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.5,
        help="Seconds between printed peak lines (default 0.5).",
    )
    args = parser.parse_args()

    print(f"[calibrate] loading wake model: {args.model}", file=sys.stderr)
    import openwakeword
    from openwakeword.model import Model as OWWModel

    if os.path.isabs(args.model) and os.path.exists(args.model):
        oww = OWWModel(wakeword_model_paths=[args.model])
    else:
        builtins = openwakeword.models
        if args.model not in builtins:
            print(
                f"[calibrate] model {args.model!r} not found. Built-ins: "
                f"{sorted(builtins)}",
                file=sys.stderr,
            )
            sys.exit(1)
        oww = OWWModel(wakeword_model_paths=[builtins[args.model]["model_path"]])

    key = next(iter(oww.models))
    print(f"[calibrate] active key={key}", file=sys.stderr)

    # Same audio config as app.py: 48kHz capture, downsample to 16k for oww.
    capture_rate = 48000
    vad_rate = 16000
    decim = capture_rate // vad_rate
    frame_ms = 30
    capture_frame = capture_rate * frame_ms // 1000

    state = {
        "high": 0.0,
        "high_t": time.monotonic(),
        "session_high": 0.0,
        "session_low_after_first_hit": 1.0,
        "hits": 0,
    }

    def cb(indata, frames, time_info, status) -> None:
        if status and not status.input_overflow:
            print(f"[audio] {status}", file=sys.stderr)
        mono_48 = indata[:, 0]
        if len(mono_48) % decim != 0:
            mono_48 = mono_48[: (len(mono_48) // decim) * decim]
        mono_16 = mono_48.reshape(-1, decim).mean(axis=1)
        pcm16 = np.clip(mono_16 * 32767, -32768, 32767).astype(np.int16)
        try:
            scores = oww.predict(pcm16)
        except Exception as e:
            print(f"[predict] {e}", file=sys.stderr)
            return
        score = float(scores.get(key, 0.0))
        if score > state["high"]:
            state["high"] = score
        if score > state["session_high"]:
            state["session_high"] = score

    print(
        "\n[calibrate] Speak 'hey sage' a few times. Press Ctrl-C to "
        "see the summary.\n",
        file=sys.stderr,
    )
    with sd.InputStream(
        device=args.device,
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        blocksize=capture_frame,
        callback=cb,
        latency="high",
    ):
        try:
            last_print = time.monotonic()
            while True:
                time.sleep(args.print_every)
                now = time.monotonic()
                peak = state["high"]
                state["high"] = 0.0
                if peak >= args.floor:
                    bar = "#" * int(peak * 40)
                    marker = " <-- HIT (above 0.50 by default)" if peak >= 0.5 else ""
                    print(f"  peak={peak:0.3f} |{bar:<40}|{marker}")
                last_print = now
        except KeyboardInterrupt:
            print("\n", file=sys.stderr)
            print(
                f"[calibrate] session peak score: {state['session_high']:.3f}",
                file=sys.stderr,
            )
            print(
                "[calibrate] suggested SAGE_WAKE_THRESHOLD: "
                f"{max(0.10, state['session_high'] - 0.10):.2f}  "
                "(0.10 below your peak)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
