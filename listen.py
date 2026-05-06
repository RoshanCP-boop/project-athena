"""
Live voice assistant in the terminal (no GUI).

Fully local: mic -> VAD -> local whisper.cpp server -> local Ollama.
Useful for debugging the speech pipeline without the pygame UI.

Prereq: whisper-server must be running, e.g.:
  cd ~/sage/vendor/whisper.cpp
  ./build/bin/whisper-server -m models/ggml-base.en.bin \
      --host 127.0.0.1 --port 8080 -t 3

Env vars:
  SAGE_LLM_TIER=high|low   pick LLM (default high = phi3.5)

Press Ctrl+C to exit.
"""

import io
import os
import queue
import re
import sys
import threading
import time
import wave

import numpy as np
import requests
import sounddevice as sd
import webrtcvad

MIC_INDEX = 0

CAPTURE_RATE = 48000
VAD_RATE = 16000
DECIMATION = CAPTURE_RATE // VAD_RATE

FRAME_MS = 30
CAPTURE_FRAME = CAPTURE_RATE * FRAME_MS // 1000

VAD_AGGRESSIVENESS = 3
RMS_GATE = 0.022

PREROLL_MS = 300
END_SILENCE_MS = 750
MIN_UTTERANCE_MS = 450
MAX_UTTERANCE_MS = 15000
MIN_VOICED_MS = 360
MIN_AVG_RMS = 0.018

PREROLL_FRAMES = PREROLL_MS // FRAME_MS
END_SILENCE_FRAMES = END_SILENCE_MS // FRAME_MS
MIN_UTTERANCE_FRAMES = MIN_UTTERANCE_MS // FRAME_MS
MAX_UTTERANCE_FRAMES = MAX_UTTERANCE_MS // FRAME_MS
MIN_VOICED_FRAMES = MIN_VOICED_MS // FRAME_MS

WAKE_WORDS = ("sage", "hey sage", "hi sage", "ok sage", "okay sage")
WAKE_RE = re.compile(
    r"^[\s,.!?:;\-—]*"
    r"(?:hey\s+|hi\s+|okay\s+|ok\s+)?sage"
    r"(?=$|[\s,.!?:;\-—])",
    re.IGNORECASE,
)
FOLLOW_UP_WINDOW_SECS = 15.0

HALLUCINATION_BRACKET_RE = re.compile(r"^\s*[\[\(][^\]\)]+[\]\)]\s*$")
HALLUCINATION_PHRASES = {
    "thank you", "thank you.", "thanks", "thanks.",
    "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.",
    "thanks for listening", "thanks for listening.",
    "please subscribe", "please subscribe.",
    "subscribe", "subscribe.",
    "like and subscribe", "like and subscribe.",
    "see you next time", "see you next time.",
    "you", ".", "..", "...",
    "okay", "okay.", "ok", "ok.",
    "yeah", "yeah.", "yes", "yes.", "no", "no.",
    "bye", "bye.", "goodbye", "goodbye.",
    "uh", "uh.", "um", "um.", "hmm", "hmm.", "mm", "mm.",
    "the end", "the end.",
    "music", "music.", "music playing",
    "applause", "laughter",
    "subtitles by the amara.org community",
    "transcribed by", "translated by",
}

WHISPER_URL = "http://127.0.0.1:8080/inference"
HTTP_TIMEOUT = 30

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TIMEOUT = 60

LLM_TIERS = {
    "high": "phi3.5",
    "low":  "qwen2.5:1.5b",
}
LLM_TIER = os.environ.get("SAGE_LLM_TIER", "high").strip().lower()
if LLM_TIER not in LLM_TIERS:
    print(f"[warn] unknown SAGE_LLM_TIER={LLM_TIER!r}, using 'high'", file=sys.stderr)
    LLM_TIER = "high"
OLLAMA_MODEL = LLM_TIERS[LLM_TIER]

SYSTEM_PROMPT = (
    "You are Sage, a friendly voice assistant running on a Raspberry Pi. "
    "The user's messages come from speech-to-text and may contain small "
    "transcription errors (wrong homophones, missing words, mangled names). "
    "Infer the user's likely intent charitably. If a message seems nonsensical "
    "or you're unsure, ask a brief clarifying question rather than guessing. "
    "Keep responses brief and conversational, usually one or two sentences. "
    "Respond in plain text with no markdown, lists, or special formatting, "
    "because your replies will be read aloud."
)
HISTORY_MAX_TURNS = 4

WHISPER_BASE_PROMPT = "A conversation with Sage, an AI assistant."
WHISPER_PROMPT_MAX_CHARS = 200

audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
overflow_count = 0
history: list[dict] = []
follow_up_until: float = 0.0


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    if HALLUCINATION_BRACKET_RE.match(t):
        return True
    if t in HALLUCINATION_PHRASES:
        return True
    return False


def strip_wake_word(text: str):
    m = WAKE_RE.match(text)
    if not m:
        return None
    rest = text[m.end():]
    return re.sub(r"^[\s,.!?:;\-—]*", "", rest).strip()


def normalize_gain(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(samples)))
    if peak < 0.005:
        return samples
    if peak >= target_peak:
        return samples
    return samples * (target_peak / peak)


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = VAD_RATE) -> bytes:
    pcm16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def build_whisper_prompt() -> str:
    parts = [WHISPER_BASE_PROMPT]
    for msg in history[-2:]:
        role = "User" if msg["role"] == "user" else "Sage"
        parts.append(f"{role}: {msg['content']}")
    prompt = " ".join(parts)
    if len(prompt) > WHISPER_PROMPT_MAX_CHARS:
        prompt = prompt[-WHISPER_PROMPT_MAX_CHARS:]
    return prompt


def transcribe(samples: np.ndarray) -> str:
    samples = normalize_gain(samples)
    wav_bytes = samples_to_wav_bytes(samples)
    prompt = build_whisper_prompt()
    try:
        r = requests.post(
            WHISPER_URL,
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            data={
                "response_format": "json",
                "temperature": "0",
                "prompt": prompt,
            },
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except requests.RequestException as e:
        print(f"[whisper error] {e}", file=sys.stderr)
        return ""


def ask_ollama(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    trimmed = history[-HISTORY_MAX_TURNS * 2:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 150, "num_thread": 3},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
    except requests.RequestException as e:
        history.pop()
        return f"[ollama error: {e}]"
    history.append({"role": "assistant", "content": reply})
    return reply


def audio_callback(indata, frames, time_info, status):
    global overflow_count
    if status:
        if status.input_overflow:
            overflow_count += 1
        else:
            print(f"\n[audio status] {status}", file=sys.stderr)
    mono_48k = indata[:, 0]
    mono_16k = mono_48k.reshape(-1, DECIMATION).mean(axis=1)
    audio_q.put(mono_16k.copy())


def processing_loop() -> None:
    global overflow_count, follow_up_until
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    preroll: list[np.ndarray] = []
    utterance: list[np.ndarray] = []
    voiced_count = 0
    silence_count = 0
    in_speech = False

    print(f"[llm] tier={LLM_TIER} model={OLLAMA_MODEL}")
    print("Listening. Say \u201cSage, ...\u201d to wake me up.\n")

    while True:
        frame = audio_q.get()
        pcm16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16).tobytes()
        rms = float(np.sqrt(np.mean(frame ** 2)))
        is_speech = vad.is_speech(pcm16, VAD_RATE) and rms >= RMS_GATE

        if not in_speech:
            preroll.append(frame)
            if len(preroll) > PREROLL_FRAMES:
                preroll.pop(0)
            if is_speech:
                utterance = list(preroll)
                utterance.append(frame)
                voiced_count = 1
                preroll = []
                silence_count = 0
                in_speech = True
        else:
            utterance.append(frame)
            if is_speech:
                voiced_count += 1
                silence_count = 0
            else:
                silence_count += 1
            end_of_utterance = (
                silence_count >= END_SILENCE_FRAMES
                or len(utterance) >= MAX_UTTERANCE_FRAMES
            )
            if end_of_utterance:
                follow_up_until = _handle_utterance(
                    utterance, voiced_count, follow_up_until
                )
                utterance = []
                preroll = []
                voiced_count = 0
                silence_count = 0
                in_speech = False
                overflow_count = 0
                while not audio_q.empty():
                    try:
                        audio_q.get_nowait()
                    except queue.Empty:
                        break


def _handle_utterance(
    utterance: list[np.ndarray], voiced_count: int, follow_up_until: float
) -> float:
    duration_ms = len(utterance) * FRAME_MS
    in_follow_up = time.monotonic() < follow_up_until

    if len(utterance) < MIN_UTTERANCE_FRAMES:
        print(f"[reject: too short {duration_ms}ms]")
        return follow_up_until
    if voiced_count < MIN_VOICED_FRAMES:
        print(f"[reject: only {voiced_count*FRAME_MS}ms voiced of {duration_ms}ms]")
        return follow_up_until

    audio = np.concatenate(utterance)
    avg_rms = float(np.sqrt(np.mean(audio ** 2)))
    if avg_rms < MIN_AVG_RMS:
        print(f"[reject: too quiet rms={avg_rms:.4f}]")
        return follow_up_until

    t0 = time.time()
    text = transcribe(audio)
    dt = time.time() - t0

    if not text or is_hallucination(text):
        print(f"[reject: empty/hallucination {text!r}]")
        return follow_up_until

    if in_follow_up:
        query = text
    else:
        rest = strip_wake_word(text)
        if rest is None:
            print(f"[ignored (no wake word): {text!r}]")
            return follow_up_until
        query = rest

    print(f"You ({duration_ms/1000:.1f}s, stt {dt:.1f}s): {text}")

    if not query:
        print("Sage: Yes?\n")
        return time.monotonic() + FOLLOW_UP_WINDOW_SECS

    t1 = time.time()
    reply = ask_ollama(query)
    dt_llm = time.time() - t1
    print(f"Sage ({dt_llm:.1f}s llm): {reply}\n")
    return time.monotonic() + FOLLOW_UP_WINDOW_SECS


def check_server() -> bool:
    try:
        r = requests.get(WHISPER_URL.replace("/inference", "/"), timeout=2)
        return r.status_code < 500
    except requests.RequestException:
        return False


def main() -> None:
    if not check_server():
        print(f"ERROR: whisper-server not reachable at {WHISPER_URL}", file=sys.stderr)
        print("Start it in another terminal with:", file=sys.stderr)
        print("  cd ~/sage/vendor/whisper.cpp && ./build/bin/whisper-server \\", file=sys.stderr)
        print("    -m models/ggml-base.en.bin --host 127.0.0.1 --port 8080 -t 3", file=sys.stderr)
        sys.exit(1)
    print("whisper-server reachable.")

    worker = threading.Thread(target=processing_loop, daemon=True)
    worker.start()

    try:
        with sd.InputStream(
            device=MIC_INDEX,
            samplerate=CAPTURE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CAPTURE_FRAME,
            callback=audio_callback,
            latency="high",
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
