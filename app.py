"""
Jarvis — pygame UI for the voice assistant.

Local-first pipeline (default):
  mic -> openWakeWord (wake) -> VAD -> whisper.cpp -> Ollama -> Piper TTS
Audio never leaves the device when online mode is OFF.

Optional online "deep think" mode (tap the toggle pill, or press 'O'):
  mic -> openWakeWord (still local) -> VAD
      -> Groq Whisper-large-v3-turbo  (STT)
      -> Groq llama-3.3-70b-versatile (LLM)
      -> Groq Orpheus                 (TTS)

Both modes pipe audio out through `aplay`, so any working ALSA device
(USB sound card, BT speaker, HDMI audio) plays Jarvis's voice.

Online mode is opt-in: it requires a GROQ_API_KEY in ~/sage/.env and an
explicit toggle. The UI palette shifts to amber + a "● ONLINE" badge
whenever audio or queries are leaving the device.

Wake word always stays local — it has to be always-on, and we never want
to stream raw mic audio to the cloud 24/7.

Prereqs:
  - whisper-server systemd service running (set up automatically; check
    with `systemctl --user status whisper-server`)
  - ollama running, with models pulled:
      ollama pull qwen2.5:1.5b
      ollama pull qwen2.5:0.5b
  - openwakeword installed (auto-downloads pretrained models on first init)
  - DISPLAY env set (e.g. DISPLAY=:0 when launched via SSH)
  - ~/sage/.env present (only required for online mode — see .env.example)

Wake word: "Hey Jarvis" (bundled openWakeWord model — no training required).

Controls:
  ESC  -> quit
  O    -> toggle online / deep-think mode (Groq)

Env vars (set in ~/sage/.env or shell):
  JARVIS_LLM_TIER=high|low      pick LLM (default high = qwen2.5:1.5b)
  JARVIS_WAKE_THRESHOLD=0.5     openWakeWord confidence threshold
  JARVIS_WAKE_MODEL=hey_jarvis  switch to a different pretrained model
                              (or path to a custom .onnx/.tflite file)
  JARVIS_WINDOWED=1             run windowed instead of fullscreen
  GROQ_API_KEY=...            enables online mode (required if you toggle it)
  GROQ_MODEL=...              override the Groq model
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

# Load ~/sage/.env BEFORE importing anything that might read GROQ_API_KEY.
# This must come before `import cloud` and any `os.environ.get(...)` checks.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.expanduser("~/sage/.env"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import requests  # noqa: E402
import sounddevice as sd  # noqa: E402
import webrtcvad  # noqa: E402

# openwakeword chats noisily on import; suppress the GPU-discovery warnings
# coming from onnxruntime so the heartbeat log stays readable.
os.environ.setdefault("OWW_HUSH", "1")
import openwakeword  # noqa: E402
from openwakeword import Model as OWWModel  # noqa: E402

import cloud  # noqa: E402  # Groq adapter for online / deep-think mode
import memory  # noqa: E402  # local SQLite-backed long-term memory
import tools  # noqa: E402  # local tool definitions for LLM function calls

MIC_INDEX = 0
CAPTURE_RATE = 48000
VAD_RATE = 16000
DECIMATION = CAPTURE_RATE // VAD_RATE
FRAME_MS = 30
CAPTURE_FRAME = CAPTURE_RATE * FRAME_MS // 1000

VAD_AGGRESSIVENESS = 2                   # was 3; 2 is more permissive at low volume
RMS_GATE = 0.015                         # was 0.022; lowered further for soft voices
PREROLL_FRAMES = 300 // FRAME_MS
END_SILENCE_FRAMES = 750 // FRAME_MS
MIN_UTTERANCE_FRAMES = 300 // FRAME_MS   # so a quick "Jarvis" survives
MIN_VOICED_FRAMES = 180 // FRAME_MS      # single-syllable wake words are ~200ms
MIN_AVG_RMS = 0.012                      # was 0.016; soft speakers
MAX_UTTERANCE_FRAMES = 15000 // FRAME_MS

FOLLOW_UP_WINDOW_SECS = 15.0

# --- openWakeWord configuration --------------------------------------------
# Pretrained model name from the openwakeword bundle. Built-ins:
#   alexa, hey_jarvis, hey_mycroft, timer, weather
# Or supply an absolute path to a custom .onnx / .tflite file (Path B).
WAKE_MODEL_NAME = os.environ.get("JARVIS_WAKE_MODEL", "hey_jarvis")
WAKE_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.5"))
# Any score above this floor — even if it doesn't trigger — is logged so
# you can see how close the wake word is getting and tune JARVIS_WAKE_THRESHOLD
# (or your pronunciation, mic distance, etc.) accordingly. Set very high
# (e.g. 0.99) to silence the diagnostic.
WAKE_DIAG_FLOOR = float(os.environ.get("JARVIS_WAKE_DIAG_FLOOR", "0.20"))
# Cooldown between accepted wake-word triggers — prevents the model from
# re-firing on the second syllable of the same phrase.
WAKE_REFRACTORY_SECS = 1.5

# Backup regex matcher for the rare case the user explicitly says the word
# "Jarvis" inside a transcribed query. Lets us strip it cleanly so the LLM
# doesn't see the trigger as part of the question.
# Whisper often hears the trigger as "jarvis", "jervis", "javis", etc.
# Anything in this set is treated as the wake word for prefix-stripping.
WAKE_WORD_VARIANTS = {
    "jarvis", "jarviss", "jervis", "jervice",
    "javis", "javice", "jarvice",
    "harvis", "garvis",
}
WAKE_WORD_MAX_PREFIX_WORDS = 4

WHISPER_URL = "http://127.0.0.1:8080/inference"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# --- Piper local TTS configuration -----------------------------------------
# Models live in ~/sage/vendor/piper-voices/. Override via env vars to
# swap voices without editing code:
#   PIPER_MODEL=~/sage/vendor/piper-voices/en_US-amy-medium.onnx
PIPER_BIN = shutil.which("piper")
PIPER_MODEL = os.path.expanduser(
    os.environ.get("PIPER_MODEL", "~/sage/vendor/piper-voices/en_US-lessac-medium.onnx")
)
PIPER_CONFIG = os.path.expanduser(
    os.environ.get("PIPER_CONFIG", PIPER_MODEL + ".json")
)
# Piper outputs raw 16-bit signed little-endian mono PCM at this rate when
# --output-raw is used. en_US-lessac-medium is 22050 Hz; some voices use
# 16000 or 24000. The .onnx.json file's sample_rate field is authoritative;
# we read it once at startup and use it for the aplay invocation.
PIPER_SAMPLE_RATE = 22050  # default; overridden in main() from config json

# LLM tiers — power-adaptive switching will plug in here later (an INA219
# monitor thread will flip LLM_TIER based on battery voltage; OLLAMA_MODEL is
# read at request time, so the change takes effect on the next turn).
LLM_TIERS = {
    "high": "qwen2.5:1.5b",   # ~1.5B, good balance — default daily driver
    "low":  "qwen2.5:0.5b",   # ~0.5B, very fast — battery-saver tier
}
LLM_TIER = os.environ.get("JARVIS_LLM_TIER", "high").strip().lower()
if LLM_TIER not in LLM_TIERS:
    print(f"[warn] unknown JARVIS_LLM_TIER={LLM_TIER!r}, using 'high'", file=sys.stderr)
    LLM_TIER = "high"
OLLAMA_MODEL = LLM_TIERS[LLM_TIER]
print(f"[llm] tier={LLM_TIER} model={OLLAMA_MODEL}", file=sys.stderr)

SYSTEM_PROMPT_TEMPLATE = (
    "You are Jarvis, a friendly voice assistant running on a Raspberry Pi. "
    "{facts_block}"
    "{summaries_block}"
    "You have tools available: get_time (for any time/date question, "
    "including other cities), calculator (for any math), set_timer (when "
    "the user wants a timer or reminder), remember (when the user asks "
    "you to remember something or shares a personal detail worth keeping), "
    "and forget (when they ask you to forget something). Use them "
    "naturally — don't announce that you're using a tool, just give the "
    "answer afterward. "
    "The user's messages come from speech-to-text and may contain small "
    "transcription errors (wrong homophones, missing words, inserted "
    "filler words). Infer their likely intent charitably and ignore "
    "obvious noise. If the message is genuinely nonsensical or unclear, "
    "ask one short clarifying question like 'sorry, could you repeat "
    "that?' — do not invent a topic or volunteer unrelated information. "
    "Keep replies short and conversational — usually one to three "
    "sentences. If asked for a list or recipe, give only the essentials "
    "in a single flowing sentence, not a bulleted list. Always finish "
    "your sentence; never trail off mid-thought. "
    "Never explain that you are an AI. Never add follow-up suggestions "
    "like 'is there anything else I can help with'. Never use markdown, "
    "bullet points, or numbered lists. Your reply will be read aloud, so "
    "write plain spoken English only."
)


def build_system_prompt() -> str:
    facts = memory.facts_for_prompt()
    summaries = memory.summaries_for_prompt(limit=2)
    return SYSTEM_PROMPT_TEMPLATE.format(
        facts_block=(facts + "\n") if facts else "",
        summaries_block=(summaries + "\n") if summaries else "",
    )

# IMPORTANT: do NOT feed previous turns of the conversation to Whisper as a
# biasing prompt. Whisper will hallucinate "coherent continuations" of that
# context whenever it gets near-silent / ambient audio. We learned this the
# hard way — leaving only a tiny static prompt that just teaches it the name.
#
# However, a static prompt loaded with the assistant's command vocabulary
# is great for accuracy: Whisper's beam search is biased toward words it
# sees in `initial_prompt`. We list the things Jarvis actually does so soft
# words like "wave" stop being decoded as "leave" / "weave" / etc.
WHISPER_BASE_PROMPT = (
    "Hey Jarvis. Voice command for the Jarvis assistant. Common verbs: "
    "wave, nod, shake, dance, look, sweep, blink, pulse, set, stop, "
    "remember, forget, calculate, time, weather. Common targets and "
    "modifiers: head, ring, light, red, blue, green, white, amber, "
    "left, right, up, down, around, slowly, fast, hello, yes, no."
)
HISTORY_MAX_TURNS = 4

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


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()


@dataclass
class Message:
    role: str
    text: str


@dataclass
class Snapshot:
    state: State
    audio_level: float
    state_age: float
    messages: list[Message]
    follow_up_remaining: float
    online: bool


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state: State = State.IDLE
        self.state_changed_at: float = time.monotonic()
        self.audio_level: float = 0.0
        self.messages: deque[Message] = deque(maxlen=10)
        self.follow_up_until: float = 0.0
        # Deep-think / online mode: when True, LLM calls will route to a
        # cloud provider (Groq) instead of local Ollama. Visual indicator
        # is mandatory whenever this is active. Toggled via 'O' key for
        # now; will be wired to a physical button later.
        self.online: bool = False

    def set_state(self, s: State) -> None:
        with self.lock:
            if s is not self.state:
                print(f"[state] {self.state.name} -> {s.name}", flush=True)
            self.state = s
            self.state_changed_at = time.monotonic()

    def snapshot(self) -> Snapshot:
        with self.lock:
            now = time.monotonic()
            return Snapshot(
                state=self.state,
                audio_level=self.audio_level,
                state_age=now - self.state_changed_at,
                messages=list(self.messages),
                follow_up_remaining=max(0.0, self.follow_up_until - now),
                online=self.online,
            )

    def add_message(self, role: str, text: str) -> None:
        with self.lock:
            self.messages.append(Message(role=role, text=text))

    def update_audio_level(self, rms: float) -> None:
        with self.lock:
            self.audio_level = self.audio_level * 0.7 + rms * 0.3

    def extend_follow_up(self, secs: float = FOLLOW_UP_WINDOW_SECS) -> None:
        with self.lock:
            self.follow_up_until = time.monotonic() + secs

    def in_follow_up_window(self) -> bool:
        with self.lock:
            return time.monotonic() < self.follow_up_until

    def toggle_online(self) -> bool:
        """Flip online/deep-think mode. Returns the new state."""
        with self.lock:
            self.online = not self.online
            print(f"[online] {self.online}", flush=True)
            return self.online

    def is_online(self) -> bool:
        with self.lock:
            return self.online


shared = SharedState()
# Bounded with drop-oldest so the queue can never grow without bound, even
# if the worker is stuck inside a long blocking call (model load, slow LLM
# turn, hung HTTP). 300 frames * 30 ms = 9 seconds — comfortably more than
# any single utterance, but small enough that recovery from a wedged turn
# is bounded to ~9 s of audio backlog instead of unbounded growth.
audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=300)
# Small bounded queue feeding the dedicated wake-word listener thread.
# Capacity ~= 15 * 30 ms = 450 ms — barge-in only cares about the most
# recent half-second of audio, so on overflow we drop the OLDEST frame
# and keep going. (See _push_drop_oldest below.)
wake_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=15)
# Set by wake_listener_loop when the user says the wake word while Jarvis
# is mid-reply. _handle_complete_utterance checks this between blocking
# calls (transcribe -> ask_ollama -> speak) and bails out so we don't
# speak a reply the user has already abandoned.
barge_event = threading.Event()
llm_history: list[dict] = []

# --- audio pipeline diagnostics (updated by audio_callback, read by watchdog)
last_audio_time: float = 0.0
max_rms_since_heartbeat: float = 0.0
frames_since_heartbeat: int = 0
audio_diag_lock = threading.Lock()

# --- conversation tracking (rolling SQLite-backed transcript) -------------
# A "conversation" is a contiguous span of activity. It starts on the first
# user turn after a long idle, and closes — with an auto-summary written
# back to memory.db — once CONV_IDLE_SECS pass with no new turns. The
# in-memory llm_history is cleared at the same time so the next session
# starts with a clean slate (past summaries persist via the system prompt).
CONV_IDLE_SECS = 90.0
_conv_lock = threading.Lock()
_current_conv_id: int | None = None
_last_turn_at: float = 0.0


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    if HALLUCINATION_BRACKET_RE.match(t):
        return True
    if t in HALLUCINATION_PHRASES:
        return True
    return False


def strip_wake_word(text: str) -> str | None:
    """
    Look for the wake word (or a common Whisper mishearing of it) anywhere
    in the first few words of the transcript. Allows natural phrasings like
    "Hello Jarvis", "Hey Jarvis", as well as mishears like "Hello Phage" or
    "Hey Page" that Whisper produces under noise.

    Returns None if no wake word found, "" if only the wake word was said,
    or the remaining query text otherwise.
    """
    tokens = text.split()
    limit = min(WAKE_WORD_MAX_PREFIX_WORDS, len(tokens))
    for i in range(limit):
        cleaned = re.sub(r"[^a-z]", "", tokens[i].lower())
        if cleaned in WAKE_WORD_VARIANTS:
            rest_split = text.split(None, i + 1)
            rest = rest_split[i + 1] if len(rest_split) > i + 1 else ""
            return re.sub(r"^[\s,.!?:;\-—]*", "", rest).strip()
    return None


def normalize_gain(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Scale quiet audio up to near full-scale; leaves loud audio alone."""
    peak = float(np.max(np.abs(samples)))
    if peak < 0.005:
        return samples
    if peak >= target_peak:
        return samples
    return samples * (target_peak / peak)


def samples_to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VAD_RATE)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def build_whisper_prompt() -> str:
    return WHISPER_BASE_PROMPT


def transcribe(samples: np.ndarray) -> str:
    """Transcribe an utterance. Routes to Groq Whisper if online, else local."""
    samples = normalize_gain(samples)
    wav_bytes = samples_to_wav_bytes(samples)
    prompt = build_whisper_prompt()

    if shared.is_online() and cloud.is_available():
        try:
            return cloud.transcribe(wav_bytes, prompt=prompt)
        except Exception as e:
            print(f"[groq stt failed, falling back to local] {e}", file=sys.stderr)

    duration_s = len(samples) / VAD_RATE
    t0 = time.time()
    try:
        r = requests.post(
            WHISPER_URL,
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            data={
                "response_format": "json",
                "temperature": "0",
                "prompt": prompt,
            },
            timeout=30,
        )
        r.raise_for_status()
        text = r.json().get("text", "").strip()
        print(
            f"[whisper] {duration_s:.1f}s audio -> {time.time()-t0:.1f}s "
            f"-> {text!r}",
            file=sys.stderr,
        )
        return text
    except requests.RequestException as e:
        print(
            f"[whisper error after {time.time()-t0:.1f}s] {e}",
            file=sys.stderr,
        )
        return ""


MAX_TOOL_ITERATIONS = 4


def _prewarm_ollama() -> None:
    """Send a tiny request at startup so Ollama loads the model into RAM.

    Without this, the first real user question pays a 15-30 s cold-load
    penalty (loading ~1 GB from NVMe, allocating tensors, etc.) — and
    that period of all-cores-pegged inference also starves the audio
    callback long enough to falsely trip the dead-stream watchdog.

    Best-effort: any failure is logged and ignored. Runs on its own
    thread so app startup isn't blocked.
    """
    t0 = time.time()
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": "1h",
                "options": {"num_predict": 1, "num_thread": 2},
            },
            timeout=120,
        )
        r.raise_for_status()
        print(
            f"[prewarm] {OLLAMA_MODEL} loaded in {time.time()-t0:.1f}s",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[prewarm] failed (ignored): {e}", file=sys.stderr)


def _ollama_chat(messages: list[dict]) -> dict:
    """Streamed round-trip to local Ollama. Returns the parsed `message` dict.

    Streaming buys cooperative cancellation: between every chunk (~50-200 ms
    on the Pi) we check barge_event, and bail out the moment the user has
    interrupted. Without this, a wake-word during THINKING still has to
    wait for the entire 220-token reply to generate before the worker
    notices — which can be 20-30 s on qwen2.5:1.5b, during which audio_q
    fills up and the system feels frozen.

    On cancel we close the response and raise RequestException; ask_ollama's
    error path takes it from there.
    """
    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": tools.OLLAMA_TOOL_SCHEMAS,
            "stream": True,
            # 1h keep-alive: keeps the model resident across long idle gaps
            # so we don't pay the ~15-30 s cold-load penalty on every "first
            # question after a coffee break".
            "keep_alive": "1h",
            # 2 of 4 Pi 5 cores: leaves 2 cores for audio callback + USB
            # iso interrupts + Python GIL + UI. With 3 threads the audio
            # thread gets starved during inference and the watchdog trips.
            "options": {"num_predict": 220, "num_thread": 2},
        },
        timeout=90,
        stream=True,
    )
    r.raise_for_status()

    role = "assistant"
    content_parts: list[str] = []
    tool_calls: list[dict] = []
    try:
        for line in r.iter_lines():
            if not line:
                continue
            if barge_event.is_set():
                print("[llm] cancelled mid-stream (barge-in)", file=sys.stderr)
                raise requests.RequestException("cancelled by barge-in")
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = chunk.get("message") or {}
            if msg.get("role"):
                role = msg["role"]
            piece = msg.get("content")
            if piece:
                content_parts.append(piece)
            calls = msg.get("tool_calls") or chunk.get("tool_calls")
            if calls:
                tool_calls.extend(calls)
            if chunk.get("done"):
                break
    finally:
        r.close()

    out: dict = {"role": role, "content": "".join(content_parts)}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _chat(messages: list[dict]) -> tuple[dict, str]:
    """Route a chat request to Groq (if online + key set) or local Ollama.

    Returns (message, backend_label). The label is "groq" or "ollama" — used
    only for diagnostic printing. On Groq failure we fall back to Ollama
    automatically so a flaky network never breaks Jarvis.
    """
    if shared.is_online() and cloud.is_available():
        try:
            return cloud.chat(messages), "groq"
        except Exception as e:
            print(f"[groq failed, falling back to local] {e}", file=sys.stderr)
    return _ollama_chat(messages), "ollama"


def ask_ollama(user_text: str) -> str:
    """Query the LLM with tool-calling support.

    Despite the name, this routes through `_chat` which may go to Groq when
    online mode is on. Kept named `ask_ollama` so the existing call sites
    don't need updating.

    The conversation in `llm_history` only ever stores clean user/assistant
    pairs. The intermediate tool-call dance happens in a per-turn buffer so
    the persistent history stays compact and easy to trim.
    """
    base = list(llm_history) + [{"role": "user", "content": user_text}]
    working = list(base)

    final_text: str | None = None
    backend = "ollama"
    t0 = time.time()
    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            messages = (
                [{"role": "system", "content": build_system_prompt()}]
                + working[-HISTORY_MAX_TURNS * 2 - 8:]  # +8 for tool dance headroom
            )
            msg, backend = _chat(messages)
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                final_text = (msg.get("content") or "").strip()
                break

            print(
                f"[tools] iter={iteration} backend={backend} "
                f"calls={[tc['function']['name'] for tc in tool_calls]}",
                file=sys.stderr,
            )
            working.append(msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                args = tc["function"].get("arguments", {})
                result = tools.execute(fn_name, args)
                print(f"[tools]   {fn_name}({args}) -> {result!r}", file=sys.stderr)
                tool_msg = {"role": "tool", "content": result, "name": fn_name}
                if tc.get("id"):
                    tool_msg["tool_call_id"] = tc["id"]
                working.append(tool_msg)
        else:
            final_text = "Sorry, I got tangled up there."
    except requests.RequestException as e:
        print(
            f"[llm error after {time.time()-t0:.1f}s] {e}",
            file=sys.stderr,
        )
        return f"[error: {e}]"

    if not final_text:
        final_text = "Sorry, I didn't catch that."
    print(f"[llm] {backend} {time.time()-t0:.1f}s -> {final_text!r}", file=sys.stderr)

    # Persist only clean user+assistant turns. Tool-call traces stay scoped
    # to this invocation.
    llm_history.append({"role": "user", "content": user_text})
    llm_history.append({"role": "assistant", "content": final_text})
    return final_text


# --- speech output ---------------------------------------------------------
# Two backends, both pipe audio through `aplay`:
#   - cloud.synthesize  (Groq Orpheus, ~600ms, very natural) when online
#   - _piper_speak      (local Piper, ~300-700ms after model load) when offline

# Tracks an in-flight aplay subprocess so we can stop it on shutdown / when
# the user starts speaking again.
_aplay_proc: subprocess.Popen | None = None
_piper_proc: subprocess.Popen | None = None
_speak_lock = threading.Lock()


def _play_wav_blocking(wav_bytes: bytes) -> None:
    """Pipe a WAV blob to aplay and wait for playback to finish.

    aplay reads the WAV header and figures out sample rate / format on
    its own, so we just dump bytes. It runs as a separate process so it
    doesn't share PortAudio state with our input stream — that keeps mic
    capture stable while Jarvis is talking.
    """
    global _aplay_proc
    with _speak_lock:
        try:
            _aplay_proc = subprocess.Popen(
                ["aplay", "-q"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("[tts] aplay not installed — skipping playback", file=sys.stderr)
            _aplay_proc = None
            return
    try:
        if _aplay_proc.stdin is not None:
            _aplay_proc.stdin.write(wav_bytes)
            _aplay_proc.stdin.close()
        _aplay_proc.wait(timeout=60)
    except (BrokenPipeError, subprocess.TimeoutExpired) as e:
        print(f"[tts] aplay error: {e}", file=sys.stderr)
        try:
            _aplay_proc.kill()
        except Exception:
            pass
    finally:
        with _speak_lock:
            _aplay_proc = None


def piper_available() -> bool:
    """True if Piper binary + selected voice files all exist."""
    return (
        PIPER_BIN is not None
        and os.path.isfile(PIPER_MODEL)
        and os.path.isfile(PIPER_CONFIG)
    )


def _piper_speak(text: str) -> None:
    """Stream Piper synthesis directly to aplay. Blocks until playback ends.

    Piper's --output-raw mode writes raw 16-bit PCM to stdout, which we
    pipe straight into aplay with the matching format flags. This is the
    standard low-latency Piper pipeline — no temp files, no WAV header
    overhead, audio starts within ~300 ms of submitting the text on a
    Pi 5 (most of which is onnxruntime model warm-up).
    """
    global _piper_proc, _aplay_proc
    if not piper_available():
        print(
            f"[piper] not available (bin={PIPER_BIN!r}, model="
            f"{PIPER_MODEL!r}) — skipping playback",
            file=sys.stderr,
        )
        return

    t0 = time.time()
    try:
        with _speak_lock:
            _piper_proc = subprocess.Popen(
                [
                    PIPER_BIN, "-m", PIPER_MODEL, "-c", PIPER_CONFIG,
                    "--output-raw",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            _aplay_proc = subprocess.Popen(
                [
                    "aplay", "-q",
                    "-r", str(PIPER_SAMPLE_RATE),
                    "-c", "1",
                    "-f", "S16_LE",
                ],
                stdin=_piper_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Close our handle to piper's stdout so aplay sees EOF when
            # piper finishes — otherwise aplay would wait forever.
            if _piper_proc.stdout is not None:
                _piper_proc.stdout.close()

        if _piper_proc.stdin is not None:
            try:
                _piper_proc.stdin.write(text.encode("utf-8"))
                _piper_proc.stdin.close()
            except BrokenPipeError:
                pass

        _aplay_proc.wait(timeout=120)
        _piper_proc.wait(timeout=120)
        print(
            f"[piper] {time.time()-t0:.1f}s ({len(text)} chars)",
            file=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        print("[piper] playback timed out — killing", file=sys.stderr)
        for p in (_piper_proc, _aplay_proc):
            try:
                if p is not None:
                    p.kill()
            except Exception:
                pass
    except Exception as e:
        print(f"[piper] error: {e}", file=sys.stderr)
    finally:
        with _speak_lock:
            _piper_proc = None
            _aplay_proc = None


def stop_speaking() -> None:
    """Kill any in-progress audio output (called on shutdown)."""
    with _speak_lock:
        for p in (_aplay_proc, _piper_proc):
            if p is not None:
                try:
                    p.kill()
                except Exception:
                    pass


def speak(text: str) -> None:
    """Synthesize + play a reply. Blocks until done.

    Routing:
      Online + Groq key set -> Orpheus (chunked at sentence boundaries
        so the user starts hearing audio while later chunks are still
        in flight). On any failure mid-reply, falls back to Piper for
        the remainder if it's available.
      Otherwise -> Piper local, which handles long text natively.

    State is set to SPEAKING for the duration; audio_callback drops
    incoming mic frames in this state so Jarvis doesn't hear itself talk
    and feedback-loop into a wake event.
    """
    text = (text or "").strip()
    if not text:
        return

    use_groq = shared.is_online() and cloud.is_available()
    shared.set_state(State.SPEAKING)
    try:
        if use_groq:
            try:
                for chunk in cloud.chunk_text(text):
                    # Cooperative cancel: if a barge-in flipped state away
                    # from SPEAKING, abandon the rest of the reply.
                    if shared.state is not State.SPEAKING:
                        print("[speak] cancelled (barge-in)", file=sys.stderr)
                        return
                    wav = cloud.synthesize(chunk)
                    if shared.state is not State.SPEAKING:
                        print("[speak] cancelled (barge-in)", file=sys.stderr)
                        return
                    _play_wav_blocking(wav)
                return
            except Exception as e:
                print(
                    f"[groq tts failed, falling back to piper] {e}",
                    file=sys.stderr,
                )
        if piper_available():
            # Piper streams the whole reply through one aplay subprocess,
            # so barge-in cancel is handled by stop_speaking() killing
            # both processes (no per-chunk check needed).
            _piper_speak(text)
    finally:
        # Drain audio captured while we were speaking — those frames are
        # mostly silence (or our own voice) and are no longer relevant.
        # EXCEPT during a barge-in: the user is mid-sentence ("hey jarvis,
        # what's the time?") and the back half of their utterance is
        # currently sitting in audio_q waiting to be transcribed. Dropping
        # it here is what makes the UI flash 'listening' but never hear
        # anything. Leave the queue alone; the next iteration will consume
        # the user's continuation.
        if not barge_event.is_set():
            _drain_queue()


def _ensure_conversation() -> int:
    """Lazily start a conversation row in SQLite. Idempotent within a session."""
    global _current_conv_id
    with _conv_lock:
        if _current_conv_id is None:
            _current_conv_id = memory.start_conversation()
            print(f"[conv] started id={_current_conv_id}", file=sys.stderr)
        return _current_conv_id


def record_turn(user_text: str, jarvis_reply: str) -> None:
    """Persist one user→assistant exchange to memory.db and bump the timer."""
    global _last_turn_at
    try:
        conv_id = _ensure_conversation()
        memory.add_message(conv_id, "user", user_text)
        memory.add_message(conv_id, "assistant", jarvis_reply)
    except Exception:
        traceback.print_exc()
    with _conv_lock:
        _last_turn_at = time.monotonic()


def _summarize_messages(msgs: list[dict]) -> str:
    """Ask the local LLM for a one-line summary. Best-effort — '' on failure.

    We always summarize with the local small model (qwen2.5:0.5b) regardless
    of whether the conversation used Groq, because:
      - It runs in the background while the user is idle, so latency is fine.
      - Saves an unnecessary cloud call.
      - Keeps the long-term memory layer 100% local.
    """
    if not msgs:
        return ""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in msgs
        if m.get("content")
    )
    prompt = (
        "Summarize this brief conversation between a user and Jarvis in one "
        "sentence (max 25 words), written in third person. Focus on what "
        "the user wanted, learned, or decided. If the exchange was trivial "
        "(e.g. just a greeting, time check, or one-off math), reply with "
        "exactly the word SKIP and nothing else.\n\n"
        f"{transcript}"
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_TIERS["low"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": 80, "num_thread": 2},
            },
            timeout=30,
        )
        r.raise_for_status()
        text = (r.json().get("message", {}).get("content") or "").strip()
        if not text or text.upper().strip(".") == "SKIP" or len(text) < 5:
            return ""
        return text
    except requests.RequestException as e:
        print(f"[summary error] {e}", file=sys.stderr)
        return ""


def close_conversation(reason: str = "idle") -> None:
    """Finalize the active conversation: summarize + write to memory.db.

    Safe to call multiple times — second call is a no-op once closed.
    """
    global _current_conv_id, _last_turn_at
    with _conv_lock:
        conv_id = _current_conv_id
    if conv_id is None:
        return

    try:
        msgs = memory.get_messages(conv_id)
    except Exception:
        traceback.print_exc()
        msgs = []

    summary: str | None = None
    if len(msgs) >= 2:
        summary = _summarize_messages(msgs) or None

    try:
        memory.end_conversation(conv_id, summary)
    except Exception:
        traceback.print_exc()

    print(
        f"[conv] closed id={conv_id} reason={reason} "
        f"turns={len(msgs)} summary={summary!r}",
        file=sys.stderr,
    )

    with _conv_lock:
        if _current_conv_id == conv_id:
            _current_conv_id = None
            _last_turn_at = 0.0
    llm_history.clear()


def conversation_watchdog() -> None:
    """Background thread that auto-closes idle conversations every ~10s."""
    while True:
        time.sleep(10.0)
        with _conv_lock:
            conv_id = _current_conv_id
            last = _last_turn_at
        if conv_id is None or last == 0.0:
            continue
        if (time.monotonic() - last) < CONV_IDLE_SECS:
            continue
        try:
            close_conversation(reason="idle")
        except Exception:
            traceback.print_exc()


def audio_callback(indata, frames, time_info, status) -> None:
    global last_audio_time, max_rms_since_heartbeat, frames_since_heartbeat
    if status:
        # input_overflow is common under CPU load and mostly cosmetic — only
        # log other status flags to keep the terminal readable.
        if not status.input_overflow:
            print(f"[audio status] {status}", file=sys.stderr)
    mono_48k = indata[:, 0]
    mono_16k = mono_48k.reshape(-1, DECIMATION).mean(axis=1)
    rms = float(np.sqrt(np.mean(mono_16k ** 2)))
    shared.update_audio_level(rms)

    # Frame-routing policy:
    #   audio_q  (consumed by processing_loop for wake / VAD / capture):
    #     pushed only when IDLE or LISTENING. During SPEAKING / THINKING /
    #     TRANSCRIBING the worker is blocked inside speak() / ask_ollama() /
    #     whisper, so anything we pushed would just pile up and get drained
    #     when those calls return.
    #   wake_q   (consumed by the dedicated wake_listener_loop thread):
    #     pushed ALWAYS. That thread is what catches barge-in (a wake-word
    #     hit while Jarvis is mid-reply) — the worker thread can't react during
    #     those phases because it's blocked. Bounded + drop-oldest so it
    #     never grows under CPU pressure.
    # Note: barge-in via wake word is only reliable when Jarvis's voice is on
    # earphones (or a quiet speaker). Open speakers can feed Jarvis's TTS back
    # into the mic — bump JARVIS_WAKE_THRESHOLD if you get false barge-ins.
    state = shared.state
    frame = mono_16k.copy()
    if state in (State.IDLE, State.LISTENING):
        # Drop-oldest semantics on overflow: the audio thread must NEVER
        # block (PortAudio would kill the stream). If the worker is stuck,
        # we keep the most recent ~9 s of audio and let the older stuff go.
        _push_drop_oldest(audio_q, frame)
    _push_drop_oldest(wake_q, frame)

    with audio_diag_lock:
        last_audio_time = time.monotonic()
        if rms > max_rms_since_heartbeat:
            max_rms_since_heartbeat = rms
        frames_since_heartbeat += 1


def load_wake_model() -> tuple[OWWModel, str]:
    """Initialize openWakeWord with just the configured model.

    Loading only one model (vs the default load-everything behaviour) cuts
    both memory and per-frame inference cost considerably.

    Returns (model, key) where `key` is the dictionary key returned by
    `model.predict(...)` for our wake word — note this is the versioned
    name (e.g. "hey_jarvis_v0.1"), NOT the friendly alias ("hey_jarvis").
    """
    name = WAKE_MODEL_NAME
    print(f"[wake] loading model={name} threshold={WAKE_THRESHOLD}", file=sys.stderr)
    if os.path.isabs(name) and os.path.exists(name):
        m = OWWModel(wakeword_model_paths=[name])
    else:
        builtins = openwakeword.models  # name -> {'model_path': '/abs/path.onnx'}
        if name not in builtins:
            raise FileNotFoundError(
                f"openwakeword model {name!r} not found. Expected either an "
                f"absolute path to a .onnx/.tflite file, or one of the "
                f"built-in names ({', '.join(sorted(builtins))})."
            )
        m = OWWModel(wakeword_model_paths=[builtins[name]["model_path"]])
    key = next(iter(m.models))
    print(f"[wake] active key={key}", file=sys.stderr)
    return m, key


def processing_loop() -> None:
    while True:
        try:
            _processing_iteration()
        except Exception:
            print("[processing loop crashed; restarting]", file=sys.stderr)
            traceback.print_exc()
            shared.set_state(State.IDLE)
            while not audio_q.empty():
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break
            time.sleep(0.5)


def _drain_queue() -> None:
    while not audio_q.empty():
        try:
            audio_q.get_nowait()
        except queue.Empty:
            break


_wake_log_lock = threading.Lock()
_wake_log_last = 0.0
_wake_log_high = 0.0


def _log_wake_score(score: float, source: str) -> None:
    """Throttled near-miss logger for the wake-word detector.

    Anything above WAKE_DIAG_FLOOR — even sub-threshold — gets surfaced
    so you can see how close 'hey jarvis' is to firing. We report the
    high-water mark over each ~500ms window so the log stays readable
    even when the model is firing predict() ~33 times per second.

    Use this to tune JARVIS_WAKE_THRESHOLD: if you consistently see
    'miss peak=0.42' when you say the wake word, drop the threshold to
    around 0.35. If you see no scores at all, the mic isn't picking
    you up — try speaking louder or closer.
    """
    global _wake_log_last, _wake_log_high
    if score < WAKE_DIAG_FLOOR:
        return
    with _wake_log_lock:
        if score > _wake_log_high:
            _wake_log_high = score
        now = time.monotonic()
        if (now - _wake_log_last) < 0.4:
            return
        marker = "HIT " if _wake_log_high >= WAKE_THRESHOLD else "miss"
        print(
            f"[wake-diag] {source:<8} {marker} peak={_wake_log_high:.2f} "
            f"(threshold={WAKE_THRESHOLD:.2f})",
            file=sys.stderr,
        )
        _wake_log_last = now
        _wake_log_high = 0.0


def _push_drop_oldest(q: "queue.Queue[np.ndarray]", item: np.ndarray) -> None:
    """Like put_nowait, but throws away the oldest entry if the queue is full.

    Used by audio_callback to feed wake_q without ever blocking the audio
    thread — a blocked audio callback is the fastest way to kill the mic
    stream.
    """
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def wake_listener_loop() -> None:
    """Always-on wake-word listener that handles barge-in.

    Lives on its own thread because the main worker (processing_loop) is
    blocked inside speak() / ask_ollama() / whisper transcribe whenever
    Jarvis is mid-reply — exactly when the user wants to be able to
    interrupt. This thread keeps openWakeWord fed at all times so the
    model sees continuous audio context, and acts on a hit only when
    Jarvis is busy producing output:

      SPEAKING     -> kill TTS via stop_speaking() (immediate audible stop)
      THINKING     -> set barge_event; the worker discards the LLM reply
                      when ask_ollama() finally returns
      TRANSCRIBING -> set barge_event; same idea, drop the in-flight
                      transcription instead of speaking it back

    During IDLE / LISTENING the worker thread's own oww instance handles
    the initial wake. Both threads run predict on every frame, but only
    one of them is "active" at a time depending on state, so the combined
    inference cost is roughly the same as before.
    """
    try:
        oww, key = load_wake_model()
    except Exception:
        traceback.print_exc()
        print(
            "[wake-listener] disabled (model failed to load); "
            "tap the screen to interrupt instead",
            file=sys.stderr,
        )
        return

    last_barge = 0.0
    while True:
        frame = wake_q.get()
        try:
            pcm16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
            scores = oww.predict(pcm16)
            score = float(scores.get(key, 0.0))
        except Exception:
            traceback.print_exc()
            continue

        state = shared.state
        if state not in (State.SPEAKING, State.THINKING, State.TRANSCRIBING):
            # Not Jarvis's turn — IDLE / LISTENING wakes are the worker's job.
            continue

        # Surface near-miss scores during busy phases too, so you can tell
        # whether barge-in is failing because the model didn't recognize you
        # (no log) vs because the threshold's too high (a 'miss' line).
        _log_wake_score(score, "barge-in")

        now = time.monotonic()
        if score < WAKE_THRESHOLD or (now - last_barge) < WAKE_REFRACTORY_SECS:
            continue

        last_barge = now
        print(
            f"[wake] BARGE-IN during {state.name} score={score:.2f}",
            file=sys.stderr,
        )
        # Tell the worker to throw away whatever blocking call it's
        # waiting on. Only matters for THINKING / TRANSCRIBING — speak()
        # is killed below regardless.
        barge_event.set()
        stop_speaking()
        shared.set_state(State.IDLE)
        shared.extend_follow_up()
        oww.reset()


def _processing_iteration() -> None:
    oww, oww_key = load_wake_model()
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    preroll: list[np.ndarray] = []
    utterance: list[np.ndarray] = []
    voiced_count = 0
    silence_count = 0
    in_speech = False
    utterance_in_follow_up = False
    last_wake_at = 0.0

    while True:
        frame = audio_q.get()
        state = shared.state
        in_follow_up = shared.in_follow_up_window()

        # Race-safety: by the time we pop a frame, state may have flipped
        # to a "Jarvis is busy" phase. Those frames are stale and barge-in
        # is the wake_listener thread's job, not ours — just drop.
        if state in (State.SPEAKING, State.THINKING, State.TRANSCRIBING):
            continue

        # ---- Wake-word mode -----------------------------------------------
        # When IDLE and outside the follow-up window, run openWakeWord ONLY.
        # Whisper and the LLM are not invoked until a wake event fires.
        if not in_speech and not in_follow_up:
            pcm16_int = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
            scores = oww.predict(pcm16_int)
            wake_score = float(scores.get(oww_key, 0.0))
            _log_wake_score(wake_score, "idle")

            now = time.monotonic()
            if (
                wake_score >= WAKE_THRESHOLD
                and (now - last_wake_at) >= WAKE_REFRACTORY_SECS
            ):
                print(
                    f"[wake] {oww_key} score={wake_score:.2f} -> follow-up",
                    file=sys.stderr,
                )
                last_wake_at = now
                shared.add_message("jarvis", "Yes?")
                shared.extend_follow_up()
                # Cold wake — wipe any leftover barge_event from a prior
                # interrupted reply that never produced a continuation.
                # (Otherwise drains stay disabled and audio_q grows.)
                barge_event.clear()
                # Drop everything still in flight and reset the wake model
                # so the same audio frames can't double-trigger.
                oww.reset()
                _drain_queue()
                preroll = []
            continue

        # ---- VAD / utterance-capture mode ---------------------------------
        # Reached when in the follow-up window, OR mid-utterance regardless
        # of whether the follow-up window has since expired (we honor the
        # decision captured at speech start).
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
                utterance_in_follow_up = in_follow_up
                if utterance_in_follow_up:
                    shared.set_state(State.LISTENING)
        else:
            utterance.append(frame)
            if is_speech:
                voiced_count += 1
                silence_count = 0
            else:
                silence_count += 1
            end = (
                silence_count >= END_SILENCE_FRAMES
                or len(utterance) >= MAX_UTTERANCE_FRAMES
            )
            if end:
                try:
                    _handle_complete_utterance(
                        utterance, voiced_count, utterance_in_follow_up
                    )
                finally:
                    # Always return to IDLE after an utterance. Any path
                    # through handle (success, reject, exception) ends here,
                    # so the UI can never get stuck at LISTENING/THINKING.
                    shared.set_state(State.IDLE)
                utterance = []
                preroll = []
                voiced_count = 0
                silence_count = 0
                in_speech = False
                utterance_in_follow_up = False
                # Reset the wake model since we just processed an utterance
                # — its internal buffer would otherwise still hold the most
                # recent ~3 s of audio and could spuriously fire next loop.
                oww.reset()
                # Same exception as in speak(): if a barge-in was triggered
                # mid-reply, the user's continuation utterance is in
                # audio_q right now. Don't blow it away.
                if not barge_event.is_set():
                    _drain_queue()


def _handle_complete_utterance(
    utterance: list[np.ndarray], voiced_count: int, in_follow_up: bool
) -> None:
    duration_ms = len(utterance) * FRAME_MS

    if len(utterance) < MIN_UTTERANCE_FRAMES:
        print(f"[reject: too short {duration_ms}ms]", file=sys.stderr)
        return
    if voiced_count < MIN_VOICED_FRAMES:
        print(
            f"[reject: only {voiced_count*FRAME_MS}ms voiced of {duration_ms}ms]",
            file=sys.stderr,
        )
        return

    audio = np.concatenate(utterance)
    avg_rms = float(np.sqrt(np.mean(audio ** 2)))
    if avg_rms < MIN_AVG_RMS:
        print(f"[reject: too quiet rms={avg_rms:.4f}]", file=sys.stderr)
        return

    # Fresh turn — clear any stale barge flag from a previous interruption
    # so it can't immediately abort the reply we're about to produce.
    barge_event.clear()

    if in_follow_up:
        shared.set_state(State.TRANSCRIBING)

    text = transcribe(audio)

    # Barge-in check #1: user said "hey jarvis" while we were transcribing.
    # Drop the result and let the new utterance reach us via the follow-up
    # window the wake_listener already opened.
    if barge_event.is_set():
        barge_event.clear()
        print("[barge] discarded transcription (user interrupted)", file=sys.stderr)
        return

    if not text or is_hallucination(text):
        print(f"[reject: empty/hallucination {text!r}]", file=sys.stderr)
        return

    if in_follow_up:
        query = text
        display_text = text
    else:
        rest = strip_wake_word(text)
        if rest is None:
            print(f"[ignored (no wake word): {text!r}]", file=sys.stderr)
            return
        query = rest
        display_text = text

    shared.add_message("user", display_text)

    if not query:
        shared.add_message("jarvis", "Yes?")
        shared.extend_follow_up()
        return

    shared.set_state(State.THINKING)
    reply = ask_ollama(query)

    # Barge-in check #2: user interrupted while the LLM was thinking.
    # The reply might still be a useful answer, but the user has moved
    # on — record it for context but don't speak it.
    if barge_event.is_set():
        barge_event.clear()
        print("[barge] suppressed reply (user interrupted)", file=sys.stderr)
        return

    shared.add_message("jarvis", reply)
    record_turn(query, reply)
    speak(reply)
    shared.extend_follow_up()


BG = (14, 16, 26)
FG = (230, 232, 240)
DIM = (130, 135, 155)
DIVIDER = (40, 44, 60)

STATE_COLOR = {
    State.IDLE:         (90, 100, 130),
    State.LISTENING:    (100, 180, 255),
    State.TRANSCRIBING: (180, 130, 255),
    State.THINKING:     (255, 180, 120),
    State.SPEAKING:     (120, 230, 170),
}

# Online / deep-think palette: shifts every state toward a vivid amber so
# the user gets an unmistakable visual signal that queries are now leaving
# the device.
STATE_COLOR_ONLINE = {
    State.IDLE:         (200, 130, 60),
    State.LISTENING:    (255, 180, 70),
    State.TRANSCRIBING: (255, 150, 90),
    State.THINKING:     (255, 200, 80),
    State.SPEAKING:     (255, 215, 130),
}

STATE_LABEL = {
    State.IDLE:         "Say \u201cHey Jarvis\u201d to start",
    State.LISTENING:    "Listening",
    State.TRANSCRIBING: "Transcribing",
    State.THINKING:     "Thinking",
    State.SPEAKING:     "Speaking",
}

USER_LABEL_COLOR = (100, 180, 255)
JARVIS_LABEL_COLOR = (255, 180, 120)

ONLINE_BADGE_COLOR = (255, 170, 60)
HINT_COLOR = (95, 100, 120)


def label_for(snap: Snapshot) -> str:
    if snap.state is State.IDLE and snap.follow_up_remaining > 0:
        return f"Listening ({int(snap.follow_up_remaining)}s)\u2026"
    return STATE_LABEL[snap.state]


def draw_state_orb(
    screen: pygame.Surface,
    center: tuple[int, int],
    state: State,
    audio_level: float,
    state_age: float,
    online: bool = False,
) -> None:
    palette = STATE_COLOR_ONLINE if online else STATE_COLOR
    color = palette[state]
    base_radius = 70

    if state is State.IDLE:
        r = base_radius + math.sin(state_age * 1.6) * 8
        alpha = 170
    elif state is State.LISTENING:
        r = base_radius + min(audio_level * 400, 120)
        alpha = 230
    elif state is State.TRANSCRIBING:
        r = base_radius + math.sin(state_age * 6) * 6
        alpha = 220
    elif state is State.THINKING:
        r = base_radius + math.sin(state_age * 2.5) * 14
        alpha = 220
    elif state is State.SPEAKING:
        # Faster, larger pulse so it's distinct from THINKING — reads as
        # "actively producing output" rather than "still cogitating".
        r = base_radius + math.sin(state_age * 7) * 16
        alpha = 235

    r = int(r)
    surface = pygame.Surface((2 * r + 40, 2 * r + 40), pygame.SRCALPHA)
    for mult, a_mult in [(1.4, 0.25), (1.2, 0.45), (1.0, 1.0)]:
        rr = int(r * mult)
        ca = int(alpha * a_mult)
        pygame.draw.circle(surface, (*color, ca), (r + 20, r + 20), rr)
    screen.blit(surface, (center[0] - r - 20, center[1] - r - 20))


def wrap_lines(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        test = current + " " + w
        if font.size(test)[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def draw_conversation(
    screen: pygame.Surface,
    area: pygame.Rect,
    label_font: pygame.font.Font,
    body_font: pygame.font.Font,
    messages: list[Message],
) -> None:
    margin_x = 60
    y = area.bottom - 24
    max_width = area.width - 2 * margin_x

    for msg in reversed(messages):
        is_user = msg.role == "user"
        label_text = "YOU" if is_user else "SAGE"
        label_color = USER_LABEL_COLOR if is_user else JARVIS_LABEL_COLOR

        body_lines = wrap_lines(msg.text, body_font, max_width)

        block_height = (
            body_font.get_linesize() * len(body_lines)
            + label_font.get_linesize()
            + 8
        )
        block_top = y - block_height
        if block_top < area.top + 20:
            break

        ly = block_top
        label_surface = label_font.render(label_text, True, label_color)
        screen.blit(label_surface, (margin_x, ly))
        ly += label_font.get_linesize() + 4
        for line in body_lines:
            line_surface = body_font.render(line, True, FG)
            screen.blit(line_surface, (margin_x, ly))
            ly += body_font.get_linesize()

        y = block_top - 20


def main() -> None:
    # Long-term memory: open the SQLite DB and wire timer notifications
    # so a fired timer surfaces as a "jarvis" message in the UI.
    memory.init_db()
    tools.set_timer_notifier(lambda msg: shared.add_message("jarvis", msg))

    if cloud.is_available():
        print(
            f"[cloud] Groq key loaded — online mode available "
            f"(model={os.environ.get('GROQ_MODEL') or cloud.DEFAULT_MODEL})",
            file=sys.stderr,
        )
    else:
        print(
            "[cloud] no GROQ_API_KEY set — online toggle will fall back to local",
            file=sys.stderr,
        )

    # Read the actual Piper sample rate from the voice's config json so
    # aplay gets the right rate flag (different voices use 16/22.05/24 kHz).
    global PIPER_SAMPLE_RATE
    if piper_available():
        try:
            import json as _json
            with open(PIPER_CONFIG) as _f:
                cfg = _json.load(_f)
            sr = cfg.get("audio", {}).get("sample_rate")
            if isinstance(sr, int) and sr > 0:
                PIPER_SAMPLE_RATE = sr
            print(
                f"[piper] available (model={os.path.basename(PIPER_MODEL)}, "
                f"sample_rate={PIPER_SAMPLE_RATE} Hz)",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[piper] config read failed, using default rate: {e}", file=sys.stderr)
    else:
        print(
            f"[piper] not available — install with `pip install piper-tts` and "
            f"download a voice to {os.path.dirname(PIPER_MODEL)}/",
            file=sys.stderr,
        )

    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    pygame.init()
    info = pygame.display.Info()
    if os.environ.get("JARVIS_WINDOWED"):
        screen = pygame.display.set_mode((1280, 720))
    else:
        screen = pygame.display.set_mode(
            (info.current_w, info.current_h), pygame.FULLSCREEN
        )
    pygame.display.set_caption("Jarvis")
    pygame.mouse.set_visible(False)

    w, h = screen.get_size()
    clock = pygame.time.Clock()

    label_font = pygame.font.SysFont("DejaVu Sans", 20, bold=True)
    state_font = pygame.font.SysFont("DejaVu Sans", 30)
    body_font = pygame.font.SysFont("DejaVu Sans", 24)

    def open_stream() -> sd.InputStream:
        # latency=0.5 (500ms target buffer) is a deliberate trade-off:
        # 'high' on Pi resolves to ~100ms, which is too tight for our
        # USB 1.1 mic under CPU pressure — a brief callback stall and
        # the kernel runs out of buffer, drops the iso stream, and
        # PortAudio's callback never fires again. 500ms gives the audio
        # thread 5x more headroom to ride out an Ollama cold-load or a
        # piper synth spike.
        latency = float(os.environ.get("JARVIS_INPUT_LATENCY", "0.5"))
        s = sd.InputStream(
            device=MIC_INDEX,
            samplerate=CAPTURE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CAPTURE_FRAME,
            callback=audio_callback,
            latency=latency,
        )
        s.start()
        return s

    try:
        stream_holder = [open_stream()]
    except Exception as e:
        print(f"Failed to open mic: {e}", file=sys.stderr)
        pygame.quit()
        sys.exit(1)

    worker = threading.Thread(target=processing_loop, daemon=True)
    worker.start()

    # Always-on wake-word listener for barge-in. Runs on its own thread so
    # it can interrupt Jarvis while the worker is blocked inside speak() or
    # ask_ollama(). See wake_listener_loop docstring for the full story.
    wake_listener = threading.Thread(
        target=wake_listener_loop, daemon=True, name="wake-listener"
    )
    wake_listener.start()

    # NOTE: _prewarm_ollama() intentionally NOT called. On the C-Media
    # USB 1.1 mic the 4-5s of bursty CPU during model load silently
    # kills the audio stream (PortAudio reports active=True but no
    # callbacks fire again). Pay the cold-load penalty on the first user
    # query instead, when any stream death at least coincides with the
    # THINKING phase (user isn't talking). The right long-term fix is
    # to pin Ollama to cores 1-3 via `sudo systemctl edit ollama` with
    # CPUAffinity=1 2 3 — once that's in place, this thread is safe to
    # uncomment.

    def watchdog() -> None:
        """Audio-stream liveness check + occasional heartbeat.

        Two cadences:
          - CHECK_INTERVAL (5s): dead-stream detection runs this fast so a
            real failure is caught quickly. Cheap; reads two ints under a
            lock and pokes the stream object.
          - HEARTBEAT_INTERVAL (30s): we only print a routine [heartbeat]
            line this often, to avoid drowning the log when everything is
            healthy. Anomalies (no frames, queue piling up, stream dead)
            still log immediately.

        Why this thread exists at all: PortAudio's input callback can stop
        firing under sustained CPU starvation or after some ALSA glitches.
        We saw it during the in-process whisper days. With whisper-server
        in a separate process this is rare, but the thread is cheap
        insurance and the only way Jarvis recovers without a restart.
        """
        global max_rms_since_heartbeat, frames_since_heartbeat, last_audio_time
        # Tighter cadence so a real stream death is caught in ~3 s rather
        # than ~10 s. The check itself is cheap (two int reads + one
        # PortAudio bool) so polling faster is fine.
        CHECK_INTERVAL = 1.5
        HEARTBEAT_INTERVAL = 30.0
        # 8s default: long enough that an Ollama cold-load (~5-15s of CPU
        # starvation while loading the model from NVMe into RAM) doesn't
        # falsely trip a stream recreation, but short enough that real
        # audio-stack hangs are still caught quickly. Override with
        # JARVIS_STREAM_DEAD_AFTER if you want different behavior.
        STREAM_DEAD_AFTER = float(os.environ.get("JARVIS_STREAM_DEAD_AFTER", "4.0"))
        last_print = 0.0
        while True:
            time.sleep(CHECK_INTERVAL)

            with audio_diag_lock:
                last_t = last_audio_time
                max_rms = max_rms_since_heartbeat
                frames = frames_since_heartbeat

            now = time.monotonic()
            age = (now - last_t) if last_t else -1.0
            q_depth = audio_q.qsize()
            stream = stream_holder[0]

            stream_dead = (last_t and age > STREAM_DEAD_AFTER) or (not stream.active)
            anomaly = (
                stream_dead
                or frames == 0
                or (last_t and age > 3.0)
                or q_depth > 30
                or not worker.is_alive()
            )

            if anomaly or (now - last_print) >= HEARTBEAT_INTERVAL:
                print(
                    f"[heartbeat] frames={frames} last={age:.1f}s ago "
                    f"max_rms={max_rms:.4f} q={q_depth} "
                    f"state={shared.state.name} follow_up={shared.in_follow_up_window()}"
                    + ("  ANOMALY" if anomaly else ""),
                    file=sys.stderr,
                )
                last_print = now
                with audio_diag_lock:
                    max_rms_since_heartbeat = 0.0
                    frames_since_heartbeat = 0

            if not worker.is_alive():
                print("[watchdog] processing thread is dead", file=sys.stderr)

            if stream_dead:
                print(
                    f"[watchdog] audio stream looks dead (age={age:.1f}s, "
                    f"active={stream.active}) — recreating from scratch",
                    file=sys.stderr,
                )
                try:
                    try:
                        stream.stop()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    stream_holder[0] = open_stream()
                    with audio_diag_lock:
                        last_audio_time = time.monotonic()
                    print("[watchdog] stream recreated", file=sys.stderr)
                except Exception:
                    traceback.print_exc()

    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=conversation_watchdog, daemon=True).start()

    orb_center = (w // 2, int(h * 0.28))
    label_y = int(h * 0.48)
    divider_y = int(h * 0.55)
    convo_rect = pygame.Rect(0, divider_y + 20, w, h - divider_y - 20)

    # Tappable rectangle for the online-mode toggle. Sized to be a
    # generous touch target on the 800x480 (or larger) display, and
    # centered where the hint text sits below the state label.
    online_toggle_rect = pygame.Rect(0, 0, 520, 56)
    online_toggle_rect.center = (w // 2, label_y + state_font.get_linesize() + 8)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # 'O' toggles online / deep-think mode. Kept as a keyboard
                # binding for desktops; touchscreen users get the tap
                # target below.
                elif event.key == pygame.K_o:
                    shared.toggle_online()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # On a touchscreen, taps come through as mouse-button
                # events. The online-mode pill takes priority; a tap
                # anywhere else while Jarvis is mid-speech (or thinking)
                # is a "shut up and listen" signal.
                if online_toggle_rect.collidepoint(event.pos):
                    shared.toggle_online()
                elif shared.state in (State.SPEAKING, State.THINKING):
                    print(
                        f"[tap] interrupt while {shared.state.name}",
                        file=sys.stderr,
                    )
                    stop_speaking()
                    shared.set_state(State.IDLE)
                    shared.extend_follow_up()
                    _drain_queue()

        snap = shared.snapshot()

        screen.fill(BG)
        draw_state_orb(
            screen, orb_center,
            snap.state, snap.audio_level, snap.state_age,
            online=snap.online,
        )

        label_text = label_for(snap)
        label_surface = state_font.render(label_text, True, DIM)
        screen.blit(
            label_surface,
            label_surface.get_rect(center=(w // 2, label_y)),
        )

        # Tappable online-mode toggle. Drawn as a rounded-rect "button"
        # so it reads as something you can press — both for touchscreen
        # taps and for keyboard 'O'. The text inside reflects current
        # state and tells the user how to flip it.
        if snap.online:
            badge_text = "\u25CF  ONLINE  \u2014  tap to go local"
            badge_color = ONLINE_BADGE_COLOR
            border_color = ONLINE_BADGE_COLOR
        else:
            badge_text = "TAP HERE for online (deep-think) mode"
            badge_color = HINT_COLOR
            border_color = (60, 65, 90)
        pygame.draw.rect(
            screen, border_color, online_toggle_rect,
            width=1, border_radius=14,
        )
        badge_surface = label_font.render(badge_text, True, badge_color)
        screen.blit(
            badge_surface,
            badge_surface.get_rect(center=online_toggle_rect.center),
        )

        pygame.draw.line(screen, DIVIDER, (60, divider_y), (w - 60, divider_y), 1)
        draw_conversation(screen, convo_rect, label_font, body_font, snap.messages)

        pygame.display.flip()
        clock.tick(60)

    try:
        stream_holder[0].stop()
        stream_holder[0].close()
    except Exception:
        pass
    stop_speaking()
    # Release hardware resources (PWM channel for the servo, etc.) so the
    # next run / servo_test.py invocation can claim them cleanly.
    try:
        import hardware
        hardware.shutdown_all()
    except Exception:
        traceback.print_exc()
    # Wrap up any open conversation before we exit so the user gets a
    # summary written even on a clean shutdown (not just on idle timeout).
    try:
        close_conversation(reason="shutdown")
    except Exception:
        traceback.print_exc()
    pygame.quit()


if __name__ == "__main__":
    main()
