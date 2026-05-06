"""
Cloud adapter for Jarvis's "deep think" / online mode.

Three Groq endpoints, all OpenAI-compatible:
  - chat()        -> /openai/v1/chat/completions   (LLM, llama-3.3-70b)
  - transcribe()  -> /openai/v1/audio/transcriptions (Whisper-large-v3-turbo)
  - synthesize()  -> /openai/v1/audio/speech       (Orpheus, English)

Only used when shared.online is True. The local-first promise stays intact:
- Without a GROQ_API_KEY in the environment, this module is inert.
- The toggle defaults to off; the user has to opt in.
- The UI shows a vivid amber palette + "● ONLINE" badge while active.

Where Groq diverges from Ollama's shape (chat only):
  - tool_calls[i].function.arguments is a JSON string (not a dict).
    We normalize back to a dict so the rest of app.py doesn't care.
  - assistant.content can be null when only tool_calls are present.
    We coerce to '' so downstream code can safely .strip() it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

import tools

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "austin"

# Orpheus accepts up to 200 chars per request. We chunk slightly under so
# trailing punctuation never trips the limit.
TTS_CHUNK_MAX_CHARS = 190

GROQ_TIMEOUT = 30.0


def is_available() -> bool:
    """True iff a Groq key is configured. Cheap — call freely."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def _model() -> str:
    return os.environ.get("GROQ_MODEL", "").strip() or DEFAULT_MODEL


def _stt_model() -> str:
    return os.environ.get("GROQ_STT_MODEL", "").strip() or DEFAULT_STT_MODEL


def _tts_model() -> str:
    return os.environ.get("GROQ_TTS_MODEL", "").strip() or DEFAULT_TTS_MODEL


def _voice() -> str:
    return os.environ.get("GROQ_VOICE", "").strip() or DEFAULT_VOICE


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Adjust the chat history so Groq's stricter validation accepts it.

    Two fix-ups:
      1. Tool messages need a `tool_call_id`. If we previously talked to
         Ollama (which doesn't use ids), the tool messages may only have
         `name`. We fabricate a synthetic id by matching against the most
         recent assistant tool_calls — Groq doesn't care what the id is,
         just that it lines up with an assistant message in scope.
      2. Assistant messages with content=None upset some validators.
         Coerce to ''.
    """
    out: list[dict] = []
    pending_calls: dict[str, str] = {}  # name -> id, from the latest assistant turn

    for m in messages:
        role = m.get("role")

        if role == "assistant":
            new_m = dict(m)
            if new_m.get("content") is None:
                new_m["content"] = ""
            tcs = new_m.get("tool_calls") or []
            pending_calls = {}
            patched = []
            for i, tc in enumerate(tcs):
                tc = dict(tc)
                tc.setdefault("type", "function")
                tc.setdefault("id", f"call_{i}_{int(time.time()*1000)}")
                fn = dict(tc.get("function") or {})
                # Groq wants arguments as a JSON string.
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args)
                tc["function"] = fn
                patched.append(tc)
                pending_calls[fn.get("name", "")] = tc["id"]
            if patched:
                new_m["tool_calls"] = patched
            out.append(new_m)
            continue

        if role == "tool":
            new_m = dict(m)
            if "tool_call_id" not in new_m:
                # Look up by name from the most recent assistant.
                tid = pending_calls.get(new_m.get("name", ""))
                if tid:
                    new_m["tool_call_id"] = tid
                else:
                    # Last resort: fabricate something. Groq will reject if
                    # the assistant turn just before didn't include any
                    # tool_calls, but in our flow it always does.
                    new_m["tool_call_id"] = f"orphan_{int(time.time()*1000)}"
            out.append(new_m)
            continue

        out.append(m)

    return out


def _normalize_response(msg: dict) -> dict:
    """Make the assistant message look like Ollama's shape downstream."""
    msg = dict(msg)
    if msg.get("content") is None:
        msg["content"] = ""
    tcs = msg.get("tool_calls") or []
    for tc in tcs:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except json.JSONDecodeError:
                fn["arguments"] = {}
        tc["function"] = fn
    if tcs:
        msg["tool_calls"] = tcs
    return msg


def chat(messages: list[dict], *, num_predict: int = 280) -> dict:
    """One round-trip to Groq. Returns a message dict shaped like Ollama's.

    Raises requests.RequestException / RuntimeError on failure — the caller
    is expected to fall back to local Ollama if anything goes wrong.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    payload: dict[str, Any] = {
        "model": _model(),
        "messages": _normalize_messages(messages),
        "tools": tools.OLLAMA_TOOL_SCHEMAS,
        "tool_choice": "auto",
        "temperature": 0.5,
        "max_tokens": num_predict,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(
        GROQ_CHAT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=GROQ_TIMEOUT,
    )
    if r.status_code >= 400:
        body = r.text[:400]
        raise RuntimeError(f"groq {r.status_code}: {body}")
    resp = r.json()
    print(
        f"[groq] {time.time()-t0:.1f}s model={_model()} "
        f"tokens={resp.get('usage', {}).get('total_tokens')}",
        file=sys.stderr,
    )
    return _normalize_response(resp["choices"][0]["message"])


# ---------- speech-to-text -------------------------------------------------


def transcribe(wav_bytes: bytes, *, prompt: str = "") -> str:
    """Send a WAV blob to Groq Whisper and return the transcribed text.

    Significantly more accurate than our local ggml-base.en (especially on
    soft speech and homophones), and round-trip latency on Groq is around
    150-300 ms vs ~2-3 seconds locally.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    t0 = time.time()
    data = {
        "model": _stt_model(),
        "response_format": "json",
        "temperature": "0",
        "language": "en",
    }
    if prompt:
        data["prompt"] = prompt
    r = requests.post(
        GROQ_STT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
        data=data,
        timeout=GROQ_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"groq stt {r.status_code}: {r.text[:300]}")
    text = (r.json().get("text") or "").strip()
    print(
        f"[groq stt] {time.time()-t0:.1f}s model={_stt_model()} -> {text!r}",
        file=sys.stderr,
    )
    return text


# ---------- text-to-speech -------------------------------------------------


def chunk_text(text: str, max_chars: int = TTS_CHUNK_MAX_CHARS) -> list[str]:
    """Split text at sentence boundaries so each chunk fits Orpheus's limit.

    Falls back to splitting on whitespace if a single sentence exceeds the
    limit. Empty input yields an empty list.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for s in sentences:
        if not s:
            continue
        if len(s) > max_chars:
            _flush()
            for word in s.split():
                if len(current) + len(word) + 1 > max_chars:
                    _flush()
                    current = word
                else:
                    current = (current + " " + word).strip()
            _flush()
        elif len(current) + len(s) + 1 > max_chars:
            _flush()
            current = s
        else:
            current = (current + " " + s).strip()

    _flush()
    return chunks


def synthesize(text: str, voice: str | None = None) -> bytes:
    """Synthesize speech via Groq Orpheus. Returns WAV bytes (16-bit PCM).

    Caller should pass text under TTS_CHUNK_MAX_CHARS — for longer text use
    chunk_text() and stitch the playback yourself (so the user starts
    hearing audio while the rest is still being synthesized).
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    voice = voice or _voice()
    t0 = time.time()
    r = requests.post(
        GROQ_TTS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": _tts_model(),
            "input": text,
            "voice": voice,
            "response_format": "wav",
        },
        timeout=GROQ_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"groq tts {r.status_code}: {r.text[:300]}")
    print(
        f"[groq tts] {time.time()-t0:.1f}s ({len(text)} chars, "
        f"voice={voice}, bytes={len(r.content)})",
        file=sys.stderr,
    )
    return r.content
