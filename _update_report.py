"""
One-shot update script for Midterm_Report_RoshanChackoPaul_v4.docx.

Two passes over the document:

  1. Globally replace em-dashes (U+2014) and en-dashes (U+2013) with
     plain ASCII hyphens, preserving run-level formatting (bold, italic,
     etc.) by editing run.text in place.

  2. Surgical content updates:
     - paragraphs that drifted from reality (openWakeWord deployed,
       Piper integrated, heartbeat throttled, date/time replaced by
       tool, UI labels, etc.) are rewritten in place.
     - two new subsections (6.3.7 online mode, 6.3.8 tools + memory)
       are inserted before the §7 RESULTS heading.
     - Section 8 (Work Proposed for Next Term) trims items that are
       now done.

Output: ~/sage/Midterm_Report_RoshanChackoPaul_v5.docx
"""

import re
from docx import Document

SRC = "/home/sage/sage/Midterm_Report_RoshanChackoPaul_v4.docx"
DST = "/home/sage/sage/Midterm_Report_RoshanChackoPaul_v5.docx"


# -------------------------------------------------------------------------
# Pass 1: dash replacement (run-level, preserves formatting)
# -------------------------------------------------------------------------

def fix_dashes(text: str) -> str:
    if not text:
        return text
    # Em-dash with optional surrounding whitespace -> " - "
    text = re.sub(r"\s*\u2014\s*", " - ", text)
    # En-dash -> plain hyphen (handles number ranges like "1.2-3.3")
    text = text.replace("\u2013", "-")
    # Collapse any doubled spaces created by the substitution.
    text = re.sub(r" {2,}", " ", text)
    return text


def fix_paragraph_dashes(para) -> None:
    for run in para.runs:
        if run.text:
            run.text = fix_dashes(run.text)


# -------------------------------------------------------------------------
# Pass 2: content rewrites
# -------------------------------------------------------------------------

def replace_paragraph_text(para, new_text: str) -> None:
    """Set paragraph text, preserving paragraph style.

    Keeps the first run (so its font/style settings carry over) and
    blanks the rest.
    """
    if not para.runs:
        para.add_run(new_text)
        return
    para.runs[0].text = new_text
    for run in para.runs[1:]:
        run.text = ""


def replace_cell_text(cell, new_text: str) -> None:
    """Replace a table cell's first paragraph text (single-line cells)."""
    if not cell.paragraphs:
        cell.add_paragraph(new_text)
        return
    p = cell.paragraphs[0]
    replace_paragraph_text(p, new_text)
    # Drop any extra paragraphs in the cell to keep it tidy.
    for extra in cell.paragraphs[1:]:
        replace_paragraph_text(extra, "")


# Rewrites keyed by paragraph index (verified against the v4 dump).
PARAGRAPH_UPDATES: dict[int, str] = {

    # --- Abstract: midterm status paragraph -----------------------------
    40: (
        "At the midterm stage all hardware subsystems integrated to date "
        "have been individually verified: INA219 (I\u00B2C 0x40, voltage / "
        "current / power readings validated), SG90 servo (jitter-free "
        "180\u00B0 sweeps via hardware PWM on GPIO 18), USB microphone and "
        "USB sound card (capture and playback verified), and the DSI "
        "touchscreen (60 Hz rendering, touch input working). The full "
        "speech pipeline is operational end-to-end: openWakeWord "
        "'hey_jarvis' detection on the live audio stream -> VAD-gated "
        "capture -> whisper.cpp STT (1.2-3.3 s on utterances up to 8 s) -> "
        "Ollama Qwen2.5-1.5B inference (5-12 s) -> Piper TTS playback "
        "through the USB sound card. Reliability mechanisms (stream "
        "watchdog, throttled heartbeat thread, frame-drop during "
        "inference, FSM try/finally guards) have been added in response "
        "to observed failure modes."
    ),

    # --- Abstract: results + remaining work paragraph -------------------
    41: (
        "The subsystem-level results demonstrate that a fully on-device "
        "conversational AI pipeline is achievable on consumer-grade "
        "embedded hardware with acceptable latency and reliability. "
        "Remaining work, scoped for the second half of the practice "
        "school period, comprises: PAM8403 amplifier and 40 mm speaker "
        "integration, NeoPixel ring wiring, DS-213K push-button wiring, "
        "training a custom 'Sage' openWakeWord model, and INA219-driven "
        "adaptive LLM tier switching. Software tools and hardware used: "
        "Python 3.13, whisper.cpp, Ollama, Qwen2.5 (1.5B / 0.5B), Piper "
        "TTS (en_US-lessac-medium), openWakeWord, pygame, gpiozero, "
        "rpi-hardware-pwm, adafruit-circuitpython-ina219, webrtcvad, "
        "python-dotenv, SQLite, ALSA, Raspberry Pi 5 8 GB, INA219, SG90 "
        "servo, USB microphone, USB sound card, Raspberry Pi Touch "
        "Display 2, PAM8403 amplifier, WS2812 NeoPixel ring."
    ),

    # --- §1.3: Voice Output Layer ---------------------------------------
    118: (
        "3.\tVoice Output Layer: Piper TTS (en_US-lessac-medium voice) "
        "synthesises assistant responses on-device and pipes directly to "
        "aplay; audio output currently routes through the USB sound "
        "card. The dedicated PAM8403 amplifier and 40 mm speaker are "
        "scoped for second-half integration."
    ),

    # --- §1.3: closing paragraph (mention optional online mode) ---------
    122: (
        "The physical platform includes a 5-inch Official Raspberry Pi "
        "Touch Display 2 connected via DSI, a WS2812 12-LED NeoPixel ring "
        "for visual state indication, and an SG90 servo motor for "
        "rotational head movement. By default all inference runs locally "
        "on the Raspberry Pi 5 with no external API calls. An optional "
        "'deep-think' mode (described in \u00A76.3.7) lets the user "
        "explicitly route the LLM, STT, and TTS through Groq's hosted "
        "models for higher quality at the cost of network dependency; "
        "the user must opt in via an on-screen toggle, and the UI "
        "provides unmistakable visual feedback whenever queries leave "
        "the device."
    ),

    # --- §6.3.3: LLM system prompt: replace date-injection with tools ---
    220: (
        "The system prompt enforces the following constraints to suit a "
        "voice interface: replies are kept to 1-3 sentences (num_predict "
        "= 220 token cap); markdown, lists, and follow-up suggestions "
        "('anything else I can help with') are forbidden, as they are "
        "discordant when read aloud; the model is given access to a "
        "small set of locally executed tools (see \u00A76.3.8) including "
        "get_time, which the model is instructed to call when asked "
        "about the current date or time, removing the need to inject "
        "volatile time values into every prompt; charitable "
        "interpretation of likely Whisper transcription errors is "
        "encouraged, with explicit instructions to ask one short "
        "clarifying question when the input is genuinely nonsensical "
        "rather than guessing. Conversation history is capped at the "
        "last four turns to bound prompt size and inference cost."
    ),

    # --- §6.3.3 status --------------------------------------------------
    221: (
        "Status: Both tiers pulled, tested, and integrated. Tool calling "
        "(\u00A76.3.8) and online deep-think routing (\u00A76.3.7) are also "
        "live. Adaptive switching from INA219 voltage readings is the "
        "next planned integration step."
    ),

    # --- §6.3.4: wake-word section --------------------------------------
    225: (
        "In the IDLE state, the assistant ignores all utterances except "
        "those triggered by the openWakeWord 'hey_jarvis' detector "
        "running in parallel on the live 16 kHz audio stream (a "
        "~200 KB TFLite model with negligible CPU cost on the Pi 5). "
        "After a successful reply, a 15-second follow-up window is "
        "opened during which the wake word is not required, allowing "
        "natural multi-turn dialogue without repeated prompting. Within "
        "this window a backup phonetic matcher catches the literal word "
        "'Sage' (and common Whisper mishearings: page, phage, paige, "
        "stage, siege, sayge, saje) so the user can re-cue the assistant "
        "naturally if the dialogue lapses. The follow-up window is "
        "captured at the start of speech and carried through "
        "end-of-utterance to prevent a race condition where the window "
        "expires mid-utterance and orphans the UI in a non-IDLE state."
    ),
    226: (
        "Two openWakeWord variants were scoped at the design stage. "
        "Path A, now deployed, uses the built-in 'hey_jarvis' pre-trained "
        "model from openWakeWord: free, fully offline, and a ~200 KB "
        "TFLite model that is production-ready immediately. Path B, "
        "planned for the second half of the practice school period, "
        "trains a custom 'Sage' wake-word model using openWakeWord's "
        "synthetic-data pipeline (Piper-TTS-generated positive samples "
        "and ~30,000 hours of negative audio); estimated 1-2 hours of "
        "GPU compute on a separate workstation, producing a ~200 KB "
        "TFLite model deployable to the Pi to give the assistant its "
        "eponymous trigger word."
    ),

    # --- §6.3.4 status --------------------------------------------------
    229: (
        "Status: openWakeWord 'hey_jarvis' integration deployed and "
        "operational. The phonetic-tolerance regex remains as a fallback "
        "matcher within the post-wake follow-up window. Custom 'Sage' "
        "wake-word training (Path B) is scoped for the next term."
    ),

    # --- §6.3.5: heartbeat (every 5 s -> 30 s, anomaly-aware) -----------
    232: (
        "\u2022\tHeartbeat thread - prints the number of audio frames "
        "received, time since last frame, peak RMS, queue depth, and the "
        "current FSM state once every 30 seconds, and immediately on "
        "anomaly (zero frames received, queue depth above 30, or stream "
        "age exceeding 3 s). Provides at-a-glance pipeline health "
        "without flooding the log under normal operation."
    ),

    # --- §6.3.6: state labels include SPEAKING --------------------------
    237: (
        "The pygame-based UI displays an animated state orb whose "
        "colour, pulse rate, and amplitude reflect the FSM state and "
        "live audio level; a status label below the orb (e.g. 'Say Hey "
        "Jarvis to start', 'Listening (12 s)...', 'Thinking', "
        "'Speaking'); and a live conversation transcript area showing "
        "the most recent user / assistant exchanges, colour-coded by "
        "speaker."
    ),
    238: (
        "The UI is rendered fullscreen on the dedicated touchscreen "
        "display at 60 Hz, with the audio pipeline and LLM inference "
        "running in dedicated background threads to keep render latency "
        "under 16 ms regardless of speech-processing load. A "
        "pill-shaped tap target underneath the state label exposes the "
        "optional online deep-think toggle (see \u00A76.3.7); when active it "
        "shifts the UI to a vivid amber palette and shows a "
        "\u25CF ONLINE badge so the user can never lose track of "
        "whether queries are leaving the device. A debug-only CLI variant "
        "(listen.py) mirrors the speech pipeline without the graphical "
        "surface, used during development to validate transcripts and "
        "timings on a headless connection."
    ),

    # --- §7.1 Results: VAD/wake paragraph -------------------------------
    249: (
        "VAD and capture: the empirically tuned VAD parameters (Table 3) "
        "reliably accept soft-spoken English utterances within the "
        "post-wake follow-up window, while rejecting micro-noise blips, "
        "door clicks, and quiet ambient speech. Wake-word detection "
        "itself is now performed by the openWakeWord 'hey_jarvis' model "
        "rather than by Whisper-and-regex; this eliminated the class of "
        "false negatives where soft 'Sage' utterances were transcribed "
        "as 'page' or 'paige' and rejected by the literal-word matcher. "
        "The 750 ms end-of-utterance silence threshold has been "
        "comfortable in real-world testing, with no observed truncations "
        "of natural speech and no observed endless captures."
    ),

    # --- §7.2 Conclusions -----------------------------------------------
    258: (
        "The principal challenges identified at this stage are the "
        "dedicated speaker hardware path (PAM8403 amplifier and 40 mm "
        "speaker, deferred pending soldering; current playback uses the "
        "USB sound card), the wiring of the WS2812 NeoPixel ring "
        "(pending component arrival), the wiring of the DS-213K push "
        "buttons, training of the custom 'Sage' openWakeWord model "
        "(Path B, scoped on a separate workstation), and the automation "
        "of LLM tier switching from INA219 voltage readings (specified "
        "but not yet wired into the runtime). All of these are scoped "
        "for the second half of the practice school period."
    ),

    # --- §8.1 (TTS / speaker / amp) -------------------------------------
    262: "8.1  Speaker and Amplifier Hardware Integration (Weeks 12-14)",
    263: (
        "\u2022\tSolder the PAM8403 class-D amplifier module and wire to "
        "the 40 mm 4 \u03A9 speaker. Verify quiescent draw and frequency "
        "response across the speech band."
    ),
    264: (
        "\u2022\tRoute Piper / Orpheus playback through the soldered "
        "speaker once the amplifier is wired (the software path - "
        "Piper or Orpheus output through aplay - is already operational "
        "on the USB sound card)."
    ),
    265: (
        "\u2022\tAdd a brief audio tone signalling 'ready to listen' to "
        "address the disorientation caused by silent processing gaps "
        "identified in supervisor feedback."
    ),

    # --- §8.2 (Wake word) -----------------------------------------------
    266: "8.2  Custom 'Sage' Wake-Word Model (Weeks 12-14)",
    267: (
        "\u2022\tTrain a custom 'Sage' wake-word model using "
        "openWakeWord's synthetic-data pipeline (Piper-generated "
        "positives plus ~30,000 hours of negatives) on a separate "
        "workstation. Deploy the resulting ~200 KB TFLite model as the "
        "production wake-word detector, replacing the currently active "
        "'hey_jarvis' Path A model."
    ),
    268: (
        "\u2022\tEvaluate false-accept and false-reject rates of the "
        "custom model against the deployed 'hey_jarvis' baseline under "
        "the same domestic-noise conditions used in the present testing."
    ),
}


# Table cell rewrites: (table_index, row_index, col_index) -> new text.
# Used to refresh status columns in the hardware summary now that more
# software pipeline items are operational; statuses themselves are
# unchanged because the work that's been done is software-side.
TABLE_UPDATES: dict[tuple[int, int, int], str] = {}


# -------------------------------------------------------------------------
# Pass 3: insert new subsections before the §7 RESULTS heading
# -------------------------------------------------------------------------

NEW_SECTIONS = [
    "6.3.7  Optional Online \"Deep-Think\" Mode",
    (
        "When network connectivity is available, the system supports an "
        "opt-in online mode that routes inference through Groq's hosted "
        "models for higher quality at the cost of leaving the device. "
        "The feature is disabled by default and requires a GROQ_API_KEY "
        "in a local .env file (loaded via python-dotenv at startup); the "
        "user must explicitly toggle it on at runtime via an on-screen "
        "pill in the touchscreen UI. While the toggle is active, the "
        "entire UI palette shifts to amber and a \u25CF ONLINE badge "
        "appears below the state label so the user has constant visual "
        "confirmation that audio and queries are leaving the device."
    ),
    (
        "The toggle flips three pipeline stages simultaneously: "
        "speech-to-text moves from local whisper.cpp ggml-base.en to "
        "Groq whisper-large-v3-turbo; LLM inference moves from local "
        "Ollama qwen2.5:1.5b to Groq llama-3.3-70b-versatile; and "
        "text-to-speech moves from local Piper en_US-lessac-medium to "
        "Groq Orpheus (canopylabs/orpheus-v1-english, 'austin' voice). "
        "Each of the three remote stages falls back to its local "
        "equivalent on any failure (network error, rate limit, API "
        "outage), so a flaky link never breaks the assistant."
    ),
    (
        "The wake-word detector remains local in both modes; raw "
        "microphone audio is never streamed to the cloud unless an "
        "utterance has been explicitly captured following a wake event. "
        "A user with no internet, or simply with no key configured, "
        "gets exactly the same offline behaviour described in "
        "\u00A76.3.1-\u00A76.3.6, preserving the project's local-first "
        "ethos while making the higher-quality option available when "
        "the user wants it."
    ),
    (
        "Status: Implemented and operational. Default state is off; the "
        "feature is opt-in by tap or by keyboard 'O'."
    ),

    "6.3.8  Tool Calling and Long-Term Memory",
    (
        "Both LLM tiers (local Ollama and online Groq) are extended with "
        "a small set of locally executed tools made available through "
        "the standard OpenAI function-calling schema. Five tools are "
        "currently registered: get_time returns the current date and "
        "time in any IANA timezone or named city; calculator is a "
        "sandboxed AST-based arithmetic evaluator (no use of eval()) "
        "supporting standard operators and math functions; set_timer "
        "schedules a background notification after a given number of "
        "seconds, surfaced as a Sage bubble in the conversation "
        "transcript on fire; remember and forget write and delete "
        "entries in a local SQLite-backed long-term memory store at "
        "~/sage/memory.db. Stored facts are injected into the system "
        "prompt for subsequent sessions, giving the assistant continuity "
        "across reboots."
    ),
    (
        "In addition to the explicit remember and forget facts, the same "
        "SQLite database tracks every conversation as a sequence of "
        "user / assistant messages and writes a one-sentence "
        "auto-summary to disk when a conversation goes idle (default "
        "90 seconds with no new turn) or on clean shutdown. Summaries "
        "are produced by a background thread that calls the smaller "
        "qwen2.5:0.5b model so the user-facing pipeline is not blocked. "
        "The two most recent summaries are made available to future "
        "system prompts, providing rolling cross-session context "
        "without inflating the prompt for every individual turn."
    ),
    (
        "Status: Implemented and operational. Tool calls and memory "
        "writes work in both offline and online modes."
    ),
]


def insert_new_sections_before(target_para, paragraph_texts):
    """Insert paragraphs immediately before target_para, in order.

    Each new paragraph inherits Normal style. The target paragraph
    remains in place.
    """
    for txt in paragraph_texts:
        target_para.insert_paragraph_before(txt)


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------

def main():
    doc = Document(SRC)

    # Pass 1: dashes, everywhere.
    for p in doc.paragraphs:
        fix_paragraph_dashes(p)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    fix_paragraph_dashes(p)

    # Pass 2: surgical paragraph rewrites.
    for idx, new_text in PARAGRAPH_UPDATES.items():
        if idx >= len(doc.paragraphs):
            print(f"WARN: paragraph index {idx} out of range, skipping")
            continue
        replace_paragraph_text(doc.paragraphs[idx], new_text)

    # Optional table updates.
    for (ti, ri, ci), new_text in TABLE_UPDATES.items():
        if ti >= len(doc.tables):
            print(f"WARN: table index {ti} out of range")
            continue
        tbl = doc.tables[ti]
        if ri >= len(tbl.rows) or ci >= len(tbl.rows[ri].cells):
            print(f"WARN: cell ({ti},{ri},{ci}) out of range")
            continue
        replace_cell_text(tbl.rows[ri].cells[ci], new_text)

    # Pass 3: insert new sections before the BODY §7 heading.
    # The TOC also has an entry "7.  RESULTS AND CONCLUSIONS\t<page>"
    # which we must skip - match the body heading by its bare text
    # (no trailing tab + page number).
    target_para = None
    for p in doc.paragraphs:
        t = p.text.strip()
        if (t.startswith("7.")
                and "RESULTS AND CONCLUSIONS" in t.upper()
                and "\t" not in p.text
                and not any(ch.isdigit() for ch in t.split("CONCLUSIONS")[-1])):
            target_para = p
            break
    if target_para is None:
        print("ERROR: could not find §7 RESULTS heading; "
              "skipping new-section insertion")
    else:
        insert_new_sections_before(target_para, NEW_SECTIONS)

    doc.save(DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
