"""
Tool definitions for Sage's LLM.

Each tool is a Python function plus a JSON Schema describing its parameters
in OpenAI/Ollama function-calling format. The LLM decides when to call a
tool; we execute it locally and feed the result back.

Tools currently registered:
  get_time         - current date/time, optionally in another timezone
  calculator       - safe arithmetic evaluator (no eval(), AST-based)
  set_timer        - schedule a notification after N seconds
  remember         - store a long-term fact in SQLite (memory.py)
  forget           - remove facts matching a query
  servo_set_angle  - point the servo head to a specific angle
  servo_center     - return the head to 0 deg
  servo_sweep      - smooth back-and-forth between two angles
  servo_nod        - small wobble (yes-ish, given pan-only motion)
  servo_shake      - wide back-and-forth (no)
  servo_wave       - friendly hello sweep
  servo_look_around- slow scan one side to the other
  servo_dance      - random goofy sequence for N seconds
  servo_stop       - cancel any current servo motion
  led_set          - solid colour (stubbed - ring not wired yet)
  led_blink        - blink at rate_hz for duration_s (stubbed)
  led_pulse        - breathing effect (stubbed)
  led_rainbow      - rainbow chase (stubbed)
  led_off          - turn ring off (stubbed)

All tools are 100% local. Web tools (weather, search) can be added later
following the same pattern.
"""

from __future__ import annotations

import ast
import math
import operator as op
import sys
import threading
from datetime import datetime as dt
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import hardware  # local module: sibling file in ~/sage/
import memory  # local module: sibling file in ~/sage/


# ---------- timer notification hook -----------------------------------------
# The app supplies a callback that gets invoked when a timer fires. It's
# expected to surface the message to the user (UI bubble + speech later).
_timer_notifier: Optional[Callable[[str], None]] = None


def set_timer_notifier(fn: Callable[[str], None]) -> None:
    global _timer_notifier
    _timer_notifier = fn


# ---------- get_time --------------------------------------------------------


def get_time(timezone: str = "local") -> str:
    """Return current date and time, optionally in a specific timezone.

    Accepts:
      - "local" (default) for the device's local time.
      - An IANA timezone name like "Asia/Tokyo", "Europe/London", "UTC".
      - A loose city name like "Tokyo" or "New York" (best-effort match
        against common cities — the LLM will usually emit the IANA form
        but we accept the human form too).
    """
    tz_arg = (timezone or "local").strip()
    if tz_arg.lower() in ("", "local", "here", "system"):
        now = dt.now()
        return now.strftime("It is %A, %B %-d %Y at %-I:%M %p in your local time.")

    tz = _resolve_timezone(tz_arg)
    if tz is None:
        return (
            f"I couldn't find a timezone matching '{tz_arg}'. Try an IANA "
            f"name like 'Asia/Tokyo' or 'America/New_York'."
        )

    now = dt.now(tz)
    label = tz_arg if "/" in tz_arg else f"{tz_arg} ({tz.key})"
    return now.strftime(f"It is %A, %B %-d %Y at %-I:%M %p in {label}.")


# Common loose-name → IANA mappings. The LLM usually emits IANA names
# directly, but voice transcription often hands us "Tokyo" or "New York".
_CITY_ALIASES = {
    "tokyo": "Asia/Tokyo",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "moscow": "Europe/Moscow",
    "new york": "America/New_York",
    "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "delhi": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "manipal": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
    "singapore": "Asia/Singapore",
    "dubai": "Asia/Dubai",
    "hong kong": "Asia/Hong_Kong",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "seoul": "Asia/Seoul",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "est": "America/New_York",
    "edt": "America/New_York",
    "utc": "UTC",
    "gmt": "UTC",
}


def _resolve_timezone(name: str) -> Optional[ZoneInfo]:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        pass
    alias = _CITY_ALIASES.get(name.lower().strip())
    if alias:
        try:
            return ZoneInfo(alias)
        except ZoneInfoNotFoundError:
            return None
    return None


# ---------- calculator ------------------------------------------------------
# Safe arithmetic via AST. Never use eval() on user input.

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.USub: op.neg, ast.UAdd: op.pos}

_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}
_SAFE_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "pow": pow,
}


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants allowed")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"disallowed operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"disallowed unary op: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise ValueError("only safe math functions are allowed")
        return _SAFE_FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"disallowed expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    expression = (expression or "").strip()
    if not expression:
        return "Please give me an expression."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as e:
        return f"Couldn't evaluate that: {e}"
    # Tidy up float results.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


# ---------- set_timer -------------------------------------------------------


def set_timer(seconds: float, label: str = "timer") -> str:
    """Schedule a notification to fire after `seconds`."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "I need a number of seconds."
    if seconds <= 0:
        return "Timer needs a positive number of seconds."
    if seconds > 24 * 3600:
        return "I can only set timers up to 24 hours."

    label = (label or "timer").strip() or "timer"

    def _fire() -> None:
        msg = f"Heads up — your {label} just finished."
        if _timer_notifier is not None:
            try:
                _timer_notifier(msg)
            except Exception as e:  # pragma: no cover
                print(f"[timer] notifier error: {e}", file=sys.stderr)
        else:
            print(f"[timer fired] {msg}", file=sys.stderr)

    threading.Timer(seconds, _fire).start()
    pretty = _pretty_duration(seconds)
    return f"OK, I'll let you know in {pretty}."


def _pretty_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''}"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}" + (
            f" and {s} second{'s' if s != 1 else ''}" if s else ""
        )
    h, m = divmod(m, 60)
    parts = [f"{h} hour{'s' if h != 1 else ''}"]
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    return " ".join(parts)


# ---------- memory tools ----------------------------------------------------


def remember(fact: str) -> str:
    """Store a fact in long-term memory."""
    fact = (fact or "").strip()
    if not fact:
        return "I need something to remember."
    if memory.add_fact(fact):
        return f"Got it. I'll remember: {fact}"
    return f"I already had that one noted."


def forget(query: str) -> str:
    """Forget any stored facts matching the query."""
    query = (query or "").strip()
    if not query:
        return "What should I forget?"
    n = memory.remove_facts(query)
    if n == 0:
        return f"I don't have any memories matching '{query}'."
    return f"Forgot {n} fact{'s' if n != 1 else ''}."


# ---------- hardware tools (servo + LED ring) -------------------------------
# All hardware actions are non-blocking: the function returns immediately
# with a confirmation string; the actual motion runs on a background
# thread inside the Servo / LedRing singleton. A new action cancels any
# in-flight one - so "stop wiggling and look left" works naturally.

_SERVO_NOT_READY = (
    "the servo isn't responding - check that the signal wire is on GPIO 18 "
    "(physical pin 12), red on pin 2 (5V), brown on pin 6 (GND), and that "
    "the dtoverlay=pwm-2chan line is in /boot/firmware/config.txt"
)


def servo_set_angle(angle: float = 0) -> str:
    """Point the servo head at a specific angle in degrees (-90..+90)."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    a = max(-90.0, min(90.0, float(angle)))
    s.set_angle(a)
    return f"OK, pointing to {int(a)} degrees."


def servo_center() -> str:
    """Return the head to 0 degrees (straight ahead)."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.center()
    return "OK, centering."


def servo_sweep(
    start: float = -45,
    end: float = 45,
    duration_s: float = 1.0,
    cycles: int = 3,
) -> str:
    """Sweep smoothly between two angles, repeated `cycles` times."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.sweep(start, end, duration_s, cycles)
    return (
        f"Sweeping {int(start)} to {int(end)} "
        f"x{int(cycles)} cycles, {duration_s:.1f}s each."
    )


def servo_nod(count: int = 2) -> str:
    """A soft 'yes' wobble."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.nod(count)
    return f"Nodding {int(count)} times."


def servo_shake(count: int = 3) -> str:
    """A wide 'no' shake."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.shake(count)
    return f"Shaking my head {int(count)} times."


def servo_wave(count: int = 1) -> str:
    """A friendly hello-style wide wave."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.wave(count)
    return f"Waving {int(count)} time(s)."


def servo_look_around(duration_s: float = 4.0) -> str:
    """A slow scan from one side to the other and back."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.look_around(duration_s)
    return f"Looking around for {float(duration_s):.1f} s."


def servo_dance(duration_s: float = 4.0) -> str:
    """A random goofy sequence for the given duration."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.dance(duration_s)
    return f"Dancing for {float(duration_s):.1f} s."


def servo_stop() -> str:
    """Cancel any current servo motion and hold position."""
    s = hardware.get_servo()
    if not s.available:
        return _SERVO_NOT_READY
    s.stop()
    return "Stopped."


# --- LED ring (stubbed - ring not on hand yet) -----------------------------

def led_set(color: str = "white", brightness: float = 0.5) -> str:
    return hardware.get_leds().set_color(color, brightness)


def led_blink(
    color: str = "white", rate_hz: float = 2.0, duration_s: float = 5.0
) -> str:
    return hardware.get_leds().blink(color, rate_hz, duration_s)


def led_pulse(
    color: str = "white", period_s: float = 2.0, duration_s: float = 10.0
) -> str:
    return hardware.get_leds().pulse(color, period_s, duration_s)


def led_rainbow(duration_s: float = 5.0) -> str:
    return hardware.get_leds().rainbow(duration_s)


def led_off() -> str:
    return hardware.get_leds().off()


# ---------- registry & schemas ---------------------------------------------

_REGISTRY: dict[str, Callable[..., str]] = {
    "get_time":         get_time,
    "calculator":       calculator,
    "set_timer":        set_timer,
    "remember":         remember,
    "forget":           forget,
    "servo_set_angle":  servo_set_angle,
    "servo_center":     servo_center,
    "servo_sweep":      servo_sweep,
    "servo_nod":        servo_nod,
    "servo_shake":      servo_shake,
    "servo_wave":       servo_wave,
    "servo_look_around":servo_look_around,
    "servo_dance":      servo_dance,
    "servo_stop":       servo_stop,
    "led_set":          led_set,
    "led_blink":        led_blink,
    "led_pulse":        led_pulse,
    "led_rainbow":      led_rainbow,
    "led_off":          led_off,
}

OLLAMA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": (
                "Return the current date and time. Call this whenever the "
                "user asks about the time, date, day of the week, or what "
                "time it is in another city or timezone."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": (
                            "Timezone or city to get the time for. Use "
                            "'local' (or omit) for the device's local "
                            "time. Otherwise prefer IANA names like "
                            "'Asia/Tokyo' or 'Europe/London'. City names "
                            "like 'Tokyo' or 'New York' also work."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression. Use for any arithmetic "
                "the user asks for. Supports +, -, *, /, **, %, parens, and "
                "math functions like sqrt, sin, cos, log."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to evaluate, e.g. '2*(3+4)' or 'sqrt(16)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": (
                "Set a timer that will notify the user after the given "
                "number of seconds. Use whenever the user asks for a "
                "timer, alarm, or reminder in N minutes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "Duration in seconds.",
                    },
                    "label": {
                        "type": "string",
                        "description": (
                            "Short label for what the timer is for "
                            "(e.g. 'pasta', 'meeting'). Optional."
                        ),
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a fact to long-term memory. Call this when the user "
                "explicitly asks Sage to remember something, or shares a "
                "personal detail Sage should retain across conversations "
                "(e.g. their name, preferences, important dates)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": (
                            "The fact to remember, written as a complete "
                            "sentence in third person about the user. "
                            "Example: 'The user's dog is named Bella.'"
                        ),
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Remove facts from long-term memory. Call this when the "
                "user asks Sage to forget something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Substring or topic to forget about. Any "
                            "stored fact containing this substring is "
                            "deleted."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },

    # ---- Servo (physical head movement) -----------------------------------
    {
        "type": "function",
        "function": {
            "name": "servo_set_angle",
            "description": (
                "Point Sage's head at a specific angle. 0 is straight "
                "ahead, negative angles look one way, positive the other. "
                "Useful for 'look left', 'look right', 'face me'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Angle in degrees, from -90 to +90.",
                    }
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_center",
            "description": "Return the head to 0 degrees (straight ahead).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_sweep",
            "description": (
                "Sweep the head smoothly back and forth between two angles. "
                "Use for 'go up and down', 'move side to side'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "number",
                        "description": "Start angle, -85 to +85.",
                    },
                    "end": {
                        "type": "number",
                        "description": "End angle, -85 to +85.",
                    },
                    "duration_s": {
                        "type": "number",
                        "description": "Seconds for one full back-and-forth cycle.",
                    },
                    "cycles": {
                        "type": "integer",
                        "description": "How many cycles to repeat.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_nod",
            "description": (
                "Soft yes-style wobble. Use for agreement, "
                "acknowledgement, 'nod your head'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of nods (1-10). Default 2.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_shake",
            "description": (
                "Wide no-style shake. Use for disagreement, "
                "'shake your head', refusal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of shakes (1-10). Default 3.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_wave",
            "description": (
                "Friendly hello sweep. Use when the user says hello, "
                "asks Sage to wave, or wants a greeting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of wave cycles (1-5). Default 1.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_look_around",
            "description": (
                "A slow curious scan from one side to the other and back. "
                "Use for 'look around', 'survey the room', curious mood."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_s": {
                        "type": "number",
                        "description": "Total scan duration (1-20). Default 4.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_dance",
            "description": (
                "Random goofy sequence of fast moves for fun. Use for "
                "'dance', 'celebrate', 'be silly', 'show off'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_s": {
                        "type": "number",
                        "description": "Dance duration in seconds (1-30). Default 4.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "servo_stop",
            "description": (
                "Stop any currently running servo motion immediately and "
                "hold position. Use for 'stop', 'hold still', 'freeze'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },

    # ---- LED ring (stubbed - ring not on hand yet) ------------------------
    {
        "type": "function",
        "function": {
            "name": "led_set",
            "description": (
                "Set the LED ring to a solid colour. The ring isn't wired "
                "yet so this will currently respond that it's not "
                "connected, but the user is welcome to ask for it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": (
                            "Colour name (e.g. 'red', 'blue', 'green', "
                            "'white', 'amber') or a hex like '#ff8800'."
                        ),
                    },
                    "brightness": {
                        "type": "number",
                        "description": "0.0 to 1.0. Default 0.5.",
                    },
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "led_blink",
            "description": (
                "Blink the LED ring at a given rate for a given duration. "
                "Stubbed until the WS2812 ring is wired."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color":      {"type": "string", "description": "Colour name or hex."},
                    "rate_hz":    {"type": "number", "description": "Blinks per second. Default 2."},
                    "duration_s": {"type": "number", "description": "Total seconds. Default 5."},
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "led_pulse",
            "description": (
                "Pulse / breathe the LED ring (smooth fade in and out). "
                "Stubbed until the WS2812 ring is wired."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color":      {"type": "string", "description": "Colour name or hex."},
                    "period_s":   {"type": "number", "description": "Seconds per breath. Default 2."},
                    "duration_s": {"type": "number", "description": "Total seconds. Default 10."},
                },
                "required": ["color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "led_rainbow",
            "description": (
                "Run a rainbow chase pattern on the LED ring for the "
                "given duration. Stubbed until the WS2812 ring is wired."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_s": {"type": "number", "description": "Total seconds. Default 5."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "led_off",
            "description": (
                "Turn the LED ring off. Stubbed until the WS2812 ring is "
                "wired."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute(name: str, arguments: Any) -> str:
    """Dispatch a tool call by name. `arguments` is whatever the LLM gave us."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"[no such tool: {name}]"
    if isinstance(arguments, str):
        # Some models pass JSON-as-string. Try to be permissive.
        import json
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        return str(fn(**arguments))
    except TypeError as e:
        return f"[tool {name} got bad arguments: {e}]"
    except Exception as e:  # pragma: no cover
        return f"[tool {name} failed: {e}]"
