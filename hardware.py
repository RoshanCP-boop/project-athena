"""
Hardware singletons for Jarvis (servo + LED ring).

Two devices, both wired to fixed Pi resources (servo on PWM chip 0
channel 2 / GPIO 18, LED ring on SPI MOSI / GPIO 10 once the part
shows up). Each is a single resource — two threads can't drive them
in parallel without conflicts — so we keep one persistent instance
of each for the whole app process and serialise actions through a
per-device lock.

Why an action queue:
Tools need to return promptly so the LLM can speak its reply.  But
some hardware actions (sweeps, dances, blinks) are inherently
long-running.  Each device has a worker thread that runs the most
recent action; calling sweep(...) returns immediately ("OK, sweeping
for 3 s") and the actual motion happens in the background.  A new
action cancels the in-flight one - so "stop dancing and look left"
just works.

LED ring is stubbed today (the part isn't on hand yet).  Tools
return a friendly "not connected" message instead of pretending.
When the WS2812 ring arrives, only the LedRing class needs to grow
real implementations - the tool layer doesn't change.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from typing import Callable, Optional


# --- Servo configuration (matches servo_test.py) ----------------------------
PWM_CHIP = 0
PWM_CHANNEL = 2
PWM_FREQ_HZ = 50

MIN_PULSE_MS = 0.5
MAX_PULSE_MS = 2.5
PERIOD_MS = 1000 / PWM_FREQ_HZ

# Soft limits used by the gestures - keeps motions safely inside the
# servo's physical range and avoids slamming the end-stops.
SAFE_MIN = -85.0
SAFE_MAX = 85.0


def _angle_to_duty(angle: float) -> float:
    angle = max(-90.0, min(90.0, angle))
    frac = (angle + 90.0) / 180.0
    pulse_ms = MIN_PULSE_MS + frac * (MAX_PULSE_MS - MIN_PULSE_MS)
    return (pulse_ms / PERIOD_MS) * 100.0


def _ease_in_out(t: float) -> float:
    """Cubic ease-in-out for smoother visible motion."""
    return 3 * t * t - 2 * t * t * t


class Servo:
    """Single SG90 on hardware PWM, with cooperative-cancel actions."""

    def __init__(self) -> None:
        self._pwm = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_angle = 0.0
        self._available = False

        try:
            from rpi_hardware_pwm import HardwarePWM
            self._pwm = HardwarePWM(
                pwm_channel=PWM_CHANNEL, hz=PWM_FREQ_HZ, chip=PWM_CHIP
            )
            self._pwm.start(_angle_to_duty(0))
            self._available = True
            print(
                f"[hardware.servo] ready on PWM chip={PWM_CHIP} "
                f"channel={PWM_CHANNEL} (GPIO 18)",
                file=sys.stderr,
            )
        except Exception as e:
            print(
                f"[hardware.servo] not available: {e}. "
                f"Tools will return a friendly error.",
                file=sys.stderr,
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def current_angle(self) -> float:
        return self._current_angle

    @property
    def is_busy(self) -> bool:
        """True iff a queued action is currently executing.

        Cheap to call. Used by the ambient-gesture loop to avoid stomping
        on an in-flight LLM tool call (e.g. user said 'wave hello' and the
        wave is still mid-motion).
        """
        t = self._thread
        return t is not None and t.is_alive()

    # ---- low-level ------------------------------------------------------

    def _set(self, angle: float) -> None:
        if not self._available:
            return
        angle = max(-90.0, min(90.0, angle))
        try:
            self._pwm.change_duty_cycle(_angle_to_duty(angle))
        except Exception as e:
            print(f"[hardware.servo] write failed: {e}", file=sys.stderr)
            return
        self._current_angle = angle

    def _smooth(self, from_a: float, to_a: float, duration_s: float) -> None:
        """Eased move between two angles. Honours self._cancel."""
        if not self._available:
            return
        fps = 60
        steps = max(2, int(duration_s * fps))
        dt = duration_s / steps
        for i in range(steps + 1):
            if self._cancel.is_set():
                return
            t = _ease_in_out(i / steps)
            self._set(from_a + t * (to_a - from_a))
            time.sleep(dt)

    # ---- action dispatcher ---------------------------------------------

    def _start_action(self, fn: Callable[[], None]) -> None:
        """Cancel any in-flight action and start a new one in the background."""
        with self._lock:
            self._cancel.set()
            t_old = self._thread
            if t_old is not None and t_old.is_alive():
                # Give the previous action a moment to honour the cancel.
                t_old.join(timeout=2.0)
            self._cancel.clear()
            t = threading.Thread(target=fn, daemon=True, name="servo-action")
            self._thread = t
            t.start()

    # ---- public actions ------------------------------------------------

    def set_angle(self, angle: float) -> None:
        target = max(-90.0, min(90.0, float(angle)))

        def fn() -> None:
            self._smooth(self._current_angle, target, 0.35)

        self._start_action(fn)

    def center(self) -> None:
        self.set_angle(0.0)

    def sweep(
        self,
        start: float,
        end: float,
        duration_s: float = 1.0,
        cycles: int = 3,
    ) -> None:
        start = max(SAFE_MIN, min(SAFE_MAX, float(start)))
        end = max(SAFE_MIN, min(SAFE_MAX, float(end)))
        cycles = max(1, min(20, int(cycles)))
        per_half = max(0.1, duration_s / 2.0)

        def fn() -> None:
            # Position to start cleanly.
            self._smooth(self._current_angle, start, 0.25)
            for _ in range(cycles):
                if self._cancel.is_set():
                    return
                self._smooth(start, end, per_half)
                if self._cancel.is_set():
                    return
                self._smooth(end, start, per_half)
            self._smooth(self._current_angle, 0.0, 0.3)

        self._start_action(fn)

    def nod(self, count: int = 2) -> None:
        """A soft 'yes' wobble. Pan-only servo, so this is a small back-and-forth."""
        count = max(1, min(10, int(count)))

        def fn() -> None:
            base = self._current_angle
            for _ in range(count):
                if self._cancel.is_set():
                    return
                self._smooth(self._current_angle, base - 12, 0.18)
                if self._cancel.is_set():
                    return
                self._smooth(self._current_angle, base + 12, 0.22)
            self._smooth(self._current_angle, base, 0.18)

        self._start_action(fn)

    def shake(self, count: int = 3) -> None:
        """A wide 'no' shake."""
        count = max(1, min(10, int(count)))

        def fn() -> None:
            base = self._current_angle
            for _ in range(count):
                if self._cancel.is_set():
                    return
                self._smooth(self._current_angle, base - 55, 0.28)
                if self._cancel.is_set():
                    return
                self._smooth(self._current_angle, base + 55, 0.36)
            self._smooth(self._current_angle, 0.0, 0.3)

        self._start_action(fn)

    def wave(self, count: int = 1) -> None:
        """A friendly hello wave."""
        count = max(1, min(5, int(count)))

        def fn() -> None:
            for _ in range(count):
                if self._cancel.is_set():
                    return
                self._smooth(self._current_angle, -65, 0.45)
                if self._cancel.is_set():
                    return
                self._smooth(-65, 65, 0.6)
                if self._cancel.is_set():
                    return
                self._smooth(65, -65, 0.6)
            self._smooth(self._current_angle, 0.0, 0.4)

        self._start_action(fn)

    def look_around(self, duration_s: float = 4.0) -> None:
        """Slow, curious scan from one side to the other and back."""
        duration_s = max(1.0, min(20.0, float(duration_s)))

        def fn() -> None:
            half = duration_s / 4.0
            self._smooth(self._current_angle, -70, half)
            if self._cancel.is_set():
                return
            self._smooth(-70, 70, 2 * half)
            if self._cancel.is_set():
                return
            self._smooth(70, 0, half)

        self._start_action(fn)

    def twitch(self) -> None:
        """A single quick startle - sharp turn and snap back."""
        def fn() -> None:
            target = random.choice([-45, 45])
            self._smooth(self._current_angle, target, 0.08)
            if self._cancel.is_set():
                return
            self._smooth(target, 0, 0.18)

        self._start_action(fn)

    def dance(self, duration_s: float = 4.0) -> None:
        """Random goofy sequence for the given duration, then return to center."""
        duration_s = max(1.0, min(30.0, float(duration_s)))

        def fn() -> None:
            end_t = time.monotonic() + duration_s
            while time.monotonic() < end_t and not self._cancel.is_set():
                move = random.choice(["snap", "sweep", "wiggle", "pause"])
                if move == "snap":
                    target = random.uniform(-70, 70)
                    self._smooth(self._current_angle, target, 0.18)
                elif move == "sweep":
                    a = random.uniform(-70, -10)
                    b = random.uniform(10, 70)
                    self._smooth(self._current_angle, a, 0.3)
                    if self._cancel.is_set():
                        return
                    self._smooth(a, b, 0.45)
                elif move == "wiggle":
                    base = self._current_angle
                    for _ in range(random.randint(2, 5)):
                        if self._cancel.is_set():
                            return
                        self._smooth(
                            self._current_angle,
                            base + random.uniform(-18, 18),
                            0.1,
                        )
                else:
                    time.sleep(random.uniform(0.1, 0.3))
            self._smooth(self._current_angle, 0.0, 0.4)

        self._start_action(fn)

    def stop(self) -> None:
        """Cancel any motion and hold position (does not power down PWM)."""
        with self._lock:
            self._cancel.set()

    def shutdown(self) -> None:
        """Release PWM cleanly. Call once on app exit."""
        self._cancel.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._pwm is not None:
            try:
                self._pwm.stop()
            except Exception:
                pass
            self._pwm = None
        self._available = False


# --- LED ring (stubbed until the WS2812 12-LED ring is wired) -------------


_LED_NOT_WIRED_MSG = (
    "the LED ring isn't wired up yet - it'll be on SPI MOSI (GPIO 10) "
    "once the part arrives"
)


class LedRing:
    """Placeholder for the WS2812 ring. All actions return a polite error."""

    def __init__(self) -> None:
        self._available = False
        print(
            "[hardware.leds] stubbed - actions will return 'not connected'",
            file=sys.stderr,
        )

    @property
    def available(self) -> bool:
        return self._available

    def set_color(self, color: str = "white", brightness: float = 0.5) -> str:
        return _LED_NOT_WIRED_MSG

    def blink(
        self, color: str = "white", rate_hz: float = 2.0, duration_s: float = 5.0
    ) -> str:
        return _LED_NOT_WIRED_MSG

    def pulse(
        self, color: str = "white", period_s: float = 2.0, duration_s: float = 10.0
    ) -> str:
        return _LED_NOT_WIRED_MSG

    def rainbow(self, duration_s: float = 5.0) -> str:
        return _LED_NOT_WIRED_MSG

    def off(self) -> str:
        return _LED_NOT_WIRED_MSG

    def shutdown(self) -> None:
        pass


# --- module-level singletons ------------------------------------------------

_servo: Optional[Servo] = None
_leds: Optional[LedRing] = None
_init_lock = threading.Lock()


def get_servo() -> Servo:
    global _servo
    with _init_lock:
        if _servo is None:
            _servo = Servo()
        return _servo


def get_leds() -> LedRing:
    global _leds
    with _init_lock:
        if _leds is None:
            _leds = LedRing()
        return _leds


def shutdown_all() -> None:
    """Release every initialised device. Safe to call multiple times."""
    global _servo, _leds
    with _init_lock:
        if _servo is not None:
            try:
                _servo.shutdown()
            except Exception:
                pass
            _servo = None
        if _leds is not None:
            try:
                _leds.shutdown()
            except Exception:
                pass
            _leds = None


if __name__ == "__main__":
    # Tiny smoke test: nod, shake, wave, dance, then exit.
    s = get_servo()
    if not s.available:
        print("Servo not available - bailing.")
        raise SystemExit(1)
    print("nod"); s.nod(2);            time.sleep(2.0)
    print("shake"); s.shake(2);        time.sleep(2.5)
    print("wave"); s.wave(1);          time.sleep(3.0)
    print("look_around"); s.look_around(3.0); time.sleep(3.5)
    print("dance"); s.dance(3.0);      time.sleep(3.5)
    print("center"); s.center();       time.sleep(0.5)
    shutdown_all()
