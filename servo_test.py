"""
SG90 servo sweep test (hardware PWM on Pi 5 / RP1).

Wiring (unchanged):
  Red    -> Pi pin 2  (5V)
  Brown  -> Pi pin 6  (GND)
  Orange -> Pi pin 12 (GPIO 18 = PWM0 via dtoverlay=pwm-2chan)

Requires:
  /boot/firmware/config.txt: dtoverlay=pwm-2chan,pin=18,func=2,pin2=19,func2=2
  pip install rpi-hardware-pwm

Press Ctrl+C to stop.
"""

import time
from rpi_hardware_pwm import HardwarePWM

PWM_CHANNEL = 2
PWM_CHIP = 0
PWM_FREQ_HZ = 50

MIN_PULSE_MS = 0.5
MAX_PULSE_MS = 2.5
PERIOD_MS = 1000 / PWM_FREQ_HZ


def angle_to_duty(angle: float) -> float:
    angle = max(-90.0, min(90.0, angle))
    frac = (angle + 90.0) / 180.0
    pulse_ms = MIN_PULSE_MS + frac * (MAX_PULSE_MS - MIN_PULSE_MS)
    return (pulse_ms / PERIOD_MS) * 100.0


def sweep(pwm: HardwarePWM, start: float, end: float, step: float = 2.0, dwell: float = 0.02) -> None:
    direction = 1 if end >= start else -1
    steps = int(abs(end - start) / step)
    for i in range(steps + 1):
        angle = start + direction * step * i
        pwm.change_duty_cycle(angle_to_duty(angle))
        time.sleep(dwell)


pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=PWM_FREQ_HZ, chip=PWM_CHIP)
pwm.start(angle_to_duty(0))

print("SG90 sweep test (hardware PWM). Ctrl-C to exit.\n")

try:
    print("  -> center (0 deg)")
    time.sleep(0.8)

    while True:
        print("  -> sweep  0  ->  +90")
        sweep(pwm, 0, 90)
        time.sleep(0.3)

        print("  -> sweep +90 ->  -90")
        sweep(pwm, 90, -90)
        time.sleep(0.3)

        print("  -> sweep -90 ->   0")
        sweep(pwm, -90, 0)
        time.sleep(0.6)
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    pwm.stop()
    print("PWM released.")
