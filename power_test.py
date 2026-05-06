import time
import board
from adafruit_ina219 import INA219

i2c = board.I2C()
ina = INA219(i2c)

print("Reading INA219. Ctrl-C to exit.\n")

while True:
    bus_v = ina.bus_voltage
    shunt_v = ina.shunt_voltage
    current = ina.current
    power = ina.power
    source_v = bus_v + shunt_v
    print(
        f"Source: {source_v:6.3f} V  |  Bus: {bus_v:6.3f} V  |  "
        f"Shunt: {shunt_v*1000:6.2f} mV  |  "
        f"Current: {current:7.2f} mA  |  Power: {power:7.2f} mW"
    )
    time.sleep(0.5)
