# OwnStride Foot-Worn Sensory Node Firmware (ESP32 + MPU-6050)

This directory contains the edge firmware for the foot-worn IMU sensor module, designed to capture 6-axis kinematic waveforms (accelerations and angular rates) and deliver adaptive haptic biofeedback cues.

---

## 📌 Hardware Pinout & Wiring

| ESP32 Pin | MPU-6050 (GY-521) | Function / Notes |
| :--- | :--- | :--- |
| **3V3** | **VCC** | Power supply (3.3V) |
| **GND** | **GND** | Ground reference |
| **GPIO 21** | **SDA** | I2C Data line (Hardware I2C0) |
| **GPIO 22** | **SCL** | I2C Clock line (400 kHz Fast Mode) |
| **GPIO 18** | **Motor (+) / Gate** | Haptic ERM Vibration motor driver (via NPN transistor) |

---

## ⚙️ Specifications

> **Status:** not yet compiled or tested on physical hardware. The host-side parser and stride analysis are tested with synthetic data (`tests/test_serial_bridge.py`).

- **Sampling**: 100 Hz fixed-rate schedule (`micros()`-based, no cumulative drift); MPU-6050 DLPF ≈ 44 Hz.
- **Sensor ranges**: accelerometer ±8 g (4096 LSB/g); gyroscope ±1000 °/s (32.8 LSB/°/s).
- **Boot calibration**: keep the foot still for 2 s after power-on; the gyroscope bias is averaged and subtracted from every sample.
- **Mounting**: sensor x-axis toward the toes, y-axis toward the left side of the foot, z-axis up from the shoe. If mounted differently, remap on the host with `--axes` (e.g. `--axes=-y,x,z`).
- **Stream (USB serial, 115200 baud)**, one JSON line per sample (acc in g, gyro in °/s, `t` = `millis()`):
  ```json
  {"seq":1024,"t":10240,"ax":0.082,"ay":-0.412,"az":1.050,"gx":2.1,"gy":-48.2,"gz":-11.8}
  ```
- **Commands (host → device)**:
  | Command | Effect |
  |---|---|
  | `VIBE` or `VIBE,<ms>` | Non-blocking vibration pulse (30–500 ms), refused once the daily budget is used |
  | `STRATEGY,<budget>,<threshold>` | Apply the latest strategy packet (daily cue budget) |
  | `RESET_DAY` | Reset the daily cue counter |

**Stage I split of work**: the MCU samples and executes cues; per-stride foot progression angle and the cue decision run on the host (`src/hardware/serial_bridge.py`). Moving both onto the MCU, so raw samples never leave the device, is Stage II work.

---

## 🧪 Host-side usage

```bash
pip install pyserial
# record one trial (e.g. walking along a tape line at -10°)
python -m src.hardware.serial_bridge record --port COM5 --out trial_minus10.csv --seconds 60
# analyse it against the nominal tape angle
python -m src.hardware.serial_bridge analyze trial_minus10.csv --nominal -10
# live: print every stride, cue when FPA leaves 3°..11°
python -m src.hardware.serial_bridge live --port COM5 --fpa-band 3,11
```

---

## 🛠️ How to Flash (Arduino IDE)

1. Open `firmware/esp32_gait_node.ino` in Arduino IDE.
2. Select Board: **ESP32 Dev Module** (or ESP32-S3 Dev Module).
3. Connect ESP32 via USB and select the appropriate COM port.
4. Click **Upload**.
