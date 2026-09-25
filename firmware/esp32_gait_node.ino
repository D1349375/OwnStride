/*
 * firmware/esp32_gait_node.ino
 * =============================
 * OwnStride Foot-worn Sensory Node Firmware (Stage I prototype)
 * Target MCU: ESP32 / ESP32-S3
 * IMU Sensor: MPU-6050 (GY-521 breakout board via I2C)
 * Haptic Motor: Coreless ERM vibration motor on GPIO 18 (via NPN transistor)
 *
 * Stage I role (see src/hardware/serial_bridge.py):
 * - 100 Hz fixed-rate sampling, streamed as compact JSON over USB serial
 * - Gyroscope bias calibration at boot (keep the foot still for 2 s)
 * - Executes vibration cues requested by the host, enforcing the daily cue budget
 *   from the latest strategy packet on the device side
 *
 * Stage II plan: move per-stride FPA computation and the cue decision onto the MCU,
 * so raw IMU samples no longer need to leave the device.
 *
 * NOTE: not yet compiled or tested on physical hardware.
 */

#include <Wire.h>

// --- Pin Definitions ---
#define SDA_PIN 21
#define SCL_PIN 22
#define HAPTIC_PIN 18
#define LED_STATUS_PIN 2

// --- MPU6050 I2C Address and Registers ---
#define MPU_ADDR 0x68
#define PWR_MGMT_1 0x6B
#define CONFIG_REG 0x1A
#define ACCEL_CONFIG 0x1C
#define GYRO_CONFIG 0x1B
#define ACCEL_XOUT_H 0x3B

// --- Sampling Configuration ---
const unsigned long SAMPLE_INTERVAL_US = 10000;  // 100 Hz
const int CALIBRATION_SAMPLES = 200;              // 2 s at 100 Hz
unsigned long lastSampleMicros = 0;
unsigned long packetSequence = 0;

// Gyroscope bias (deg/s), measured at boot while the foot is still
float gx_bias = 0.0, gy_bias = 0.0, gz_bias = 0.0;

// Strategy packet state (budget enforced on-device)
int daily_cue_budget = 50;
int cues_used_today = 0;
float cue_threshold = 0.0;  // informational in Stage I; the host decides when to cue

// Non-blocking haptic pulse
unsigned long hapticOffAtMs = 0;
bool hapticOn = false;

bool readRaw(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true);
  if (Wire.available() < 14) return false;

  int16_t raw_ax = Wire.read() << 8 | Wire.read();
  int16_t raw_ay = Wire.read() << 8 | Wire.read();
  int16_t raw_az = Wire.read() << 8 | Wire.read();
  Wire.read(); Wire.read();  // temperature (unused)
  int16_t raw_gx = Wire.read() << 8 | Wire.read();
  int16_t raw_gy = Wire.read() << 8 | Wire.read();
  int16_t raw_gz = Wire.read() << 8 | Wire.read();

  // +/-8 g -> 4096 LSB/g, +/-1000 dps -> 32.8 LSB/(deg/s)
  ax = raw_ax / 4096.0; ay = raw_ay / 4096.0; az = raw_az / 4096.0;
  gx = raw_gx / 32.8;   gy = raw_gy / 32.8;   gz = raw_gz / 32.8;
  return true;
}

void writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

void initMPU() {
  writeReg(PWR_MGMT_1, 0x01);    // wake up, PLL with X-gyro reference
  writeReg(CONFIG_REG, 0x03);    // DLPF ~44 Hz (below Nyquist of 100 Hz sampling)
  writeReg(ACCEL_CONFIG, 0x10);  // +/-8 g
  writeReg(GYRO_CONFIG, 0x10);   // +/-1000 deg/s
}

void calibrateGyro() {
  float sx = 0, sy = 0, sz = 0, ax, ay, az, gx, gy, gz;
  int n = 0;
  while (n < CALIBRATION_SAMPLES) {
    if (readRaw(ax, ay, az, gx, gy, gz)) {
      sx += gx; sy += gy; sz += gz;
      n++;
    }
    delay(10);
  }
  gx_bias = sx / n; gy_bias = sy / n; gz_bias = sz / n;
}

void startHaptic(int durationMs) {
  if (cues_used_today >= daily_cue_budget) {
    Serial.println("{\"ack\":\"cue_blocked\",\"reason\":\"daily_budget_exhausted\"}");
    return;
  }
  digitalWrite(HAPTIC_PIN, HIGH);
  hapticOn = true;
  hapticOffAtMs = millis() + durationMs;
  cues_used_today++;
  Serial.print("{\"ack\":\"cue\",\"used\":");
  Serial.print(cues_used_today);
  Serial.print(",\"budget\":");
  Serial.print(daily_cue_budget);
  Serial.println("}");
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000);

  pinMode(HAPTIC_PIN, OUTPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);
  digitalWrite(HAPTIC_PIN, LOW);
  digitalWrite(LED_STATUS_PIN, LOW);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  initMPU();

  Serial.println("{\"status\":\"calibrating\",\"hold_still_ms\":2000}");
  calibrateGyro();

  digitalWrite(LED_STATUS_PIN, HIGH);
  Serial.print("{\"status\":\"ready\",\"node\":\"esp32_foot_imu\",\"fs_hz\":100,\"gyro_bias\":[");
  Serial.print(gx_bias, 2); Serial.print(",");
  Serial.print(gy_bias, 2); Serial.print(",");
  Serial.print(gz_bias, 2); Serial.println("]}");
  lastSampleMicros = micros();
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;  // fixed-rate schedule, no cumulative drift
    streamSample();
  }

  if (hapticOn && (long)(millis() - hapticOffAtMs) >= 0) {
    digitalWrite(HAPTIC_PIN, LOW);
    hapticOn = false;
  }

  if (Serial.available()) {
    handleIncomingCommand();
  }
}

void streamSample() {
  float ax, ay, az, gx, gy, gz;
  if (!readRaw(ax, ay, az, gx, gy, gz)) return;
  gx -= gx_bias; gy -= gy_bias; gz -= gz_bias;
  packetSequence++;

  // ~75 chars x 100 Hz ≈ 7.5 kB/s, within 115200 baud (~11.5 kB/s)
  char line[128];
  snprintf(line, sizeof(line),
           "{\"seq\":%lu,\"t\":%lu,\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,\"gx\":%.1f,\"gy\":%.1f,\"gz\":%.1f}",
           packetSequence, millis(), ax, ay, az, gx, gy, gz);
  Serial.println(line);
}

void handleIncomingCommand() {
  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.startsWith("VIBE")) {
    // "VIBE" or "VIBE,<ms>"
    int comma = line.indexOf(',');
    int durationMs = comma > 0 ? line.substring(comma + 1).toInt() : 120;
    startHaptic(constrain(durationMs, 30, 500));
  } else if (line.startsWith("STRATEGY")) {
    // "STRATEGY,<daily_budget>,<cue_threshold>"
    int c1 = line.indexOf(',');
    int c2 = line.indexOf(',', c1 + 1);
    if (c1 > 0 && c2 > 0) {
      daily_cue_budget = line.substring(c1 + 1, c2).toInt();
      cue_threshold = line.substring(c2 + 1).toFloat();
      Serial.print("{\"ack\":\"strategy_updated\",\"budget\":");
      Serial.print(daily_cue_budget);
      Serial.print(",\"threshold\":");
      Serial.print(cue_threshold, 2);
      Serial.println("}");
    }
  } else if (line.startsWith("RESET_DAY")) {
    cues_used_today = 0;
    Serial.println("{\"ack\":\"day_reset\"}");
  }
}
