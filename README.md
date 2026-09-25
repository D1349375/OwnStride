# OwnStride — On-Device AI Walking-Form Trainer
### ASUS UGen AI League 2026 | Battlefield Lightning Track (Everyday AI)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-55%20passed-success.svg)](https://pytest.org)
[![Hailo-10H](https://img.shields.io/badge/Target%20NPU-Hailo--10H%20(UGen300)-orange.svg)](https://hailo.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **One-liner**: **OwnStride** is an on-device AI walking-form trainer that learns *your own* best days, fades its guidance only as you truly improve, and turns your progress into a weekly training plan through an on-device LLM — with no cloud dependency, ever.

---

## 💡 The Problem: Three Fundamental Industry Flaws

Current smart insole and gait tracking devices (e.g., Digitsole, Moticon, and the defunct Nurvv Run) fail users due to three structural flaws:

```
┌───────────────────────────┬───────────────────────────────────┬──────────────────────────────────────┐
│ Conventional Flaw         │ Why It Fails                      │ OwnStride Design Solution            │
├───────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
│ 1. Wrong Reference        │ Compares user against arbitrary   │ Learns the user's own best days      │
│    Baseline               │ population norms (high false +)   │ (Ledoit-Wolf, one-sided, ratchet)    │
├───────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
│ 2. Feedback Dependency    │ Continuous vibration creates      │ Adaptive Faded Feedback FSM:         │
│    (Guidance Hypothesis)  │ dependency; form reverts once off │ Cues fade 60-90% as form stabilizes  │
├───────────────────────────┼───────────────────────────────────┼──────────────────────────────────────┤
│ 3. Data-Action Gap        │ Gives raw sensor numbers users    │ On-Device LLM transforms trends into │
│                           │ don't know how to train           │ actionable weekly movement routines  │
└───────────────────────────┴───────────────────────────────────┴──────────────────────────────────────┘
```

---

## 🏛️ System Architecture

OwnStride partitions computation strictly by **evolutionary necessity** rather than physical placement:

```
┌────────────────────────────────────────────────────────────────────────┐
│  Sensory Layer — Foot-Worn Unit (All-Day Wear, Continuous Stream)      │
│  Hardware: 6-Axis IMU (MPU-6050) + MCU (ESP32-S3) + Coreless ERM Motor │
├────────────────────────────────────────────────────────────────────────┤
│    ① Deterministic Signal Processing (No Neural Network)               │
│       Zero-velocity detection -> strapdown attitude integration + ZUPT │
│       -> per-stride foot progression angle, eversion, stride length    │
│    ② Closed-Loop Strategy Executor                                     │
│       Evaluates current stride against Strategy Packet threshold       │
│       Triggers haptic cue if deviation > threshold & daily budget left │
└────────────────────────────────────────────────────────────────────────┘
                                    │
             Sync via BLE / USB (Periodic: Daily / Weekly, ~15 KB)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Evaluation Layer — Home Dock Base Station + ASUS UGen300 (Hailo-10H)  │
│  Frequency: Weekly / Daily Non-realtime | Latency Budget: Seconds     │
├────────────────────────────────────────────────────────────────────────┤
│    ③ Personal Movement Baseline (Ledoit-Wolf Shrinkage Estimation)    │
│       Calibrates robust inverse covariance matrix on small samples     │
│       Computes Mahalanobis Distance (DM) for true coordinated variance │
│    ④ Adaptive Faded Controller (Schmitt Trigger Hysteresis FSM)        │
│       Phase 1 (Acquisition, 50 cues/day)                               │
│       -> Phase 2 (Faded Feedback, 20 cues/day, 60% reduction)          │
│       -> Phase 3 (Retention, 5 cues/day, sparse proprioceptive check)  │
│       Emits compact Strategy Packet to Sensory Layer                   │
│    ⑤ On-Device LLM Training Planner (Qwen2.5-1.5B INT4)                │
│       Transforms weekly metrics into structured JSON & movement routine│
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Interface Layer — Responsive Web Dashboard (Mobile & Desktop)         │
│    L1: Today's 3-Second Glance (Deviation, Phase, Remaining Budget)     │
│    L2: Personal Corridor, Personal-Best Ratchet, Weekly Trend, z-Scores│
│    L3: Structured Weekly Training Routine (Target Muscles & Rationale) │
│    Toggle: Personal mode <-> Share-mode concept (Phase 2, not built)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 What Is Real, What Is Synthetic (Stage I)

| Component | Status |
|---|---|
| Foot progression angle / eversion / stride length | Computed from 6-axis IMU signals by strapdown attitude integration with zero-velocity updates (`src/features/foot_imu.py`). Magnetometer-free: each stride's heading reference is re-set at foot-flat. |
| Algorithm validation | On a **kinematics-first synthetic simulator** (`src/dataset/synthetic_stream.py`): foot motion is defined first, IMU readings are derived from it, and the analyzer must recover the known angles from the signals alone. Mean FPA error < 0.4° across normal / in-toeing / out-toeing / over-pronation, both feet. This checks the mathematics, **not real-world accuracy**. |
| Real human data | WISDM v1.1 (Kwapisz et al. 2011) smartphone accelerometer only — no gyroscope — so it is used **only** to check gait-cycle segmentation on real walking. It cannot produce a foot progression angle. MAREA requires a signed data-release agreement and has not been obtained. |
| Hardware | ESP32 + MPU-6050 firmware (`firmware/`) and host serial bridge (`src/hardware/serial_bridge.py`) with a tape-line validation mode. **Not yet tested on a worn device.** |
| Personal baseline | **Personal-best days**: during a 14-day calibration the user walks as usual; the best 5 days in the user-chosen training direction (reduce in-toeing / reduce out-toeing / maintain) form the Ledoit-Wolf baseline. Deviation uses **only the trained feature** (foot progression angle) and is **one-sided** — doing better than your best is never a deviation; other features (stride length, impact, …) change naturally as gait improves, so they are displayed as z-scores but not scored (with a “maintain” goal all 6 features are scored by Ledoit-Wolf Mahalanobis distance). Every week the best 5 of the last 14 days are re-selected and the baseline only moves up (ratchet). Stepping to a lighter feedback phase requires at least one ratchet in the current phase, so cues fade only after real progress. No population norm is used. |
| Six-week trajectory | Synthetic, with an assumed S-shaped learning curve. Every number on the dashboard is computed by the full pipeline; none are hard-coded. |
| Weekly plan | Local Ollama (Qwen2.5-1.5B, the same model targeted for Hailo-10H in Stage II), streamed token by token with measured tokens/s — about 8–10 tok/s and ~500 tokens (~60–80 s) on a laptop CPU. Numbers are interpreted in code; the LLM only writes the narrative (a 1.5B model misreads raw numbers). Chinese output is converted to Traditional Chinese with OpenCC. If Ollama is unavailable, a rule-based fallback is used and **labelled as such** in the UI. |

---

## ⚡ Quickstart & How to Run

### 1. Installation
```bash
git clone https://github.com/D1349375/OwnStride.git
cd OwnStride
pip install -r requirements.txt
```

### 2. (Optional) Real-data check: WISDM
The WISDM v1.1 dataset (Kwapisz et al. 2011) is not redistributed in this repository. Download it once (~11 MB) to enable the "WISDM real data" button and its tests:
```bash
python -c "from src.dataset.wisdm_loader import WISDMLoader; WISDMLoader().ensure_downloaded(); WISDMLoader().load_walking_data()"
```
Without it, the dashboard still runs and the WISDM tests are skipped.

### 3. (Optional) Local LLM
```bash
ollama pull qwen2.5:1.5b
```
On Windows machines where the default `%USERPROFILE%\.ollama` folder lacks delete permission, start Ollama with `scripts/start_ollama.ps1` (keeps models in `%LOCALAPPDATA%\OwnStride\ollama-models`).
Environment variables: `OWNSTRIDE_OLLAMA_URL`, `OWNSTRIDE_LLM_MODEL`, `OWNSTRIDE_LLM_TIMEOUT`, `OWNSTRIDE_SKIP_LLM_WARMUP`. The API pre-loads the model in the background at start-up so the first plan does not wait for model loading.

### 4. Run the test suite
```bash
pytest tests/ -v
```
55 tests cover FPA/eversion recovery from IMU signals, the personal-best baseline (best-day selection, one-sided deviation, ratchet), the progress-rate-driven FSM with its progress gate, LLM streaming/fallback (against a fake Ollama server), bilingual plans, the REST API, the WISDM loader and the serial bridge.

### 5. Launch the dashboard
```bash
python run_demo.py
```
Open **`http://127.0.0.1:8000`** (中 / EN toggle top right; every card has a **?** with a plain-language explanation and a technical definition):
- **L1**: share of today's strides within the personal best, current phase, progress rate, cue budget, foot angle vs. personal best.
- **Pipeline dock**: add a synthetic walk (strides are cued only above the day's threshold and within budget), close the day to run the daily evaluation, or load WISDM real data (segmentation only).
- **L2**: personal corridor (best-days waveform band vs. today's mean), personal best over time (ratchet), weekly shortfall vs. cues, per-feature z-scores.
- **L3**: weekly plan, streamed from the local LLM.

### 6. Headless CLI
```bash
python run_demo.py --cli
```

### 7. Hardware (ESP32 + MPU-6050)
See [`firmware/README.md`](firmware/README.md) for wiring, flashing and the `record` / `analyze --nominal` / `live` bridge commands.

---

## 🔬 Scientific Foundations & References

- **Guidance Hypothesis**: Salmoni, A. W., Schmidt, R. A., & Walter, C. B. (1984). *Knowledge of results and motor learning: A review and critical reappraisal.* Psychological Bulletin, 95(3), 355–386. [DOI: 10.1037/0033-2909.95.3.355](https://doi.org/10.1037/0033-2909.95.3.355)
- **Direct Walking Task Evidence**: Sato-Klemm, M., et al. (2026). *Comparing the effects of faded vs. constant knowledge of results on the acquisition, retention, and transfer of a skilled walking task.* Human Movement Science, 105, 103442. [DOI: 10.1016/j.humov.2025.103442](https://doi.org/10.1016/j.humov.2025.103442)
- **Gait as Individual Biometric**: Horst, F., et al. (2019). *Explaining the unique nature of individual gait patterns with deep learning.* Scientific Reports, 9, 2391. [DOI: 10.1038/s41598-019-38748-8](https://doi.org/10.1038/s41598-019-38748-8)
- **Foot Progression Angle Accuracy**: Tan, T., et al. (2021). *Magnetometer-Free, IMU-Based Foot Progression Angle Estimation for Real-Life Walking Conditions.* IEEE TNSRE, 29, 282–289. [DOI: 10.1109/TNSRE.2020.3047402](https://doi.org/10.1109/TNSRE.2020.3047402)
- **WISDM Benchmark Dataset**: Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). *Activity recognition using cell phone accelerometers.* ACM SIGKDD Explorations Newsletter, 12(2), 74–82. [DOI: 10.1145/1964897.1964918](https://doi.org/10.1145/1964897.1964918)
- **MAREA Database & Detection Protocol**: Khandelwal, S., & Wickström, N. (2017). *Evaluation of accelerometer-based gait event detection algorithms.* Gait & Posture, 51, 84–90. [DOI: 10.1016/j.gaitpost.2016.09.023](https://doi.org/10.1016/j.gaitpost.2016.09.023)

---

## ⚖️ Regulatory Positioning & Commercialization

OwnStride executes a strict phased commercialization roadmap:
- **Phase 1 (Stage I / General Wellness)**: Personal fitness and movement training tool. Strictly non-medical, requiring no FDA/TFDA clearance, minimizing legal and liability friction.
- **Phase 2 (Post-Competition / Clinical Expansion)**: Remote Patient Monitoring (RPM) assistant with Clinician-in-the-Loop review for physical therapy clinics.
