"""
src/hardware/serial_bridge.py
=============================
ESP32 + MPU-6050 足部節點的主機端橋接程式（Stage I 原型）

韌體（firmware/esp32_gait_node.ino）以 100 Hz 經 USB 序列埠串流 JSON：
    {"seq":1024,"t":10240,"ax":0.082,"ay":-0.412,"az":1.050,"gx":2.1,"gy":-48.2,"gz":-11.8}
（加速度 g、角速度 deg/s；t 為韌體 millis()）

三種模式：
1. record  — 錄製原始資料成 CSV（例如做地面膠帶角度驗證時每個角度錄一段）
       python -m src.hardware.serial_bridge record --port COM5 --out trial_minus10.csv --seconds 60
2. analyze — 離線分析 CSV：逐步 FPA／外翻角／步長；給 --nominal 時計算與膠帶標稱角度的誤差
       python -m src.hardware.serial_bridge analyze trial_minus10.csv --nominal -10
3. live    — 即時分析；給 --fpa-band 時，單步 FPA 超出區間就送 VIBE 指令讓馬達震動
       python -m src.hardware.serial_bridge live --port COM5 --fpa-band 3,11
   （原型階段提示判斷在主機端；每日配額由韌體端強制執行）

安裝方向：預設感測器 x 軸指向腳尖、y 軸指向足部左側、z 軸垂直鞋面向上。
實際黏貼方向不同時以 --axes 重新對應，例如 --axes=-y,x,z 表示「新 x = −原 y、新 y = 原 x、新 z = 原 z」。

⚠️ 此程式尚未在實際配戴的裝置上驗證；解析與分析流程已用合成資料測試（tests/test_serial_bridge.py）。
"""

import argparse
import csv
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features.foot_imu import FootIMUStrideAnalyzer

SAMPLE_RATE = 100.0
FIELDS = ["seq", "t", "ax", "ay", "az", "gx", "gy", "gz"]


def parse_line(line: str) -> Optional[Dict[str, float]]:
    """解析一行韌體 JSON；非資料行（ready／ack 訊息、斷行雜訊）回傳 None。"""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not all(k in obj for k in ("ax", "ay", "az", "gx", "gy", "gz")):
        return None
    return {k: float(obj[k]) for k in FIELDS if k in obj}


def parse_axes(spec: str) -> Tuple[List[int], List[float]]:
    """'-y,x,z' → 來源軸索引 [1, 0, 2] 與正負號 [-1, 1, 1]"""
    index, sign = [], []
    for token in spec.split(","):
        token = token.strip().lower()
        s = -1.0 if token.startswith("-") else 1.0
        axis = token.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError(f"invalid axis '{token}' in --axes")
        index.append("xyz".index(axis))
        sign.append(s)
    if sorted(index) != [0, 1, 2]:
        raise ValueError("--axes must use each of x, y, z exactly once")
    return index, sign


def samples_to_arrays(samples: List[Dict[str, float]], axes: str = "x,y,z") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """樣本清單 → (acc_g, gyro_dps)，並回報掉包數（依 seq 連續性）"""
    idx, sign = parse_axes(axes)
    acc = np.array([[s["ax"], s["ay"], s["az"]] for s in samples])[:, idx] * sign
    gyro = np.array([[s["gx"], s["gy"], s["gz"]] for s in samples])[:, idx] * sign
    seqs = [int(s["seq"]) for s in samples if "seq" in s]
    dropped = int(sum(max(0, b - a - 1) for a, b in zip(seqs[:-1], seqs[1:])))
    return acc, gyro, {"samples": len(samples), "dropped_packets": dropped}


def load_csv(path: str) -> List[Dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]


def analyze_samples(samples: List[Dict[str, float]], side: str = "right", axes: str = "x,y,z",
                    nominal_fpa: Optional[float] = None) -> Dict:
    acc, gyro, info = samples_to_arrays(samples, axes)
    strides = FootIMUStrideAnalyzer(sample_rate=SAMPLE_RATE, side=side).analyze(acc, gyro)
    fpas = np.array([s["fpa"] for s in strides])
    result = {
        **info,
        "strides": len(strides),
        "fpa_mean": round(float(fpas.mean()), 2) if len(fpas) else None,
        "fpa_sd": round(float(fpas.std()), 2) if len(fpas) else None,
        "eversion_mean": round(float(np.mean([s["eversion"] for s in strides])), 2) if strides else None,
        "stride_length_mean": round(float(np.mean([s["stride_length"] for s in strides])), 3) if strides else None,
        "per_stride_fpa": [round(float(v), 2) for v in fpas],
    }
    if nominal_fpa is not None and len(fpas):
        err = fpas - nominal_fpa
        result["nominal_fpa"] = nominal_fpa
        result["bias_deg"] = round(float(err.mean()), 2)
        result["mae_deg"] = round(float(np.abs(err).mean()), 2)
    return result


# ----------------------------------------------------------------------
# 需要 pyserial 的模式
# ----------------------------------------------------------------------

def _open_port(port: str, baud: int):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("pyserial is required: pip install pyserial")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2.0)  # ESP32 開機後會做 2 秒陀螺儀偏移校正，期間必須靜止
    ser.reset_input_buffer()
    return ser


def cmd_record(args):
    ser = _open_port(args.port, args.baud)
    deadline = time.time() + args.seconds
    count = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        while time.time() < deadline:
            sample = parse_line(ser.readline().decode("utf-8", errors="ignore"))
            if sample:
                writer.writerow(sample)
                count += 1
    print(f"recorded {count} samples to {args.out}")


def cmd_analyze(args):
    result = analyze_samples(load_csv(args.csv), side=args.side, axes=args.axes, nominal_fpa=args.nominal)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_live(args):
    ser = _open_port(args.port, args.baud)
    analyzer = FootIMUStrideAnalyzer(sample_rate=SAMPLE_RATE, side=args.side)
    band = tuple(float(v) for v in args.fpa_band.split(",")) if args.fpa_band else None
    buffer: List[Dict[str, float]] = []
    reported_until = -1
    window = int(10 * SAMPLE_RATE)
    print("walking... (Ctrl+C to stop)")
    try:
        while True:
            sample = parse_line(ser.readline().decode("utf-8", errors="ignore"))
            if not sample:
                continue
            buffer.append(sample)
            if len(buffer) % int(SAMPLE_RATE) or len(buffer) < 3 * SAMPLE_RATE:
                continue
            recent = buffer[-window:]
            offset = int(recent[0]["seq"])
            acc, gyro, _ = samples_to_arrays(recent, args.axes)
            try:
                strides = analyzer.analyze(acc, gyro)
            except ValueError:
                continue
            for s in strides:
                end_seq = offset + s["end_sample"]
                if end_seq <= reported_until:
                    continue
                reported_until = end_seq
                out_of_band = band is not None and not (band[0] <= s["fpa"] <= band[1])
                if out_of_band:
                    ser.write(b"VIBE\n")
                print(f"stride  FPA {s['fpa']:+6.1f}°  eversion {s['eversion']:+5.1f}°  length {s['stride_length']:.2f} m"
                      + ("  → cue" if out_of_band else ""))
    except KeyboardInterrupt:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="OwnStride ESP32 serial bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--side", choices=["right", "left"], default="right")
    common.add_argument("--axes", default="x,y,z", help="sensor→foot axis mapping, e.g. -y,x,z")

    p = sub.add_parser("record", parents=[common])
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("analyze", parents=[common])
    p.add_argument("csv")
    p.add_argument("--nominal", type=float, default=None, help="tape-line angle walked (deg, + = toe-out)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("live", parents=[common])
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--fpa-band", default=None, help="lo,hi in degrees; outside → send VIBE")
    p.set_defaults(func=cmd_live)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
