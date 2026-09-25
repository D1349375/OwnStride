"""
src/features/foot_imu.py
========================
足部 IMU 逐步分析器 (Foot-Mounted IMU Stride Analyzer)

以足部慣性導航的標準做法（strapdown integration + ZUPT 零速度更新）計算每一步的：
足偏角（FPA）、站立期外翻角、步長、步時／步頻、支撐比例、著地衝擊。
全程為確定性訊號處理，不含神經網路（規劃書 v0.7 3.3：感測層不跑 NN）。

演算法（對應 Tan et al. 2021 無磁力計 FPA 估算的核心概念）：
1. 靜止偵測：|角速度| 與 |比力 − 1g| 同時低於門檻的連續區段 = 足部平貼（零速度）。
2. 相鄰兩段靜止之間 = 一個積分區段（本步站立末 → 擺動 → 下一步站立）。
3. 起點姿態：roll / pitch 由重力方向求得，yaw 設為 0（即以「起點足部朝向」為參考方向）。
4. 以陀螺儀積分姿態、以比力扣除重力積分速度；終點速度必須為 0，殘差以線性漂移修正。
5. 位移向量 d 的水平方向 = 這一步的實際前進方向；FPA = 足部朝向與前進方向的夾角。
   因為參考方向是每步重設的，不需要磁力計，也不受跨步 yaw 漂移影響。

限制：此模組在合成資料上驗證的是「數學正確性」；真實硬體的精度受感測器安裝角、
軟組織晃動、靜止偵測門檻影響，須以實測（例如地面貼角度膠帶行走）另行驗證。
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665
WAVEFORM_POINTS = 21  # 0%, 5%, ..., 100%


def _chain_quaternions(q0: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """
    依序右乘機體座標增量旋轉：q[k+1] = q[k] ⊗ step[k]（scipy 的 [x, y, z, w] 慣例）。
    以純浮點數運算展開，避免逐筆建立 scipy Rotation 物件的額外開銷。
    """
    out = np.empty((len(steps) + 1, 4))
    x, y, z, w = (float(v) for v in q0)
    out[0] = (x, y, z, w)
    for k, (bx, by, bz, bw) in enumerate(steps.tolist()):
        x, y, z, w = (
            w * bx + x * bw + y * bz - z * by,
            w * by - x * bz + y * bw + z * bx,
            w * bz + x * by - y * bx + z * bw,
            w * bw - x * bx - y * by - z * bz,
        )
        out[k + 1] = (x, y, z, w)
    return out


class FootIMUStrideAnalyzer:
    """
    :param sample_rate: 取樣率（Hz），MPU-6050 韌體為 100 Hz
    :param side: 感測器配戴於 'right' 或 'left' 腳（決定 FPA／外翻角的正負號）
    :param mount_roll_offset_deg: 安裝校正——受試者自然站立時量到的 roll，會從外翻角中扣除
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        side: str = "right",
        gyro_still_dps: float = 40.0,
        acc_still_g: float = 0.08,
        min_still_sec: float = 0.05,
        swing_speed_mps: float = 0.10,
        mount_roll_offset_deg: float = 0.0,
    ):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.fs = float(sample_rate)
        self.side = side
        self.gyro_still_dps = gyro_still_dps
        self.acc_still_g = acc_still_g
        self.min_still_samples = max(3, int(min_still_sec * self.fs))
        self.swing_speed_mps = swing_speed_mps
        self.mount_roll_offset_deg = mount_roll_offset_deg
        self._fpa_sign = 1.0 if side == "right" else -1.0
        self._eversion_sign = -1.0 if side == "right" else 1.0

    # ------------------------------------------------------------------
    # 1. 靜止區段偵測
    # ------------------------------------------------------------------

    def detect_stationary_runs(self, acc: np.ndarray, gyro: np.ndarray) -> List[tuple]:
        """回傳 [(start, end_exclusive), ...]，每段為足部平貼的連續靜止樣本。"""
        gyro_norm = np.linalg.norm(gyro, axis=1)
        acc_dev = np.abs(np.linalg.norm(acc, axis=1) - 1.0)
        still = (gyro_norm < self.gyro_still_dps) & (acc_dev < self.acc_still_g)

        runs, start = [], None
        for i, flag in enumerate(still):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= self.min_still_samples:
                    runs.append((start, i))
                start = None
        if start is not None and len(still) - start >= self.min_still_samples:
            runs.append((start, len(still)))
        return runs

    # ------------------------------------------------------------------
    # 2. 單一區段的姿態與位移積分
    # ------------------------------------------------------------------

    @staticmethod
    def _attitude_from_gravity(acc_mean: np.ndarray) -> Rotation:
        ax, ay, az = acc_mean
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.hypot(ay, az))
        return Rotation.from_euler("ZYX", [0.0, pitch, roll])

    def _integrate_segment(self, acc: np.ndarray, gyro: np.ndarray, still_before: tuple, still_after: tuple) -> Optional[Dict]:
        dt = 1.0 / self.fs
        i0 = still_before[1] - 1           # 前一段靜止的最後一個樣本
        i1 = still_after[0] + (still_after[1] - still_after[0]) // 2  # 下一段靜止的中點
        if i1 - i0 < int(0.3 * self.fs) or i1 - i0 > int(2.5 * self.fs):
            return None

        acc_seg = acc[i0:i1 + 1]
        gyro_seg = np.radians(gyro[i0:i1 + 1])
        n = len(acc_seg)

        # 起點姿態：以前一段靜止的平均比力定 roll / pitch
        r0 = self._attitude_from_gravity(np.mean(acc[still_before[0]:still_before[1]], axis=0))
        steps = Rotation.from_rotvec(gyro_seg[:-1] * dt).as_quat()
        rots = Rotation.from_quat(_chain_quaternions(r0.as_quat(), steps))

        # 比力轉到導航座標並扣除重力 → 線加速度（m/s²）
        lin_acc = (rots.apply(acc_seg) - np.array([0.0, 0.0, 1.0])) * GRAVITY
        vel = np.zeros((n, 3))
        vel[1:] = np.cumsum(0.5 * (lin_acc[1:] + lin_acc[:-1]) * dt, axis=0)

        # ZUPT：終點（下一段靜止）速度應為 0，殘差視為線性漂移扣除
        drift = vel[-1]
        vel -= np.outer(np.linspace(0.0, 1.0, n), drift)
        pos = np.zeros((n, 3))
        pos[1:] = np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dt, axis=0)

        disp = pos[-1]
        horizontal = float(np.hypot(disp[0], disp[1]))
        if horizontal < 0.2:  # 原地踏步或非行走動作
            return None

        progression = np.arctan2(disp[1], disp[0])
        foot_x = rots.apply(np.array([1.0, 0.0, 0.0]))
        heading = np.unwrap(np.arctan2(foot_x[:, 1], foot_x[:, 0]))
        # 足部相對前進方向的角度；換成 FPA 慣例（正 = 外八）
        rel_yaw = np.degrees(np.angle(np.exp(1j * (heading - progression))))
        fpa_curve = -self._fpa_sign * rel_yaw

        euler = rots.as_euler("ZYX", degrees=True)
        eversion_curve = self._eversion_sign * (euler[:, 2] - self.mount_roll_offset_deg)

        speed = np.hypot(vel[:, 0], vel[:, 1])
        moving = np.where(speed > self.swing_speed_mps)[0]
        if len(moving) == 0:
            return None
        toe_off, heel_strike = int(moving[0]), int(moving[-1])

        return {
            "start": i0,
            "end": i1,
            "toe_off": i0 + toe_off,
            "heel_strike": i0 + heel_strike,
            "stride_length": horizontal,
            "fpa": float(fpa_curve[0]),
            "eversion": float(eversion_curve[0]),
            "fpa_curve": fpa_curve,
            "eversion_curve": eversion_curve,
            "toe_off_pct": 100.0 * toe_off / (n - 1),
            "heel_strike_pct": 100.0 * heel_strike / (n - 1),
        }

    # ------------------------------------------------------------------
    # 3. 整段分析
    # ------------------------------------------------------------------

    def analyze(self, acc: np.ndarray, gyro: np.ndarray) -> List[Dict]:
        """
        分析一段連續 IMU 資料，回傳逐步結果清單。每一步含：
        cadence, stride_time, stride_length, stance_ratio, fpa, eversion, impact_magnitude,
        以及 0–100% 正規化波形 fpa_waveform / eversion_waveform（0% = 足部平貼結束，100% = 下一步足部平貼）。
        """
        acc = np.asarray(acc, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        if acc.shape != gyro.shape or acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("acc and gyro must both be (N, 3)")
        if not np.any(np.abs(gyro) > 1.0):
            raise ValueError("gyroscope data required: FPA cannot be computed from accelerometer-only data")

        runs = self.detect_stationary_runs(acc, gyro)
        segments = []
        for before, after in zip(runs[:-1], runs[1:]):
            seg = self._integrate_segment(acc, gyro, before, after)
            if seg is not None:
                segments.append(seg)

        strides = []
        grid = np.linspace(0.0, 1.0, WAVEFORM_POINTS)
        for idx, seg in enumerate(segments):
            # 步時以相鄰兩次腳尖離地的間隔計；最後一步沿用前一步
            if idx + 1 < len(segments):
                stride_time = (segments[idx + 1]["toe_off"] - seg["toe_off"]) / self.fs
            elif idx > 0:
                stride_time = (seg["toe_off"] - segments[idx - 1]["toe_off"]) / self.fs
            else:
                continue
            if not 0.6 <= stride_time <= 2.2:
                continue
            swing_time = (seg["heel_strike"] - seg["toe_off"]) / self.fs
            stance_ratio = 1.0 - swing_time / stride_time

            hs = seg["heel_strike"]
            window = acc[hs:min(hs + int(0.15 * self.fs), len(acc))]
            impact = float(np.max(np.linalg.norm(window, axis=1))) if len(window) else 1.0

            src = np.linspace(0.0, 1.0, len(seg["fpa_curve"]))
            strides.append({
                "stride_index": len(strides),
                "start_sample": seg["start"],
                "end_sample": seg["end"],
                "stride_time": float(stride_time),
                "cadence": float(120.0 / stride_time),  # 單腳一個步態週期 = 左右各一步
                "stride_length": seg["stride_length"],
                "stance_ratio": float(stance_ratio),
                "fpa": seg["fpa"],
                "eversion": seg["eversion"],
                "impact_magnitude": impact,
                "toe_off_pct": seg["toe_off_pct"],
                "heel_strike_pct": seg["heel_strike_pct"],
                "fpa_waveform": np.interp(grid, src, seg["fpa_curve"]),
                "eversion_waveform": np.interp(grid, src, seg["eversion_curve"]),
            })
        return strides
