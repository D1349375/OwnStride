"""
src/features/event_detector.py
==============================
僅加速度計的步態週期切分器 (Accelerometer-Only Gait Cycle Segmenter)

用途限定：處理**沒有陀螺儀**的資料（例如 WISDM 手機加速度資料集），只能切出步態週期、
估算步時與支撐比例，**無法計算足偏角或外翻角**。
足部 IMU（含陀螺儀）的完整逐步分析請用 src/features/foot_imu.py。
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import signal


class GaitEventDetector:
    """
    步態事件偵測器：基於足部/雙踝 IMU 訊號切分步態週期。
    
    支援事件：
    - Initial Contact / Heel Strike (HS): 腳跟著地事件（步態週期起始與結束）
    - Toe Off (TO): 腳尖離地事件（支撐相與擺動相分界）
    """

    def __init__(self, sample_rate: float = 128.0):
        """
        :param sample_rate: IMU 取樣率 (Hz)，MAREA 預設 128Hz，MPU6050 預設 100Hz。
        """
        self.fs = sample_rate
        # 步態週期的生理時間約束 (一般成人步態週期約 0.8s ~ 1.5s，最小生理不反應期 0.4s)
        self.min_step_samples = int(0.4 * self.fs)
        self.max_step_samples = int(2.5 * self.fs)

    def bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 15.0, order: int = 4) -> np.ndarray:
        """
        雙向 Butterworth 帶通濾波器：去除重力靜態分量與高頻震動雜訊。
        具備 SciPy filtfilt padlen 防禦。
        """
        if len(data) <= 27:
            return data

        nyq = 0.5 * self.fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)

    def detect_heel_strikes(self, acc: np.ndarray, gyro: Optional[np.ndarray] = None) -> np.ndarray:
        """
        偵測腳跟著地事件 (Heel Strike, HS)。
        
        :param acc: (N, 3) 加速度訊號 [ax, ay, az] (g 或 m/s^2)
        :param gyro: 可選 (N, 3) 角速度訊號 [gx, gy, gz] (deg/s 或 rad/s)
        :return: HS 發生之樣本索引陣列 (int)
        """
        n_samples = len(acc)
        if n_samples < self.min_step_samples or n_samples <= 27:
            return np.array([], dtype=int)

        # 1. 計算合加速度純量 norm
        acc_mag = np.linalg.norm(acc, axis=1)

        # 2. 濾波平滑
        filtered_mag = self.bandpass_filter(acc_mag, lowcut=0.8, highcut=12.0)

        # 3. 峰值偵測：著地瞬間有強烈衝擊波
        threshold = np.mean(filtered_mag) + 0.5 * np.std(filtered_mag)
        peaks, properties = signal.find_peaks(
            filtered_mag,
            height=threshold,
            distance=self.min_step_samples,
            prominence=0.3 * np.std(filtered_mag)
        )

        return peaks

    def detect_toe_offs(self, acc: np.ndarray, hs_indices: np.ndarray) -> np.ndarray:
        """
        全域掃描離地點 (備援用)。
        """
        if len(hs_indices) < 2:
            return np.array([], dtype=int)

        acc_mag = np.linalg.norm(acc, axis=1)
        filtered_mag = self.bandpass_filter(acc_mag, lowcut=0.8, highcut=12.0)

        to_indices = []
        for i in range(len(hs_indices) - 1):
            start = hs_indices[i]
            end = hs_indices[i + 1]
            cycle_len = end - start

            window_start = start + int(0.45 * cycle_len)
            window_end = start + int(0.75 * cycle_len)

            if window_end <= window_start or window_end > len(filtered_mag):
                continue

            sub_mag = filtered_mag[window_start:window_end]
            min_rel_idx = np.argmin(sub_mag)
            to_indices.append(window_start + min_rel_idx)

        return np.array(to_indices, dtype=int)

    def segment_gait_cycles(self, acc: np.ndarray, gyro: Optional[np.ndarray] = None) -> List[Dict[str, int]]:
        """
        完整分割步態週期。
        修正審計缺陷一：改為針對每個相鄰 HS 局部搜尋 TO，徹底防止單一缺漏導致後續週期雪崩式遺失。
        """
        hs = self.detect_heel_strikes(acc, gyro)
        if len(hs) < 2:
            return []

        acc_mag = np.linalg.norm(acc, axis=1)
        filtered_mag = self.bandpass_filter(acc_mag, lowcut=0.8, highcut=12.0)
        cycles = []

        for i in range(len(hs) - 1):
            hs_start = int(hs[i])
            hs_end = int(hs[i + 1])
            cycle_len = hs_end - hs_start

            if cycle_len <= 0:
                continue

            # 局部搜尋當前週期的離地點 (約 45% ~ 75% 區間)
            w_start = hs_start + int(0.45 * cycle_len)
            w_end = min(hs_start + int(0.75 * cycle_len), len(filtered_mag))

            if w_end > w_start:
                sub_mag = filtered_mag[w_start:w_end]
                to_pt = int(w_start + np.argmin(sub_mag))
            else:
                to_pt = hs_start + int(0.60 * cycle_len)  # 生理 60% 兜底

            duration_samples = hs_end - hs_start
            stance_samples = max(to_pt - hs_start, 1)
            swing_samples = max(hs_end - to_pt, 1)
            duration_sec = duration_samples / self.fs
            stance_ratio = stance_samples / duration_samples

            # 生理合理區間過濾
            if 0.5 <= duration_sec <= 2.2 and 0.40 <= stance_ratio <= 0.80:
                cycles.append({
                    "cycle_index": len(cycles),
                    "hs_start": hs_start,
                    "to": to_pt,
                    "hs_end": hs_end,
                    "duration_samples": duration_samples,
                    "duration_sec": float(duration_sec),
                    "stance_samples": stance_samples,
                    "swing_samples": swing_samples,
                    "stance_ratio": float(stance_ratio)
                })

        return cycles
