"""
src/features/event_detector.py
==============================
僅加速度計的跨步切分器 (Accelerometer-Only Stride Segmenter)

用途限定：處理**沒有陀螺儀**的資料（例如 WISDM 手機放大腿口袋的加速度），只能切出跨步並估算
跨步時間與步頻，**無法計算足偏角或外翻角**。足部 IMU（含陀螺儀）的完整逐步分析請用 src/features/foot_imu.py。

方法（2026-09-25 修正）：
- 手機在口袋裡時，左右腳著地都會在加速度合量上留下峰值，同側腳的峰值通常較大。
  舊版把每個峰值都當成「腳跟著地」，切出來的週期一下是一步、一下是一跨步（WISDM 受試者 #1：0.62 s / 1.04 s 交替）。
- 新版先用自相關找出「跨步週期」（同一隻腳兩次著地的間隔），再以該週期的 0.7 倍作為峰值最小間距，
  每個跨步只取一個峰值；與估計週期相差超過 ±25% 的間隔視為漏抓或多抓，剔除並回報數量。
- 不再輸出腳尖離地與支撐比例：大腿上的加速度無法可靠定出腳尖離地時間。
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal


class GaitEventDetector:
    """以自相關估計跨步週期，再逐跨步切分加速度訊號。"""

    MIN_STRIDE_SEC = 0.7   # 跨步週期搜尋範圍（一般成人步行約 0.9–1.3 s）
    MAX_STRIDE_SEC = 2.0
    PEAK_SPACING = 0.7     # 峰值最小間距 = 0.7 × 跨步週期（排除對側腳的較小峰值）
    TOLERANCE = 0.25       # 與跨步週期相差超過 ±25% 的間隔視為切分失敗

    def __init__(self, sample_rate: float = 128.0):
        self.fs = sample_rate

    def _filtered_magnitude(self, acc: np.ndarray) -> np.ndarray:
        mag = np.linalg.norm(acc, axis=1)
        nyq = 0.5 * self.fs
        b, a = signal.butter(4, [0.5 / nyq, min(5.0 / nyq, 0.99)], btype="band")
        return signal.filtfilt(b, a, mag)

    def estimate_stride_period(self, filtered: np.ndarray) -> Optional[float]:
        """自相關在 0.7–2.0 s 內的最高峰 = 跨步週期（秒）。訊號太短時回傳 None。"""
        lo, hi = int(self.MIN_STRIDE_SEC * self.fs), int(self.MAX_STRIDE_SEC * self.fs)
        if len(filtered) < 2 * hi:
            return None
        x = filtered - filtered.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        return (lo + int(np.argmax(ac[lo:hi]))) / self.fs

    def segment_strides(self, acc: np.ndarray, gyro: Optional[np.ndarray] = None) -> Dict:
        """
        :param acc: (N, 3) 加速度（g）
        :return: {"stride_period_sec", "strides": [{"start", "end", "duration_sec"}], "rejected"}
        """
        empty = {"stride_period_sec": None, "strides": [], "rejected": 0}
        if len(acc) <= 27:
            return empty
        filtered = self._filtered_magnitude(acc)
        period = self.estimate_stride_period(filtered)
        if period is None:
            return empty
        peaks, _ = signal.find_peaks(filtered, distance=int(self.PEAK_SPACING * period * self.fs),
                                     prominence=0.3 * np.std(filtered))
        strides, rejected = [], 0
        for start, end in zip(peaks[:-1], peaks[1:]):
            duration = (end - start) / self.fs
            if abs(duration - period) <= self.TOLERANCE * period:
                strides.append({"start": int(start), "end": int(end), "duration_sec": float(duration)})
            else:
                rejected += 1
        return {"stride_period_sec": float(period), "strides": strides, "rejected": rejected}
