"""
src/features/gait_metrics.py
============================
步態特徵向量 (Gait Feature Vector) 與萃取介面。
依據規劃書 v0.7：維度控制在 6 維，供評估層 Ledoit-Wolf 與 Mahalanobis 距離穩定計算。

特徵選擇（2026-09-24 修訂）：
- 移除「左右對稱性」：單一足部 IMU 只量得到一隻腳，無法計算雙側對稱，舊版的 1.0 是寫死的佔位值。
- 移除「步時」出向量：步時與步頻互為倒數、完全共線，放進共變異數矩陣沒有資訊量，改用步長。
- 所有特徵皆由 src/features/foot_imu.py 的姿態積分實際算出，不再由合成器參數反解。
"""

from typing import Dict, List

import numpy as np
from pydantic import BaseModel, Field

from src.features.foot_imu import FootIMUStrideAnalyzer


class GaitFeatureVector(BaseModel):
    """單一步態週期之特徵向量 (Data Contract)"""
    cycle_index: int
    cadence: float = Field(..., description="步頻 (steps/min)")
    stride_time: float = Field(..., description="步態週期耗時 (seconds)，不進入統計向量")
    stride_length: float = Field(..., description="步長 (m)，ZUPT 積分位移")
    stance_ratio: float = Field(..., description="支撐相比例 (0.0 ~ 1.0)")
    fpa: float = Field(..., description="足偏角 Foot Progression Angle (度): 負值內八, 正值外八")
    eversion_angle: float = Field(..., description="站立期足部額狀面傾角 (度): 正值外翻，旋前近似指標")
    impact_magnitude: float = Field(..., description="著地後 150ms 內比力峰值 (g)")

    def to_numpy(self) -> np.ndarray:
        """轉為 6 維數值向量供統計模型計算"""
        return np.array([
            self.cadence,
            self.stride_length,
            self.stance_ratio,
            self.fpa,
            self.eversion_angle,
            self.impact_magnitude,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> List[str]:
        return ["cadence", "stride_length", "stance_ratio", "fpa", "eversion_angle", "impact_magnitude"]


class GaitMetricsExtractor:
    """
    由連續足部 IMU 資料萃取逐步特徵。回傳特徵向量，另保留原始逐步結果（含 0–100% 波形）
    於 last_strides 供介面層繪製個人走廊曲線。
    """

    def __init__(self, sample_rate: float = 100.0, side: str = "right"):
        self.analyzer = FootIMUStrideAnalyzer(sample_rate=sample_rate, side=side)
        self.last_strides: List[Dict] = []

    def extract(self, acc: np.ndarray, gyro: np.ndarray) -> List[GaitFeatureVector]:
        self.last_strides = self.analyzer.analyze(acc, gyro)
        return [
            GaitFeatureVector(
                cycle_index=s["stride_index"],
                cadence=round(s["cadence"], 2),
                stride_time=round(s["stride_time"], 3),
                stride_length=round(s["stride_length"], 3),
                stance_ratio=round(s["stance_ratio"], 3),
                fpa=round(s["fpa"], 2),
                eversion_angle=round(s["eversion"], 2),
                impact_magnitude=round(s["impact_magnitude"], 2),
            )
            for s in self.last_strides
        ]
