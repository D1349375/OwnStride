"""
src/baseline/personal_baseline.py
=================================
個人化步態基線模組 (Personal Gait Baseline Module)

基線定義（2026-09-24 定案：「個人最佳日」基線）：
1. 參照基準是「使用者自己表現最好的那幾天」，不是群體常模，也不是全部日子的平均。
   - 使用者在建立時選擇訓練方向 goal_direction：+1 = 減少內八（FPA 越大越好）、
     −1 = 減少外八（FPA 越小越好）、0 = 維持現狀（以步態最穩定的日子為基線）。
   - 從候選日中依方向挑出最好的 best_k 天，合併這些天的逐步特徵擬合 Ledoit-Wolf。
2. 偏差只看正在訓練的項目（2026-09-25 定案）：
   - 有訓練方向時，偏差只用訓練目標特徵（FPA）計算，且只計算「比最佳狀態更差」的那一側；
     比最佳更好不算偏差，所以使用者進步時不會被提示。
   - 其他特徵（步長、著地衝擊等）在步態改善時本來就會跟著改變，若一起計算會把「進步的副作用」
     誤判為偏差，因此只顯示（z 分數），不參與偏差、提示與 Phase 判斷。
   - 方向 0（維持）時沒有任何項目應該改變，6 個特徵一起以雙側 Mahalanobis 距離計算。
3. 棘輪更新（ratchet）：定期以最近的日子重新挑最佳日，若新的最佳明顯更好才上調基線；
   表現變差時基線不下調。
4. 小樣本穩定：Ledoit-Wolf 收縮估計保證共變異數可逆（6 維完整模型用於 z 分數與維持模式）。
"""

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from sklearn.covariance import LedoitWolf

from src.features.gait_metrics import GaitFeatureVector

GOAL_LABELS = {1: "reduce_in_toeing", -1: "reduce_out_toeing", 0: "maintain"}


class BaselineState(BaseModel):
    """個人基線狀態封包"""
    is_calibrated: bool = False
    n_samples: int = 0
    feature_names: List[str] = Field(default_factory=GaitFeatureVector.feature_names)
    mean_vector: List[float] = Field(default_factory=list)
    shrinkage_coefficient: float = 0.0
    normal_threshold: float = 2.5
    severe_threshold: float = 4.5
    goal: str = "maintain"
    source_days: List[int] = Field(default_factory=list)
    scored_features: List[str] = Field(default_factory=list)


class PersonalGaitBaseline:
    """個人步態基線引擎：學習使用者「最佳狀態」的步態分佈。"""

    def __init__(
        self,
        normal_percentile: float = 95.0,
        goal_direction: int = 0,
        goal_feature: str = "fpa",
        scored_features: Optional[List[str]] = None,
    ):
        """
        :param normal_percentile: 基線樣本自身偏差的第幾百分位作為個人閾值 τ
        :param goal_direction: +1 減少內八、−1 減少外八、0 維持
        :param scored_features: 參與偏差計算的特徵；預設有訓練方向時只用 goal_feature，維持時用全部
        """
        if goal_direction not in (-1, 0, 1):
            raise ValueError("goal_direction must be -1, 0 or 1")
        self.normal_percentile = normal_percentile
        self.goal_direction = goal_direction
        self.feature_names = GaitFeatureVector.feature_names()
        self.goal_feature = goal_feature
        self.goal_index = self.feature_names.index(goal_feature)
        if scored_features is None:
            scored_features = [goal_feature] if goal_direction != 0 else list(self.feature_names)
        self.scored_features = list(scored_features)
        self.score_idx = [self.feature_names.index(f) for f in self.scored_features]
        # 偏差計算中「訓練目標特徵」的位置（不在計分特徵中時為 None）
        self.score_goal_pos = self.scored_features.index(goal_feature) if goal_feature in self.scored_features else None
        self.model: Optional[LedoitWolf] = None
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None
        self.score_mean_: Optional[np.ndarray] = None
        self.score_precision_: Optional[np.ndarray] = None
        self.shrinkage_: float = 0.0
        self.n_samples_ = 0
        self.source_days: List[int] = []
        self.corridor_: Dict[str, Dict] = {}
        self.corridor_band_sd: float = 2.0
        self.normal_threshold: float = 2.5
        self.severe_threshold: float = 4.5

    @property
    def is_calibrated(self) -> bool:
        return self.precision_ is not None and self.mean_ is not None

    @property
    def goal(self) -> str:
        return GOAL_LABELS[self.goal_direction]

    # ------------------------------------------------------------------
    # 擬合
    # ------------------------------------------------------------------

    def fit(self, baseline_features: List[GaitFeatureVector]) -> "PersonalGaitBaseline":
        """以一組逐步特徵直接擬合（不做最佳日篩選）。"""
        if len(baseline_features) < 5:
            raise ValueError(f"At least 5 strides required to fit baseline, got {len(baseline_features)}")
        X = np.array([f.to_numpy() for f in baseline_features], dtype=np.float64)
        return self.fit_numpy(X)

    def fit_numpy(self, X: np.ndarray) -> "PersonalGaitBaseline":
        lw = LedoitWolf().fit(X)
        self.model = lw
        self.mean_ = lw.location_.copy()
        self.precision_ = lw.get_precision().copy()
        self.shrinkage_ = float(lw.shrinkage_)
        self.n_samples_ = X.shape[0]

        # 偏差計分用的子模型（只含 scored_features）
        lw_score = LedoitWolf().fit(X[:, self.score_idx])
        self.score_mean_ = lw_score.location_.copy()
        self.score_precision_ = lw_score.get_precision().copy()

        # 個人閾值 τ：基線樣本自身（單側）偏差分佈的百分位
        train_distances = [self._mahalanobis(x) for x in X]
        self.normal_threshold = float(np.percentile(train_distances, self.normal_percentile))
        self.severe_threshold = float(self.normal_threshold * 1.6)
        return self

    def rank_days(self, days: List[Dict]) -> List[Dict]:
        """
        依訓練方向排序候選日（最好的在前）。
        days: [{"day_index": int, "features": [GaitFeatureVector, ...], ...}, ...]
        方向 ±1：以當日平均 FPA 往目標方向的程度排序；方向 0：以當日 FPA 標準差（越穩定越好）排序。
        """
        def score(day):
            fpa = np.array([f.fpa for f in day["features"]])
            if self.goal_direction == 0:
                return -float(np.std(fpa))
            return self.goal_direction * float(np.mean(fpa))
        return sorted((d for d in days if len(d["features"]) >= 5), key=score, reverse=True)

    def fit_best_days(self, days: List[Dict], best_k: int = 5) -> "PersonalGaitBaseline":
        """從候選日中挑最好的 best_k 天擬合基線；若日資料含 strides 波形，同時建立個人走廊。"""
        chosen = self.rank_days(days)[:best_k]
        if not chosen:
            raise ValueError("No usable days to fit the baseline")
        self.fit([f for d in chosen for f in d["features"]])
        self.source_days = sorted(int(d["day_index"]) for d in chosen)
        strides = [s for d in chosen for s in d.get("strides", [])]
        if strides:
            self.fit_corridor({
                "fpa": [s["fpa_waveform"] for s in strides],
                "eversion": [s["eversion_waveform"] for s in strides],
            })
        return self

    def maybe_ratchet(self, recent_days: List[Dict], best_k: int = 5, min_gain_deg: float = 0.5) -> bool:
        """
        棘輪更新：用最近的日子重新挑最佳日；只有在新最佳往訓練方向至少進步 min_gain_deg 度時才上調。
        方向 0（維持）不做棘輪。回傳是否更新。
        """
        if self.goal_direction == 0 or not self.is_calibrated:
            return False
        candidate = PersonalGaitBaseline(self.normal_percentile, self.goal_direction, self.goal_feature, self.scored_features)
        try:
            candidate.fit_best_days(recent_days, best_k)
        except ValueError:
            return False
        gain = self.goal_direction * (candidate.mean_[self.goal_index] - self.mean_[self.goal_index])
        if gain < min_gain_deg:
            return False
        self.__dict__.update(candidate.__dict__)
        return True

    # ------------------------------------------------------------------
    # 偏差計算
    # ------------------------------------------------------------------

    def _mahalanobis(self, x: np.ndarray) -> float:
        """計分特徵上的 Mahalanobis 距離；訓練目標特徵往「更好」方向的偏移視為 0（單側）。"""
        diff = x[self.score_idx] - self.score_mean_
        g = self.score_goal_pos
        if self.goal_direction != 0 and g is not None and self.goal_direction * diff[g] > 0:
            diff[g] = 0.0
        dist_sq = float(diff @ self.score_precision_ @ diff)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def compute_deviation_score(self, feature: GaitFeatureVector) -> float:
        """偏差分數 = 這一步比個人最佳狀態「差」了多少（Mahalanobis 單位）"""
        if not self.is_calibrated:
            raise RuntimeError("Baseline model has not been calibrated. Call fit() first.")
        return self._mahalanobis(feature.to_numpy())

    def compute_session_deviations(self, features: List[GaitFeatureVector]) -> Dict:
        """批次計算整段行走的偏差統計"""
        if not features:
            return {"mean_deviation": 0.0, "aberrant_ratio": 0.0, "scores": []}
        scores = [self.compute_deviation_score(f) for f in features]
        return {
            "mean_deviation": round(float(np.mean(scores)), 3),
            "std_deviation": round(float(np.std(scores)), 3),
            "max_deviation": round(float(np.max(scores)), 3),
            "normal_threshold": round(self.normal_threshold, 3),
            "severe_threshold": round(self.severe_threshold, 3),
            "aberrant_ratio": round(sum(1 for s in scores if s > self.normal_threshold) / len(scores), 3),
            "stride_count": len(scores),
            "scores": [round(s, 2) for s in scores],
        }

    def feature_sd(self, name: str) -> float:
        i = self.feature_names.index(name)
        return float(np.sqrt(self.model.covariance_[i, i]))

    def feature_zscores(self, features: List[GaitFeatureVector]) -> Dict[str, float]:
        """
        各特徵的平均值相對個人基線的標準化偏移 (z-score)，標準差取自收縮後共變異數對角線。
        讓使用者看到是哪一個維度造成偏差，而不只是一個合成後的距離。
        """
        if not self.is_calibrated or not features:
            return {}
        x = np.mean([f.to_numpy() for f in features], axis=0)
        sd = np.sqrt(np.diag(self.model.covariance_))
        z = (x - self.mean_) / np.maximum(sd, 1e-9)
        return {name: round(float(v), 2) for name, v in zip(self.feature_names, z)}

    def fit_corridor(self, waveforms: Dict[str, List[np.ndarray]], band_sd: float = 2.0) -> None:
        """以基線樣本逐步的 0–100% 波形建立個人走廊帶（平均 ± band_sd 個標準差）"""
        self.corridor_ = {}
        for channel, curves in waveforms.items():
            arr = np.vstack(curves)
            mean, sd = arr.mean(axis=0), arr.std(axis=0)
            self.corridor_[channel] = {"mean": mean, "upper": mean + band_sd * sd,
                                       "lower": mean - band_sd * sd, "n_strides": len(curves)}
        self.corridor_band_sd = band_sd

    def get_state(self) -> BaselineState:
        return BaselineState(
            is_calibrated=self.is_calibrated,
            n_samples=self.n_samples_,
            feature_names=self.feature_names,
            mean_vector=[round(float(v), 3) for v in (self.mean_ if self.mean_ is not None else [])],
            shrinkage_coefficient=round(self.shrinkage_, 4),
            normal_threshold=round(self.normal_threshold, 3),
            severe_threshold=round(self.severe_threshold, 3),
            goal=self.goal,
            source_days=self.source_days,
            scored_features=self.scored_features,
        )
