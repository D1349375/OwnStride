"""
src/controller/faded_fsm.py
===========================
漸退策略控制器 (Adaptive Faded Feedback Controller)
依據規劃書 v0.7 第二節與護城河 #1：「回饋頻率 = f(進步速率, 當前 Phase, 偏差嚴重程度)」。

每日輸入一個「當日平均偏差分數」，控制器維護最近 7 天的歷史，計算：
- 水準 level：最近 3 天平均偏差
- 進步速率 progress_rate：最近 7 天「偏差 + τ」的指數趨勢，換算成「每週相對下降比例」
  （0.3 = 每週偏差下降 30%；上限 1；負值 = 退步）

Phase 轉換同時看水準與速率（遲滯 + 連續天數確認 + 每階段最短停留天數，防止 Phase 抖動）：
- 升階門檻會隨進步速率放寬：進步越快的人越早開始漸退（這是「由個人進步速率驅動」的具體實作）
- 退步速率過大時直接降階，不必等水準惡化到門檻
- 同一 Phase 內，每日提示配額也依進步速率再往下調
- 進步閘門（2026-09-25 定案，require_progress=True 時）：升階前，本階段內個人最佳基線必須至少上調過一次。
  個人最佳基線下的「偏差」代表「比自己的最佳差多少」，一個維持在最佳、但完全沒進步的人偏差也很低；
  若只看偏差就會被漸退。閘門確保漸退綁定「真的有進步」。選「維持」目標（無棘輪）時不啟用。

所有偏差門檻皆以「個人正常閾值 τ」（基線訓練樣本 Mahalanobis 距離的第 95 百分位）為單位，
不使用跨使用者共用的絕對數值。

⚠️ 下方 PHASE_CONFIGS 與 TransitionRules 的數值是設計參數，尚未經使用者資料校準。
"""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


class FeedbackPhase(IntEnum):
    PHASE_1_ACQUISITION = 1  # 密集提示期：建立正確步態感知
    PHASE_2_FADED = 2        # 漸退適應期：提示減量，本體感覺主導
    PHASE_3_RETENTION = 3    # 稀疏維持期：僅明顯偏差介入


# 每個 Phase 的基礎控制參數；cue_threshold_tau 為「偏差超過幾倍 τ 才提示」
PHASE_CONFIGS: Dict[FeedbackPhase, Dict] = {
    FeedbackPhase.PHASE_1_ACQUISITION: {"name": "Phase 1: Acquisition", "base_budget": 50, "cue_threshold_tau": 1.0, "intensity": 1.0},
    FeedbackPhase.PHASE_2_FADED:       {"name": "Phase 2: Faded Feedback", "base_budget": 20, "cue_threshold_tau": 1.2, "intensity": 0.7},
    FeedbackPhase.PHASE_3_RETENTION:   {"name": "Phase 3: Retention", "base_budget": 5, "cue_threshold_tau": 1.5, "intensity": 0.5},
}


@dataclass
class TransitionRules:
    """Phase 轉換規則（單位：水準為 τ 的倍數；速率為每週相對下降比例）"""
    p1_to_p2_level: float = 1.2        # 基本升階水準
    p2_to_p3_level: float = 0.85
    rate_relax: float = 0.8            # 每 1.0 進步速率可放寬的升階水準（速率上限 0.5 → 最多放寬 0.4τ）
    rate_cap: float = 0.5
    min_rate_to_advance: float = -0.05  # 升階前提：沒有在退步
    p2_to_p1_level: float = 1.9        # 降階水準（與升階門檻之間保留遲滯帶）
    p3_to_p2_level: float = 1.25
    p2_to_p1_rate: float = -0.30       # 退步速率過大直接降階
    p3_to_p2_rate: float = -0.20
    budget_fade_per_rate: float = 0.6  # Phase 內配額隨進步速率再縮減的比例


class StrategyPacket(BaseModel):
    """
    策略封包：評估層每日發送給足部穿戴裝置的緊湊控制參數。
    Stage I 原型中提示判斷在主機端執行（見 src/hardware/serial_bridge.py）；
    封包格式同時保留給 Stage II 移到 MCU 端執行。
    """
    packet_id: int
    user_id: str = "user_default"
    phase: int = Field(..., description="目前漸退階段 (1, 2, 3)")
    phase_name: str
    cue_threshold: float = Field(..., description="觸發提示的 Mahalanobis 偏差閾值 (= cue_threshold_tau × τ)")
    daily_cue_budget: int = Field(..., description="每日提示總配額上限")
    haptic_intensity: float = Field(..., description="馬達震動強度 (0.0 ~ 1.0)")
    faded_ratio: float = Field(..., description="配額相對 Phase 1 基礎配額的比例")
    progress_rate: float = Field(..., description="每週偏差相對下降比例 (正 = 進步)")
    deviation_level: float = Field(..., description="最近 3 日平均偏差")
    active_target: str = Field(..., description="當前主要訓練目標")


class FadedFeedbackFSM:
    """具備遲滯帶、連續天數確認，且由進步速率調節的漸退狀態機。"""

    def __init__(
        self,
        initial_phase: FeedbackPhase = FeedbackPhase.PHASE_1_ACQUISITION,
        debounce_days: int = 2,
        personal_threshold: float = 2.5,
        rules: Optional[TransitionRules] = None,
        window_days: int = 7,
        min_days_in_phase: int = 7,
        require_progress: bool = False,
    ):
        self.current_phase = initial_phase
        self.debounce_days = debounce_days
        self.personal_threshold = float(personal_threshold)
        self.rules = rules or TransitionRules()
        self.history: Deque[float] = deque(maxlen=window_days)
        self.days_in_pending_state = 0
        self.pending_phase: Optional[FeedbackPhase] = None
        self.packet_counter = 0
        self.min_days_in_phase = min_days_in_phase
        self.days_in_phase = 0
        self.require_progress = require_progress
        self.progress_events_in_phase = 0
        self.phase_configs = PHASE_CONFIGS

    # ------------------------------------------------------------------
    # 指標
    # ------------------------------------------------------------------

    def level(self) -> float:
        if not self.history:
            return float("nan")
        return float(np.mean(list(self.history)[-3:]))

    def progress_rate(self) -> float:
        """
        最近 7 天偏差的指數趨勢，換算為每週相對下降比例：rate = 1 − exp(7 × d(log(偏差 + τ))/d天)。
        加上 τ 作為底：偏差遠大於 τ 時等同相對下降比例；偏差接近 0（使用者已在個人最佳）時，
        0.00 → 0.12 這類微小波動不會被對數放大成劇烈「退步」。上限為 1，退步時為負值。
        資料少於 3 天時視為 0。
        """
        if len(self.history) < 3:
            return 0.0
        y = np.log(np.array(self.history, dtype=np.float64) + self.personal_threshold)
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        return float(1.0 - np.exp(7.0 * slope))

    # ------------------------------------------------------------------
    # 轉換邏輯
    # ------------------------------------------------------------------

    def evaluate_next_phase(self, level: float, rate: float) -> FeedbackPhase:
        """純函數：依水準（τ 倍數）與速率計算目標 Phase，不改變狀態。"""
        if np.isnan(level) or np.isinf(level):
            return self.current_phase
        r = self.rules
        lv = level / self.personal_threshold
        relax = r.rate_relax * float(np.clip(rate, 0.0, r.rate_cap))
        improving_or_stable = rate >= r.min_rate_to_advance

        if self.current_phase == FeedbackPhase.PHASE_1_ACQUISITION:
            if lv <= r.p1_to_p2_level + relax and improving_or_stable:
                return FeedbackPhase.PHASE_2_FADED
        elif self.current_phase == FeedbackPhase.PHASE_2_FADED:
            if lv >= r.p2_to_p1_level or rate <= r.p2_to_p1_rate:
                return FeedbackPhase.PHASE_1_ACQUISITION
            if lv <= r.p2_to_p3_level + relax * 0.5 and improving_or_stable:
                return FeedbackPhase.PHASE_3_RETENTION
        elif self.current_phase == FeedbackPhase.PHASE_3_RETENTION:
            if lv >= r.p3_to_p2_level or rate <= r.p3_to_p2_rate:
                return FeedbackPhase.PHASE_2_FADED
        return self.current_phase

    def record_progress(self) -> None:
        """個人最佳基線上調（棘輪）時由評估層呼叫，作為升階的進步證據。"""
        self.progress_events_in_phase += 1

    def progress_gate_open(self) -> bool:
        return not self.require_progress or self.progress_events_in_phase >= 1

    def _daily_budget(self, phase: FeedbackPhase, rate: float) -> int:
        base = self.phase_configs[phase]["base_budget"]
        if phase == FeedbackPhase.PHASE_1_ACQUISITION or rate <= 0:
            return base
        fade = self.rules.budget_fade_per_rate * float(np.clip(rate, 0.0, self.rules.rate_cap))
        return max(1, int(round(base * (1.0 - fade))))

    def build_packet(self, target_symptom: str = "fpa") -> StrategyPacket:
        rate = self.progress_rate()
        cfg = self.phase_configs[self.current_phase]
        budget = self._daily_budget(self.current_phase, rate)
        self.packet_counter += 1
        level = self.level()
        return StrategyPacket(
            packet_id=self.packet_counter,
            phase=int(self.current_phase),
            phase_name=cfg["name"],
            cue_threshold=round(cfg["cue_threshold_tau"] * self.personal_threshold, 3),
            daily_cue_budget=budget,
            haptic_intensity=cfg["intensity"],
            faded_ratio=round(budget / self.phase_configs[FeedbackPhase.PHASE_1_ACQUISITION]["base_budget"], 3),
            progress_rate=round(rate, 3),
            deviation_level=round(level, 3) if not np.isnan(level) else 0.0,
            active_target=target_symptom,
        )

    def step(self, daily_mean_deviation: float, target_symptom: str = "fpa") -> Tuple[FeedbackPhase, bool, StrategyPacket]:
        """
        每日評估一次：記錄當日偏差 → 計算水準與速率 → 連續 debounce_days 天指向同一新 Phase 才轉換。
        :return: (當前 Phase, 本日是否轉換, 次日策略封包)
        """
        if not (np.isnan(daily_mean_deviation) or np.isinf(daily_mean_deviation)):
            self.history.append(max(0.0, float(daily_mean_deviation)))

        self.days_in_phase += 1
        candidate = self.evaluate_next_phase(self.level(), self.progress_rate())
        # 升階需在目前 Phase 停留滿 min_days_in_phase 天；降階不受限
        if candidate > self.current_phase and (self.days_in_phase < self.min_days_in_phase or not self.progress_gate_open()):
            candidate = self.current_phase
        changed = False
        if candidate != self.current_phase:
            if candidate == self.pending_phase:
                self.days_in_pending_state += 1
            else:
                self.pending_phase = candidate
                self.days_in_pending_state = 1
            if self.days_in_pending_state >= self.debounce_days:
                self.current_phase = candidate
                self.pending_phase = None
                self.days_in_pending_state = 0
                self.days_in_phase = 0
                self.progress_events_in_phase = 0
                changed = True
        else:
            self.pending_phase = None
            self.days_in_pending_state = 0

        return self.current_phase, changed, self.build_packet(target_symptom)
