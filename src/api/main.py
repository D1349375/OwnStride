"""
src/api/main.py
================
OwnStride 評估層後端服務 (Evaluation Layer REST API)
依據規劃書 v0.7：整合特徵萃取、個人基線、漸退狀態機、策略封包與每週訓練計畫，
提供 RWD Dashboard 的 L1 今日一眼、L2 多週軌跡與個人走廊、L3 訓練計畫。

資料來源誠實標註（介面必須如實顯示 data_provenance）：
- 啟動時的「個人基線」與「六週軌跡」皆為合成資料（src/dataset/synthetic_stream.py），
  但每一個數字都是由完整管線（姿態積分 → 特徵 → 基線 → 狀態機）實際算出，沒有寫死的展示數值。
- 真人資料僅有 WISDM 手機加速度（無陀螺儀），只能驗證步態週期切分，無法計算足偏角。
"""

import json
import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.baseline.personal_baseline import PersonalGaitBaseline
from src.controller.faded_fsm import FadedFeedbackFSM, FeedbackPhase, StrategyPacket
from src.dataset.synthetic_stream import SyntheticGaitStream
from src.features.gait_metrics import GaitFeatureVector, GaitMetricsExtractor
from src.llm.training_planner import LLMTrainingPlanner, WeeklyTrainingPlan, WeekSummary

app = FastAPI(title="OwnStride Evaluation Layer API", version="0.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 100.0
SIM_WEEKS = 6
SIM_STRIDES_PER_DAY = 120
CALIBRATION_DAYS = 14
BEST_K_DAYS = 5
RATCHET_WINDOW_DAYS = 14
GOAL_DIRECTION = 1          # 示範使用者：訓練目標為「減少內八」
CALIBRATION_FPA = -13.0     # 合成使用者建立期的平均足偏角
CALIBRATION_DAY_SD = 1.5    # 建立期日與日之間的自然波動（度）

BASELINE_ASSUMPTION = (
    f"Personal-best baseline: the best {BEST_K_DAYS} of {CALIBRATION_DAYS} calibration days (synthetic user who in-toes, "
    "goal = reduce in-toeing), re-selected weekly from the last "
    f"{RATCHET_WINDOW_DAYS} days and only moved up when the new best is better (ratchet). "
    "Deviation uses the trained feature only (foot progression angle, one-sided: improvement is never a deviation); "
    "other features are displayed but not scored. Stepping to a lighter feedback phase requires at least one "
    "personal-best raise in the current phase."
)

DATA_PROVENANCE = {
    "baseline": BASELINE_ASSUMPTION,
    "six_week_trajectory": (
        "Synthetic: a kinematics-first foot-IMU simulator with an assumed S-shaped learning curve "
        f"({SIM_STRIDES_PER_DAY} sampled strides/day). Every number shown is computed by the full pipeline "
        "(attitude integration → features → baseline → faded-feedback FSM); none are hard-coded."
    ),
    "walk_button": "Synthetic session generated on demand and processed by the same pipeline.",
    "real_data": (
        "WISDM v1.1 (Kwapisz et al. 2011): smartphone accelerometer only, no gyroscope. "
        "Used to check gait-cycle segmentation on real human data; foot progression angle cannot be computed from it."
    ),
    "hardware": "ESP32 + MPU-6050 firmware and serial bridge are provided (firmware/, src/hardware/); not yet validated on a worn device.",
}


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


class DayLog:
    """單日累積資料（評估層每日關帳一次）"""

    def __init__(self, day_index: int, packet: StrategyPacket):
        self.day_index = day_index
        self.packet = packet
        self.features: List[GaitFeatureVector] = []
        self.scores: List[float] = []
        self.strides: List[Dict] = []
        self.cues_used = 0

    def add(self, features: List[GaitFeatureVector], strides: List[Dict], scores: List[float]) -> List[bool]:
        """加入一段行走；依當日策略封包逐步判斷是否提示（受每日配額限制），回傳每步是否提示。"""
        cued = []
        for s in scores:
            fire = s > self.packet.cue_threshold and self.cues_used < self.packet.daily_cue_budget
            if fire:
                self.cues_used += 1
            cued.append(fire)
        self.features.extend(features)
        self.strides.extend(strides)
        self.scores.extend(scores)
        return cued

    def summary(self, tau: float) -> Dict[str, Any]:
        f = self.features
        return {
            "day_index": self.day_index,
            "week": self.day_index // 7 + 1,
            "strides": len(f),
            "mean_deviation": round(_mean(self.scores), 3),
            "in_range_ratio": round(float(np.mean([s <= tau for s in self.scores])), 3) if self.scores else None,
            "mean_fpa": round(_mean([x.fpa for x in f]), 2),
            "mean_eversion": round(_mean([x.eversion_angle for x in f]), 2),
            "cadence": round(_mean([x.cadence for x in f]), 1),
            "stride_length": round(_mean([x.stride_length for x in f]), 3),
            "stance_ratio": round(_mean([x.stance_ratio for x in f]), 3),
            "cues_used": self.cues_used,
            "daily_cue_budget": self.packet.daily_cue_budget,
            "phase": self.packet.phase,
        }


class SystemState:
    """系統執行期狀態：啟動時以合成資料跑完整管線建立基線與六週歷史。"""

    def __init__(self, seed: int = 42):
        self.fs = SAMPLE_RATE
        self.stream = SyntheticGaitStream(sample_rate=self.fs, seed=seed)
        self.extractor = GaitMetricsExtractor(sample_rate=self.fs)
        self.baseline = PersonalGaitBaseline(normal_percentile=95.0, goal_direction=GOAL_DIRECTION)
        self.planner = LLMTrainingPlanner()
        self.daily_history: List[Dict[str, Any]] = []
        self.baseline_history: List[Dict[str, Any]] = []
        self.recent_days: Deque[Dict[str, Any]] = deque(maxlen=RATCHET_WINDOW_DAYS)
        self.current_plan: Optional[WeeklyTrainingPlan] = None
        self._calibrate(seed)
        self.fsm = FadedFeedbackFSM(personal_threshold=self.baseline.normal_threshold,
                                    require_progress=self.baseline.goal_direction != 0)
        self._simulate_history(seed + 1)
        self.current_plan = self._build_plan(use_llm=False)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _calibrate(self, seed: int):
        """建立期：使用者照平常走 CALIBRATION_DAYS 天（不提示），取最好的 BEST_K_DAYS 天作為個人最佳基線。"""
        rng = np.random.default_rng(seed)
        days = []
        for d in range(CALIBRATION_DAYS):
            acc, gyro, _ = self.stream.generate_walk_session(
                n_strides=SIM_STRIDES_PER_DAY, gait_type="in_toeing",
                custom_fpa=CALIBRATION_FPA + rng.normal(0.0, CALIBRATION_DAY_SD), fpa_sd=3.5)
            features = self.extractor.extract(acc, gyro)
            days.append({"day_index": d - CALIBRATION_DAYS, "features": features, "strides": self.extractor.last_strides})
        self.calibration_summary = [
            {"day_index": d["day_index"], "mean_fpa": round(_mean([f.fpa for f in d["features"]]), 2)} for d in days
        ]
        self.baseline.fit_best_days(days, BEST_K_DAYS)
        self.recent_days.extend(days)
        self._on_baseline_updated(day_index=-1, reason="calibration")

    def _on_baseline_updated(self, day_index: int, reason: str):
        """基線建立或上調後：更新走廊時間點、記錄演進歷史、同步狀態機的個人閾值 τ。"""
        source = [d for d in self.recent_days if d["day_index"] in self.baseline.source_days]
        self.baseline_fpa = float(self.baseline.mean_[self.baseline.goal_index])
        strides = [s for d in source for s in d.get("strides", [])]
        self.baseline_timing = {
            "toe_off_pct": round(_mean([s["toe_off_pct"] for s in strides]), 1),
            "heel_strike_pct": round(_mean([s["heel_strike_pct"] for s in strides]), 1),
        }
        self.baseline_history.append({
            "day_index": day_index,
            "week": day_index // 7 + 1 if day_index >= 0 else 0,
            "reason": reason,
            "best_fpa": round(self.baseline_fpa, 2),
            "tau": round(self.baseline.normal_threshold, 3),
            "source_days": list(self.baseline.source_days),
        })
        if hasattr(self, "fsm"):
            self.fsm.personal_threshold = self.baseline.normal_threshold

    def _end_of_day(self, day: "DayLog"):
        """每日關帳：記錄、推進狀態機；每週最後一天檢查棘輪（新最佳明顯更好才上調基線）。"""
        self.daily_history.append(day.summary(self.baseline.normal_threshold))
        self.recent_days.append({"day_index": day.day_index, "features": day.features, "strides": day.strides})
        _, changed, packet = self.fsm.step(_mean(day.scores))
        ratcheted = False
        if day.day_index % 7 == 6 and self.baseline.maybe_ratchet(list(self.recent_days), BEST_K_DAYS):
            self._on_baseline_updated(day.day_index, "weekly ratchet")
            self.fsm.record_progress()
            packet = self.fsm.build_packet()
            ratcheted = True
        return changed, packet, ratcheted

    def _process(self, acc: np.ndarray, gyro: np.ndarray):
        features = self.extractor.extract(acc, gyro)
        strides = self.extractor.last_strides
        scores = [self.baseline.compute_deviation_score(f) for f in features]
        return features, strides, scores

    def _simulate_history(self, seed: int):
        sessions = SyntheticGaitStream(sample_rate=self.fs, seed=seed).generate_longitudinal_recovery_stream(
            weeks=SIM_WEEKS, strides_per_day=SIM_STRIDES_PER_DAY
        )
        packet = self.fsm.build_packet()
        for d in sessions[:-1]:
            day = DayLog(d["day_index"], packet)
            day.add(*self._process(d["acc"], d["gyro"]))
            _, packet, _ = self._end_of_day(day)
        # 最後一天為「今天」，尚未關帳（每日評估尚未執行）
        last = sessions[-1]
        self.today = DayLog(last["day_index"], packet)
        self.today.add(*self._process(last["acc"], last["gyro"]))
        self.latest_packet = packet

    # ------------------------------------------------------------------
    # 彙整
    # ------------------------------------------------------------------

    def weekly_trajectory(self) -> List[Dict[str, Any]]:
        days = self.daily_history + [self.today.summary(self.baseline.normal_threshold)]
        weeks: Dict[int, List[Dict]] = {}
        for d in days:
            if d["strides"]:
                weeks.setdefault(d["week"], []).append(d)
        out = []
        for w, items in sorted(weeks.items()):
            out.append({
                "week": w,
                "days": len(items),
                "partial": len(items) < 7,
                "fpa": round(_mean([x["mean_fpa"] for x in items]), 2),
                "dev": round(_mean([x["mean_deviation"] for x in items]), 3),
                "cues": round(_mean([x["cues_used"] for x in items]), 1),
                "budget": items[-1]["daily_cue_budget"],
                "phase": items[-1]["phase"],
            })
        return out

    def week_summaries(self) -> List[WeekSummary]:
        return [
            WeekSummary(week=w["week"], mean_fpa=w["fpa"], mean_deviation=w["dev"], phase=w["phase"], cues_per_day=w["cues"])
            for w in self.weekly_trajectory()
        ]

    def plan_inputs(self) -> Dict[str, Any]:
        traj = self.weekly_trajectory()
        current = traj[-1] if traj else {"week": 1, "fpa": 0.0, "dev": 0.0}
        return {
            "week_number": current["week"],
            "phase": int(self.fsm.current_phase),
            "mean_fpa": current["fpa"],
            "mean_deviation": current["dev"],
            "improvement_rate": round(self.fsm.progress_rate() * 100.0, 1),
            "daily_cue_budget": self.latest_packet.daily_cue_budget,
            "history": self.week_summaries()[:-1],
            "baseline_fpa": round(self.baseline_fpa, 2),
            "baseline_fpa_sd": round(self.baseline.feature_sd("fpa"), 2),
            "goal_direction": self.baseline.goal_direction,
        }

    def _build_plan(self, use_llm: bool, lang: str = "en") -> WeeklyTrainingPlan:
        return self.planner.generate_plan(**self.plan_inputs(), use_llm=use_llm, lang=lang)

    def corridor(self) -> Dict[str, Any]:
        """個人走廊：基線期逐步波形的平均 ± 2SD，對照今天所有步的平均波形（皆為實際計算）。"""
        out: Dict[str, Any] = {
            "cycle_percentages": list(range(0, 101, 5)),
            "cycle_definition": "0% = end of foot-flat (before heel-off) → 100% = next foot-flat of the same foot",
            "band_sd": self.baseline.corridor_band_sd,
            "baseline_timing": self.baseline_timing,
            "baseline_strides": self.baseline.corridor_.get("fpa", {}).get("n_strides", 0),
            "baseline_source_days": list(self.baseline.source_days),
            "today_strides": len(self.today.strides),
        }
        names = {"fpa": ("Foot angle vs. walking direction (+ = toe-out)", "°"),
                 "eversion": ("Foot frontal-plane tilt (+ = eversion)", "°")}
        for ch, (label, unit) in names.items():
            band = self.baseline.corridor_[ch]
            current = (np.mean([s[f"{ch}_waveform"] for s in self.today.strides], axis=0)
                       if self.today.strides else None)
            entry = {
                "name": label,
                "unit": unit,
                "baseline_mean": np.round(band["mean"], 2).tolist(),
                "baseline_upper": np.round(band["upper"], 2).tolist(),
                "baseline_lower": np.round(band["lower"], 2).tolist(),
                "current_cycle": np.round(current, 2).tolist() if current is not None else [],
                "out_of_bound": [],
                "out_of_bound_ranges": [],
            }
            if current is not None:
                above = current > band["upper"]
                below = current < band["lower"]
                entry["out_of_bound"] = [bool(a or b) for a, b in zip(above, below)]
                entry["out_of_bound_ranges"] = _contiguous_ranges(above, below, current, band)
            out[ch] = entry
        return out


def _contiguous_ranges(above, below, current, band) -> List[Dict[str, Any]]:
    ranges, start, direction = [], None, None
    flags = ["above" if a else "below" if b else None for a, b in zip(above, below)]
    for i, flag in enumerate(flags + [None]):
        if flag != direction:
            if direction is not None:
                seg = slice(start, i)
                excess = (current[seg] - band["upper"][seg]) if direction == "above" else (band["lower"][seg] - current[seg])
                ranges.append({"start_pct": start * 5, "end_pct": (i - 1) * 5, "direction": direction,
                               "max_excess_deg": round(float(np.max(excess)), 2)})
            start, direction = i, flag
    return ranges


state = SystemState()


def _warm_up_llm():
    state.llm_warmup = {"status": "loading"}
    reason = state.planner.warm_up()
    state.llm_warmup = {"status": "ready"} if reason is None else {"status": "unavailable", "reason": reason}


state.llm_warmup = {"status": "not started"}
if os.environ.get("OWNSTRIDE_SKIP_LLM_WARMUP") != "1":
    import threading
    threading.Thread(target=_warm_up_llm, daemon=True).start()


# --- API 端點 ---

@app.get("/api/status")
def get_status():
    """L1 今日一眼：今日步數、落在個人範圍內比例、偏差、Phase、配額與策略封包"""
    tau = state.baseline.normal_threshold
    cfg = state.fsm.phase_configs[state.fsm.current_phase]
    return {
        "today_metrics": state.today.summary(tau),
        "strategy_packet": state.latest_packet.model_dump(),
        "baseline": {**state.baseline.get_state().model_dump(), "mean_fpa": round(state.baseline_fpa, 2),
                     "fpa_sd": round(state.baseline.feature_sd("fpa"), 2),
                     "goal_direction": state.baseline.goal_direction,
                     "last_update": state.baseline_history[-1],
                     "source": BASELINE_ASSUMPTION},
        "phase_info": {
            "current_phase": int(state.fsm.current_phase),
            "phase_name": cfg["name"],
            "daily_budget": state.latest_packet.daily_cue_budget,
            "base_budget": cfg["base_budget"],
            "faded_ratio": state.latest_packet.faded_ratio,
            "progress_rate": state.latest_packet.progress_rate,
            "deviation_level": state.latest_packet.deviation_level,
            "personal_threshold": round(tau, 3),
            "pending_phase": int(state.fsm.pending_phase) if state.fsm.pending_phase else None,
            "days_in_pending_state": state.fsm.days_in_pending_state,
            "debounce_days": state.fsm.debounce_days,
            "days_in_phase": state.fsm.days_in_phase,
            "requires_progress": state.fsm.require_progress,
            "progress_events_in_phase": state.fsm.progress_events_in_phase,
        },
        "data_provenance": DATA_PROVENANCE,
        "llm": {**state.llm_warmup, "model": state.planner.model},
    }


@app.get("/api/history")
def get_history():
    """L2：每週軌跡、每日明細、各特徵相對個人基線的 z 分數、個人走廊曲線"""
    return {
        "trajectory": state.weekly_trajectory(),
        "daily": state.daily_history + [state.today.summary(state.baseline.normal_threshold)],
        "baseline_normal_threshold": state.baseline.normal_threshold,
        "baseline_severe_threshold": state.baseline.severe_threshold,
        "feature_zscores": state.baseline.feature_zscores(state.today.features),
        "baseline_history": state.baseline_history,
        "calibration_days": state.calibration_summary,
        "kinematic_corridor": state.corridor(),
        "data_provenance": DATA_PROVENANCE,
    }


def _lang(lang: str) -> str:
    return "zh" if lang == "zh" else "en"


@app.get("/api/plan")
def get_current_plan(lang: str = "en"):
    """
    L3 訓練計畫。規則式備援的計畫可即時換語言；LLM 生成的計畫維持生成時的語言
    （介面會提示「重新產生」以取得另一種語言）。
    """
    plan = state.current_plan
    if not plan.is_generated_by_ollama and plan.language != _lang(lang):
        reason = plan.fallback_reason
        plan = state._build_plan(use_llm=False, lang=_lang(lang))
        plan.fallback_reason = reason
        state.current_plan = plan
    return plan.model_dump()


@app.post("/api/plan/generate")
def generate_fresh_plan(lang: str = "en"):
    """非串流生成（Ollama 不可用時回傳規則式備援並附原因）"""
    state.current_plan = state._build_plan(use_llm=True, lang=_lang(lang))
    return state.current_plan.model_dump()


@app.get("/api/plan/stream")
def stream_plan(lang: str = "en"):
    """串流生成（Server-Sent Events）：逐字輸出 LLM token，完成時附實測 tok/s"""
    def events():
        for event in state.planner.stream_plan(**state.plan_inputs(), lang=_lang(lang)):
            if event["type"] in ("done", "fallback"):
                state.current_plan = WeeklyTrainingPlan(**event["plan"])
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


class WalkSimulationRequest(BaseModel):
    n_strides: int = Field(default=20, ge=5, le=100)
    gait_type: str = Field(default="in_toeing", description="normal | in_toeing | out_toeing | over_pronation | custom")
    custom_fpa: Optional[float] = None
    custom_eversion: Optional[float] = None


@app.post("/api/walk")
def simulate_walk_session(req: WalkSimulationRequest):
    """
    加入今天的一段行走（合成）：IMU 訊號 → 姿態積分 → 特徵 → 偏差 → 依當日策略封包判斷提示。
    Phase 不在這裡改變——Phase 由每日評估（/api/day/close）決定，與架構一致。
    """
    gait_type = "normal" if req.gait_type == "custom" else req.gait_type
    acc, gyro, truth = state.stream.generate_walk_session(
        n_strides=req.n_strides, gait_type=gait_type,
        custom_fpa=req.custom_fpa, custom_eversion=req.custom_eversion,
    )
    features, strides, scores = state._process(acc, gyro)
    if not features:
        raise HTTPException(status_code=400, detail="Could not segment strides from session")
    cued = state.today.add(features, strides, scores)
    tau = state.baseline.normal_threshold

    strides_detail = [{
        "stride_index": i + 1,
        "fpa": round(f.fpa, 1),
        "fpa_truth": round(float(truth["target_fpa"].iloc[i]), 1) if i < len(truth) else None,
        "eversion": round(f.eversion_angle, 1),
        "stride_length": round(f.stride_length, 2),
        "mahalanobis_score": round(s, 2),
        "within_personal_range": bool(s <= tau),
        "cued": c,
    } for i, (f, s, c) in enumerate(zip(features, scores, cued))]

    fpa_err = [abs(d["fpa"] - d["fpa_truth"]) for d in strides_detail if d["fpa_truth"] is not None]
    return {
        "message": "Walking session processed",
        "data_source": "synthetic",
        "strides_processed": len(features),
        "session_summary": {
            "mean_deviation": round(_mean(scores), 3),
            "in_range_ratio": round(float(np.mean([s <= tau for s in scores])), 3),
            "cues_this_session": int(sum(cued)),
            "fpa_mae_vs_ground_truth_deg": round(_mean(fpa_err), 2),
        },
        "updated_today_metrics": state.today.summary(tau),
        "strategy_packet": state.latest_packet.model_dump(),
        "strides_detail": strides_detail,
    }


@app.post("/api/day/close")
def close_day():
    """每日評估：以今日平均偏差推進漸退狀態機，產生明日策略封包，開始新的一天。"""
    if not state.today.scores:
        raise HTTPException(status_code=400, detail="No strides recorded today")
    summary = state.today.summary(state.baseline.normal_threshold)
    changed, packet, ratcheted = state._end_of_day(state.today)
    state.latest_packet = packet
    state.today = DayLog(summary["day_index"] + 1, packet)
    return {
        "closed_day": summary,
        "phase": int(state.fsm.current_phase),
        "phase_changed": changed,
        "pending_phase": int(state.fsm.pending_phase) if state.fsm.pending_phase else None,
        "days_in_pending_state": state.fsm.days_in_pending_state,
        "debounce_days": state.fsm.debounce_days,
        "baseline_ratcheted": ratcheted,
        "baseline": state.baseline_history[-1],
        "strategy_packet": packet.model_dump(),
    }


def _stride_stats(durations: List[float]) -> Dict[str, Any]:
    d = np.array(durations)
    return {
        "median_stride_sec": round(float(np.median(d)), 3),
        "cadence_steps_per_min": round(float(120.0 / np.median(d)), 1),   # 1 跨步 = 2 步
        "stride_time_cv": round(float(np.std(d) / np.mean(d)), 3),
    }


_wisdm_cohort_cache: Optional[Dict[str, Any]] = None


def _wisdm_cohort() -> Dict[str, Any]:
    """全部 WISDM 受試者各取 20 秒連續行走，以同一切分器計算跨步時間（結果快取）。"""
    global _wisdm_cohort_cache
    if _wisdm_cohort_cache is None:
        from src.dataset.wisdm_loader import WISDMLoader, WISDM_TARGET_RATE
        from src.features.event_detector import GaitEventDetector

        loader, detector = WISDMLoader(), GaitEventDetector(sample_rate=WISDM_TARGET_RATE)
        n_users = loader.load_walking_data()["user"].nunique()
        medians, cvs, kept, total = [], [], 0, 0
        for sid in range(1, n_users + 1):
            acc, _, _ = loader.get_subject_session(subject_id=sid, n_samples=int(20 * WISDM_TARGET_RATE))
            seg = detector.segment_strides(acc)
            if len(seg["strides"]) < 3:
                continue
            stats = _stride_stats([x["duration_sec"] for x in seg["strides"]])
            medians.append(stats["median_stride_sec"])
            cvs.append(stats["stride_time_cv"])
            kept += len(seg["strides"])
            total += len(seg["strides"]) + seg["rejected"]
        _wisdm_cohort_cache = {
            "subjects": len(medians),
            "median_stride_sec_range": [round(min(medians), 2), round(max(medians), 2)],
            "median_stride_sec": round(float(np.median(medians)), 2),
            "median_stride_time_cv": round(float(np.median(cvs)), 3),
            "strides_kept_ratio": round(kept / total, 3),
        }
    return _wisdm_cohort_cache


@app.get("/api/benchmark/real")
def get_real_walking_benchmark(subject_id: int = 1, n_samples: int = 1280):
    """
    真人資料檢查：WISDM v1.1 手機加速度（無陀螺儀，手機在大腿口袋）→ 僅加速度的跨步切分。
    只能驗證跨步時間與步頻在真人訊號上可切分；此資料無法計算足偏角，也無法可靠定出腳尖離地。
    """
    from src.dataset.wisdm_loader import WISDMLoader, WISDM_TARGET_RATE
    from src.features.event_detector import GaitEventDetector

    loader = WISDMLoader()
    if not loader.is_downloaded():
        raise HTTPException(status_code=404, detail="WISDM data not available locally")
    acc, _, meta = loader.get_subject_session(subject_id=subject_id, n_samples=n_samples)
    seg = GaitEventDetector(sample_rate=WISDM_TARGET_RATE).segment_strides(acc)
    strides = seg["strides"]

    return {
        "is_synthetic": False,
        "metadata": meta,
        "limitations": ("Thigh-pocket phone accelerometer only (no gyroscope): stride time and cadence only; "
                        "no foot progression angle and no reliable toe-off timing."),
        "samples_count": len(acc),
        "stride_period_sec": seg["stride_period_sec"],
        "strides_detected": len(strides),
        "strides_rejected": seg["rejected"],
        "stride_stats": _stride_stats([x["duration_sec"] for x in strides]) if strides else None,
        "stride_details": [
            {"stride_index": i + 1, "start_idx": x["start"], "end_idx": x["end"], "duration_sec": round(x["duration_sec"], 3)}
            for i, x in enumerate(strides[:12])
        ],
        "cohort": _wisdm_cohort(),
        "acc_waveform_preview": {
            "x": [round(float(v), 3) for v in acc[:128, 0]],
            "y": [round(float(v), 3) for v in acc[:128, 1]],
            "z": [round(float(v), 3) for v in acc[:128, 2]],
        },
    }


@app.middleware("http")
async def no_cache_for_pages(request, call_next):
    """避免瀏覽器快取舊版頁面（展示時頁面與 API 版本必須一致）"""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store"
    return response


# 掛載 Web 靜態網頁資源
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
