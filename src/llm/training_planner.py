"""
src/llm/training_planner.py
===========================
每週訓練計畫生成器 (Weekly Training Planner)
依據規劃書 v0.7：
1. 評估層非同步執行，不在即時路徑上（每週一次、約 400 token）。
2. Stage I：本機 Ollama 跑 Qwen2.5-1.5B（與 Stage II 在 Hailo-10H 上部署的是同一顆模型）；
   Stage II 改接 UGen300 推論端點，介面不變。
3. Ollama 不可用時切換到規則式備援，輸出會標記 is_generated_by_ollama=False，介面必須如實顯示。
4. 措辭遵守 Phase 1 定位（一般健康／動作訓練）：不使用 diagnosis / clinical / rehabilitation / patient。
5. 基線為「個人最佳日」：計畫以「相對自己的最佳狀態」描述本週表現，訓練動作依使用者選擇的訓練方向挑選。

環境變數：
- OWNSTRIDE_OLLAMA_URL（預設 http://localhost:11434）
- OWNSTRIDE_LLM_MODEL（預設 qwen2.5:1.5b）
- OWNSTRIDE_LLM_TIMEOUT 秒（預設 300；CPU 上 1.5B 模型生成 400 token 可能需要 30 秒以上）
"""

import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class ExerciseItem(BaseModel):
    name: str = Field(..., description="動作名稱")
    target_muscle: str = Field(..., description="主要訓練肌群")
    frequency: str = Field(..., description="執行頻率")
    dosage: str = Field(..., description="組數 × 次數")
    rationale: str = Field(..., description="這個動作和本週步態數據的關聯")


class WeeklyTrainingPlan(BaseModel):
    week_number: int
    current_phase: int
    phase_title: str
    faded_feedback_guidance: str
    movement_analysis: str = Field(..., description="本週步態數據的白話解讀")
    primary_focus: str
    daily_cue_budget: int
    exercises: List[ExerciseItem]
    weekly_summary: str
    is_generated_by_ollama: bool = False
    model_name: Optional[str] = None
    generation_stats: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None
    language: str = "en"


class WeekSummary(BaseModel):
    """提供給 LLM 的單週摘要（由評估層實際計算）"""
    week: int
    mean_fpa: float
    mean_deviation: float
    phase: int
    cues_per_day: float


# 交給 Ollama structured outputs 的 JSON schema（只含模型需要產生的欄位）
_LLM_OUTPUT_FIELDS = [
    "week_number", "current_phase", "phase_title", "faded_feedback_guidance",
    "movement_analysis", "primary_focus", "daily_cue_budget", "exercises", "weekly_summary",
]


def _llm_output_schema() -> Dict[str, Any]:
    schema = WeeklyTrainingPlan.model_json_schema()
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in _LLM_OUTPUT_FIELDS}
    schema["required"] = _LLM_OUTPUT_FIELDS
    return schema


def classify_fpa(
    mean_fpa: float,
    baseline_fpa: Optional[float],
    baseline_fpa_sd: Optional[float],
    goal_direction: Optional[int] = None,
) -> str:
    """
    相對「個人最佳基線」描述足偏角，超出 ±2SD 才算有差別：
    - 有訓練方向（±1）：'worse'（比最佳退步）/ 'at_best' / 'better'（超越最佳）
    - 方向 0（維持）：'outside' / 'within'
    - 沒有基線資訊：退回以 0° 為界的 'toe_in' / 'toe_out'（僅供單元測試或基線尚未建立時）
    """
    if baseline_fpa is None or baseline_fpa_sd is None:
        return "toe_in" if mean_fpa < 0 else "toe_out"
    margin = 2 * baseline_fpa_sd
    if not goal_direction:
        return "within" if abs(mean_fpa - baseline_fpa) <= margin else "outside"
    gain = goal_direction * (mean_fpa - baseline_fpa)
    if gain < -margin:
        return "worse"
    if gain > margin:
        return "better"
    return "at_best"


PHASE_TITLES = {
    "en": {1: "Phase 1: Acquisition (frequent cues)", 2: "Phase 2: Faded Feedback (fewer cues)", 3: "Phase 3: Retention (rare cues)"},
    "zh": {1: "Phase 1：建立期（提示頻繁）", 2: "Phase 2：漸退期（提示減少）", 3: "Phase 3：維持期（很少提示）"},
}

GOAL_TEXT = {
    1: "reduce in-toeing (a larger foot angle is better)",
    -1: "reduce out-toeing (a smaller foot angle is better)",
    0: "maintain the current walking pattern",
}

RELATION_TEXT = {
    "worse": "slipped back compared with the user's personal best",
    "at_best": "at the user's personal best",
    "better": "better than the user's personal best (a new best)",
    "within": "within the user's usual range",
    "outside": "outside the user's usual range",
    "toe_in": "toes turned in",
    "toe_out": "toes turned out",
}


def describe_trend(improvement_rate: float) -> str:
    """把每週趨勢數字先判讀成文字，避免小模型誤讀正負號。"""
    if improvement_rate > 5:
        return "improving: getting closer to (or beyond) the personal best"
    if improvement_rate < -5:
        return "slipping: drifting further from the personal best than last week"
    return "steady: about the same as last week"


KEEP_ALIVE = "30m"  # 模型常駐記憶體時間，避免每週生成時重新載入

_TO_TRADITIONAL = None


def to_traditional(text: str) -> str:
    """
    Qwen2.5-1.5B 即使被要求繁體仍常輸出簡體；以 OpenCC s2twp（簡體→台灣繁體含用詞）後處理。
    未安裝 opencc 時原樣回傳。
    """
    global _TO_TRADITIONAL
    if _TO_TRADITIONAL is None:
        try:
            from opencc import OpenCC
            _TO_TRADITIONAL = OpenCC("s2twp").convert
        except ImportError:
            _TO_TRADITIONAL = lambda x: x  # noqa: E731
    return _TO_TRADITIONAL(text)


class LLMTrainingPlanner:
    """結合步態數據與本機 LLM 的每週訓練計畫生成模組。"""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ):
        self.base_url = (ollama_url or os.environ.get("OWNSTRIDE_OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OWNSTRIDE_LLM_MODEL", "qwen2.5:1.5b")
        self.timeout_sec = float(timeout_sec or os.environ.get("OWNSTRIDE_LLM_TIMEOUT", 300))

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        week_number: int,
        phase: int,
        mean_fpa: float,
        mean_deviation: float,
        improvement_rate: float,
        daily_cue_budget: int,
        history: Optional[List[WeekSummary]] = None,
        baseline_fpa: Optional[float] = None,
        baseline_fpa_sd: Optional[float] = None,
        goal_direction: Optional[int] = None,
        lang: str = "en",
    ) -> str:
        relation = RELATION_TEXT[classify_fpa(mean_fpa, baseline_fpa, baseline_fpa_sd, goal_direction)]
        baseline_line = (
            f"- User's personal-best foot progression angle: {baseline_fpa:+.1f} deg (usual spread +/- {2 * baseline_fpa_sd:.1f} deg)\n"
            if baseline_fpa is not None and baseline_fpa_sd is not None else ""
        )
        goal_line = f"- Training goal chosen by the user: {GOAL_TEXT[goal_direction]}\n" if goal_direction is not None else ""
        history_lines = "\n".join(
            f"- Week {h.week}: mean foot progression angle {h.mean_fpa:+.1f} deg, "
            f"mean shortfall from personal best {h.mean_deviation:.2f}, phase {h.phase}, {h.cues_per_day:.0f} cues/day"
            for h in (history or [])
        ) or "- (no earlier weeks)"
        language_line = ("所有文字欄位（包含動作名稱與說明）一律使用繁體中文（台灣用語），不要使用英文或簡體中文。"
                         if lang == "zh" else "Write every text field in English.")
        return f"""You are the weekly planner inside OwnStride, a general-wellness walking-form training aid.
You are not a medical device. Do not diagnose, do not mention diseases, patients, treatment or rehabilitation.
Write in plain, encouraging language for an everyday user. {language_line}

OwnStride compares the user only with their own best days, never with other people.

Measured by a foot-worn motion sensor this week:
- Week: {week_number}
- Feedback phase: {phase} ({PHASE_TITLES['en'].get(phase, '')})
- Vibration cue budget next week: {daily_cue_budget} cues/day
{goal_line}- Mean foot progression angle: {mean_fpa:+.1f} deg ({relation}; positive = toes point outward)
{baseline_line}- Mean shortfall from personal best (Mahalanobis distance, 0 = at or above best): {mean_deviation:.2f}
- Weekly trend (already interpreted, use as is): {describe_trend(improvement_rate)}

Earlier weeks:
{history_lines}

Return a JSON object with:
- phase_title, primary_focus (one sentence)
- movement_analysis: explain what the numbers and the multi-week trend mean compared with the personal best, in two or three sentences
- faded_feedback_guidance: how the user should treat the vibration cues this week given the phase
- exercises: 2 to 4 simple strength or balance exercises with target_muscle, frequency, dosage, and a rationale tied to the training goal
- weekly_summary: two encouraging sentences
Use week_number={week_number}, current_phase={phase}, daily_cue_budget={daily_cue_budget}.
Only restate the interpretations given above; do not compute new trends from the numbers.
{language_line}"""

    # ------------------------------------------------------------------
    # Ollama 連線
    # ------------------------------------------------------------------

    def check_available(self) -> Optional[str]:
        """回傳 None 表示可用；否則回傳無法使用的原因（供介面顯示）。"""
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=1.5) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return f"Ollama not reachable at {self.base_url} ({type(e).__name__})"
        names = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names and self.model not in names:
            return f"model '{self.model}' not pulled in Ollama (run: ollama pull {self.model})"
        return None

    def warm_up(self) -> Optional[str]:
        """
        預先把模型載入記憶體（首次載入在筆電上可能需要數分鐘）。回傳 None 表示成功，否則回傳原因。
        API 啟動時在背景執行緒呼叫，讓第一次按下「產生計畫」不必等待模型載入。
        """
        reason = self.check_available()
        if reason:
            return reason
        payload = json.dumps({"model": self.model, "prompt": "", "keep_alive": KEEP_ALIVE}).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec * 3) as resp:
                resp.read()
        except OSError as e:
            return f"warm-up failed: {type(e).__name__}: {e}"
        return None

    def _request(self, prompt: str, stream: bool) -> urllib.request.Request:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": _llm_output_schema(),
            "stream": stream,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.3},
        }
        return urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def _finalize(self, raw_text: str, final_chunk: Dict[str, Any], wall_sec: float, lang: str) -> WeeklyTrainingPlan:
        parsed = json.loads(to_traditional(raw_text) if lang == "zh" else raw_text)
        eval_count = final_chunk.get("eval_count")
        eval_ns = final_chunk.get("eval_duration")
        stats = {
            "output_tokens": eval_count,
            "tokens_per_sec": round(eval_count / (eval_ns / 1e9), 2) if eval_count and eval_ns else None,
            "time_to_first_token_sec": final_chunk.get("_ttft_sec"),
            "wall_time_sec": round(wall_sec, 2),
            "prompt_tokens": final_chunk.get("prompt_eval_count"),
        }
        return WeeklyTrainingPlan(
            **{k: parsed[k] for k in _LLM_OUTPUT_FIELDS if k in parsed},
            is_generated_by_ollama=True,
            model_name=self.model,
            generation_stats=stats,
            language=lang,
        )

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        week_number: int,
        phase: int,
        mean_fpa: float,
        mean_deviation: float,
        improvement_rate: float,
        daily_cue_budget: int,
        history: Optional[List[WeekSummary]] = None,
        use_llm: bool = True,
        baseline_fpa: Optional[float] = None,
        baseline_fpa_sd: Optional[float] = None,
        goal_direction: Optional[int] = None,
        lang: str = "en",
    ) -> WeeklyTrainingPlan:
        """非串流生成：優先使用本機 Ollama，不可用或輸出不合 schema 時切換規則式備援並註明原因。"""
        args = (week_number, phase, mean_fpa, mean_deviation, improvement_rate, daily_cue_budget)
        context = dict(baseline_fpa=baseline_fpa, baseline_fpa_sd=baseline_fpa_sd, goal_direction=goal_direction, lang=lang)
        if not use_llm:
            return self._generate_deterministic_plan(*args, **context, reason="LLM not requested (startup default)")
        reason = self.check_available()
        if reason is None:
            try:
                t0 = time.time()
                request = self._request(self.build_prompt(*args, history=history, **context), stream=False)
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._finalize(body.get("response", ""), body, time.time() - t0, lang)
            except (OSError, json.JSONDecodeError, ValidationError, KeyError) as e:
                reason = f"LLM call failed: {type(e).__name__}: {e}"
        logger.info("Falling back to rule-based planner: %s", reason)
        return self._generate_deterministic_plan(*args, **context, reason=reason)

    def stream_plan(
        self,
        week_number: int,
        phase: int,
        mean_fpa: float,
        mean_deviation: float,
        improvement_rate: float,
        daily_cue_budget: int,
        history: Optional[List[WeekSummary]] = None,
        baseline_fpa: Optional[float] = None,
        baseline_fpa_sd: Optional[float] = None,
        goal_direction: Optional[int] = None,
        lang: str = "en",
    ) -> Iterator[Dict[str, Any]]:
        """
        串流生成，逐一產出事件：
        {"type": "token", "text": ...}  — 模型逐字輸出
        {"type": "done", "plan": {...}} — 完成，含實測 tok/s
        {"type": "fallback", "reason": ..., "plan": {...}} — 無法使用 LLM，改用規則式備援
        """
        args = (week_number, phase, mean_fpa, mean_deviation, improvement_rate, daily_cue_budget)
        context = dict(baseline_fpa=baseline_fpa, baseline_fpa_sd=baseline_fpa_sd, goal_direction=goal_direction, lang=lang)
        reason = self.check_available()
        if reason is None:
            raw, final_chunk = [], {}
            t0 = time.time()
            ttft = None
            try:
                request = self._request(self.build_prompt(*args, history=history, **context), stream=True)
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as resp:
                    for line in resp:
                        if not line.strip():
                            continue
                        chunk = json.loads(line.decode("utf-8"))
                        piece = chunk.get("response", "")
                        if piece:
                            if ttft is None:
                                ttft = round(time.time() - t0, 3)
                            raw.append(piece)
                            yield {"type": "token", "text": to_traditional(piece) if lang == "zh" else piece}
                        if chunk.get("done"):
                            final_chunk = chunk
                            break
                final_chunk["_ttft_sec"] = ttft
                plan = self._finalize("".join(raw), final_chunk, time.time() - t0, lang)
                yield {"type": "done", "plan": plan.model_dump()}
                return
            except (OSError, json.JSONDecodeError, ValidationError, KeyError) as e:
                reason = f"LLM call failed: {type(e).__name__}: {e}"
        plan = self._generate_deterministic_plan(*args, **context, reason=reason)
        yield {"type": "fallback", "reason": reason, "plan": plan.model_dump()}

    # ------------------------------------------------------------------
    # 規則式備援（無 LLM 時）
    # ------------------------------------------------------------------

    def _generate_deterministic_plan(
        self,
        week_number: int,
        phase: int,
        mean_fpa: float,
        mean_deviation: float,
        improvement_rate: float,
        daily_cue_budget: int,
        baseline_fpa: Optional[float] = None,
        baseline_fpa_sd: Optional[float] = None,
        goal_direction: Optional[int] = None,
        lang: str = "en",
        reason: Optional[str] = None,
    ) -> WeeklyTrainingPlan:
        """
        規則式範本：動作組合依訓練方向挑選，描述依「相對個人最佳」的關係挑選；只有數字會隨輸入變動。
        """
        lang = "zh" if lang == "zh" else "en"
        t = _TEMPLATES[lang]
        relation = classify_fpa(mean_fpa, baseline_fpa, baseline_fpa_sd, goal_direction)
        # 訓練方向：有明確目標時依目標；沒有時（舊介面／單元測試）依 FPA 正負推定
        goal = goal_direction if goal_direction is not None else (1 if relation == "toe_in" else -1 if relation == "toe_out" else 0)
        best = f"{baseline_fpa:+.1f}°" if baseline_fpa is not None else "—"
        fmt = dict(fpa=f"{mean_fpa:+.1f}°", best=best, budget=daily_cue_budget, dev=mean_deviation,
                   rate=improvement_rate, week=week_number, phase_title=PHASE_TITLES[lang].get(phase, f"Phase {phase}"))

        return WeeklyTrainingPlan(
            week_number=week_number,
            current_phase=phase,
            phase_title=fmt["phase_title"],
            faded_feedback_guidance=t["guide"][min(max(phase, 1), 3)].format(**fmt),
            movement_analysis=t["analysis"].format(**fmt),
            primary_focus=t["focus"][relation].format(**fmt),
            daily_cue_budget=daily_cue_budget,
            exercises=[ExerciseItem(**e) for e in t["exercises"][goal]],
            weekly_summary=t["summary"].format(**fmt),
            is_generated_by_ollama=False,
            model_name=None,
            fallback_reason=reason,
            language=lang,
        )


_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "en": {
        "guide": {
            1: "Frequent cues this week ({budget}/day). Each time the sensor vibrates, notice where your toes are pointing.",
            2: "Cues are reduced to {budget}/day. Try to check your foot direction yourself before the sensor has to remind you.",
            3: "Only {budget} cues/day, kept for larger slips. Your own sense of foot position is doing most of the work now.",
        },
        "focus": {
            "worse": "Get back to your personal best (this week {fpa}, your best {best})",
            "at_best": "Hold your personal best and nudge it a little further (this week {fpa}, your best {best})",
            "better": "New personal best! This week {fpa} beats your best {best}; the baseline moves up at the weekly review",
            "within": "Keep your usual pattern (this week {fpa}, your baseline {best})",
            "outside": "Return to your usual pattern (this week {fpa}, your baseline {best})",
            "toe_in": "Point the toes closer to straight ahead (current foot angle {fpa}, turned in)",
            "toe_out": "Reduce outward foot turn (current foot angle {fpa}, turned out)",
        },
        "analysis": "Your average foot angle this week was {fpa} against your personal best of {best}. "
                    "Your average shortfall from that best was {dev:.2f} (0 means at or beyond your best), "
                    "trending {rate:+.1f}% per week.",
        "summary": "Week {week}: {phase_title} with {budget} cues per day. Keep practising the drills between walks.",
        "exercises": {
            1: [
                {"name": "Clamshell with Resistance Band", "target_muscle": "Gluteus Medius & Hip External Rotators",
                 "frequency": "3-4 days / week", "dosage": "3 sets of 15 reps per side",
                 "rationale": "Stronger hip external rotators make it easier to keep the knee and foot from turning inward."},
                {"name": "Straight-Line Walking Drill", "target_muscle": "Foot & Ankle Stabilizers",
                 "frequency": "Daily, 5 minutes", "dosage": "3 sets of 20 steps along a floor line",
                 "rationale": "Practising foot direction without the sensor builds the habit the cues are fading toward."},
                {"name": "Single-Leg Balance", "target_muscle": "Ankle Stabilizers & Peroneals",
                 "frequency": "3 days / week", "dosage": "3 sets of 30 seconds per leg",
                 "rationale": "Better balance on one leg supports a steadier foot position during each step."},
            ],
            -1: [
                {"name": "Short Foot Exercise", "target_muscle": "Intrinsic Foot Muscles",
                 "frequency": "Daily", "dosage": "3 sets of 10 holds (5 seconds each)",
                 "rationale": "Activating the arch helps the foot stay stable instead of rolling and turning outward."},
                {"name": "Step-Down with Knee Tracking", "target_muscle": "Quadriceps & Gluteus Maximus",
                 "frequency": "3 days / week", "dosage": "3 sets of 12 reps per leg",
                 "rationale": "Controlling knee direction on a step makes it easier to keep the foot pointing forward."},
            ],
            0: [
                {"name": "Straight-Line Walking Drill", "target_muscle": "Foot & Ankle Stabilizers",
                 "frequency": "3 days / week, 5 minutes", "dosage": "3 sets of 20 steps along a floor line",
                 "rationale": "Keeps your usual foot direction without relying on cues."},
                {"name": "Single-Leg Balance", "target_muscle": "Ankle Stabilizers & Peroneals",
                 "frequency": "3 days / week", "dosage": "3 sets of 30 seconds per leg",
                 "rationale": "Balance practice maintains steady foot placement as cues become rare."},
            ],
        },
    },
    "zh": {
        "guide": {
            1: "本週提示較頻繁（每天 {budget} 次）。每次感測器震動時，留意一下腳尖朝向哪裡。",
            2: "提示減為每天 {budget} 次。試著在感測器提醒前，先自己檢查腳的方向。",
            3: "每天只剩 {budget} 次提示，只在明顯退步時出現。現在主要靠你自己的身體感覺維持。",
        },
        "focus": {
            "worse": "回到你的個人最佳（本週 {fpa}，你的最佳 {best}）",
            "at_best": "維持你的個人最佳，並再往前推一點（本週 {fpa}，你的最佳 {best}）",
            "better": "新的個人最佳！本週 {fpa} 超越原本的最佳 {best}，每週檢查時基線會往上調",
            "within": "維持你平常的步態（本週 {fpa}，你的基線 {best}）",
            "outside": "回到你平常的步態（本週 {fpa}，你的基線 {best}）",
            "toe_in": "讓腳尖更朝向正前方（目前足偏角 {fpa}，偏內）",
            "toe_out": "減少腳尖外轉（目前足偏角 {fpa}，偏外）",
        },
        "analysis": "本週平均足偏角 {fpa}，你的個人最佳是 {best}。和最佳狀態相比平均落後 {dev:.2f}"
                    "（0 代表達到或超越最佳），趨勢為每週 {rate:+.1f}%。",
        "summary": "第 {week} 週：{phase_title}，每天 {budget} 次提示。走路之外的時間，記得持續做下面的練習。",
        "exercises": {
            1: [
                {"name": "彈力帶蚌殼式", "target_muscle": "臀中肌與髖外旋肌群",
                 "frequency": "每週 3–4 天", "dosage": "每側 3 組 × 15 下",
                 "rationale": "髖外旋肌群更有力，比較容易讓膝蓋和腳不往內轉。"},
                {"name": "直線行走練習", "target_muscle": "足踝穩定肌群",
                 "frequency": "每天 5 分鐘", "dosage": "沿地板直線走 3 組 × 20 步",
                 "rationale": "不靠感測器練習腳的方向，建立提示逐漸減少後仍能維持的習慣。"},
                {"name": "單腳站立平衡", "target_muscle": "足踝穩定肌群與腓骨肌",
                 "frequency": "每週 3 天", "dosage": "每腳 3 組 × 30 秒",
                 "rationale": "單腳平衡更好，每一步的腳掌位置會更穩定。"},
            ],
            -1: [
                {"name": "短足運動", "target_muscle": "足部內在肌",
                 "frequency": "每天", "dosage": "3 組 × 10 次（每次維持 5 秒）",
                 "rationale": "啟動足弓有助於腳掌穩定，減少向外翻轉。"},
                {"name": "下階梯膝蓋對位", "target_muscle": "股四頭肌與臀大肌",
                 "frequency": "每週 3 天", "dosage": "每腳 3 組 × 12 下",
                 "rationale": "下階梯時控制膝蓋方向，比較容易讓腳尖朝前。"},
            ],
            0: [
                {"name": "直線行走練習", "target_muscle": "足踝穩定肌群",
                 "frequency": "每週 3 天，每次 5 分鐘", "dosage": "沿地板直線走 3 組 × 20 步",
                 "rationale": "不依賴提示也能維持平常的腳尖方向。"},
                {"name": "單腳站立平衡", "target_muscle": "足踝穩定肌群與腓骨肌",
                 "frequency": "每週 3 天", "dosage": "每腳 3 組 × 30 秒",
                 "rationale": "提示變少時，平衡練習有助於維持穩定的落腳位置。"},
            ],
        },
    },
}
