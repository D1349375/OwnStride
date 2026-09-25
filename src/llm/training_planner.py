"""
src/llm/training_planner.py
===========================
每週訓練計畫生成器 (Weekly Training Planner)
依據規劃書 v0.7：
1. 評估層非同步執行，不在即時路徑上（每週一次）。
2. Stage I：本機 Ollama 跑 Qwen2.5-1.5B（與 Stage II 在 Hailo-10H 上部署的是同一顆模型）；
   Stage II 改接 UGen300 推論端點，介面不變。
3. Ollama 不可用時切換到規則式備援，輸出會標記 is_generated_by_ollama=False，介面必須如實顯示。
4. 措辭遵守 Phase 1 定位（一般健康／動作訓練）：不使用 diagnosis / clinical / rehabilitation / patient。
5. 基線為「個人最佳日」：計畫以「相對自己的最佳狀態」描述本週表現，訓練動作依使用者選擇的訓練方向挑選。
6. 程式與 LLM 分工（2026-09-25 修正）：1.5B 模型即使拿到「已判讀」的提示，仍會把數字講錯
   （實測把本週平均 +5.4° 說成「進步了 +5.4°」、把落後 0.31 說成「接近最佳 31%」）。因此：
   - 程式寫：週次、Phase、配額、數據解讀（movement_analysis）與動作內容（名稱、劑量、理由）；
   - LLM 寫：本週重點、提示使用建議、週總結，並從審核過的動作清單中挑選 2–3 項（JSON schema 以 enum 限制）；
   - prompt 不提供任何原始數字，並要求 LLM 不寫數字。

環境變數：
- OWNSTRIDE_OLLAMA_URL（預設 http://localhost:11434）
- OWNSTRIDE_LLM_MODEL（預設 qwen2.5:1.5b）
- OWNSTRIDE_LLM_TIMEOUT 秒（預設 300；CPU 上 1.5B 模型生成可能需要 30 秒以上）
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
    rationale: str = Field(..., description="這個動作和訓練方向的關聯")


class WeeklyTrainingPlan(BaseModel):
    week_number: int
    current_phase: int
    phase_title: str
    faded_feedback_guidance: str
    movement_analysis: str = Field(..., description="本週步態數據的白話解讀（程式產生）")
    primary_focus: str
    daily_cue_budget: int
    exercises: List[ExerciseItem]
    weekly_summary: str
    is_generated_by_ollama: bool = False
    model_name: Optional[str] = None
    generation_stats: Optional[Dict[str, Any]] = None
    fallback_reason: Optional[str] = None
    language: str = "en"
    llm_fields: List[str] = Field(default_factory=list, description="由 LLM 撰寫／挑選的欄位；其餘由程式產生")


class WeekSummary(BaseModel):
    """單週摘要（由評估層實際計算）"""
    week: int
    mean_fpa: float
    mean_deviation: float
    phase: int
    cues_per_day: float


# LLM 只產生這些欄位；exercise_ids 會在程式端換成動作庫內容
_LLM_TEXT_FIELDS = ["primary_focus", "faded_feedback_guidance", "weekly_summary"]
LLM_FIELDS = ["primary_focus", "exercises", "faded_feedback_guidance", "weekly_summary"]


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


def resolve_goal(goal_direction: Optional[int], relation: str) -> int:
    """訓練方向：有明確目標時依目標；沒有時（舊介面／單元測試）依 FPA 正負推定。"""
    if goal_direction is not None:
        return goal_direction
    return 1 if relation == "toe_in" else -1 if relation == "toe_out" else 0


def trend_key(improvement_rate: float) -> str:
    if improvement_rate > 5:
        return "improving"
    if improvement_rate < -5:
        return "slipping"
    return "steady"


def multi_week_key(history: Optional[List[WeekSummary]], mean_fpa: float, goal: int,
                   baseline_fpa_sd: Optional[float]) -> Optional[str]:
    """第一週到本週的足偏角變化，依訓練方向判讀為 toward / away / flat（差距小於 2SD 或 1° 視為持平）。"""
    if not history:
        return None
    margin = max(2 * baseline_fpa_sd, 1.0) if baseline_fpa_sd else 1.0
    change = mean_fpa - history[0].mean_fpa
    if goal == 0 or abs(change) <= margin:
        return "flat"
    return "toward" if goal * change > 0 else "away"


PHASE_TITLES = {
    "en": {1: "Phase 1: Acquisition (frequent cues)", 2: "Phase 2: Faded Feedback (fewer cues)", 3: "Phase 3: Retention (rare cues)"},
    "zh": {1: "Phase 1：建立期（提示頻繁）", 2: "Phase 2：漸退期（提示減少）", 3: "Phase 3：維持期（很少提示）"},
}

GOAL_TEXT = {
    1: "reduce in-toeing (turn the toes a little more outward over time)",
    -1: "reduce out-toeing (turn the toes a little more inward over time)",
    0: "maintain the current walking pattern",
}
# 繁中計畫時一併提供中文說法，避免小模型自行翻譯走樣（實測曾譯成「增加腳向後傾斜」）
GOAL_TEXT_ZH = {1: "減少內八：讓腳尖逐漸稍微朝外", -1: "減少外八：讓腳尖逐漸稍微朝內", 0: "維持目前的走路方式"}

# 給 LLM 的判讀文字（不含數字）
RELATION_TEXT = {
    "worse": "slipped back compared with the user's personal best",
    "at_best": "at the user's personal best",
    "better": "better than the user's personal best (a new best)",
    "within": "within the user's usual range",
    "outside": "outside the user's usual range",
    "toe_in": "toes turned in",
    "toe_out": "toes turned out",
}

TREND_TEXT = {
    "improving": "improving: getting closer to (or beyond) the personal best",
    "slipping": "slipping: drifting further from the personal best than last week",
    "steady": "steady: about the same as last week",
}

MULTI_WEEK_TEXT = {
    "toward": "over the weeks so far, walking has clearly moved toward the training goal",
    "away": "over the weeks so far, walking has moved away from the training goal",
    "flat": "over the weeks so far, walking has stayed about the same",
}


def describe_trend(improvement_rate: float) -> str:
    """把每週趨勢數字先判讀成文字，避免小模型誤讀正負號。"""
    return TREND_TEXT[trend_key(improvement_rate)]


# ----------------------------------------------------------------------
# 動作庫：LLM 只能從對應訓練方向的清單挑選（schema enum），內容與劑量由程式提供
# ----------------------------------------------------------------------

EXERCISE_LIBRARY: Dict[str, Dict[str, Dict[str, str]]] = {
    "clamshell": {
        "en": {"name": "Clamshell with Resistance Band", "target_muscle": "Gluteus Medius & Hip External Rotators",
               "frequency": "3-4 days / week", "dosage": "3 sets of 15 reps per side",
               "rationale": "Stronger hip external rotators make it easier to keep the knee and foot from turning inward."},
        "zh": {"name": "彈力帶蚌殼式", "target_muscle": "臀中肌與髖外旋肌群",
               "frequency": "每週 3–4 天", "dosage": "每側 3 組 × 15 下",
               "rationale": "髖外旋肌群更有力，比較容易讓膝蓋和腳不往內轉。"},
    },
    "band_side_step": {
        "en": {"name": "Banded Side Steps", "target_muscle": "Gluteus Medius",
               "frequency": "3 days / week", "dosage": "3 sets of 10 steps each direction",
               "rationale": "Side-stepping against a band trains the hip muscles that keep the leg from rolling inward while walking."},
        "zh": {"name": "彈力帶側向走", "target_muscle": "臀中肌",
               "frequency": "每週 3 天", "dosage": "左右各 3 組 × 10 步",
               "rationale": "對抗彈力帶側走，訓練走路時讓腿不往內轉的髖部肌群。"},
    },
    "side_lying_abduction": {
        "en": {"name": "Side-Lying Hip Abduction", "target_muscle": "Gluteus Medius & Minimus",
               "frequency": "3 days / week", "dosage": "3 sets of 12 reps per side",
               "rationale": "Hip abductor strength supports a steadier leg and foot direction during each step."},
        "zh": {"name": "側躺抬腿", "target_muscle": "臀中肌與臀小肌",
               "frequency": "每週 3 天", "dosage": "每側 3 組 × 12 下",
               "rationale": "髖外展肌群更有力，每一步的腿與腳方向更穩定。"},
    },
    "short_foot": {
        "en": {"name": "Short Foot Exercise", "target_muscle": "Intrinsic Foot Muscles",
               "frequency": "Daily", "dosage": "3 sets of 10 holds (5 seconds each)",
               "rationale": "Activating the arch helps the foot stay stable instead of rolling and turning outward."},
        "zh": {"name": "短足運動", "target_muscle": "足部內在肌",
               "frequency": "每天", "dosage": "3 組 × 10 次（每次維持 5 秒）",
               "rationale": "啟動足弓有助於腳掌穩定，減少向外翻轉。"},
    },
    "step_down": {
        "en": {"name": "Step-Down with Knee Tracking", "target_muscle": "Quadriceps & Gluteus Maximus",
               "frequency": "3 days / week", "dosage": "3 sets of 12 reps per leg",
               "rationale": "Controlling knee direction on a step makes it easier to keep the foot pointing forward."},
        "zh": {"name": "下階梯膝蓋對位", "target_muscle": "股四頭肌與臀大肌",
               "frequency": "每週 3 天", "dosage": "每腳 3 組 × 12 下",
               "rationale": "下階梯時控制膝蓋方向，比較容易讓腳尖朝前。"},
    },
    "straight_line_walk": {
        "en": {"name": "Straight-Line Walking Drill", "target_muscle": "Foot & Ankle Stabilizers",
               "frequency": "Daily, 5 minutes", "dosage": "3 sets of 20 steps along a floor line",
               "rationale": "Practising foot direction without the sensor builds the habit the cues are fading toward."},
        "zh": {"name": "直線行走練習", "target_muscle": "足踝穩定肌群",
               "frequency": "每天 5 分鐘", "dosage": "沿地板直線走 3 組 × 20 步",
               "rationale": "不靠感測器練習腳的方向，建立提示逐漸減少後仍能維持的習慣。"},
    },
    "single_leg_balance": {
        "en": {"name": "Single-Leg Balance", "target_muscle": "Ankle Stabilizers & Peroneals",
               "frequency": "3 days / week", "dosage": "3 sets of 30 seconds per leg",
               "rationale": "Better balance on one leg supports a steadier foot position during each step."},
        "zh": {"name": "單腳站立平衡", "target_muscle": "足踝穩定肌群與腓骨肌",
               "frequency": "每週 3 天", "dosage": "每腳 3 組 × 30 秒",
               "rationale": "單腳平衡更好，每一步的腳掌位置會更穩定。"},
    },
}

# 各訓練方向可選的動作；前 DEFAULT_PICK 項是規則式備援的固定組合
GOAL_EXERCISES: Dict[int, List[str]] = {
    1: ["clamshell", "straight_line_walk", "single_leg_balance", "band_side_step", "side_lying_abduction"],
    -1: ["short_foot", "step_down", "straight_line_walk", "single_leg_balance"],
    0: ["straight_line_walk", "single_leg_balance", "clamshell"],
}
DEFAULT_PICK = {1: 3, -1: 2, 0: 2}


def exercise_items(ids: List[str], goal: int, lang: str) -> List[ExerciseItem]:
    """把 LLM 挑選的動作 id 換成動作庫內容：只接受該方向清單內的 id、去除重複，不足 2 項時以預設組合補齊。"""
    allowed = GOAL_EXERCISES[goal]
    picked: List[str] = []
    for i in ids:
        if i in allowed and i not in picked:
            picked.append(i)
    for i in allowed:
        if len(picked) >= 2:
            break
        if i not in picked:
            picked.append(i)
    return [ExerciseItem(**EXERCISE_LIBRARY[i][lang]) for i in picked[:3]]


def _llm_output_schema(goal: int) -> Dict[str, Any]:
    """交給 Ollama structured outputs 的 JSON schema：動作以 enum 限制在該訓練方向的清單內。"""
    return {
        "type": "object",
        "properties": {
            "primary_focus": {"type": "string"},
            "exercise_ids": {"type": "array", "items": {"type": "string", "enum": GOAL_EXERCISES[goal]},
                             "minItems": 2, "maxItems": 3},
            "faded_feedback_guidance": {"type": "string"},
            "weekly_summary": {"type": "string"},
        },
        "required": ["primary_focus", "exercise_ids", "faded_feedback_guidance", "weekly_summary"],
    }


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
    # 程式端判讀（LLM 與規則式備援共用）
    # ------------------------------------------------------------------

    @staticmethod
    def _context(
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
    ) -> Dict[str, Any]:
        lang = "zh" if lang == "zh" else "en"
        relation = classify_fpa(mean_fpa, baseline_fpa, baseline_fpa_sd, goal_direction)
        goal = resolve_goal(goal_direction, relation)
        return dict(
            week_number=week_number, phase=phase, mean_fpa=mean_fpa, mean_deviation=mean_deviation,
            daily_cue_budget=daily_cue_budget, history=history or [], baseline_fpa=baseline_fpa,
            lang=lang, relation=relation, goal=goal, trend=trend_key(improvement_rate),
            multi_week=multi_week_key(history, mean_fpa, goal, baseline_fpa_sd),
        )

    @staticmethod
    def _analysis(c: Dict[str, Any]) -> str:
        """數據解讀一律由程式寫，數字與判讀永遠一致。"""
        t = _TEMPLATES[c["lang"]]
        fpa = f"{c['mean_fpa']:+.1f}°"
        text = t["analysis_head"].format(fpa=fpa)
        if c["baseline_fpa"] is not None:
            text += t["analysis_best"].format(best=f"{c['baseline_fpa']:+.1f}°", relation=t["relation"][c["relation"]],
                                              dev=c["mean_deviation"])
        text += t["analysis_trend"].format(trend=t["trend"][c["trend"]])
        if c["multi_week"]:
            first = c["history"][0]
            text += t["analysis_history"].format(first_week=first.week, first_fpa=f"{first.mean_fpa:+.1f}°", fpa=fpa,
                                                 direction=t["multi_week"][c["multi_week"]])
        return text

    def _code_fields(self, c: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            week_number=c["week_number"],
            current_phase=c["phase"],
            phase_title=PHASE_TITLES[c["lang"]].get(c["phase"], f"Phase {c['phase']}"),
            daily_cue_budget=c["daily_cue_budget"],
            movement_analysis=self._analysis(c),
            language=c["lang"],
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt(self, *args, **kwargs) -> str:
        return self._prompt(self._context(*args, **kwargs))

    @staticmethod
    def _prompt(c: Dict[str, Any]) -> str:
        """prompt 只含已判讀的文字，不含任何原始數字（小模型會誤讀數字）。"""
        language_line = ("所有文字欄位一律使用繁體中文（台灣用語），不要使用英文或簡體中文。"
                         if c["lang"] == "zh" else "Write every text field in English.")
        menu = "\n".join(
            f"- {i}: {EXERCISE_LIBRARY[i]['en']['name']} ({EXERCISE_LIBRARY[i]['en']['target_muscle']})"
            for i in GOAL_EXERCISES[c["goal"]]
        )
        multi_week = f"- Multi-week picture: {MULTI_WEEK_TEXT[c['multi_week']]}\n" if c["multi_week"] else ""
        goal = GOAL_TEXT[c["goal"]] + (f"（中文：{GOAL_TEXT_ZH[c['goal']]}）" if c["lang"] == "zh" else "")
        return f"""You are the weekly coach inside OwnStride, a general-wellness walking-form training aid.
You are not a medical device. Do not diagnose, do not mention diseases, patients, treatment or rehabilitation.
Write in plain, encouraging language for an everyday user. {language_line}

OwnStride compares the user only with their own best days, never with other people.
The app already shows the user every measured number next to your text, so do not write any numbers,
angles, percentages or scores yourself. Describe things in words only.

This week, already interpreted from a foot-worn motion sensor:
- Training goal chosen by the user: {goal}
- Foot direction this week: {RELATION_TEXT[c["relation"]]}
- Weekly trend: {TREND_TEXT[c["trend"]]}
{multi_week}- Feedback phase: {PHASE_TITLES['en'].get(c["phase"], '')}. The vibration cues fade as the user improves, so they rely more on their own sense of foot position.

Exercise menu for this goal (choose 2 or 3 ids that fit this week):
{menu}

Return a JSON object with:
- primary_focus: one sentence on what to focus on this week
- exercise_ids: 2 or 3 ids from the menu above
- faded_feedback_guidance: one or two sentences on how to treat the vibration cues in this phase
- weekly_summary: two encouraging sentences that match the trend above
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

    def _request(self, c: Dict[str, Any], stream: bool) -> urllib.request.Request:
        payload = {
            "model": self.model,
            "prompt": self._prompt(c),
            "format": _llm_output_schema(c["goal"]),
            "stream": stream,
            "keep_alive": KEEP_ALIVE,
            "options": {"temperature": 0.3},
        }
        return urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def _finalize(self, raw_text: str, final_chunk: Dict[str, Any], wall_sec: float, c: Dict[str, Any]) -> WeeklyTrainingPlan:
        parsed = json.loads(to_traditional(raw_text) if c["lang"] == "zh" else raw_text)
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
            **self._code_fields(c),
            **{k: parsed[k] for k in _LLM_TEXT_FIELDS},
            exercises=exercise_items(parsed.get("exercise_ids", []), c["goal"], c["lang"]),
            is_generated_by_ollama=True,
            model_name=self.model,
            generation_stats=stats,
            llm_fields=LLM_FIELDS,
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
        c = self._context(week_number, phase, mean_fpa, mean_deviation, improvement_rate, daily_cue_budget, history,
                          baseline_fpa, baseline_fpa_sd, goal_direction, lang)
        if not use_llm:
            return self._generate_deterministic_plan(c, reason="LLM not requested (startup default)")
        reason = self.check_available()
        if reason is None:
            try:
                t0 = time.time()
                with urllib.request.urlopen(self._request(c, stream=False), timeout=self.timeout_sec) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return self._finalize(body.get("response", ""), body, time.time() - t0, c)
            except (OSError, json.JSONDecodeError, ValidationError, KeyError) as e:
                reason = f"LLM call failed: {type(e).__name__}: {e}"
        logger.info("Falling back to rule-based planner: %s", reason)
        return self._generate_deterministic_plan(c, reason=reason)

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
        c = self._context(week_number, phase, mean_fpa, mean_deviation, improvement_rate, daily_cue_budget, history,
                          baseline_fpa, baseline_fpa_sd, goal_direction, lang)
        reason = self.check_available()
        if reason is None:
            raw, final_chunk = [], {}
            t0 = time.time()
            ttft = None
            try:
                with urllib.request.urlopen(self._request(c, stream=True), timeout=self.timeout_sec) as resp:
                    for line in resp:
                        if not line.strip():
                            continue
                        chunk = json.loads(line.decode("utf-8"))
                        piece = chunk.get("response", "")
                        if piece:
                            if ttft is None:
                                ttft = round(time.time() - t0, 3)
                            raw.append(piece)
                            yield {"type": "token", "text": to_traditional(piece) if c["lang"] == "zh" else piece}
                        if chunk.get("done"):
                            final_chunk = chunk
                            break
                final_chunk["_ttft_sec"] = ttft
                plan = self._finalize("".join(raw), final_chunk, time.time() - t0, c)
                yield {"type": "done", "plan": plan.model_dump()}
                return
            except (OSError, json.JSONDecodeError, ValidationError, KeyError) as e:
                reason = f"LLM call failed: {type(e).__name__}: {e}"
        plan = self._generate_deterministic_plan(c, reason=reason)
        yield {"type": "fallback", "reason": reason, "plan": plan.model_dump()}

    # ------------------------------------------------------------------
    # 規則式備援（無 LLM 時）
    # ------------------------------------------------------------------

    def _generate_deterministic_plan(self, c: Dict[str, Any], reason: Optional[str] = None) -> WeeklyTrainingPlan:
        """規則式範本：動作組合依訓練方向挑選，描述依「相對個人最佳」的關係挑選；只有數字會隨輸入變動。"""
        t = _TEMPLATES[c["lang"]]
        fields = self._code_fields(c)
        fmt = dict(fpa=f"{c['mean_fpa']:+.1f}°",
                   best=f"{c['baseline_fpa']:+.1f}°" if c["baseline_fpa"] is not None else "—",
                   budget=c["daily_cue_budget"], week=c["week_number"], phase_title=fields["phase_title"])
        return WeeklyTrainingPlan(
            **fields,
            faded_feedback_guidance=t["guide"][min(max(c["phase"], 1), 3)].format(**fmt),
            primary_focus=t["focus"][c["relation"]].format(**fmt),
            exercises=exercise_items(GOAL_EXERCISES[c["goal"]][:DEFAULT_PICK[c["goal"]]], c["goal"], c["lang"]),
            weekly_summary=t["summary"].format(**fmt),
            is_generated_by_ollama=False,
            model_name=None,
            fallback_reason=reason,
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
        "analysis_head": "Your average foot angle this week was {fpa}",
        "analysis_best": "; your personal best is {best}, so you are {relation}. "
                         "Average shortfall from your best: {dev:.2f} (0 means at or beyond your best)",
        "analysis_trend": ". Weekly trend: {trend}.",
        "analysis_history": " Since week {first_week} your weekly average has moved from {first_fpa} to {fpa} ({direction}).",
        "relation": {
            "worse": "below your personal best", "at_best": "at your personal best (within your usual spread)",
            "better": "beyond your personal best", "within": "within your usual range",
            "outside": "outside your usual range", "toe_in": "turned in", "toe_out": "turned out",
        },
        "trend": {"improving": "improving", "slipping": "slipping compared with last week", "steady": "steady"},
        "multi_week": {"toward": "toward your goal", "away": "away from your goal", "flat": "about the same"},
        "summary": "Week {week}: {phase_title} with {budget} cues per day. Keep practising the drills between walks.",
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
        "analysis_head": "本週平均足偏角 {fpa}",
        "analysis_best": "，你的個人最佳是 {best}，目前{relation}。和最佳狀態相比平均落後 {dev:.2f}（0 代表達到或超越最佳）",
        "analysis_trend": "。每週趨勢：{trend}。",
        "analysis_history": "從第 {first_week} 週到現在，每週平均由 {first_fpa} 變為 {fpa}（{direction}）。",
        "relation": {
            "worse": "低於個人最佳", "at_best": "在個人最佳範圍內", "better": "超越個人最佳",
            "within": "在平常範圍內", "outside": "超出平常範圍", "toe_in": "偏內", "toe_out": "偏外",
        },
        "trend": {"improving": "進步中", "slipping": "比上週退步", "steady": "持平"},
        "multi_week": {"toward": "朝訓練目標前進", "away": "偏離訓練目標", "flat": "大致持平"},
        "summary": "第 {week} 週：{phase_title}，每天 {budget} 次提示。走路之外的時間，記得持續做下面的練習。",
    },
}
