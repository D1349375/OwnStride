"""
tests/test_llm_planner.py
=========================
驗證每週計畫生成：規則式備援、相對個人基線的方向判斷、Phase 1 措辭、程式與 LLM 的分工
（數字由程式寫、LLM 只寫敘述並從動作庫挑選），以及以本機假 Ollama 伺服器驗證串流／非串流解析與 tok/s 統計。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.training_planner import (GOAL_EXERCISES, LLMTrainingPlanner, WeekSummary, WeeklyTrainingPlan,
                                      _llm_output_schema, classify_fpa, exercise_items)

UNREACHABLE = "http://127.0.0.1:9"
BANNED = ("diagnos", "clinical", "rehabilitat", "patient", "treatment", "prescription")

# 假 LLM 的輸出：只有敘述欄位與動作 id（其中一個不在清單內，應被忽略）
FAKE_PLAN = {
    "primary_focus": "point toes forward", "exercise_ids": ["band_side_step", "not_in_menu"],
    "faded_feedback_guidance": "fewer cues", "weekly_summary": "good week",
}


class _FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:1.5b"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = json.dumps(FAKE_PLAN)
        self.send_response(200)
        self.end_headers()
        final = {"done": True, "eval_count": 40, "eval_duration": 2_000_000_000, "prompt_eval_count": 300}
        if req["stream"]:
            for i in range(0, len(text), 20):
                self.wfile.write((json.dumps({"response": text[i:i + 20], "done": False}) + "\n").encode())
            self.wfile.write((json.dumps(final) + "\n").encode())
        else:
            self.wfile.write(json.dumps({"response": text, **final}).encode())


@pytest.fixture
def fake_ollama():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


ARGS = dict(week_number=3, phase=2, mean_fpa=-5.2, mean_deviation=2.35, improvement_rate=16.8, daily_cue_budget=20)


def test_fallback_when_ollama_unreachable_and_reason_reported():
    plan = LLMTrainingPlanner(ollama_url=UNREACHABLE).generate_plan(**ARGS)
    assert isinstance(plan, WeeklyTrainingPlan)
    assert plan.is_generated_by_ollama is False
    assert "not reachable" in plan.fallback_reason
    assert plan.week_number == 3 and plan.current_phase == 2 and plan.daily_cue_budget == 20
    assert len(plan.exercises) >= 2


def test_relation_is_relative_to_personal_best_and_goal():
    # 目標：減少內八（FPA 越大越好），個人最佳 -9°
    assert classify_fpa(-8.0, -9.0, 1.5, goal_direction=1) == "at_best"
    assert classify_fpa(-14.0, -9.0, 1.5, goal_direction=1) == "worse"
    assert classify_fpa(-3.0, -9.0, 1.5, goal_direction=1) == "better"
    # 目標：減少外八，方向相反
    assert classify_fpa(8.0, 14.0, 1.5, goal_direction=-1) == "better"
    # 維持：只看是否在範圍內
    assert classify_fpa(5.5, 6.9, 1.6, goal_direction=0) == "within"
    assert classify_fpa(-3.0, 6.9, 1.6, goal_direction=0) == "outside"

    planner = LLMTrainingPlanner(ollama_url=UNREACHABLE)
    better = planner.generate_plan(**{**ARGS, "mean_fpa": -3.0}, baseline_fpa=-9.0, baseline_fpa_sd=1.5, goal_direction=1)
    assert "New personal best" in better.primary_focus
    assert better.exercises[0].name == "Clamshell with Resistance Band"  # 動作依訓練方向挑選


def test_fallback_plan_in_traditional_chinese():
    plan = LLMTrainingPlanner(ollama_url=UNREACHABLE).generate_plan(
        **ARGS, baseline_fpa=-9.0, baseline_fpa_sd=1.5, goal_direction=1, lang="zh")
    assert plan.language == "zh"
    assert "個人最佳" in plan.primary_focus
    assert plan.exercises[0].name == "彈力帶蚌殼式"


def test_prompt_contains_interpretations_but_no_raw_numbers():
    """1.5B 模型會誤讀數字：prompt 只給判讀後的文字，不給任何測量數值"""
    planner = LLMTrainingPlanner(ollama_url=UNREACHABLE)
    history = [WeekSummary(week=1, mean_fpa=-11.6, mean_deviation=0.37, phase=1, cues_per_day=5.9)]
    prompt = planner.build_prompt(**{**ARGS, "improvement_rate": -12.0}, history=history,
                                  baseline_fpa=-9.0, baseline_fpa_sd=1.5, goal_direction=1)
    assert "slipping" in prompt and "moved toward the training goal" in prompt
    for number in ("-12.0", "-5.2", "2.35", "-9.0", "-11.6", "0.37", "5.9", "20"):
        assert number not in prompt
    assert "繁體中文" in planner.build_prompt(**ARGS, lang="zh")


def test_llm_can_only_pick_exercises_from_the_goal_menu():
    schema = _llm_output_schema(1)
    assert schema["properties"]["exercise_ids"]["items"]["enum"] == GOAL_EXERCISES[1]
    assert set(schema["required"]) == {"primary_focus", "exercise_ids", "faded_feedback_guidance", "weekly_summary"}
    # 清單外或重複的 id 被忽略，不足 2 項時補上預設動作
    items = exercise_items(["short_foot", "clamshell", "clamshell"], goal=1, lang="en")
    assert [e.name for e in items] == ["Clamshell with Resistance Band", "Straight-Line Walking Drill"]


def test_analysis_is_written_by_code_with_consistent_numbers():
    history = [WeekSummary(week=1, mean_fpa=-11.6, mean_deviation=0.37, phase=1, cues_per_day=5.9)]
    plan = LLMTrainingPlanner(ollama_url=UNREACHABLE).generate_plan(
        **{**ARGS, "mean_fpa": 5.4, "improvement_rate": 0.0}, history=history,
        baseline_fpa=5.8, baseline_fpa_sd=1.2, goal_direction=1)
    text = plan.movement_analysis
    assert "+5.4°" in text and "+5.8°" in text and "at your personal best" in text
    assert "steady" in text and "from -11.6° to +5.4° (toward your goal)" in text
    assert "-0.0" not in text


def test_simplified_chinese_output_is_converted_to_traditional():
    from src.llm.training_planner import to_traditional
    assert to_traditional("持续改进脚部训练") == "持續改進腳部訓練"


def test_phase1_wording_in_prompt_and_fallback():
    planner = LLMTrainingPlanner(ollama_url=UNREACHABLE)
    plan = planner.generate_plan(**ARGS)
    text = json.dumps(plan.model_dump()).lower()
    assert not any(word in text for word in BANNED)
    prompt = planner.build_prompt(**ARGS).lower()
    # prompt 只在禁止清單裡提到這些詞
    assert "do not diagnose" in prompt
    assert "clinical" not in prompt and "prescription" not in prompt


def test_non_streaming_llm_path_with_stats(fake_ollama):
    plan = LLMTrainingPlanner(ollama_url=fake_ollama).generate_plan(**ARGS)
    assert plan.is_generated_by_ollama is True
    assert plan.model_name == "qwen2.5:1.5b"
    assert plan.generation_stats["tokens_per_sec"] == 20.0
    # 敘述來自 LLM；週次、Phase、配額、數據解讀與動作內容來自程式
    assert plan.primary_focus == "point toes forward"
    assert plan.week_number == 3 and plan.current_phase == 2 and plan.daily_cue_budget == 20
    assert "-5.2°" in plan.movement_analysis
    assert [e.name for e in plan.exercises] == ["Banded Side Steps", "Clamshell with Resistance Band"]
    assert "movement_analysis" not in plan.llm_fields and "exercises" in plan.llm_fields


def test_streaming_llm_path_emits_tokens_then_done(fake_ollama):
    events = list(LLMTrainingPlanner(ollama_url=fake_ollama).stream_plan(**ARGS))
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) > 3
    assert "".join(e["text"] for e in tokens) == json.dumps(FAKE_PLAN)
    assert events[-1]["type"] == "done"
    assert events[-1]["plan"]["generation_stats"]["output_tokens"] == 40


def test_streaming_falls_back_when_unreachable():
    events = list(LLMTrainingPlanner(ollama_url=UNREACHABLE).stream_plan(**ARGS))
    assert len(events) == 1 and events[0]["type"] == "fallback"
    assert events[0]["plan"]["is_generated_by_ollama"] is False
