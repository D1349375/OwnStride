"""
tests/test_llm_planner.py
=========================
驗證每週計畫生成：規則式備援、相對個人基線的方向判斷、Phase 1 措辭，
以及以本機假 Ollama 伺服器驗證串流／非串流解析與 tok/s 統計（不需要真的安裝 Ollama）。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.training_planner import LLMTrainingPlanner, WeeklyTrainingPlan, classify_fpa

UNREACHABLE = "http://127.0.0.1:9"
BANNED = ("diagnos", "clinical", "rehabilitat", "patient", "treatment", "prescription")

FAKE_PLAN = {
    "week_number": 3, "current_phase": 2, "phase_title": "Phase 2", "faded_feedback_guidance": "fewer cues",
    "movement_analysis": "toes turned in", "primary_focus": "point toes forward", "daily_cue_budget": 20,
    "exercises": [{"name": "Clamshell", "target_muscle": "Glutes", "frequency": "3x/wk", "dosage": "3x15",
                   "rationale": "hip rotation"}],
    "weekly_summary": "good week",
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


def test_prompt_pre_interprets_trend_for_small_model():
    planner = LLMTrainingPlanner(ollama_url=UNREACHABLE)
    prompt = planner.build_prompt(**{**ARGS, "improvement_rate": -12.0})
    assert "slipping" in prompt
    assert "-12.0%" not in prompt
    assert "繁體中文" in planner.build_prompt(**ARGS, lang="zh")


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
    assert plan.primary_focus == "point toes forward"


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
