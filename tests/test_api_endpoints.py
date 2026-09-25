"""
tests/test_api_endpoints.py
===========================
FastAPI 端點整合測試：所有數字必須來自管線計算，並附資料來源標註。
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app
from src.dataset.wisdm_loader import WISDMLoader


@pytest.fixture(scope="module")
def client():
    main.state.planner.base_url = "http://127.0.0.1:9"  # 測試不依賴本機 Ollama
    return TestClient(app)


def test_status_is_computed_and_labelled(client):
    data = client.get("/api/status").json()
    m, p = data["today_metrics"], data["phase_info"]
    assert m["strides"] > 0 and 0.0 <= m["in_range_ratio"] <= 1.0
    assert p["current_phase"] in (1, 2, 3)
    assert p["personal_threshold"] == pytest.approx(main.state.baseline.normal_threshold, abs=1e-3)
    assert data["strategy_packet"]["cue_threshold"] > 0
    assert "synthetic" in data["baseline"]["source"].lower()
    assert data["baseline"]["goal"] == "reduce_in_toeing"
    assert data["phase_info"]["days_in_phase"] >= 0
    assert "baseline" in data["data_provenance"] and "real_data" in data["data_provenance"]
    assert "symmetry_pct" not in m  # 單腳感測量不到對稱性


def test_history_trajectory_matches_daily_log(client):
    data = client.get("/api/history").json()
    weeks = data["trajectory"]
    assert len(weeks) == main.SIM_WEEKS
    week1_days = [d for d in data["daily"] if d["week"] == 1]
    assert weeks[0]["dev"] == pytest.approx(sum(d["mean_deviation"] for d in week1_days) / len(week1_days), abs=1e-2)
    # 合成軌跡從內八開始逐週改善；個人最佳基線只往目標方向上調（棘輪）
    assert weeks[0]["fpa"] < 0 < weeks[-1]["fpa"]
    history = data["baseline_history"]
    assert history[0]["reason"] == "calibration" and len(data["calibration_days"]) == main.CALIBRATION_DAYS
    assert len(history[0]["source_days"]) == main.BEST_K_DAYS
    bests = [b["best_fpa"] for b in history]
    assert bests == sorted(bests) and bests[-1] > bests[0] + 5
    calib = [d["mean_fpa"] for d in data["calibration_days"]]
    assert history[0]["best_fpa"] > sum(calib) / len(calib)  # 最佳日優於建立期平均
    assert set(data["feature_zscores"]) == {"cadence", "stride_length", "stance_ratio", "fpa", "eversion_angle", "impact_magnitude"}

    corridor = data["kinematic_corridor"]
    assert len(corridor["cycle_percentages"]) == 21
    for ch in ("fpa", "eversion"):
        assert len(corridor[ch]["baseline_mean"]) == 21
        assert len(corridor[ch]["current_cycle"]) == 21


def test_walk_adds_strides_and_respects_budget(client):
    before = client.get("/api/status").json()
    res = client.post("/api/walk", json={"n_strides": 15, "gait_type": "in_toeing", "custom_fpa": -14.0})
    assert res.status_code == 200
    data = res.json()
    assert data["data_source"] == "synthetic"
    assert data["strides_processed"] >= 12
    assert data["session_summary"]["fpa_mae_vs_ground_truth_deg"] < 1.0
    after = data["updated_today_metrics"]
    assert after["strides"] == before["today_metrics"]["strides"] + data["strides_processed"]
    assert after["cues_used"] <= after["daily_cue_budget"]
    assert all(d["mahalanobis_score"] > 0 for d in data["strides_detail"])


def test_close_day_steps_fsm_and_starts_new_day(client):
    n_days = len(client.get("/api/history").json()["daily"])
    res = client.post("/api/day/close")
    assert res.status_code == 200
    data = res.json()
    assert data["phase"] in (1, 2, 3)
    status = client.get("/api/status").json()
    assert status["today_metrics"]["strides"] == 0
    assert status["today_metrics"]["cues_used"] == 0
    assert len(client.get("/api/history").json()["daily"]) == n_days + 1
    # 空的一天不能關帳
    assert client.post("/api/day/close").status_code == 400
    client.post("/api/walk", json={"n_strides": 10, "gait_type": "normal"})


def test_plan_language_switch_for_fallback(client):
    zh = client.get("/api/plan?lang=zh").json()
    assert zh["language"] == "zh" and zh["is_generated_by_ollama"] is False
    en = client.get("/api/plan?lang=en").json()
    assert en["language"] == "en"


def test_plan_endpoints_report_provenance(client):
    plan = client.get("/api/plan").json()
    assert plan["is_generated_by_ollama"] is False and plan["fallback_reason"]
    fresh = client.post("/api/plan/generate").json()
    assert fresh["is_generated_by_ollama"] is False
    assert "not reachable" in fresh["fallback_reason"]


def test_plan_stream_fallback_event(client):
    res = client.get("/api/plan/stream")
    assert res.status_code == 200
    events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["type"] == "fallback"
    assert events[-1]["plan"]["week_number"] >= 1


@pytest.mark.skipif(not WISDMLoader().is_downloaded(), reason="WISDM data not downloaded (see README)")
def test_real_benchmark_is_labelled_accelerometer_only(client):
    res = client.get("/api/benchmark/real?subject_id=1&n_samples=500")
    assert res.status_code == 200
    data = res.json()
    assert data["is_synthetic"] is False
    assert data["samples_count"] == 500
    assert "Kwapisz" in data["metadata"]["citation"]
    assert "no gyroscope" in data["limitations"]
    assert len(data["acc_waveform_preview"]["x"]) == 128
