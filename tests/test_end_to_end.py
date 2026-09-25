"""
tests/test_end_to_end.py
========================
端到端管線：IMU 訊號 ➔ 姿態積分特徵 ➔ 個人基線 ➔ 多日漸退狀態機 ➔ 策略封包 ➔ 每週計畫
"""

import numpy as np

from src.baseline.personal_baseline import PersonalGaitBaseline
from src.controller.faded_fsm import FadedFeedbackFSM, FeedbackPhase
from src.dataset.synthetic_stream import SyntheticGaitStream
from src.features.gait_metrics import GaitMetricsExtractor
from src.llm.training_planner import LLMTrainingPlanner


def test_full_pipeline_end_to_end():
    extractor = GaitMetricsExtractor()
    acc, gyro, _ = SyntheticGaitStream(seed=2026).generate_walk_session(n_strides=60, gait_type="normal")
    baseline = PersonalGaitBaseline().fit(extractor.extract(acc, gyro))
    fsm = FadedFeedbackFSM(personal_threshold=baseline.normal_threshold)

    # 兩週合成軌跡：內八逐日改善
    sessions = SyntheticGaitStream(seed=7).generate_longitudinal_recovery_stream(
        weeks=2, strides_per_day=25, initial_fpa=-13.0, target_fpa=6.5)
    daily = []
    for d in sessions:
        feats = extractor.extract(d["acc"], d["gyro"])
        dev = float(np.mean([baseline.compute_deviation_score(f) for f in feats]))
        phase, _, packet = fsm.step(dev)
        daily.append((float(np.mean([f.fpa for f in feats])), dev, phase, packet))

    # 第一天明顯內八且偏差遠高於個人閾值，最後偏差下降
    assert daily[0][0] < -8.0
    assert daily[0][1] > 3 * baseline.normal_threshold
    assert daily[-1][1] < daily[0][1]
    assert daily[0][2] == FeedbackPhase.PHASE_1_ACQUISITION
    assert daily[-1][3].progress_rate > 0

    fpa, dev, phase, packet = daily[-1]
    plan = LLMTrainingPlanner(ollama_url="http://127.0.0.1:9").generate_plan(
        week_number=2, phase=int(phase), mean_fpa=fpa, mean_deviation=dev,
        improvement_rate=packet.progress_rate * 100, daily_cue_budget=packet.daily_cue_budget,
        baseline_fpa=6.9, baseline_fpa_sd=1.8,
    )
    assert plan.week_number == 2
    assert plan.is_generated_by_ollama is False and plan.fallback_reason
    assert len(plan.exercises) >= 2
