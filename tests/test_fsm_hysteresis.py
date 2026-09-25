"""
tests/test_fsm_hysteresis.py
============================
驗證漸退狀態機：水準 + 進步速率驅動、遲滯帶、連續天數確認、個人閾值縮放。
"""

from src.controller.faded_fsm import FadedFeedbackFSM, FeedbackPhase

TAU = 2.0


def _run(fsm, values):
    out = None
    for v in values:
        out = fsm.step(v)
    return out


def test_advance_requires_level_and_debounce():
    fsm = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU, debounce_days=2)
    # 持平在 1.1τ（低於 1.2τ 升階線、速率 0）：第一天待確認、第二天轉入 Phase 2
    phase, changed, _ = fsm.step(2.2)
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION and not changed
    phase, changed, packet = fsm.step(2.2)
    assert phase == FeedbackPhase.PHASE_2_FADED and changed
    assert packet.cue_threshold == round(1.2 * TAU, 3)  # 閾值以個人 τ 為單位


def test_no_advance_while_worsening():
    fsm = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU)
    # 水準低於升階線，但一路變差（速率為負）→ 不升階
    phase, _, packet = _run(fsm, [1.4, 1.6, 1.8, 2.0, 2.2, 2.3])
    assert packet.progress_rate < -0.05
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION


def test_faster_progress_relaxes_advancement_threshold():
    """同樣停在 1.45τ：持平的人留在 Phase 1，快速進步中的人已可升階"""
    flat = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU)
    phase_flat, _, _ = _run(flat, [2.9] * 8)
    fast = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU)
    phase_fast, _, packet = _run(fast, [6.0, 5.2, 4.4, 3.7, 3.2, 2.9, 2.9, 2.9])
    assert packet.progress_rate > 0.3
    assert phase_flat == FeedbackPhase.PHASE_1_ACQUISITION
    assert phase_fast == FeedbackPhase.PHASE_2_FADED


def test_hysteresis_deadband_anti_jitter():
    fsm = FadedFeedbackFSM(initial_phase=FeedbackPhase.PHASE_2_FADED, personal_threshold=TAU)
    # 1.5τ 高於升階線 1.2τ、低於降階線 1.9τ：留在 Phase 2
    for _ in range(6):
        phase, changed, _ = fsm.step(3.0)
        assert phase == FeedbackPhase.PHASE_2_FADED and not changed
    # 惡化：近 3 日平均超過 1.9τ 降階線後，連續 2 天確認 → 降回 Phase 1
    fsm.step(4.4)
    fsm.step(4.8)
    phase, changed, packet = fsm.step(5.0)
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION and changed
    assert packet.daily_cue_budget == 50


def test_sharp_regression_rate_triggers_downgrade_before_level():
    fsm = FadedFeedbackFSM(initial_phase=FeedbackPhase.PHASE_3_RETENTION, personal_threshold=TAU)
    # 一週內快速變差：第一次降階發生時，水準仍低於 1.25τ 的水準降階線 → 是速率觸發的
    for value in [1.0, 1.1, 1.3, 1.6, 1.9, 2.1, 2.3, 2.4]:
        phase, changed, packet = fsm.step(value)
        if changed:
            break
    assert changed and phase == FeedbackPhase.PHASE_2_FADED
    assert packet.progress_rate <= -0.2
    assert packet.deviation_level < 1.25 * TAU


def test_budget_fades_with_progress_rate_within_phase():
    fsm = FadedFeedbackFSM(initial_phase=FeedbackPhase.PHASE_2_FADED, personal_threshold=TAU)
    _, _, steady = _run(fsm, [2.8] * 7)
    fsm2 = FadedFeedbackFSM(initial_phase=FeedbackPhase.PHASE_2_FADED, personal_threshold=TAU)
    _, _, improving = _run(fsm2, [3.5, 3.4, 3.3, 3.1, 3.0, 2.9, 2.8])
    assert steady.daily_cue_budget == 20
    assert improving.daily_cue_budget < 20
    assert improving.faded_ratio < steady.faded_ratio


def test_invalid_input_is_ignored():
    fsm = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU)
    phase, changed, _ = fsm.step(float("nan"))
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION and not changed
    assert len(fsm.history) == 0


def test_progress_gate_blocks_fading_without_a_personal_best_raise():
    """維持在個人最佳（偏差很低）但完全沒有進步的人，不應該被漸退"""
    stuck = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU, require_progress=True)
    phase, _, _ = _run(stuck, [0.1] * 14)
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION

    improving = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU, require_progress=True)
    _run(improving, [0.1] * 5)
    improving.record_progress()  # 個人最佳上調
    phase, _, _ = _run(improving, [0.1] * 2)
    assert phase == FeedbackPhase.PHASE_2_FADED
    assert improving.progress_events_in_phase == 0  # 換階段後重新計算


def test_near_zero_shortfall_noise_is_not_read_as_regression():
    """偏差在 0 附近小幅波動（已在個人最佳）不應被算成劇烈退步"""
    fsm = FadedFeedbackFSM(min_days_in_phase=0, personal_threshold=TAU)
    _, _, packet = _run(fsm, [0.0, 0.0, 0.01, 0.0, 0.12, 0.04, 0.02])
    assert packet.progress_rate > -0.1


def test_minimum_days_in_phase_blocks_early_step_up_but_not_step_down():
    fsm = FadedFeedbackFSM(personal_threshold=TAU)  # 預設每階段至少 7 天
    for day in range(1, 7):
        phase, _, _ = fsm.step(1.0)
        assert phase == FeedbackPhase.PHASE_1_ACQUISITION, f"stepped up on day {day}"
    fsm.step(1.0)
    phase, _, _ = fsm.step(1.0)
    assert phase == FeedbackPhase.PHASE_2_FADED
    # 剛升到 Phase 2 就大幅惡化：降階不受最短停留限制
    fsm.step(4.5)
    phase, changed, _ = fsm.step(4.8)
    assert phase == FeedbackPhase.PHASE_1_ACQUISITION and changed
