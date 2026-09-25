"""
run_demo.py
===========
OwnStride 端到端展示啟動入口腳本 (Main Entry Point)
支援：
1. Web 互動模式 (預設)：啟動 FastAPI 服務並提供 RWD 響應式儀表板 (http://127.0.0.1:8000)
2. CLI 終端模式 (--cli)：在命令列執行端到端特徵萃取、基線比對、漸退狀態機與每週計畫生成
"""

import sys
import io
import argparse
import uvicorn
import numpy as np

# 確保 Windows 主控台正確輸出 UTF-8 字元與表情符號
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_cli_demo():
    print("=" * 70)
    print("OwnStride: On-Device AI Walking-Form Trainer (CLI demonstration)")
    print("資料來源：合成足部 IMU（src/dataset/synthetic_stream.py），所有數字由管線實際計算")
    print("=" * 70)

    from src.dataset.synthetic_stream import SyntheticGaitStream
    from src.features.gait_metrics import GaitMetricsExtractor
    from src.baseline.personal_baseline import PersonalGaitBaseline
    from src.controller.faded_fsm import FadedFeedbackFSM
    from src.llm.training_planner import LLMTrainingPlanner

    extractor = GaitMetricsExtractor()

    # 1. 個人最佳日基線：14 天建立期（內八使用者照平常走），取「最不內八」的 5 天
    print(">>> [1] 建立期 14 天 → 依訓練方向（減少內八）取最佳 5 天擬合個人基線")
    rng = np.random.default_rng(2026)
    stream = SyntheticGaitStream(seed=2026)
    days = []
    for d in range(14):
        acc, gyro, _ = stream.generate_walk_session(n_strides=30, gait_type="in_toeing",
                                                    custom_fpa=-13.0 + rng.normal(0, 1.5), fpa_sd=3.5)
        days.append({"day_index": d, "features": extractor.extract(acc, gyro), "strides": extractor.last_strides})
    baseline = PersonalGaitBaseline(goal_direction=1).fit_best_days(days, best_k=5)
    b = baseline.get_state()
    base_fpa = float(baseline.mean_[baseline.goal_index])
    calib_fpa = float(np.mean([f.fpa for d in days for f in d["features"]]))
    print(f"    最佳日 {[d + 1 for d in b.source_days]}，個人最佳 FPA {base_fpa:+.1f}°（建立期平均 {calib_fpa:+.1f}°），個人閾值 τ = {b.normal_threshold:.2f}\n")

    # 2. 一段更內八的行走：FPA 由姿態積分算出，並與合成真值比較
    print(">>> [2] 分析一段比個人最佳更內八的行走（演算法未讀取合成參數，只看 IMU 訊號）")
    acc, gyro, truth = SyntheticGaitStream(seed=7).generate_walk_session(n_strides=20, gait_type="in_toeing", custom_fpa=-18.0)
    feats = extractor.extract(acc, gyro)
    fpa_err = np.abs(np.array([f.fpa for f in feats]) - truth["target_fpa"].values[:len(feats)])
    summary = baseline.compute_session_deviations(feats)
    print(f"    切出 {len(feats)} 步，平均 FPA {np.mean([f.fpa for f in feats]):+.1f}°（與合成真值平均誤差 {fpa_err.mean():.2f}°）")
    print(f"    平均落後 {summary['mean_deviation']:.2f}（τ = {b.normal_threshold:.2f}），落後超過 τ 的步數 {summary['aberrant_ratio'] * 100:.0f}%\n")

    # 3. 三週合成軌跡：每日推進漸退狀態機，每週檢查個人最佳（棘輪）
    print(">>> [3] 三週合成軌跡 → 漸退狀態機（水準 + 進步速率 + 進步閘門）＋ 每週棘輪")
    fsm = FadedFeedbackFSM(personal_threshold=b.normal_threshold, require_progress=True)
    recent = list(days)
    packet = None
    for d in SyntheticGaitStream(seed=8).generate_longitudinal_recovery_stream(weeks=3, strides_per_day=30):
        day_feats = extractor.extract(d["acc"], d["gyro"])
        recent = (recent + [{"day_index": 100 + d["day_index"], "features": day_feats, "strides": extractor.last_strides}])[-14:]
        dev = float(np.mean([baseline.compute_deviation_score(f) for f in day_feats]))
        phase, changed, packet = fsm.step(dev)
        note = "  ← 轉換" if changed else ""
        if d["day_of_week"] == 7 and baseline.maybe_ratchet(recent, best_k=5):
            fsm.personal_threshold = baseline.normal_threshold
            fsm.record_progress()
            note += f"  ← 個人最佳上調至 {baseline.mean_[baseline.goal_index]:+.1f}°"
        print(f"    Day {d['day_index'] + 1:2d}  FPA {np.mean([f.fpa for f in day_feats]):+6.1f}°  落後 {dev:5.2f}  "
              f"速率 {packet.progress_rate * 100:+6.1f}%/週  Phase {int(phase)}  配額 {packet.daily_cue_budget}{note}")
    print()

    # 4. 每週計畫
    print(">>> [4] 每週訓練計畫（本機 Ollama；不可用時為規則式備援）")
    plan = LLMTrainingPlanner().generate_plan(
        week_number=3, phase=packet.phase, mean_fpa=float(np.mean([f.fpa for f in day_feats])),
        mean_deviation=dev, improvement_rate=packet.progress_rate * 100,
        daily_cue_budget=packet.daily_cue_budget, baseline_fpa=float(baseline.mean_[baseline.goal_index]),
        baseline_fpa_sd=baseline.feature_sd("fpa"), goal_direction=baseline.goal_direction,
    )
    source = f"本機 LLM {plan.model_name}" if plan.is_generated_by_ollama else f"規則式備援（{plan.fallback_reason}）"
    print(f"    來源：{source}")
    print(f"    重點：{plan.primary_focus}")
    for idx, ex in enumerate(plan.exercises, 1):
        print(f"      [{idx}] {ex.name} ({ex.target_muscle}) - {ex.frequency}, {ex.dosage}")
    print(f"    摘要：{plan.weekly_summary}")
    print("=" * 70)


def run_web_demo():
    print("=" * 70)
    print("🚀 啟動 OwnStride 評估層與 RWD 儀表板服務...")
    print("📍 本地伺服器位址: http://127.0.0.1:8000")
    print("📱 支援電腦與手機瀏覽器自適應 (RWD)")
    print("=" * 70)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OwnStride Demo Runner")
    parser.add_argument("--cli", action="store_true", help="以命令列模式執行端到端驗證")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo()
    else:
        run_web_demo()
