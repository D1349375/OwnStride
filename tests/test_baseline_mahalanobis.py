"""
tests/test_baseline_mahalanobis.py
==================================
驗證個人基線模組 (Ledoit-Wolf + Mahalanobis Distance)
"""

import numpy as np

from src.baseline.personal_baseline import PersonalGaitBaseline
from src.dataset.synthetic_stream import SyntheticGaitStream
from src.features.gait_metrics import GaitMetricsExtractor


def _features(seed, n, **kwargs):
    acc, gyro, _ = SyntheticGaitStream(seed=seed).generate_walk_session(n_strides=n, **kwargs)
    extractor = GaitMetricsExtractor()
    return extractor.extract(acc, gyro), extractor.last_strides


def test_baseline_fit_and_small_sample_stability():
    features, _ = _features(789, 25, gait_type="normal")
    baseline = PersonalGaitBaseline(normal_percentile=95.0).fit(features)

    assert baseline.is_calibrated
    assert baseline.precision_.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(baseline.precision_) > 0)


def test_aberrant_gait_deviation_separation():
    base, _ = _features(999, 40, gait_type="normal")
    baseline = PersonalGaitBaseline().fit(base)

    res_norm = baseline.compute_session_deviations(_features(1000, 15, gait_type="normal")[0])
    res_aberrant = baseline.compute_session_deviations(_features(1001, 15, gait_type="in_toeing", custom_fpa=-14.0)[0])

    assert res_aberrant["mean_deviation"] > res_norm["mean_deviation"] * 1.8
    assert res_aberrant["aberrant_ratio"] > 0.6


def test_feature_zscores_point_to_the_deviating_dimension():
    base, _ = _features(21, 60, gait_type="normal")
    baseline = PersonalGaitBaseline().fit(base)
    z = baseline.feature_zscores(_features(22, 20, gait_type="normal", custom_fpa=-8.0)[0])
    assert z["fpa"] < -3
    assert max(abs(v) for k, v in z.items() if k != "fpa") < abs(z["fpa"])


def test_corridor_band_from_baseline_waveforms():
    base, strides = _features(31, 40, gait_type="normal")
    baseline = PersonalGaitBaseline().fit(base)
    baseline.fit_corridor({"fpa": [s["fpa_waveform"] for s in strides]})
    band = baseline.corridor_["fpa"]
    assert band["n_strides"] == len(strides)
    assert np.all(band["upper"] >= band["mean"]) and np.all(band["lower"] <= band["mean"])
    # 站立期（0%）的走廊中心應接近基線 FPA
    assert abs(band["mean"][0] - np.mean([f.fpa for f in base])) < 0.5


# ----------------------------------------------------------------------
# 個人最佳日基線（2026-09-24 定案）
# ----------------------------------------------------------------------

def _day(day_index, fpa, seed, n=30):
    features, strides = _features(seed, n, gait_type="in_toeing", custom_fpa=fpa)
    return {"day_index": day_index, "features": features, "strides": strides}


def _in_toe_days():
    return [_day(i, fpa, 100 + i) for i, fpa in enumerate([-15.0, -12.0, -14.0, -10.0, -13.0, -11.0, -16.0])]


def test_best_days_are_selected_in_goal_direction():
    days = _in_toe_days()
    reduce_in_toe = PersonalGaitBaseline(goal_direction=1).fit_best_days(days, best_k=3)
    assert reduce_in_toe.source_days == [1, 3, 5]  # -12, -10, -11：最不內八的三天
    assert reduce_in_toe.goal == "reduce_in_toeing"
    assert "fpa" in reduce_in_toe.corridor_

    maintain = PersonalGaitBaseline(goal_direction=0).fit_best_days(days, best_k=3)
    assert len(maintain.source_days) == 3


def test_improvement_beyond_best_is_not_a_deviation():
    baseline = PersonalGaitBaseline(goal_direction=1).fit_best_days(_in_toe_days(), best_k=3)
    two_sided = PersonalGaitBaseline(goal_direction=0)
    two_sided.fit_numpy(np.array([f.to_numpy() for d in _in_toe_days() if d["day_index"] in (1, 3, 5) for f in d["features"]]))
    better_feats = _features(300, 20, gait_type="in_toeing", custom_fpa=-2.0)[0]
    better = baseline.compute_session_deviations(better_feats)
    better_two_sided = two_sided.compute_session_deviations(better_feats)
    worse = baseline.compute_session_deviations(_features(301, 20, gait_type="in_toeing", custom_fpa=-20.0)[0])
    assert better["mean_deviation"] < baseline.normal_threshold
    assert better_two_sided["mean_deviation"] > 2 * better["mean_deviation"]
    assert worse["mean_deviation"] > 1.5 * baseline.normal_threshold


def test_ratchet_only_moves_toward_the_goal():
    baseline = PersonalGaitBaseline(goal_direction=1).fit_best_days(_in_toe_days(), best_k=3)
    start = baseline.mean_[baseline.goal_index]
    worse_week = [_day(10 + i, -17.0, 400 + i) for i in range(5)]
    assert baseline.maybe_ratchet(worse_week, best_k=3) is False
    assert baseline.mean_[baseline.goal_index] == start

    better_week = [_day(20 + i, -5.0, 500 + i) for i in range(5)]
    assert baseline.maybe_ratchet(better_week, best_k=3) is True
    assert baseline.mean_[baseline.goal_index] > start + 3
    assert baseline.source_days[0] >= 20
