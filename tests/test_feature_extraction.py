"""
tests/test_feature_extraction.py
================================
驗證足部 IMU 逐步分析：演算法必須從 IMU 訊號（非合成器參數）把已知真值算回來。
這驗證的是演算法的數學正確性；真實硬體精度須另以實測驗證。
"""

import numpy as np
import pytest

from src.dataset.synthetic_stream import SyntheticGaitStream
from src.features.event_detector import GaitEventDetector
from src.features.foot_imu import FootIMUStrideAnalyzer
from src.features.gait_metrics import GaitFeatureVector, GaitMetricsExtractor


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("gait_type", ["normal", "in_toeing", "out_toeing", "over_pronation"])
def test_fpa_and_eversion_recovered_from_imu(side, gait_type):
    acc, gyro, truth = SyntheticGaitStream(seed=11, side=side).generate_walk_session(n_strides=25, gait_type=gait_type)
    strides = FootIMUStrideAnalyzer(side=side).analyze(acc, gyro)

    assert len(strides) == len(truth)
    fpa_err = np.array([s["fpa"] for s in strides]) - truth["target_fpa"].values
    ev_err = np.array([s["eversion"] for s in strides]) - truth["target_eversion"].values
    len_err = np.array([s["stride_length"] for s in strides]) - truth["stride_length"].values

    assert np.abs(fpa_err).mean() < 1.0
    assert np.abs(fpa_err).max() < 2.0
    assert np.abs(ev_err).mean() < 0.5
    assert np.abs(len_err).mean() < 0.03


def test_fpa_sign_convention():
    """負值 = 內八、正值 = 外八，左右腳皆同"""
    for side in ("right", "left"):
        for target in (-15.0, 20.0):
            acc, gyro, _ = SyntheticGaitStream(seed=3, side=side).generate_walk_session(
                n_strides=10, custom_fpa=target)
            fpas = [s["fpa"] for s in FootIMUStrideAnalyzer(side=side).analyze(acc, gyro)]
            assert np.sign(np.mean(fpas)) == np.sign(target)


def test_temporal_features_physiological():
    acc, gyro, truth = SyntheticGaitStream(seed=5).generate_walk_session(n_strides=20)
    strides = FootIMUStrideAnalyzer().analyze(acc, gyro)
    for s in strides:
        assert 0.8 <= s["stride_time"] <= 1.4
        assert 0.5 <= s["stance_ratio"] <= 0.75
        assert 85 <= s["cadence"] <= 150
        assert len(s["fpa_waveform"]) == 21
        assert s["impact_magnitude"] > 1.0
    assert abs(np.mean([s["stance_ratio"] for s in strides]) - truth["stance_ratio"].mean()) < 0.04


def test_accelerometer_only_data_rejected():
    acc, _, _ = SyntheticGaitStream(seed=1).generate_walk_session(n_strides=10)
    with pytest.raises(ValueError, match="gyroscope"):
        FootIMUStrideAnalyzer().analyze(acc, np.zeros_like(acc))


def test_feature_vector_contract():
    acc, gyro, _ = SyntheticGaitStream(seed=456).generate_walk_session(n_strides=20, gait_type="in_toeing")
    extractor = GaitMetricsExtractor()
    features = extractor.extract(acc, gyro)
    assert len(features) == len(extractor.last_strides) >= 15
    for feat in features:
        assert isinstance(feat, GaitFeatureVector)
        assert feat.fpa < 0.0
        arr = feat.to_numpy()
        assert arr.shape == (6,)
        assert not np.isnan(arr).any()
    assert GaitFeatureVector.feature_names() == [
        "cadence", "stride_length", "stance_ratio", "fpa", "eversion_angle", "impact_magnitude"]


def test_accelerometer_only_segmenter_finds_stride_period():
    """僅加速度的切分器（供 WISDM 使用）：自相關找出的跨步週期要接近合成真值，且每個跨步只切一次"""
    acc, gyro, truth = SyntheticGaitStream(sample_rate=128.0, seed=9).generate_walk_session(n_strides=20)
    seg = GaitEventDetector(sample_rate=128.0).segment_strides(acc)
    true_period = float(truth["stride_time"].mean())
    assert abs(seg["stride_period_sec"] - true_period) < 0.1 * true_period
    assert len(seg["strides"]) >= 10
    durations = np.array([s["duration_sec"] for s in seg["strides"]])
    assert np.all(np.abs(durations - true_period) < 0.25 * true_period)
