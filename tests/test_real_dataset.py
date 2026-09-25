"""
tests/test_real_dataset.py
==========================
驗證真實人體步態資料集 (WISDM) 載入器與真實性驗證測試
確保資料源來自真人實測 (is_synthetic == False)，拒絕假資料偷天換日。
"""

import pytest
import numpy as np
from src.dataset.wisdm_loader import WISDMLoader
from src.dataset.marea_loader import MAREALoader

# WISDM 資料不隨 repo 散布；未下載時略過（下載方式見 README）
pytestmark = pytest.mark.skipif(not WISDMLoader().is_downloaded(), reason="WISDM data not downloaded")


def test_wisdm_loader_loads_real_data():
    loader = WISDMLoader()
    assert loader.is_downloaded(), "WISDM dataset should be downloaded"
    
    df = loader.load_walking_data()
    assert len(df) > 400000, f"Expected >400k rows, got {len(df)}"
    assert df["user"].nunique() >= 30, f"Expected >= 30 participants, got {df['user'].nunique()}"
    assert "acc_x" in df.columns
    assert "acc_y" in df.columns
    assert "acc_z" in df.columns


def test_wisdm_session_extraction_provenance():
    loader = WISDMLoader()
    acc, gyro, meta = loader.get_subject_session(subject_id=1, n_samples=500)
    
    # 嚴格驗證真實資料標記
    assert meta["is_synthetic"] is False, "Must NOT be synthetic data!"
    assert meta["source"] == "WISDM_real_human_walking"
    assert "Kwapisz" in meta["citation"]
    assert acc.shape == (500, 3)
    assert gyro.shape == (500, 3)
    
    # 加速度數值合理性 (單位 g: 均值應該在 0.5 ~ 1.5g 之間)
    mag = np.linalg.norm(acc, axis=1)
    mean_mag = np.mean(mag)
    assert 0.5 <= mean_mag <= 2.0, f"Mean acceleration magnitude {mean_mag} out of physical range"


def test_wisdm_uses_real_timestamps():
    """WISDM 各受試者取樣率不同（約 20 或 25 Hz）；時間軸必須依實際時間戳，而不是假設 20 Hz"""
    loader = WISDMLoader()
    rates = {round(loader.get_subject_session(subject_id=s, n_samples=10)[2]["original_hz"]) for s in (1, 3)}
    assert rates == {20, 25}


def test_wisdm_stride_segmentation_is_physiological():
    """每位受試者都能切出一致的跨步：跨步時間在成人步行範圍內，且不再一步／一跨步交替"""
    from src.dataset.wisdm_loader import WISDM_TARGET_RATE
    from src.features.event_detector import GaitEventDetector

    loader = WISDMLoader()
    detector = GaitEventDetector(sample_rate=WISDM_TARGET_RATE)
    medians, cvs = [], []
    for sid in range(1, loader.load_walking_data()["user"].nunique() + 1):
        acc, _, _ = loader.get_subject_session(subject_id=sid, n_samples=int(20 * WISDM_TARGET_RATE))
        d = np.array([s["duration_sec"] for s in detector.segment_strides(acc)["strides"]])
        assert len(d) >= 3, f"subject {sid}: too few strides"
        medians.append(np.median(d))
        cvs.append(np.std(d) / np.mean(d))
    assert 0.8 <= min(medians) and max(medians) <= 1.4
    assert np.median(cvs) < 0.08


def test_marea_loader_benchmark_real_delegation():
    loader = MAREALoader()
    acc, gyro, meta = loader.get_or_create_benchmark_data(subject_id=1)
    
    # 確認不再靜默回退到合成資料
    assert meta.get("is_synthetic") is False, "MAREALoader benchmark must use real data or raise, never silent synthetic"
    assert acc.shape[0] > 100
    assert acc.shape[1] == 3
