"""
tests/test_serial_bridge.py
===========================
驗證 ESP32 序列埠橋接：韌體 JSON 解析、安裝方向重新對應、掉包偵測、CSV 離線分析。
（以合成資料模擬韌體輸出；尚未以實體裝置驗證。）
"""

import csv
import json

import numpy as np
import pytest

from src.dataset.synthetic_stream import SyntheticGaitStream
from src.hardware.serial_bridge import FIELDS, analyze_samples, load_csv, parse_axes, parse_line, samples_to_arrays


def _firmware_samples(acc, gyro):
    return [{"seq": i + 1, "t": i * 10, "ax": a[0], "ay": a[1], "az": a[2], "gx": g[0], "gy": g[1], "gz": g[2]}
            for i, (a, g) in enumerate(zip(acc, gyro))]


def test_parse_line_accepts_data_and_ignores_status():
    line = '{"seq":1024,"t":10240,"ax":0.082,"ay":-0.412,"az":1.050,"gx":2.1,"gy":-48.2,"gz":-11.8}'
    sample = parse_line(line)
    assert sample["seq"] == 1024 and sample["gy"] == -48.2
    assert parse_line('{"status":"ready","node":"esp32_foot_imu"}') is None
    assert parse_line('{"ack":"cue","used":3}') is None
    assert parse_line('{"seq":1,"ax":0.1') is None
    assert parse_line("garbage") is None


def test_axis_remap_and_dropped_packets():
    assert parse_axes("-y,x,z") == ([1, 0, 2], [-1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        parse_axes("x,x,z")
    samples = [{"seq": s, "ax": 1, "ay": 2, "az": 3, "gx": 4, "gy": 5, "gz": 6} for s in (1, 2, 5, 6)]
    acc, gyro, info = samples_to_arrays(samples, "-y,x,z")
    assert acc[0].tolist() == [-2, 1, 3]
    assert info["dropped_packets"] == 2


def test_csv_roundtrip_and_tape_validation(tmp_path):
    acc, gyro, truth = SyntheticGaitStream(seed=4).generate_walk_session(n_strides=20, custom_fpa=-10.0, fpa_sd=0.5)
    path = tmp_path / "trial.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(_firmware_samples(np.round(acc, 3), np.round(gyro, 1)))

    result = analyze_samples(load_csv(str(path)), nominal_fpa=-10.0)
    assert result["strides"] == 20
    assert result["dropped_packets"] == 0
    assert result["mae_deg"] < 1.0


def test_mounting_rotation_is_undone_by_axes_option():
    """感測器繞 z 軸轉 90° 黏貼（x 朝左側）：以 --axes 對應回足部座標後結果不變"""
    acc, gyro, _ = SyntheticGaitStream(seed=6).generate_walk_session(n_strides=15, custom_fpa=12.0)
    # 實體讀值：sensor_x = foot_y, sensor_y = -foot_x
    mounted_acc = np.column_stack([acc[:, 1], -acc[:, 0], acc[:, 2]])
    mounted_gyro = np.column_stack([gyro[:, 1], -gyro[:, 0], gyro[:, 2]])
    result = analyze_samples(_firmware_samples(mounted_acc, mounted_gyro), axes="-y,x,z", nominal_fpa=12.0)
    assert result["strides"] == 15
    assert result["mae_deg"] < 2.5
