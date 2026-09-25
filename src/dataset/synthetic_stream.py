"""
src/dataset/synthetic_stream.py
===============================
足部 IMU 合成資料產生器 (Kinematics-First Synthetic Foot IMU Generator)

設計原則（2026-09-24 重寫）：
1. **先定義足部真實運動，再推導感測器讀值**——每一步先給定足部在世界座標中的位置軌跡與姿態
   （yaw / pitch / roll），再以剛體運動學推導 IMU 應讀到的比力（specific force）與角速度。
   因此特徵萃取端（src/features/foot_imu.py）必須用真正的姿態積分才能把 FPA 算回來，
   不存在「合成器把答案編碼進某個軸、萃取器再解碼」的循環驗證。
2. **每一步都附真值（ground truth）**：FPA、外翻角、步長、步時、支撐比例，供演算法驗證。
3. **限制（誠實揭露）**：此模型是理想化剛體足部——站立期感測器位置固定（未模擬腳跟／腳尖滾動
   的平移）、無軟組織晃動、雜訊為高斯白雜訊加常數偏移。它能驗證「演算法數學上正確」，
   **不能**代表真實世界精度；真實精度須以硬體實測驗證。

座標慣例：
- 世界座標：X 前進方向、Y 左、Z 上。
- 感測器／足部座標：x 指向腳尖、y 指向足部左側、z 垂直鞋面向上（感測器對齊足部安裝）。
- 姿態：R = Rz(yaw)·Ry(pitch)·Rx(roll)（scipy intrinsic 'ZYX'）；pitch > 0 為腳尖朝下。
- FPA 正值 = 外八（toe-out）、負值 = 內八（toe-in）；外翻角正值 = 外翻（旋前方向）。
- 輸出單位：加速度 g、角速度 deg/s（與 firmware/esp32_gait_node.ino 串流格式一致）。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

GRAVITY = 9.80665

# 各步態樣態的預設參數（FPA / 外翻角 / 步時 / 支撐比例 / 步長 / 著地衝擊）
GAIT_PRESETS: Dict[str, Dict[str, float]] = {
    "normal":         {"fpa": 7.0,   "eversion": 3.0,  "stride_time": 1.05, "stance": 0.62, "length": 1.30, "impact": 2.0},
    "in_toeing":      {"fpa": -12.0, "eversion": 5.5,  "stride_time": 1.10, "stance": 0.64, "length": 1.18, "impact": 2.5},
    "out_toeing":     {"fpa": 22.0,  "eversion": 2.0,  "stride_time": 1.02, "stance": 0.63, "length": 1.25, "impact": 2.3},
    "over_pronation": {"fpa": 9.0,   "eversion": 12.0, "stride_time": 1.08, "stance": 0.66, "length": 1.22, "impact": 2.8},
}

# 步態週期內的相位節點（占步時比例）
LOADING_END = 0.10       # 腳跟著地 → 足部平貼
HEEL_OFF_BEFORE_TO = 0.22  # 腳跟離地發生在腳尖離地前 22% 步時
PITCH_HEEL_STRIKE = -18.0  # 著地時腳尖上翹（度）
PITCH_TOE_OFF = 45.0       # 離地時腳尖朝下（度）


def _ease(x: np.ndarray) -> np.ndarray:
    """0→1 平滑過渡（餘弦緩動），端點斜率為 0。"""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, 1.0)))


def _min_jerk(x: np.ndarray) -> np.ndarray:
    """最小急動度（minimum-jerk）位移曲線，端點速度與加速度皆為 0。"""
    x = np.clip(x, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


class SyntheticGaitStream:
    """
    足部 IMU 合成器。每次呼叫產生一段連續行走（含開頭站立 1 秒、結尾著地站立 0.5 秒），
    以便 ZUPT（零速度更新）演算法有明確的靜止區段作為積分起訖點。
    """

    def __init__(self, sample_rate: float = 100.0, seed: Optional[int] = None, side: str = "right"):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.fs = float(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.side = side
        # 右腳外八 = 足部相對前進方向順時針（yaw 為負）；左腳相反
        self._yaw_sign = -1.0 if side == "right" else 1.0
        # 右腳外翻 = 外側（右緣）抬高 = roll 為負；左腳相反
        self._roll_sign = -1.0 if side == "right" else 1.0

    # ------------------------------------------------------------------
    # 單步運動學
    # ------------------------------------------------------------------

    def _stride_kinematics(
        self,
        u: np.ndarray,
        stance: float,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        yaw_start: float,
        yaw_end: float,
        roll_start: float,
        roll_end: float,
        swing_height: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        給定步內正規化時間 u ∈ [0, 1)（0 = 腳跟著地），回傳足部位置 (N,3) 與歐拉角 (N,3)[yaw, pitch, roll]（度）。
        """
        n = len(u)
        pos = np.tile(start_pos, (n, 1)).astype(np.float64)
        yaw = np.full(n, yaw_start)
        roll = np.full(n, roll_start)
        pitch = np.zeros(n)

        heel_off = stance - HEEL_OFF_BEFORE_TO

        loading = u < LOADING_END
        pitch[loading] = PITCH_HEEL_STRIKE * (1.0 - _ease(u[loading] / LOADING_END))

        push = (u >= heel_off) & (u < stance)
        pitch[push] = PITCH_TOE_OFF * _ease((u[push] - heel_off) / (stance - heel_off))

        swing = u >= stance
        tau = (u[swing] - stance) / (1.0 - stance)
        pitch[swing] = PITCH_TOE_OFF + (PITCH_HEEL_STRIKE - PITCH_TOE_OFF) * _ease(tau)
        s = _min_jerk(tau)
        pos[swing] = start_pos + np.outer(s, end_pos - start_pos)
        pos[swing, 2] += swing_height * np.sin(np.pi * tau) ** 2
        # 擺動期足部繞垂直軸的額外轉動（真實足部擺動並非純平移），落地時回到下一步的站立姿態
        yaw[swing] = yaw_start + (yaw_end - yaw_start) * _ease(tau) + 4.0 * np.sin(np.pi * tau)
        roll[swing] = roll_start + (roll_end - roll_start) * _ease(tau) + 3.0 * np.sin(np.pi * tau)

        return pos, np.column_stack([yaw, pitch, roll])

    # ------------------------------------------------------------------
    # 由運動學推導 IMU 讀值
    # ------------------------------------------------------------------

    def _kinematics_to_imu(
        self, pos: np.ndarray, euler_deg: np.ndarray, impact_events: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt = 1.0 / self.fs
        rot = Rotation.from_euler("ZYX", euler_deg, degrees=True)

        # 角速度（機體座標）：相鄰兩姿態的相對旋轉向量 / dt，第 k 筆代表 k → k+1 的轉動
        rel = rot[:-1].inv() * rot[1:]
        omega = rel.as_rotvec() / dt
        omega = np.vstack([omega, omega[-1:]])
        gyro_dps = np.degrees(omega)

        # 比力（機體座標）：f = Rᵀ (a_world + g)，單位 g
        vel = np.gradient(pos, dt, axis=0)
        acc_world = np.gradient(vel, dt, axis=0)
        specific_force_world = acc_world / GRAVITY + np.array([0.0, 0.0, 1.0])
        acc_g = rot.inv().apply(specific_force_world)

        # 著地衝擊：沿感測器 z 軸的零均值衰減振盪（模擬鞋底受力瞬態，不改變整體位移）
        n = len(acc_g)
        for idx, amplitude in impact_events:
            length = int(0.08 * self.fs)
            end = min(idx + length, n)
            t = np.arange(end - idx) / self.fs
            acc_g[idx:end, 2] += (amplitude - 1.0) * np.exp(-t / 0.015) * np.cos(2 * np.pi * 30 * t)

        # 感測器雜訊：白雜訊 + 每段固定的殘餘陀螺偏移（韌體開機校正後的殘差量級）
        gyro_bias = self.rng.normal(0.0, 0.2, 3)
        acc_g += self.rng.normal(0.0, 0.01, acc_g.shape)
        gyro_dps += self.rng.normal(0.0, 0.3, gyro_dps.shape) + gyro_bias
        return acc_g, gyro_dps

    # ------------------------------------------------------------------
    # 公開介面
    # ------------------------------------------------------------------

    def generate_walk_session(
        self,
        n_strides: int = 40,
        gait_type: str = "normal",
        custom_fpa: Optional[float] = None,
        custom_eversion: Optional[float] = None,
        fpa_sd: float = 1.8,
        eversion_sd: float = 1.2,
        blend_to: Optional[str] = None,
        blend: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        產生一段連續行走。

        :return: (acc_g (N,3), gyro_dps (N,3), ground_truth_df)
                 ground_truth_df 每列為一步的真值：target_fpa 為該步「站立期」足偏角。
        """
        preset = dict(GAIT_PRESETS.get(gait_type, GAIT_PRESETS["normal"]))
        if blend_to is not None:
            # 兩種樣態之間線性過渡（避免多週軌跡中步長、衝擊等參數在切換樣態時突然跳變）
            target = GAIT_PRESETS[blend_to]
            preset = {k: (1.0 - blend) * v + blend * target[k] for k, v in preset.items()}
        base_fpa = preset["fpa"] if custom_fpa is None else float(custom_fpa)
        base_ev = preset["eversion"] if custom_eversion is None else float(custom_eversion)

        # 逐步參數（含步間自然變異）；多產生一組作為「最後落地站立」的姿態
        fpas = base_fpa + self.rng.normal(0.0, fpa_sd, n_strides + 1)
        evs = base_ev + self.rng.normal(0.0, eversion_sd, n_strides + 1)
        stride_times = np.clip(preset["stride_time"] + self.rng.normal(0.0, 0.03, n_strides), 0.8, 1.6)
        stances = np.clip(preset["stance"] + self.rng.normal(0.0, 0.012, n_strides), 0.55, 0.72)
        lengths = np.clip(preset["length"] + self.rng.normal(0.0, 0.04, n_strides), 0.6, 1.8)
        impacts = np.maximum(1.3, preset["impact"] + self.rng.normal(0.0, 0.2, n_strides))

        # 行進方向緩慢漂移（真實行走不是完美直線；FPA 以每步實際前進方向為準）
        heading = np.cumsum(self.rng.normal(0.0, 1.5, n_strides + 1))
        foot_yaw = heading + self._yaw_sign * fpas
        foot_roll = self._roll_sign * evs

        positions = [np.zeros(3)]
        for k in range(n_strides):
            h = np.radians(heading[k])
            positions.append(positions[-1] + lengths[k] * np.array([np.cos(h), np.sin(h), 0.0]))

        pos_chunks, euler_chunks, impact_events, rows = [], [], [], []
        sample_cursor = 0

        # 開頭站立 1 秒（足部平貼，對應第 0 步的站立姿態）
        n_stand = int(1.0 * self.fs)
        pos_chunks.append(np.tile(positions[0], (n_stand, 1)))
        euler_chunks.append(np.tile([foot_yaw[0], 0.0, foot_roll[0]], (n_stand, 1)))
        sample_cursor += n_stand

        for k in range(n_strides):
            n = int(round(stride_times[k] * self.fs))
            u = np.arange(n) / n
            # 第 0 步從足部平貼開始（開頭站立已涵蓋著地），其餘從腳跟著地開始
            if k == 0:
                u = u[u >= LOADING_END]
            pos, euler = self._stride_kinematics(
                u, stances[k], positions[k], positions[k + 1],
                foot_yaw[k], foot_yaw[k + 1], foot_roll[k], foot_roll[k + 1],
                swing_height=0.10,
            )
            if k > 0:
                impact_events.append((sample_cursor, float(impacts[k])))
            rows.append({
                "stride_index": k,
                "start_time": sample_cursor / self.fs,
                "stride_time": float(stride_times[k]),
                "stance_ratio": float(stances[k]),
                "stride_length": float(lengths[k]),
                "target_fpa": float(fpas[k]),
                "target_eversion": float(evs[k]),
                "gait_type": gait_type,
            })
            pos_chunks.append(pos)
            euler_chunks.append(euler)
            sample_cursor += len(u)

        # 結尾：最後一次著地（負重期）後站立 0.5 秒
        n_land = int(round(LOADING_END * preset["stride_time"] * self.fs))
        u_land = np.linspace(0.0, LOADING_END, n_land, endpoint=False)
        land_pitch = PITCH_HEEL_STRIKE * (1.0 - _ease(u_land / LOADING_END))
        impact_events.append((sample_cursor, float(impacts[-1])))
        n_end = int(0.5 * self.fs)
        pos_chunks.append(np.tile(positions[-1], (n_land + n_end, 1)))
        euler_chunks.append(np.column_stack([
            np.full(n_land + n_end, foot_yaw[-1]),
            np.concatenate([land_pitch, np.zeros(n_end)]),
            np.full(n_land + n_end, foot_roll[-1]),
        ]))

        pos_all = np.vstack(pos_chunks)
        euler_all = np.vstack(euler_chunks)
        acc, gyro = self._kinematics_to_imu(pos_all, euler_all, impact_events)
        return acc, gyro, pd.DataFrame(rows)

    def generate_longitudinal_recovery_stream(
        self,
        weeks: int = 6,
        days_per_week: int = 7,
        strides_per_day: int = 120,
        initial_fpa: float = -13.0,
        target_fpa: float = -5.0,
        initial_eversion: float = 6.0,
        target_eversion: float = 4.5,
        initial_fpa_sd: float = 3.5,
        target_fpa_sd: float = 2.6,
    ) -> List[Dict]:
        """
        產生多週的「合成訓練軌跡」：足偏角依 S 型學習曲線由 initial_fpa 往 target_fpa 移動，
        步間變異同步下降。**此軌跡的形狀與幅度是假設，不是實測或文獻擬合結果**——
        用途是驗證「基線 → 偏差 → 漸退狀態機」在多週資料上的行為，不代表真實學習速度。

        參數取保守值（2026-09-25 調整）：六週改善約 8°、仍略為內八，而不是一路走到族群常態的 +7°
        （舊版 −13° → +6.5°，約 20°，且第 18 天就進入 Phase 3，過於樂觀）；第一週幾乎還沒進步。

        strides_per_day 是每日抽樣步數（真實一天可達數千步），用於控制運算量。
        """
        sessions = []
        total_days = weeks * days_per_week
        normal_fpa = GAIT_PRESETS["normal"]["fpa"]
        for day in range(total_days):
            progress = day / max(total_days - 1, 1)
            learned = 1.0 / (1.0 + np.exp(-6.0 * (progress - 0.5)))
            mean_fpa = initial_fpa + (target_fpa - initial_fpa) * learned
            mean_ev = initial_eversion + (target_eversion - initial_eversion) * learned
            fpa_sd = initial_fpa_sd + (target_fpa_sd - initial_fpa_sd) * learned
            # 步長、步時等其他參數依「足偏角往常態走了多少」同比例過渡
            blend = float(np.clip((mean_fpa - initial_fpa) / (normal_fpa - initial_fpa), 0.0, 1.0))

            acc, gyro, meta = self.generate_walk_session(
                n_strides=strides_per_day,
                gait_type="in_toeing",
                custom_fpa=mean_fpa,
                custom_eversion=mean_ev,
                fpa_sd=fpa_sd,
                blend_to="normal",
                blend=blend,
            )
            sessions.append({
                "day_index": day,
                "week_num": day // days_per_week + 1,
                "day_of_week": day % days_per_week + 1,
                "acc": acc,
                "gyro": gyro,
                "mean_fpa": float(mean_fpa),
                "mean_eversion": float(mean_ev),
                "meta": meta,
            })
        return sessions
