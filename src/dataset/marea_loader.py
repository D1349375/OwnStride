"""
src/dataset/marea_loader.py
===========================
Dataset Benchmark Adapter

MAREA gait database (Khandelwal & Wickstrom 2017) requires a signed
data-release agreement and cannot be downloaded programmatically.
See: https://www.hh.se/english/research/research-projects/caisr/gait-database.html

This module now delegates to WISDMLoader for real human walking data,
which is freely available without registration. When the user obtains
MAREA .mat files manually, load_subject_mat() can parse them directly.

Real data source used: WISDM Activity Recognition v1.1
  - 36 human participants, smartphone accelerometer, ~20 Hz walking data
  - 418,393 real walking measurements
  - Kwapisz et al. (2011), ACM SIGKDD Explorations, 12(2), 74-82
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import io


class MAREALoader:
    """
    Adapter for MAREA .mat files (when manually obtained) or
    falls back to WISDMLoader for real-data benchmark sessions.

    MAREA requires manual registration at Halmstad University.
    WISDM is used as the openly-available real-data substitute.
    """

    def __init__(self, data_dir: str = "data/raw/marea"):
        self.data_dir = data_dir
        self.sample_rate = 128.0

    # ------------------------------------------------------------------
    # MAREA .mat file loading (only if user has manually downloaded)
    # ------------------------------------------------------------------

    def load_subject_mat(self, mat_file_path: str) -> Dict:
        """
        Load a manually downloaded MAREA .mat file.
        Obtain the file from: https://www.hh.se/english/research/...
        """
        if not os.path.exists(mat_file_path):
            raise FileNotFoundError(
                f"MAREA .mat file not found: {mat_file_path}\n"
                "MAREA requires manual registration. Visit:\n"
                "https://www.hh.se/english/research/research-projects/caisr/gait-database.html"
            )
        mat = io.loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
        return mat

    def parse_ankle_signals(self, mat_data: Dict) -> Dict[str, np.ndarray]:
        """Parse left/right ankle (LF/RF) signals from a MAREA mat dict."""
        res = {}
        for key in ["LF", "RF", "treadmill", "outdoor"]:
            if key in mat_data:
                res[key] = mat_data[key]
        return res

    # ------------------------------------------------------------------
    # Real-data benchmark (WISDM as open substitute for MAREA)
    # ------------------------------------------------------------------

    def get_or_create_benchmark_data(
        self,
        subject_id: int = 1,
        activity: str = "indoor_walk",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Obtain a benchmark session of REAL human walking data.

        Priority:
        1. If MAREA .mat file exists locally -> load it (user must obtain manually)
        2. If WISDM real data is available -> use real WISDM walking session
        3. NEVER fall back to synthetic data silently

        Raises
        ------
        RuntimeError if neither MAREA nor WISDM data is available.
        Call WISDMLoader.ensure_downloaded() once to fetch WISDM.
        """
        # --- Priority 1: MAREA mat file ---
        candidate_path = os.path.join(self.data_dir, f"Subject{subject_id}.mat")
        if os.path.exists(candidate_path):
            try:
                mat = self.load_subject_mat(candidate_path)
                if hasattr(mat.get("LF", None), "acc"):
                    acc = np.array(mat["LF"].acc, dtype=np.float64)
                    gyro = np.zeros_like(acc)
                    return acc, gyro, {
                        "source": "MAREA_mat_file",
                        "is_synthetic": False,
                        "path": candidate_path,
                    }
            except Exception:
                pass

        # --- Priority 2: WISDM real data ---
        from src.dataset.wisdm_loader import WISDMLoader
        wisdm = WISDMLoader(data_dir=self.data_dir)
        if wisdm.is_downloaded():
            acc, gyro, meta = wisdm.get_subject_session(
                subject_id=subject_id,
                resample_to_hz=self.sample_rate,
            )
            meta["marea_requested_activity"] = activity
            return acc, gyro, meta

        # --- Priority 3: Raise, do NOT silently use synthetic ---
        raise RuntimeError(
            "No real gait data available.\n"
            "Run: from src.dataset.wisdm_loader import WISDMLoader; "
            "WISDMLoader().ensure_downloaded()\n"
            "This downloads 36-subject real walking accelerometer data (~11 MB)."
        )
